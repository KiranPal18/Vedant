import cv2
import numpy as np
import math
import os
from Helper import KalmanTracker, CentroidTracker, contour_centroid

class Solution:
    def __init__(self):
        # We store unique IDs here so we don't double-count cars.
        self.counted_ids = set()
        
    def determine_mode(self, cap):
        """
        The 'Smart Switch'.
        It looks at the first frame to decide if we are in 'Night Mode' or 'Day Mode'.
        Logic: Night videos usually have low color saturation (grayish) and low brightness.
        """
        # Read the very first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind video back to start
        
        if not ret: return "day" # Safety fallback
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # --- BORDER MASKING ---
        # Crucial Trick: We only analyze the borders of the image.
        # Why? At night, headlights in the center make the avg brightness high.
        # The borders (sky/trees) tell the true story of the ambient light.

        mask = np.zeros((h, w), dtype="uint8")
        margin_x = int(w * 0.20)
        margin_y = int(h * 0.20)
        
        # Draw rectangles on the edges to calc the mean there
        cv2.rectangle(mask, (0, 0), (w, margin_y), 255, -1)       # Top
        cv2.rectangle(mask, (0, h-margin_y), (w, h), 255, -1)     # Bottom
        cv2.rectangle(mask, (0, 0), (margin_x, h), 255, -1)       # Left
        cv2.rectangle(mask, (w-margin_x, 0), (w, h), 255, -1)     # Right
        
        # Calculate stats only in the masked area
        saturation_mean = cv2.mean(hsv[..., 1], mask=mask)[0]
        brightness_mean = cv2.mean(hsv[..., 2], mask=mask)[0]
        
        # Thresholds: Less than 25 saturation (very gray) or < 80 brightness = Night
        if saturation_mean < 25 or brightness_mean < 80:
            print(f"🌙 Mode: NIGHT (Sat: {saturation_mean:.1f}, Val: {brightness_mean:.1f})")
            return "night"
        else:
            print(f"☀️ Mode: DAY (Sat: {saturation_mean:.1f}, Val: {brightness_mean:.1f})")
            return "day"

    def adjust_gamma(self, image, gamma=1.0):
        """
        Night Vision Goggles.
        It artificially boosts the brightness of dark pixels (Gamma Correction).
        This helps us see black cars that would otherwise blend into the dark road.
        """
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_roi_mask(self, frame):
        """
        The 'Blinders'.
        We ignore the sky and distant trees by applying a trapezoid mask.
        If we don't look at the trees, we don't count shaking leaves as cars.(was a great issue druing testing)
        """

        h, w = frame.shape[:2]
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        # Define a trapezoid focusing on the road
        pts = np.array([
            [int(w * 0.20), int(h * 0.15)],  # Top Left
            [int(w * 0.85), int(h * 0.15)],  # Top Right
            [int(w * 0.95), h],              # Bottom Right
            [int(w * 0.10), h]               # Bottom Left
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return cv2.bitwise_and(frame, frame, mask=mask)

    # ======================================================
    #                 NIGHT MODE ENGINE
    # ======================================================
    def run_night_engine(self, cap, w, h):
        """
        Strategy: Use Kalman Filters to smooth jumpy headlight detections.
        """
        # Night objects are often just small headlights, so we lower the area requirement.
        min_blob_area = 250 
        min_vehicle_area = 800
        trip_y = int(h * 0.55) # The finish line
        
        # KalmanTracker handles the "jumping" detections common at night

        tracker = KalmanTracker(max_disappeared=30, max_distance=150)
        
        # MOG2 Subtractor: We use a lower threshold (30) to catch faint lights

        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=30, detectShadows=True)
        
        # Kernels for cleaning up the image
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Special vertical kernel to merge separate headlights into one car
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15)) 
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Enhance the image
            frame_blurred = cv2.GaussianBlur(frame, (9, 9), 2)
            frame_masked = self.apply_roi_mask(frame_blurred)
            # Boost brightness so we can see dark cars
            frame_processed = self.adjust_gamma(frame_masked, gamma=1.5)
            
            # 2. Detect Motion
            fg = bg_subtractor.apply(frame_processed)
            # Cut off shadows (gray pixels)
            _, fg = cv2.threshold(fg, 210, 255, cv2.THRESH_BINARY)
            
            # 3. Clean Noise
            fg = cv2.erode(fg, kernel_erode, iterations=1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_vertical, iterations=2)   
            fg = cv2.dilate(fg, kernel_dilate, iterations=1)

            # 4. Find Blobs
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_centroids = []
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_blob_area: continue    
                if area < min_vehicle_area: continue 
                
                # Aspect Ratio Check: Ignore tall thin lines (probably reflections and noise)
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw / float(bh) < 0.25: continue 
                
                cent = contour_centroid(cnt)
                if cent: current_centroids.append(cent)

            # 5. Track
            objects_coords, object_instances = tracker.update(current_centroids)

            for obj_id, (cx, cy) in objects_coords.items():
                kf_vehicle = object_instances[obj_id]
                
                # Speed Check: If it's not moving, it's probably a streetlight reflection.
                vx, vy = kf_vehicle.get_velocity()
                speed = math.hypot(vx, vy)
                if speed < 1.0: continue
                # Stability Check: Wait 5 frames before trusting a new object
                if kf_vehicle.age < 5: continue 

                # Counting Rule: Must cross the line moving DOWN
                if len(kf_vehicle.history) > 3:
                    prev_y = kf_vehicle.history[-3][1]
                    if prev_y < trip_y and cy > trip_y: 
                         if obj_id not in self.counted_ids:
                            self.counted_ids.add(obj_id)

    # ======================================================
    #                   DAY MODE ENGINE
    # ======================================================
    def run_day_engine(self, cap, w, h):
        """
        Strategy: Use heavy filtering (Shape, Solidity, Speed) to ignore trees/bushes.
        """
        min_area = 750            
        vehicle_area_thresh = 1500 
        trip_y = int(h * 0.55)
        buffer_y = trip_y - 20          
        border_margin = 50
        
        # Day footage is clean, so we use the simpler CentroidTracker
        tracker = CentroidTracker(max_disappeared=20, max_distance=150)
        # High threshold (60) prevents swaying leaves from being detected (was an issue due to wind)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=60, detectShadows=True)
        
        start_positions = {} # To track where objects spawned
        object_dims = {}     # To store width/height for shape analysis
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            fg = bg_subtractor.apply(frame)
            
            # Anti-Flash: If >30% of screen changes at once (e.g., camera auto-adjust), skip frame (overhead cloud and random jitter)
            if cv2.countNonZero(fg) > (w * h * 0.30): continue 
                
            _, fg = cv2.threshold(fg, 240, 255, cv2.THRESH_BINARY)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open, iterations=1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            fg = cv2.dilate(fg, kernel_close, iterations=1)

            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_centroids = []
            frame_dims = {}

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area: continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                # Solidity Check:
                # Cars are solid rectangles. Bushes are "messy" with holes.
                # If the blob area is much smaller than its convex hull, it's a bush.
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    if solidity < 0.4: continue 

                cent = contour_centroid(cnt)
                if cent is not None:
                    current_centroids.append(cent)
                    frame_dims[cent] = (bw, bh)

            objects = tracker.update(current_centroids)

            for object_id, centroid in objects.items():
                cx, cy = centroid
                if centroid in frame_dims:
                    object_dims[object_id] = frame_dims[centroid]

                # Border Logic: 
                # Only trust objects that enter from the edge of the screen.
                # Objects appearing in the middle are usually glitches ("Ghosts").
                if object_id not in start_positions:
                    in_margin = (cx < border_margin) or (cx > w - border_margin) or \
                                (cy < border_margin) or (cy > h - border_margin)
                    if in_margin: start_positions[object_id] = (cx, cy) 
                    else: start_positions[object_id] = None 
                
                if start_positions[object_id] is None: continue 

                frames_alive = len(tracker.history.get(object_id, []))
                start_x, start_y = start_positions[object_id]
                net_disp = math.hypot(start_x - cx, start_y - cy)
                
                # Speed Normalization:
                # A car far away moves slowly in pixels, but fast relative to its size.
                # A bush moves slowly in pixels AND slowly relative to its size.
                current_w, _ = object_dims.get(object_id, (1, 1))
                pixel_speed = net_disp / frames_alive if frames_alive > 0 else 0
                norm_speed = pixel_speed / current_w if current_w > 0 else 0
                
                # If it moves > 15% of its width per frame, it's truly fast.
                is_fast_relative = norm_speed > 0.15

                # 1. Stationary Filter: If alive long but moved little -> NOISE
                if frames_alive > 15 and net_disp < 30: continue 

                # 2. Path Efficiency: Real cars move straight. Noise jitters back and forth.
                hist = tracker.history[object_id]
                total_path = sum([math.hypot(hist[i][0]-hist[i-1][0], hist[i][1]-hist[i-1][1]) for i in range(1, len(hist))])
                efficiency = net_disp / total_path if total_path > 0 else 0
                if frames_alive > 10 and efficiency < 0.35: continue 
                
                # 3. Shape Logic:
                # If it's a wide blob (like a car) but moving slowly relative to size, 
                # it's probably just a wide bush. Reject it.
                area_approx = current_w * object_dims.get(object_id, (1,1))[1]
                is_valid = True
                if area_approx < 1500:
                     aspect_ratio = current_w / float(object_dims.get(object_id, (1,1))[1])
                     if aspect_ratio > 0.7 and not is_fast_relative:
                         is_valid = False
                if not is_valid: continue

                # --- COUNTING RULES ---
                should_count = False
                
                # Rule A: Buffer Zone Crossing (Standard)
                # Did it cross from below the line to above the buffer zone?
                was_below = len(hist) >= 2 and hist[-2][1] > trip_y
                is_above_buffer = cy < buffer_y
                if was_below and is_above_buffer: should_count = True
                
                # Rule B: Pure Distance 
                if net_disp > 100: should_count = True
                
                # Rule C: Fast Mover (Save Distant Cars)
                # If it's small but zooming, count it early.
                if is_fast_relative and net_disp > 20: should_count = True
                
                if should_count:
                    if object_id not in self.counted_ids:
                        self.counted_ids.add(object_id)

    def forward(self, video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Decide which engine to use
        mode = self.determine_mode(cap)
        
        try:
            if mode == "night":
                self.run_night_engine(cap, w, h)
            else:
                self.run_day_engine(cap, w, h)
        except Exception as e:
            print(f"Processing warning: {e}")
        finally:
            cap.release()
            
        return len(self.counted_ids)