# Vehicle Count Challenge - Adaptive Computer Vision Solution

## 📝 Overview
This project is a solution for the Vehant Technologies 2-Day Hackathon. It addresses the challenge of counting vehicles moving away from a static camera using **classical computer vision techniques** exclusively.

Per the challenge constraints, **no deep learning models** (YOLO, SSD, etc.) were used. Instead, we engineered an **Adaptive Hybrid Engine** that intelligently switches between two specialized processing pipelines (Day vs. Night) to handle varying lighting conditions robustly.

## 📂 Submission Structure
The submission is modularized for clarity and maintainability:

* **`main.py`**: The entry point containing the `Solution` class and the mandatory `forward()` method.
* **`trackers.py`**: A helper module encapsulating the mathematical logic for the `KalmanTracker` and `CentroidTracker` classes.
* **`requirements.txt`**: A list of standard Python library dependencies.
* **`README.md`**: This documentation file.

## ⚙️ Setup & Execution

### Prerequisites
* Python 3.x
* PIP (Python Package Installer)

### Installation
Open your terminal in the project directory and run:

```bash
pip install -r requirements.txt

```

### How to Run

1. **Execute the Script** Run the main file. The script acts as a standalone executable.
```bash
python main.py

```


2. **Provide Input** The script will prompt you for the video file path. Paste the path (relative or absolute) and press Enter.
```text
Please enter the path to the video file:
Dataset/Dataset/vehant_hackathon_video_1.avi

```


3. **View Output** The script will process the video and output the final integer count.
```text
Final Vehicle Count: 42

```

## 🧠 Methodology & Algorithm

The core innovation of this solution is the **Adaptive Scene Analysis**, which allows the system to "understand" the environment before processing it.

### 1. Scene Analysis (The "Smart Switch")

Standard brightness checks fail in traffic analysis because headlight glare at night often inflates the average pixel intensity, mimicking daylight.

* **Technique:** We analyze only the outer 20% borders of the first frame, applying a mask to the center to ignore direct headlight glare.
* **Metrics:** We calculate:
* **Saturation (Color):** Night/IR footage is typically monochrome (low saturation).
* **Brightness (Value):** Ambient light levels in the sky/environment.


* **Decision:** If `Saturation < 25` or `Brightness < 80`, the **Night Engine** is activated. Otherwise, the **Day Engine** is used.

### 2. Day Mode Engine (Precision & Speed)

In daylight, objects are distinct, but environmental noise (swaying trees, shadows) is significant.

* **Detection:** Background Subtraction (MOG2) with shadow detection.
* **Tracking:** Centroid Tracking. Since visibility is clear, geometric distance matching is computationally efficient and accurate.
* **Noise Filtering:**
* **Path Efficiency:** Real vehicles move in straight lines; noise "jitters" in place. We filter out objects with low displacement efficiency.
* **Solidity Check:** We discard "hollow" contours (like leafy bushes) by comparing contour area to convex hull area.



### 3. Night Mode Engine (Robustness & Prediction)

At night, visibility is low, and vehicles often appear fragmented (e.g., two disconnected headlights).

* **Preprocessing:** Gamma Correction (1.5x) is applied to non-linearly boost dark regions without washing out highlights.
* **Tracking:** Kalman Filter. Unlike simple centroid tracking, the Kalman Filter maintains an internal state (velocity/position). If a vehicle enters a shadow and detection is lost, the filter predicts its new location, ensuring continuity and preventing double-counting.
* **Flash Suppression:** We detect global brightness spikes (camera auto-exposure adjustment) and skip those frames to prevent false positives.

## 📌 Assumptions & Design Choices

* **Static Camera:** The solution assumes a fixed camera position. No ego-motion compensation is applied.
* **Directionality:** The logic counts vehicles moving away from the camera (downward flow).
* **Validation Strategy:** In the absence of an automated ground truth file, the algorithm was validated via manual human verification. Representative clips were reviewed frame-by-frame to tune thresholds (e.g., `min_blob_area`, `trip_y`) to match human perception.

## 🛠 Dependencies

* **opencv-python:** For image processing and computer vision tasks.
* **numpy:** For matrix operations and efficient masking.

```