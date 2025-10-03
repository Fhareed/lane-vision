# Lane-Vision

Lane-Vision is a simple perception system for autonomous navigation.  
It combines computer vision and deep learning to detect lanes and track obstacles in real time.

##  Features
- Lane Detection** using CV:
  
  - HSV color filtering (white/yellow lanes)
  - Canny edge detection
  - Hough Transform for line detection
  - Lane averaging + smoothing for stability
    
- Obstacle Detection with YOLOv8 (Ultralytics)
- Object Tracking using SORT (Kalman Filter + Hungarian Algorithm)
- Apple Silicon Acceleration (`device="mps"`) supported
- Real-time overlay of **lanes + detected vehicles + stable IDs**


##  Demo
Example output (lanes + tracked vehicles):  

> ![Demo Screenshot]
> <img width="1800" height="1169" alt="image" src="https://github.com/user-attachments/assets/fd914fa4-ca35-4f33-af77-6d84c9361187" />
 



##  Installation

Clone the repository:
```bash
git clone https://github.com/Fhareed/lane-vision.git
cd lane-vision
```

Install dependencies:

pip install -r requirements.txt

Usage

Run lane & obstacle detection on your video:

python main.py

Press Q to quit the video window.
