# YOLO Object Detection Project

This project is a web application for object detection using YOLO (You Only Look Once) models, implemented with Python and Streamlit. It supports three versions of YOLO (v3, v5, and v8) for detecting objects in images.

## Features

- **YOLOv3**: Traditional object detection with pre-defined class names.
- **YOLOv5**: Enhanced performance using Torch Hub.
- **YOLOv8**: Advanced detection with the latest YOLO capabilities.
- **Interactive UI**: Upload an image, choose a YOLO model, and view detection results.

## Project Structure

- **Sourcecode.py**: Main Python script containing all detection logic and Streamlit app.
- **YOLO Models**: Supports weights and configurations for YOLOv3, YOLOv5, and YOLOv8.
- **Class Names File**: A text file (e.g., `yolo.txt`) containing class names for object detection.

## Requirements

To run this project, you need:

- Python 3.x
- Libraries: `numpy`, `cv2`, `Pillow`, `streamlit`, `torch`, `ultralytics`
- Pre-trained YOLO weights and configuration files:
  - `yolov3.cfg`, `yolov3.weights`
  - `yolov5x.pt`
  - `yolov8x.pt`
Please download the required files from the rspective Official GitHub Repositories.

## How to Use

1. Clone this repository.
2. Install required dependencies:
   ```bash
   pip install numpy opencv-python pillow streamlit torch ultralytics
   ```
3. Place the required YOLO weights and configuration files in the same directory.
4. Run the Streamlit app:
   ```bash
   streamlit run Sourcecode.py
   ```
5. Upload an image through the app, select a YOLO model, and view the detection results.

## Customization

- Modify the detection thresholds (e.g., confidence levels) in the script.
- Add support for more YOLO models or fine-tune existing weights.

## Future Work

- Integrate video or real-time detection.
- Enhance the UI for more functionality.
- Optimize for larger datasets or applications.



