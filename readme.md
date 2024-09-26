# Face Scan and 3D Model Generation

## Overview
This project implements a face scan feature using OpenCV and MediaPipe, combined with YOLOv5 for multiple detection. It integrates with a 3D Model App to generate `.obj` and `.gltf` files based on the captured image. The face scan checks various conditions such as lighting, headwear, glasses, and hair before capturing the image.

## Features
- Face detection and facial landmark tracking using MediaPipe.
- YOLOv5 for hair, glasses, and headwear detection.
- Integration with a 3D model generation pipeline.
- Real-time webcam interface.
- Capturing photo when all coniditions met
- `.obj` file preview with Accept/Decline functionality.

## Setup Instructions

### Prerequisites
- Python 3.8+
- OpenCV
- MediaPipe
- YOLOv5
- Open3D (for .obj file preview)

### Installation
1.  python3 -m venv venv
   source venv/bin/activate

2. Clone the repository:
   ```bash
   git clone https://github.com/rafayshahood/face-scan-feature.git
   cd face-scan-feature

3. Download the SF-Pro.dmg font file: The SF-Pro.dmg file exceeds GitHub's size limit. Please download it     
   from Google Drive: https://drive.google.com/file/d/1cqpi7NKVau0ZCeupeJjcHmL3sbtBoHkX/view?usp=sharing and place it in the fonts/ directory.
   
3. pip install -r requirements.txt


4. Run the face scan feature:
    python main.py


# Project Structure

face-scan-3d-model/
│
├── main.py               # Main script to run the face scan feature
├── README.md             # Project documentation
├── requirements.txt      # Required libraries for the project
├── captureButton.py      # Handles the capture button logic
├── countdown.py          # Handles countdown logic and image capture
├── conditions.py         # Defines conditions for the face scan (lighting, hair, etc.)
├── displayFrame.py       # Handles frame processing and displaying
├── loadingFrame.py       # Handles frame processing and displaying of loading frame
├── maskedFrame.py        # Handles frame processing 
├── effects.py            # Visual effects for the face scan
├── utilities.py          # Utility functions (e.g., scan effect)
