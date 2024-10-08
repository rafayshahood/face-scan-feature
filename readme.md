
# Face Scan Feature

This repository contains a face scan feature that uses FastAPI as the backend and various models such as YOLOv5 and MediaPipe for detecting facial landmarks, headwear, glasses, and other conditions like mouth being closed.

## Table of Contents
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Testing the Application](#testing-the-application)
- [Common Issues](#common-issues)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Facial Landmark Detection**: Using MediaPipe to detect face and key facial landmarks.
- **Headwear and Glasses Detection**: Using YOLOv5 models for headwear and glasses detection.
- **Condition Validation**: Detecting whether the mouth is closed, and validating if the face is inside a defined oval region.
- **Backend with FastAPI**: API-based architecture for efficient processing of frames.

---

## Setup Instructions
## Python 3.10 and Tkinter Installation Instructions

Before proceeding with the project setup, make sure you have Python 3.10 and Tkinter installed on your system. Follow the instructions below based on your operating system:

### macOS
For macOS, you can use **Homebrew** to install Python and Tkinter:

1. **Install Python 3.10 and tkinter**:
   ```bash
   brew install python@3.10
   brew install python-tk@3.10
   python3.10 --version

### Ubuntu/Linux
1. **Install Python 3.10 and tkinter**:
   ```bash
   sudo apt update
   sudo apt install python3.10
   sudo apt-get install python3-tk
   python3.10 --version



### 1. Clone the Repository
Start by cloning the repository from GitHub:
```bash
git clone https://github.com/rafayshahood/face-scan-feature.git
cd face_scan
```

### 2. Set Up a Virtual Environment
It is recommended to create and use a virtual environment to manage project dependencies:

#### Using Python `venv`:
```bash
python3.10 -m venv face_scan_env
source face_scan_env/bin/activate  # On Windows: face_scan_env\Scripts\activate
```

### 3. Install Project Dependencies
Once the virtual environment is active, install the required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Set Up the FastAPI Backend
Make sure the backend server is running to handle requests for condition checks, headwear, and glasses detection.

#### Start the FastAPI Server:
```bash
cd fastapi_backend
uvicorn main:app --reload
```
The API will be running at `http://127.0.0.1:8000`.

---

## Testing the Application
1. **Ensure the Backend is Running**:
   Ensure the FastAPI server is running on the correct port.
   
2. **Run the Face Scan Feature**:
   If the backend is running and you have the face scan feature from main directory. Inside the face_scan run:
   ```bash
   python main.py
   ```

3. **View Results**:
   A pop up window will open with real time webcam which will check multiple conditions (e.g. checks for headwear, glasses, mouth open/close,lighting, face inside oval and front facing) that will be returned as responses from the API.

---

## Common Issues

1. **Missing Dependencies**:
   If you encounter issues related to missing libraries, ensure all dependencies are installed using the `requirements.txt` file.

2. **Python Version**:
   This project requires Python 3.10. Ensure you are using the correct Python version by running:
   ```bash
   python --version
   ```

3. **FastAPI or Uvicorn Errors**:
   Ensure that FastAPI and Uvicorn are installed and the correct version is being used. Check the `requirements.txt` file.

---

## Technologies Used

- **FastAPI**: Backend framework for creating the API.
- **YOLOv5**: Object detection models used for headwear and glasses detection.
- **MediaPipe**: Facial landmark detection library for detecting face positions and features.
- **OpenCV**: Used for image processing and handling webcam/video input.
- **PyTorch**: Framework used for deep learning models like YOLOv5.

---

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any changes or improvements.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
