# Import necessary libraries
import cv2
import tkinter as tk
# from conditions import evaluate_conditions
from loadingFrame import show_loading_frame
from displayFrame import process_display_frame
from maskedFrame import create_masked_frame
from countdown import countdown_sequence  # Import the countdown logic
from captureButton import draw_button, button_callback, update_button_state, is_button_pressed  # Import button logic
import mediapipe as mp
import time
import requests


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True  )
# Initialize the MTCNN detector for face and landmark detection

# Fetch screen dimensions using tkinter for dynamic frame sizing
root = tk.Tk()
screen_width = root.winfo_screenwidth()  # Get screen width
screen_height = root.winfo_screenheight()  # Get screen height
root.destroy()


frame_size = (int(640), int(480))  # 40% width, 100% height

# Ratio for the oval size in the frame
oval_size_ratio = (0.4, 0.8)

# loadscreen duration
duration = 3
start_time = time.time()  # Record the start time of the loading sequence

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set webcam width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set webcam height

# Check if webcam opens successfully
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("Press '1' to take a picture, '2' to quit.")


# Create a named window for the webcam display
cv2.namedWindow('Webcam')
cv2.setMouseCallback('Webcam', button_callback)  # Set callback for button click detection

def evaluate_conditions_via_api(frame):
    """
    Send the frame to the FastAPI backend for condition evaluation.
    """
    _, img_encoded = cv2.imencode('.jpg', frame)
    try:
        response = requests.post(
            'http://127.0.0.1:8000/evaluate_conditions/',
            files={"file": img_encoded.tobytes()}  # Only sending the image now
        )
        response.raise_for_status()

        result = response.json()
        conditions_met = result.get('conditions_met', False)
        prompt = result.get('prompt', "Unknown error")
        return conditions_met, prompt
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return False, "API call failed"



def process_frame(frame):
    """
    Process the captured frame by resizing, applying a mask, detecting faces, and checking conditions.
    
    :param frame: The current frame captured from the webcam.
    :return: Processed display frame, condition check result, success flag, prompt, oval center, and axes.
    """

    # Create an oval mask around the face region
    original_frame = frame.copy()
    masked_frame, center, axes = create_masked_frame(frame, oval_size_ratio)


    # Convert the image to RGB for MTCNN face detection
    # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # img_rgb2 = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

    # faces = detector_mtcnn.detect_faces(img_rgb)

    # Perform face detection with MediaPipe
    # faces = detect_face_via_api(frame)

    # result = face_mesh.process(img_rgb)
    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)


    # # Extract face landmarks
    # faces = []
    # if result.multi_face_landmarks:
    #     for face_landmarks in result.multi_face_landmarks:
    #         faces.append(face_landmarks)

    # Evaluate conditions such as lighting, face positioning, etc.
    # conditions_met, prompt = evaluate_conditions(frame, original_frame, faces, center, axes)
    conditions_met, prompt = evaluate_conditions_via_api(frame)


    if time.time() - start_time > duration:
        # Process the frame for display with conditions results
        display_frame = process_display_frame(frame,center, axes, conditions_met, prompt)

        # Update and draw the button based on whether conditions are met
        update_button_state(conditions_met)

        display_frame = draw_button(display_frame, conditions_met)
    else:
         display_frame = None

    return display_frame, conditions_met, True, prompt, center, axes

    

# Main loop to capture and process frames from the webcam
while True:
    ret, frame = cap.read()  # Capture frame-by-frame from webcam
    if not ret:
        print("Error: Failed to capture image.")
        break

    if time.time() - start_time < duration:
        # Display loading screen before starting the webcam feed
        show_loading_frame(frame_size, oval_size_ratio)
        

    # Process the captured frame
    display_frame, conditions_met, process_success, prompt, center, axes = process_frame(frame)
    if time.time() - start_time > duration:
        cv2.imshow('Webcam', display_frame)  # Display the display frame

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    if (key == ord('1') and conditions_met) or (is_button_pressed() and conditions_met):
        # Trigger the countdown sequence when conditions are met and '1' or button is pressed
        countdown_sequence(cap, frame, center, axes, process_frame, frame_size)

    elif key == ord('2'):
        # Quit the program if '2' is pressed
        print("Quitting...")
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
