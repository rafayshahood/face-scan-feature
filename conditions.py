# Import necessary libraries
import cv2
import numpy as np
import torch

# Load the trained YOLOv5 model for glasses detection using custom weights
glasses_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/glasses_trained.pt', force_reload=True)
headwear_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/headwear.pt', force_reload=True)
hair_segment_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/hair_segment.pt')  # Replace with your YOLOv5 model path

# Function to check if the face is front-facing
def is_front_facing(landmarks):
    left_eye = landmarks[33]  # Left eye
    right_eye = landmarks[263]  # Right eye
    nose = landmarks[1]  # Nose

    # Calculate the distance between the eyes (horizontal distance)
    eye_dist = np.linalg.norm(np.array([right_eye.x, right_eye.y]) - np.array([left_eye.x, left_eye.y]))

    # Calculate the vertical midpoint between the eyes
    eye_mid_y = (left_eye.y + right_eye.y) / 2

    # Adjust the target position for the nose to be slightly below the eye midpoint
    adjusted_eye_mid_y = eye_mid_y + eye_dist * 0.6  # Allowing a slight downward shift (10% of eye distance)

    # Calculate the horizontal and vertical distances
    nose_midpoint_dist = np.abs(nose.x - (left_eye.x + right_eye.x) / 2)
    vertical_nose_dist = np.abs(nose.y - adjusted_eye_mid_y)

    # Check horizontal and vertical alignment
    front_facing_horizontal = nose_midpoint_dist < eye_dist * 0.2
    front_facing_vertical = vertical_nose_dist < eye_dist * 0.3

    # Return True if both horizontal and vertical conditions are met
    return front_facing_horizontal and front_facing_vertical

    # Return True if the nose is centered between the eyes (indicating a front-facing face)
    # return nose_dist < eye_dist * 0.2


def check_lighting(frame):
    """
    Check the lighting condition of the frame by calculating the average brightness.
    
    :param frame: The frame to check lighting for.
    :return: Boolean indicating if the lighting is adequate.
    """
    # Convert frame to grayscale to measure brightness
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the mean brightness
    mean_brightness = np.mean(gray_frame)
    
    # Print brightness for debugging purposes
    print(f"Average Brightness: {mean_brightness:.2f}")

    # Return True if brightness is above a threshold of 80
    return mean_brightness > 80

def check_glasses(frame):
    """
    Check if the person is wearing glasses using the YOLOv5 model.
    
    :param frame: The frame to check for glasses.
    :return: Boolean indicating if glasses are detected.
    """
    # Run the YOLOv5 model on the frame to detect objects
    results = glasses_model(frame)
    detected_objects = results.pandas().xyxy[0]
    
    # Check if any detected object is labeled as 'with_glasses'
    glasses_detected = detected_objects[detected_objects['name'] == 'with_glasses']
    
    # Return True if glasses are not detected, False otherwise
    return not glasses_detected.empty

def check_headwear(frame):
    """
    Detects headwear (e.g., helmet) using YOLOv5.
    :param frame: The frame to check for headwear.
    :return: Boolean indicating if headwear is detected and corresponding message.
    """
    results = headwear_model(frame)
    detected_objects = results.xyxy[0]


    for det in detected_objects:
        class_id = int(det[5])
        print(class_id)
        class_name = results.names[class_id]
        print(class_name)
        confidence = det[4]  # Confidence score
        print(confidence)


        if class_id == 1 and confidence > 0.65:
            return True
        else:
            return False

def check_hair(frame, face_landmarks, frame_width, frame_height):
    """
    Detect hair using YOLOv5 and check if it is inside the defined ellipse region.
    
    :param frame: The current frame captured from the webcam.
    :param face_landmarks: Detected face landmarks from MediaPipe.
    :param frame_width: Width of the frame.
    :param frame_height: Height of the frame.
    :return: Boolean indicating if hair is detected inside the ellipse region.
    """
    # Get landmarks for the eyes
    left_eye_idxs = [33, 160, 158, 133, 153, 144]
    right_eye_idxs = [362, 385, 387, 263, 373, 380]
    landmarks = np.array([(int(pt.x * frame_width), int(pt.y * frame_height)) for pt in face_landmarks.landmark])

    # Calculate the ellipse around the eyes
    left_eye = landmarks[left_eye_idxs].mean(axis=0)
    right_eye = landmarks[right_eye_idxs].mean(axis=0)
    midpoint = ((left_eye + right_eye) / 2).astype(int)
    eye_distance = np.linalg.norm(left_eye - right_eye)
    ellipse_width = int(eye_distance * 1.9)
    ellipse_height = int(eye_distance * 0.8)

    # Shift the midpoint upwards
    shifted_midpoint = (midpoint[0], midpoint[1] - int(ellipse_height * 0.3))

    # Crop the frame for hair detection
    crop_x1 = max(0, shifted_midpoint[0] - ellipse_width // 2)
    crop_x2 = min(frame_width, shifted_midpoint[0] + ellipse_width // 2)
    cropped_frame = frame[:, crop_x1:crop_x2]

    # Run YOLOv5 hair detection on cropped frame
    results_yolo = hair_segment_model(cropped_frame)

    # Check for hair inside the ellipse
    for *xyxy, conf, cls in results_yolo.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        x1_full = x1 + crop_x1
        x2_full = x2 + crop_x1
        label = hair_segment_model.names[int(cls)]
        
        if label == 'hair':  # Check if the detected object is hair
            # Calculate a point 11% below the center of the bounding box
            box_center = ((x1_full + x2_full) // 2, (y1 + y2) // 2)
            box_height = y2 - y1
            point_below_center = (box_center[0], box_center[1] + int(box_height * 0.09))

            # Check if any bounding box corner or 11% below center point is inside the ellipse
            corners = [(x1_full, y1), (x2_full, y1), (x1_full, y2), (x2_full, y2)]
            hair_inside_ellipse = False

            for corner in corners:
                distance = np.linalg.norm(np.array(corner) - np.array(shifted_midpoint))
                if distance <= ellipse_width // 2:
                    hair_inside_ellipse = True
                    break

            below_center_distance = np.linalg.norm(np.array(point_below_center) - np.array(shifted_midpoint))
            if below_center_distance <= ellipse_width // 2:
                hair_inside_ellipse = True

            if hair_inside_ellipse:
                return True

    return False




def is_face_inside_oval(face_landmarks, center, axes, frame_width, frame_height):
    """
    Check if the face is inside the oval (ellipse) based on facial landmarks.
    
    :param face_landmarks: The detected face landmarks.
    :param center: The center of the ellipse.
    :param axes: The axes (radii) of the ellipse.
    :param frame_width: The width of the frame.
    :param frame_height: The height of the frame.
    :return: Boolean indicating whether the face is inside the oval.
    """
    # Normalize landmark coordinates based on the frame size
    key_landmarks = [33, 263, 1]  # landmarks: left eye, right eye, and nose
    for lm_index in key_landmarks:
        lm = face_landmarks.landmark[lm_index]
        
        # Convert normalized coordinates to pixel coordinates
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        
        # Ellipse equation: ((x - h)^2 / a^2) + ((y - k)^2 / b^2) <= 1
        if (((x - center[0]) ** 2) / (axes[0] ** 2)) + (((y - center[1]) ** 2) / (axes[1] ** 2)) > 1:
            return False  # If any key point is outside the ellipse, return False
    
    return True  # If all key points are inside, return True


def evaluate_conditions(frame, frame2, faces, center, axes):
    """
    Evaluate various conditions including lighting, glasses detection, and face positioning.
    
    :param frame: The frame captured from the webcam.
    :param faces: List of detected faces from MTCNN.
    :param center: Center of the oval.
    :param axes: Axes dimensions of the oval.
    :return: Tuple (conditions_met, prompt) where conditions_met is a boolean and prompt is a message.
    """
    # Perform lighting check
    adequate_lighting = check_lighting(frame)
    if not adequate_lighting:
        return False, "Increase lighting"


    # If no face is detected, prompt the user to bring their face inside the oval
    if not faces:
        return False, "Please bring your face inside the oval"


    frame_height,frame_width = frame.shape[:2]  # Extract the frame dimensions
    # Check for glasses
    # # Perform headwear detection (this comes after glasses detection)
    headwear_detected = check_headwear(frame2)
    glasses_detected = check_glasses(frame2)
    # # Evaluate conditions for each detected face
    # Evaluate conditions for each detected face
    for face_landmarks in faces:
        # Check if the face is inside the oval
        face_inside_oval = is_face_inside_oval(face_landmarks, center, axes, frame_width, frame_height)
        # Check if the face is front-facing
        front_facing = is_front_facing(face_landmarks.landmark)

        if not face_inside_oval:
            return False, "Position your face inside the oval"
        if not front_facing:
            return False, "Make your face front-facing"
        elif headwear_detected:
            return False, "Please remove headwear"
        elif glasses_detected:
            return False, "Please remove glasses"


        # Hair detection condition
        hair_detected = check_hair(frame2, face_landmarks, frame_width, frame_height)
        if hair_detected:
            return False, "Please remove hair from the defined region."


    return True, "Conditions met - Take a photo"
