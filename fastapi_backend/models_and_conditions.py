import torch
import numpy as np
import cv2
from utillities import resize_and_pad_image
from utillities import letterbox_image
from ultralytics import YOLO

glasses_model = None
headwear_model = None
hair_model = None

def load_models():
    global glasses_model, headwear_model, hair_model
    if glasses_model is None:
        glasses_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/glasses_trained.pt', force_reload=True)
    if headwear_model is None:
        headwear_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/headwear2.pt', force_reload=True)
    if hair_model is None:
        hair_model = YOLO('./models/hairseg-v8.pt')

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
selected_landmarks = [116, 139, 71, 68, 104, 67, 109, 10, 338, 297, 333, 298, 301, 368, 345, 195]

def detect_hair_in_forehead(image_np, overlap_threshold=1):
    """
    Detects hair in the forehead region based on selected landmarks and YOLO hair segmentation.
    The image is resized before performing both landmark detection and hair segmentation.
    """

    # Resize the frame with letterboxing
    letterbox_frame = letterbox_image(image_np)

    # Convert the resized frame to RGB for MediaPipe
    image_rgb = cv2.cvtColor(letterbox_frame, cv2.COLOR_BGR2RGB)

    # Detect face and landmarks using MediaPipe
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image_rgb)

    # Run YOLO model for hair segmentation
    results_yolo = hair_model(letterbox_frame)

    # Initialize the hair mask as None
    hair_mask = None

    # Check if there are any masks in the YOLOv8 results
    if results_yolo[0].masks is not None and len(results_yolo[0].masks.data) > 0:
        # Extract the mask from the results
        hair_mask = results_yolo[0].masks.data[0].cpu().numpy().astype(np.uint8)
        hair_mask = cv2.resize(hair_mask, (image_np.shape[1], image_np.shape[0]))  # Resize to match original frame size

        # Optionally dilate the hair mask to make it larger
        # kernel = np.ones((5, 5), np.uint8)
        # hair_mask = cv2.dilate(hair_mask, kernel, iterations=2)

    # Ensure there are face landmarks and a hair mask
    if results.multi_face_landmarks and hair_mask is not None:
        for face_landmarks in results.multi_face_landmarks:
            # Collect selected landmark points
            points = []
            h, w, _ = image_np.shape
            for idx in selected_landmarks:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append((x, y))

            # Convert to numpy array for shape drawing
            points = np.array(points, dtype=np.int32)

            # Create an empty mask for the polygonal region
            shape_mask = np.zeros_like(hair_mask)

            # Fill the polygonal region inside the selected landmarks
            cv2.fillPoly(shape_mask, [points], 1)

            # Check for overlap between the hair mask and the shape mask
            overlap = cv2.bitwise_and(hair_mask, shape_mask)

            # Calculate the percentage of hair inside the selected shape
            hair_pixels = np.sum(hair_mask)
            overlap_pixels = np.sum(overlap)

            if hair_pixels > 0:
                overlap_percentage = (overlap_pixels / hair_pixels) * 100

                # Return True if hair is detected inside the shape
                if overlap_percentage > overlap_threshold:
                    return True

    # Return False if no hair is detected
    return False



def is_mouth_closed(face_landmarks, frame_height):
    upper_lip = face_landmarks[13]  # Upper lip
    lower_lip = face_landmarks[14]  # Lower lip
    lip_distance = abs(upper_lip['y'] - lower_lip['y']) * frame_height
    mouth_closed_threshold = 2
    return lip_distance < mouth_closed_threshold


def check_lighting(image_np):
    gray_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_frame)
    return mean_brightness > 80


def is_front_facing(face_landmarks):
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]
    nose = face_landmarks[1]

    eye_dist = np.linalg.norm(np.array([right_eye['x'], right_eye['y']]) - np.array([left_eye['x'], left_eye['y']]))
    eye_mid_y = (left_eye['y'] + right_eye['y']) / 2
    adjusted_eye_mid_y = eye_mid_y + eye_dist * 0.6

    nose_midpoint_dist = np.abs(nose['x'] - (left_eye['x'] + right_eye['x']) / 2)
    vertical_nose_dist = np.abs(nose['y'] - adjusted_eye_mid_y)

    front_facing_horizontal = nose_midpoint_dist < eye_dist * 0.15
    front_facing_vertical = vertical_nose_dist < eye_dist * 0.2

    return front_facing_horizontal and front_facing_vertical


def is_face_inside_oval(face_landmarks, center, axes, frame_width, frame_height):
    key_landmarks = [33, 263, 1]
    for lm_index in key_landmarks:
        lm = face_landmarks[lm_index]
        x = int(lm['x'] * frame_width)
        y = int(lm['y'] * frame_height)
        
        if (((x - center[0]) ** 2) / (axes[0] ** 2)) + (((y - center[1]) ** 2) / (axes[1] ** 2)) > 1:
            return False
    return True


def detect_hair(face_landmarks, image_np):
    return False


def check_glasses(image_np):
    resized_image = resize_and_pad_image(image_np)
    results = glasses_model(resized_image)
    detected_objects = results.pandas().xyxy[0]
    glasses_detected = detected_objects[detected_objects['name'] == 'with_glasses']
    return not glasses_detected.empty

def check_headwear(image_np):
    """Check for headwear using the updated YOLOv5 model."""
    HEAD_CONFIDENCE_THRESHOLD = 0.45
    HEAD_MIN_BOX_SIZE = 50
    # Resize and pad image to 640x640 (letterbox)
    resized_image = resize_and_pad_image(image_np)

    # Run YOLOv5 inference
    results = headwear_model(resized_image)

    # Extract detections above the confidence threshold
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        box_w = xyxy[2] - xyxy[0]
        box_h = xyxy[3] - xyxy[1]

        # Apply custom filter: only show detections with confidence > threshold and bounding box size > MIN_BOX_SIZE
        if conf > HEAD_CONFIDENCE_THRESHOLD and box_w > HEAD_MIN_BOX_SIZE and box_h > HEAD_MIN_BOX_SIZE:
            return True  # Headwear detected

    return False  # No headwear detected