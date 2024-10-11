import torch
import numpy as np
import cv2
from utillities import resize_and_pad_image
from utillities import resize_and_pad_image2
from ultralytics import YOLO


# Load YOLOv5 models for glasses and headwear detection
glasses_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/glasses_trained.pt', force_reload=True)
headwear_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/headwear2.pt', force_reload=True)
# Load YOLOv8 hair segmentation model
hair_model = YOLO('./models/hairseg-v8.pt')

def detect_hair_in_forehead(image_np, face_landmarks, overlap_threshold=1):

    # Resize the frame with letterboxing
    letterbox_frame, scale, top, left = resize_and_pad_image2(image_np, 640)

    # Run YOLO model on the letterbox frame for hair segmentation
    results = hair_model(letterbox_frame)

    # Initialize the hair mask as None
    hair_mask = None

    # Check if there are any masks in the results
    if results[0].masks is not None and len(results[0].masks.data) > 0:
        # Extract the mask from the results
        hair_mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        hair_mask = cv2.resize(hair_mask, (letterbox_frame.shape[1], letterbox_frame.shape[0]))

        # Optionally dilate the hair mask to make it larger and more tolerant
        kernel = np.ones((5, 5), np.uint8)
        hair_mask = cv2.dilate(hair_mask, kernel, iterations=2)

    # If no hair mask or no face landmarks, return False
    if hair_mask is None or face_landmarks is None:
        return False

    # Get landmarks for the eyes (MediaPipe indexes)
    left_eye_idxs = [33, 160, 158, 133, 153, 144]
    right_eye_idxs = [362, 385, 387, 263, 373, 380]

    # Convert landmarks to numpy array (for easier calculations)
    h, w, _ = letterbox_frame.shape
    landmarks = np.array([(int(pt['x'] * w), int(pt['y'] * h)) for pt in face_landmarks])

    # Calculate the center of the ellipse (midpoint of the eyes)
    left_eye = landmarks[left_eye_idxs].mean(axis=0)
    right_eye = landmarks[right_eye_idxs].mean(axis=0)
    midpoint = ((left_eye + right_eye) / 2).astype(int)

    # Calculate the ellipse width and height for eyes
    eye_distance = np.linalg.norm(left_eye - right_eye)
    ellipse_width = int(eye_distance * 1.95)
    ellipse_height = int(eye_distance * 1)

    # Shift the midpoint upwards to increase the height above the eyes
    shifted_midpoint = (midpoint[0], midpoint[1] - int(ellipse_height * 0.3))

    # Create an empty mask for the ellipse
    ellipse_mask = np.zeros_like(hair_mask)

    # Draw the ellipse on the mask
    cv2.ellipse(ellipse_mask, shifted_midpoint, (ellipse_width // 2, ellipse_height // 2), 0, 0, 360, 1, -1)

    # Check for overlap between the hair mask and the ellipse mask
    overlap = cv2.bitwise_and(hair_mask, ellipse_mask)

    # Calculate the percentage of hair inside the ellipse
    hair_pixels = np.sum(hair_mask)
    overlap_pixels = np.sum(overlap)

    if hair_pixels > 0:
        overlap_percentage = (overlap_pixels / hair_pixels) * 100

        # Set the condition for hair being inside the ellipse
        if overlap_percentage > overlap_threshold:
            return True

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
    return mean_brightness > 100


def is_front_facing(face_landmarks):
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]
    nose = face_landmarks[1]

    eye_dist = np.linalg.norm(np.array([right_eye['x'], right_eye['y']]) - np.array([left_eye['x'], left_eye['y']]))
    eye_mid_y = (left_eye['y'] + right_eye['y']) / 2
    adjusted_eye_mid_y = eye_mid_y + eye_dist * 0.5

    nose_midpoint_dist = np.abs(nose['x'] - (left_eye['x'] + right_eye['x']) / 2)
    vertical_nose_dist = np.abs(nose['y'] - adjusted_eye_mid_y)

    front_facing_horizontal = nose_midpoint_dist < eye_dist * 0.3
    front_facing_vertical = vertical_nose_dist < eye_dist * 0.3

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