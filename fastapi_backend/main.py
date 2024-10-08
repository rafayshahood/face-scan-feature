from fastapi import FastAPI, File, UploadFile, Form  # Import Form to handle form data
import cv2
import numpy as np
import mediapipe as mp
import torch
app = FastAPI()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Load YOLOv5 models for glasses and headwear detection
glasses_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/glasses_trained.pt', force_reload=True)
headwear_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/headwear.pt', force_reload=True)

def is_mouth_closed(face_landmarks, frame_height):
    """ 
    Check if the mouth is closed based on the distance between the upper and lower lip landmarks.
    
    :param face_landmarks: The detected face landmarks from MediaPipe.
    :param frame_height: Height of the frame (used for scaling the distance).
    :return: True if the mouth is closed, False if open.
    """
    # Get the coordinates of the upper and lower lip landmarks
    upper_lip = face_landmarks[13]  # Upper lip
    lower_lip = face_landmarks[14]  # Lower lip

    # Calculate the vertical distance between the upper and lower lip
    lip_distance = abs(upper_lip['y'] - lower_lip['y']) * frame_height

    # Set a threshold for determining if the mouth is open or closed
    mouth_closed_threshold = 5  # You can adjust this threshold based on testing

    # If the lip distance is below the threshold, consider the mouth closed
    return lip_distance < mouth_closed_threshold


def create_oval_mask(image_np, center, axes):
    """
    Create an oval mask over the given frame.
    """
    mask = np.zeros_like(image_np)   
    # Create the oval mask
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
    # Apply the mask: keep the content inside the oval, make the outside black
    masked_frame = cv2.bitwise_and(image_np, mask)
    return masked_frame


def check_glasses(image_np):
    # Resize and pad the image to the target size expected by the YOLO model
    resized_image = resize_and_pad_image(image_np, target_size=(640, 640))
    # Run YOLO glasses detection on the resized image
    results = glasses_model(resized_image)
    detected_objects = results.pandas().xyxy[0]
    glasses_detected = detected_objects[detected_objects['name'] == 'with_glasses']
    return not glasses_detected.empty  # Returns True if glasses are detected

def check_headwear(image_np):
    # Resize and pad the image to the target size expected by the YOLO model
    resized_image = resize_and_pad_image(image_np, target_size=(640, 640))
    # Run YOLO headwear detection on the resized image
    results = headwear_model(resized_image)
    detected_objects = results.pandas().xyxy[0]
    for det in detected_objects.itertuples():
        if det.name in ['hat', 'helmet'] and det.confidence > 0.45:
            return True
    return False


# Add utility functions for conditions (lighting, front-facing, etc.)
def check_lighting(image_np):
    gray_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_frame)
    return mean_brightness > 80  # You can adjust this threshold

def is_front_facing(face_landmarks):
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]
    nose = face_landmarks[1]
    
    eye_dist = np.linalg.norm(np.array([right_eye['x'], right_eye['y']]) - np.array([left_eye['x'], left_eye['y']]))
    eye_mid_y = (left_eye['y'] + right_eye['y']) / 2
    adjusted_eye_mid_y = eye_mid_y + eye_dist * 0.6

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
    # Simplified example; replace with actual hair detection logic
    # Your YOLO hair detection logic will go here
    return False  # Placeholder for actual detection

def resize_and_pad_image(image_np, target_size=(640, 640)):
    """
    Resize the input image while maintaining the aspect ratio and pad it to the target size.
    
    :param image_np: Input image.
    :param target_size: Desired size for the image (default is 640x640).
    :return: Resized and padded image.
    """
    h, w, _ = image_np.shape
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize the image
    resized_image = cv2.resize(image_np, (new_w, new_h))

    # Create a new blank image with the target size
    padded_image = np.full((target_size[1], target_size[0], 3), 128, dtype=np.uint8)  # Default padding color is gray

    # Place the resized image in the center
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return padded_image

@app.post("/evaluate_conditions/")
async def evaluate_conditions(file: UploadFile = File(...)):
    image_data = await file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Calculate center and axes dynamically based on frame dimensions
    frame_height, frame_width = image_np.shape[:2]
    center = (frame_width // 2, frame_height // 2)
    axes = (int(frame_width * 0.4 / 2), int(frame_height * 0.8 / 2))

    # The oval mask is created but NOT used for condition checks
    masked_frame = create_oval_mask(image_np, center, axes)

    # Use the original frame (image_np) for all condition checks
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # Original frame for face detection
    result = face_mesh.process(img_rgb)

    faces = []
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            faces.append([{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face_landmarks.landmark])

    if not faces:
        return {"conditions_met": False, "prompt": "No face detected"}

    face_landmarks = faces[0]

    # Step 1: Check lighting condition on the original frame (not masked)
    if not check_lighting(image_np):
        return {"conditions_met": False, "prompt": "Increase lighting"}

    # Step 2: Check if face is inside oval
    if not is_face_inside_oval(face_landmarks, center, axes, frame_width, frame_height):
        return {"conditions_met": False, "prompt": "Position your face inside the oval"}

    # Step 3: Check if the face is front-facing
    if not is_front_facing(face_landmarks):
        return {"conditions_met": False, "prompt": "Make your face front-facing"}

    # Check mouth closed condition
    if not is_mouth_closed(face_landmarks, frame_height):
        return {"conditions_met": False, "prompt": "Please close your mouth"}


    # Step 4: Check hair detection (using masked frame)
    if detect_hair(face_landmarks, masked_frame):
        return {"conditions_met": False, "prompt": "Please remove hair from forehead"}

    # Step 5: Check glasses and headwear on the original frame (not masked)
    if check_glasses(image_np):
        return {"conditions_met": False, "prompt": "Please remove glasses"}

    if check_headwear(image_np):
        return {"conditions_met": False, "prompt": "Please remove headwear"}

    return {"conditions_met": True, "prompt": "Conditions met - Take a photo"}
