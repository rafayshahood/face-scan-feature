from asyncio import tasks
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import mediapipe as mp  # Initialize Mediapipe here
from models_and_conditions import (
    check_lighting,
    is_front_facing,
    is_face_inside_oval,
    check_glasses,
    check_headwear,
    is_mouth_closed,
    detect_hair_in_forehead,
    load_models
)
from utillities import create_oval_mask
import time
import os
import subprocess
from threading import Thread


app = FastAPI()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


@app.on_event("startup")
def startup_event():
    load_models()

@app.post("/evaluate_conditions/")
async def evaluate_conditions(file: UploadFile = File(...)):
    start_time = time.time()
    image_data = await file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    glasses_frame = image_np.copy()
    headwear_frame = image_np.copy()
    hair_frame = image_np.copy()

    # Calculate center and axes dynamically based on frame dimensions
    frame_height, frame_width = image_np.shape[:2]
    center = (frame_width // 2, frame_height // 2)
    axes = (int(frame_width * 0.4 / 2), int(frame_height * 0.8 / 2))

    # The oval mask is created but NOT used for condition checks
    masked_frame = create_oval_mask(image_np, center, axes)

    # Use the original frame (image_np) for all condition checks
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # Original frame for face detection
    result = face_mesh.process(img_rgb)

    # # Step 1: Check lighting condition
    if not check_lighting(image_np):
        return {"conditions_met": False, "prompt": "Increase lighting"}

    faces = []
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            faces.append([{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face_landmarks.landmark])

    if not faces:
        return {"conditions_met": False, "prompt": "No face detected"}

    face_landmarks = faces[0]

    # # # # Step 2: Check if face is inside oval
    if not is_face_inside_oval(face_landmarks, center, axes, frame_width, frame_height):
        return {"conditions_met": False, "prompt": "Position your face inside the oval"}

    # # # # Step 3: Check if the face is front-facing
    if not is_front_facing(face_landmarks):
        return {"conditions_met": False, "prompt": "Make your face front-facing"}

    # # # # Step 4: Check mouth closed condition
    if not is_mouth_closed(face_landmarks, frame_height):
        return {"conditions_met": False, "prompt": "Please close your mouth"}

    # # # # Step 6: Check glasses and headwear on the original frame
    if check_glasses(glasses_frame):
        return {"conditions_met": False, "prompt": "Please remove glasses"}

    if check_headwear(headwear_frame):
        return {"conditions_met": False, "prompt": "Please remove headwear"}

    # # # Step 5: Check hair detection (new condition, run after all others)
    if detect_hair_in_forehead(image_np):
        return {"conditions_met": False, "prompt": "Please remove hair from forehead"}

    # Stop timing the process
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing Time: {processing_time} seconds")

    return {"conditions_met": True, "prompt": "Conditions met - Take a photo"}

# Photo capturing and model generation
tasks= {}
@app.post("/generate_model/")
async def generate_model(file: UploadFile = File(...)):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    image_dir = './saved_images'
    image_path = f'{image_dir}/image_{timestamp}.jpg'

    # Ensure the 'saved_images' directory exists
    os.makedirs(image_dir, exist_ok=True)

    # Save the uploaded image
    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Create a unique task ID
    task_id = f"task_{timestamp}"
    tasks[task_id] = "in_progress"

    # Start the 3D model generation in a separate thread
    def run_model_generation(image_path, task_id):
        try:
            output_dir = f'./generated_models/model_{timestamp}'
            os.makedirs(output_dir, exist_ok=True)
            # Simulate the model generation process with an error
            subprocess.run(['python', './3dmodel/pipeline.py', '-i', image_path, '-o', output_dir], check=True)
            tasks[task_id] = "completed"
        except subprocess.CalledProcessError as e:
            tasks[task_id] = "failed"
            print(f"Error generating 3D model: {e}")
        except Exception as e:
            tasks[task_id] = "failed"
            print(f"Unexpected error during model generation: {e}")

    thread = Thread(target=run_model_generation, args=(image_path, task_id))
    thread.start()

    return {"task_id": task_id}


@app.get("/check_model_status/{task_id}")
async def check_model_status(task_id: str):
    # Return the status of the task (in_progress, completed, failed)
    status = tasks.get(task_id, "unknown")
    return {"status": status}
