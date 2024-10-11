# from fastapi import FastAPI, File, UploadFile
# from utils.face_detection import process_face_detection
# from utils.conditions import check_conditions

# app = FastAPI()

# @app.post("/evaluate_conditions/")
# async def evaluate_conditions(file: UploadFile = File(...)):
#     image_data = await file.read()
#     result = process_face_detection(image_data)
#     if not result['faces']:
#         return {"conditions_met": False, "prompt": "No face detected"}

#     conditions_met, prompt = check_conditions(result)
#     return {"conditions_met": conditions_met, "prompt": prompt}
