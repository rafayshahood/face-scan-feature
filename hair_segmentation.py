import cv2
import torch
import numpy as np

# Load your trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./hairseg.pt')

# Set model parameters
model.conf = 0.25  # confidence threshold
model.iou = 0.45   # NMS IOU threshold

# Initialize webcam
cap = cv2.VideoCapture(0)

def preprocess_image(img, input_size=(640, 640)):
    # Resize image
    img = cv2.resize(img, input_size)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Transpose to CHW format
    img = img.transpose((2, 0, 1))
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    # Convert to float32 and normalize
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = preprocess_image(frame)

    # Run inference
    results = model(input_tensor)

    # Process results
    if len(results) > 0:
        for result in results:
            if result.masks is not None:
                masks = result.masks.cpu().numpy()
                for i, mask in enumerate(masks):
                    # Resize mask to match frame size
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    colored_mask = np.zeros(frame.shape, dtype=np.uint8)
                    colored_mask[:, :, 1] = (mask > 0.5).astype(np.uint8) * 255  # Green mask
                    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

    # Display the frame
    cv2.imshow('YOLOv5 Hair Segmentation', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()