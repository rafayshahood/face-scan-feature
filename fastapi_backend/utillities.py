import cv2
import numpy as np

def create_oval_mask(image_np, center, axes):
    mask = np.zeros_like(image_np)   
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
    masked_frame = cv2.bitwise_and(image_np, mask)
    return masked_frame


def resize_and_pad_image(image_np, target_size=(640, 640)):
    h, w, _ = image_np.shape
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image_np, (new_w, new_h))
    padded_image = np.full((target_size[1], target_size[0], 3), 128, dtype=np.uint8)

    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return padded_image

def letterbox_image(image, target_size=(640, 640)):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    letterboxed_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    letterboxed_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return letterboxed_image