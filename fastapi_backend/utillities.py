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

def resize_and_pad_image2(image, target_size):
    """Resizes the image to the target size with letterbox (padding)."""
    ih, iw = image.shape[:2]
    scale = min(target_size / iw, target_size / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    # Resize image
    resized_image = cv2.resize(image, (nw, nh))

    # Create new image with the target size, padded with black (0)
    letterbox_image = np.full((target_size, target_size, 3), 0, dtype=np.uint8)

    # Calculate padding
    top = (target_size - nh) // 2
    left = (target_size - nw) // 2

    # Place the resized image onto the padded background
    letterbox_image[top:top + nh, left:left + nw] = resized_image

    return letterbox_image, scale, top, left