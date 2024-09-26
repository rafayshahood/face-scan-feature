# Import necessary libraries
import cv2
import numpy as np

def apply_sketch_filter(image, invert_colors=False):
    """
    Apply a sketch-like filter to the image, optionally inverting the colors.
    
    :param image: The original image to apply the filter to.
    :param invert_colors: Boolean flag to indicate whether colors should be inverted.
    :return: Image with the sketch filter applied.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inv_gray = cv2.bitwise_not(gray)
    
    # Apply Gaussian blur to the inverted image
    blurred = cv2.GaussianBlur(inv_gray, (21, 21), sigmaX=0, sigmaY=0)
    
    # Divide the original grayscale image by the inverted blurred image to create a sketch effect
    sketch = cv2.divide(gray, 255 - blurred, scale=256)

    # If invert_colors is True, invert the sketch colors
    if invert_colors:
        sketch = cv2.bitwise_not(sketch)

    # Convert the single-channel sketch back to a 3-channel image (BGR format)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_color_effect(image, color):
    """
    Apply a color overlay effect to the image, with an oval mask keeping the original content visible.
    
    :param image: The original image to apply the effect to.
    :param color: The color (BGR format) to overlay on the image.
    :return: Image with the color effect applied.
    """
    # Create a solid color overlay with the same size as the image
    overlay = np.full_like(image, color, dtype=np.uint8)

    # Create a mask with an elliptical shape (oval) in the center
    mask = np.zeros_like(image, dtype=np.uint8)
    center = (image.shape[1] // 2, image.shape[0] // 2)  # Center of the oval
    axes = (int(image.shape[1] * 0.3), int(image.shape[0] * 0.4))  # Size of the oval
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)  # Draw a white-filled oval in the mask

    # Invert the mask (so the oval remains visible while the rest is colored)
    mask = cv2.bitwise_not(mask)

    # Apply the color effect to areas outside the oval
    return cv2.addWeighted(image, 1, cv2.bitwise_and(overlay, mask), 0.3, 0)

def apply_combined_filter(image, color_filter):
    """
    Combine the sketch filter with a color overlay for a composite effect.
    
    :param image: The original image to apply the combined effect to.
    :param color_filter: The color filter to overlay on the image.
    :return: Image with both sketch and color effects applied.
    """
    # Apply the sketch filter to the image
    sketch = apply_sketch_filter(image)
    
    # Combine the sketch with the color filter using weighted addition
    return cv2.addWeighted(sketch, 0.25, color_filter, 0.75, 0)
