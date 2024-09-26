# Import necessary libraries
import cv2
import numpy as np
import time
from effects import apply_combined_filter, apply_sketch_filter
from PIL import ImageFont, ImageDraw, Image

def get_scaling_factor(frame):
    """
    Calculate a scaling factor based on the frame height for responsive text sizing.

    :param frame: The frame to calculate the scaling factor from.
    :return: Scaling factor based on the frame's height.
    """
    return frame.shape[0] / 321  # Use 321px as the base height to adjust scaling

def draw_text_on_frame(frame, text, center, axes, t_size=15, fill_color=(0, 0, 0)):
    """
    Draws text above the oval mask on the given frame using a custom font.

    :param frame: The frame on which the text is drawn.
    :param text: The text to display.
    :param center: Center of the oval mask.
    :param axes: Dimensions of the oval mask (used for positioning the text).
    :param t_size: Text size (default is 15).
    :param fill_color: Text color (default is black).
    :return: Frame with the text drawn.
    """
    font_path = "./fonts/SF-Pro-Text-Medium.otf"  # Path to the custom font file

    scaling_factor = get_scaling_factor(frame)  # Calculate scaling factor based on frame size
    
    # Convert the OpenCV frame (BGR) to a PIL image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(pil_image)  # Create a draw object for text rendering
    font = ImageFont.truetype(font_path, int(t_size * scaling_factor))  # Load the font with scaled size

    # Calculate the bounding box for the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Position the text above the oval mask
    text_x = center[0] - text_width // 2
    text_y = (0 + (center[1] - axes[1])) // 2 - text_height // 2

    # Draw the text onto the PIL image
    draw.text((text_x, text_y), text, font=font, fill=fill_color)

    # Convert the PIL image back to an OpenCV frame (RGB to BGR)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def draw_countdown_text(frame, text, center, t_size=100, fill_color=(0, 0, 255)):
    """
    Draws countdown text in the center of the frame.

    :param frame: The frame to draw the countdown on.
    :param text: Countdown number to display (e.g., "3", "2", "1").
    :param center: Center position for the countdown text.
    :param t_size: Text size for the countdown.
    :param fill_color: Color of the countdown text.
    :return: Frame with the countdown text drawn.
    """
    font_path = "./fonts/SF-Pro-Text-Medium.otf"  # Path to the custom font

    # Calculate the scaling factor for responsive text sizing
    scaling_factor = frame.shape[0] / 321  
    
    # Convert the OpenCV frame (BGR) to a PIL image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(pil_image)  # Create a draw object for text rendering
    font = ImageFont.truetype(font_path, int(t_size * scaling_factor))  # Load the font with scaled size

    # Calculate the bounding box for the countdown text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Center the countdown text in the frame
    text_x = center[0] - text_width // 2
    text_y = center[1] - text_height // 2 - int(text_height * 0.1)  # Fine-tune vertical position

    # Draw the countdown text on the PIL image
    draw.text((text_x, text_y), text, font=font, fill=fill_color)

    # Convert the PIL image back to an OpenCV frame (RGB to BGR)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def create_oval_mask(frame):
    """
    Create an oval mask that blurs the area outside the oval and keeps the content inside clear.

    :param frame: The original frame.
    :return: Masked frame, center of the oval, and dimensions of the oval's axes.
    """
    # Create a blank mask with the same size as the frame
    mask = np.zeros_like(frame)
    
    # Get the number of rows and columns in the frame
    rows, cols, _ = frame.shape
    
    # Define the center and size of the oval
    center = (cols // 2, rows // 2)
    axes = (int(cols * 0.35 / 2), int(rows * 0.75 / 2))
    
    color = (255, 255, 255)  # White color for the oval
    thickness = -1  # Fill the oval completely

    # Draw the filled oval on the mask
    cv2.ellipse(mask, center, axes, 0, 0, 360, color, thickness)

    # Apply a Gaussian blur to areas outside the oval
    blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)

    # Keep the original frame inside the oval and blur the rest
    masked_frame = np.where(mask == 255, frame, blurred_frame)

    # Return the masked frame, center, and axes of the oval
    return masked_frame, center, axes

def scan_effect(frame, frame_size):
    """
    Apply a scanning effect on the frame with different filters during a short animation.

    :param frame: The original frame to apply the effect to.
    :param frame_size: Size of the frame for resizing.
    """
    # Resize the frame to match the desired frame size
    frame = cv2.resize(frame, frame_size)

    start_time = time.time()  # Record the start time of the scan effect
    scan_duration = 3  # Set the scan duration (in seconds)
    line_position = 0  # Initialize the line position
    filter_switch_rate = 0.5  # Set the interval for switching filters

    while time.time() - start_time < scan_duration:
        elapsed = time.time() - start_time  # Calculate elapsed time
        line_position = int((elapsed / scan_duration) * frame.shape[0])  # Update line position

        # Define different filters to apply in sequence
        filters = [
            apply_combined_filter(frame, cv2.GaussianBlur(frame, (15, 15), 0)),  # Blur filter
            apply_combined_filter(frame, np.full_like(frame, (0, 0, 255))),  # Red color filter
            apply_combined_filter(frame, np.full_like(frame, (0, 255, 255))),  # Yellow color filter
            apply_combined_filter(frame, apply_sketch_filter(frame, invert_colors=True))  # Inverted sketch filter
        ]
        
        # Calculate the current filter index based on the elapsed time and switch rate
        current_filter_index = int(elapsed / filter_switch_rate) % len(filters)
        current_filter = filters[current_filter_index]

        # Draw a white line across the frame as part of the scan effect
        cv2.line(current_filter, (0, line_position), (frame.shape[1], line_position), (255, 255, 255), 5)

        # Display the current filter on the 'Webcam' window
        cv2.imshow('Webcam', current_filter)
        cv2.waitKey(1)

    # After the loop, display the final frame with an inverted sketch filter
    final_filter = apply_combined_filter(frame, apply_sketch_filter(frame, invert_colors=True))
    cv2.imshow('Webcam', final_filter)
    cv2.waitKey(500)  # Show the final effect briefly
