# Import necessary libraries
import cv2
import numpy as np
import time
from utilities import draw_text_on_frame

def show_loading_frame(frame_size, oval_size_ratio, color=(0, 139, 183), duration=2):
    """
    Displays a loading screen with an oval mask for a set duration.

    :param frame_size: Tuple containing the width and height of the frame.
    :param oval_size_ratio: Tuple containing the ratio for the oval's size (relative to the frame).
    :param color: Color of the oval (default is blue).
    :param duration: Duration for which the loading screen is displayed (in seconds).
    """
    # Create a blank white frame for the loading screen
    loading_frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255
    
    # Define the center and size of the oval
    center = (loading_frame.shape[1] // 2, loading_frame.shape[0] // 2)
    axes = (int(loading_frame.shape[1] * oval_size_ratio[0] / 2), int(loading_frame.shape[0] * oval_size_ratio[1] / 2))
    
    thickness = 2  # Set the thickness of the oval's border
    
    # Convert the oval color from RGB to BGR format for OpenCV
    bgr_color = (color[2], color[1], color[0])
    
    # Draw the oval on the loading frame
    cv2.ellipse(loading_frame, center, axes, 0, 0, 360, bgr_color, thickness)
    
    # Add "Loading Screen" text inside the oval
    loading_frame = draw_text_on_frame(
        loading_frame,
        text="Loading Screen",  # Display "Loading Screen" text
        center=center,  # Center the text
        axes=axes,  # Use oval axes to position the text
        t_size=12,  # Text size for the loading message
        fill_color=(0, 139, 183),  # Use the same blue color for the text
    )
    
    # Show the loading frame
    cv2.imshow('Webcam', loading_frame)
    cv2.waitKey(1)  # Allow a brief wait to display the frame
