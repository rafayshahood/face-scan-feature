# Import necessary libraries
import cv2
from effects import apply_sketch_filter, apply_color_effect
from utilities import draw_text_on_frame

def process_display_frame(frame, center, axes, conditions_met, prompt):
    """
    Process the given frame for display by applying filters, drawing the oval, and adding text prompts.
    
    :param frame: The current frame captured from the webcam.
    :param center: Center of the oval mask used for face positioning.
    :param axes: Dimensions (axes) of the oval mask.
    :param conditions_met: Boolean indicating whether conditions (like lighting, face position) are met.
    :param prompt: Text prompt to display on the frame (e.g., "Move face inside the oval").
    :return: Processed frame ready for display.
    """
    # Apply a sketch filter to the frame to give a stylized effect
    display_frame = apply_sketch_filter(frame.copy())

    # Choose oval color based on whether conditions are met (Blue if met, Red otherwise)
    oval_color = (0, 139, 183) if conditions_met else (255, 75, 51)  # Blue for success, red for failure
    bgr_color = (oval_color[2], oval_color[1], oval_color[0])  # Convert color to BGR format

    # Draw the oval mask around the face area
    cv2.ellipse(display_frame, center, axes, 0, 0, 360, bgr_color, 2)

    # Apply a color effect to highlight the frame based on conditions
    display_frame = apply_color_effect(display_frame, bgr_color)

    # If a prompt is provided (e.g., instructions), draw the text on the frame
    if prompt:
        display_frame = draw_text_on_frame(
            display_frame,
            text=prompt,  # Display the provided prompt text
            center=center,  # Position the text above the oval
            axes=axes,  # Use the oval's axes for reference
            t_size=10,  # Text size for the prompt
            fill_color=(0, 139, 183) if conditions_met else (255, 75, 51),  # Color based on conditions
        )
    
    return display_frame  # Return the fully processed frame for display
