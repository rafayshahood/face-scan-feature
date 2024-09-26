# Import necessary libraries
import cv2
import numpy as np

def create_masked_frame(frame, oval_size_ratio):
    """
    Create a masked frame with an oval mask at the center, keeping the original content inside the oval.

    :param frame: The frame captured from the webcam.
    :param oval_size_ratio: Tuple containing the ratio for the oval's size (relative to the frame).
    :return: Masked frame, center of the oval, and dimensions of the oval's axes.
    """
    # Create a blank mask with the same size as the frame
    mask = np.zeros_like(frame)
    
    # Get the number of rows and columns in the frame
    rows, cols, _ = frame.shape
    
    # Define the center and size of the oval
    center = (cols // 2, rows // 2)  # Center of the oval in the middle of the frame
    axes = (int(cols * oval_size_ratio[0] / 2), int(rows * oval_size_ratio[1] / 2))  # Axes of the oval
    
    color = (255, 255, 255)  # White color for the oval mask
    thickness = -1  # Fill the oval completely

    # Draw the filled oval on the mask
    cv2.ellipse(mask, center, axes, 0, 0, 360, color, thickness)

    # Apply the mask to the frame, keeping the original frame inside the oval
    masked_frame = np.where(mask == 255, frame, frame)  # No blurring outside the oval (optional)
    print(center, axes)
    print(f"Ellipse center: {center}, Axes: {axes}, Frame size: {masked_frame.shape}")
    print(f"Ellipse center: {center}, Axes: {axes}, Frame size: {frame.shape}")
    # Return the masked frame, oval center, and oval dimensions
    return masked_frame, center, axes


def visualize_oval(frame, center, axes):
    """
    Visualize the oval on the given frame.
    :param frame: The frame where the oval is drawn.
    :param center: Center of the oval.
    :param axes: Axes dimensions of the oval.
    """
    # Clone the frame so you can draw on it
    frame_copy = frame.copy()
    
    # Draw the oval on the frame
    cv2.ellipse(frame_copy, center, axes, 0, 0, 360, (0, 255, 0), 2)  # Green oval
    
    # Display the frame with the oval
    cv2.imshow('Oval Visualization', frame_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

