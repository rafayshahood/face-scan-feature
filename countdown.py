# Import necessary libraries
import cv2
import time
from utilities import draw_countdown_text, draw_text_on_frame
from effects import apply_color_effect

def countdown_sequence(cap, frame, center, axes, process_frame, frame_size):
    """
    Handles the countdown sequence and captures an image when conditions are met.
    
    :param cap: The video capture object (webcam).
    :param frame: The current frame from the webcam.
    :param center: The center of the oval mask used for face positioning.
    :param axes: The dimensions (axes) of the oval.
    :param process_frame: Function to process and evaluate each frame.
    :param frame_size: Size of the frame for capturing the image.
    """
    
    timer_start = time.time()  # Record the start time for the countdown
    countdown = 3  # Set countdown duration (in seconds)

    # Loop through the countdown
    while countdown > 0:
        ret, frame = cap.read()  # Capture a new frame from the webcam
        if not ret:
            break  # Exit if frame capture fails
        
        # Process the current frame (e.g., face detection, condition checks)
        display_frame, conditions_met, _, prompt, center, axes = process_frame(frame)

        # If conditions are not met, show an error message and stop the countdown
        if not conditions_met:
            # Set oval color to red (indicating conditions are not met)
            oval_color = (255, 75, 51)  # Red color for conditions not met
            bgr_color = (oval_color[2], oval_color[1], oval_color[0])
            
            # Apply a color effect and display the error prompt
            display_frame = apply_color_effect(display_frame, bgr_color)
            display_frame = draw_text_on_frame(
                display_frame,
                text=prompt,  # Display the prompt message
                center=center,
                axes=axes,
                t_size=10,  # Text size for the prompt
                fill_color=(255, 75, 51),  # Red text color for the prompt
            )
            # Show the updated frame
            cv2.imshow('Webcam', display_frame)
            cv2.waitKey(1)
            break  # Stop the countdown if conditions are not met

        # Draw the countdown number on the frame (3, 2, 1, etc.)
        display_frame = draw_countdown_text(
            display_frame,
            text=str(countdown),  # Display the countdown number
            center=center,  # Position the countdown in the center
            t_size=50,  # Large text size for countdown visibility
            fill_color=(0, 139, 183),  # Blue color for countdown
        )

        # Show the updated frame with the countdown
        cv2.imshow('Webcam', display_frame)
        cv2.waitKey(1)
        
        # Decrease the countdown number every second
        if time.time() - timer_start >= 1:
            countdown -= 1  # Decrease countdown by 1
            timer_start = time.time()  # Reset timer for the next second

    # If countdown reaches 0 and conditions are met, capture the photo
    if countdown == 0 and conditions_met:
        # Save the captured image to a file with reduced quality (50%)
        cv2.imwrite('./capturedImage/captured_image.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        print("Picture taken and saved as 'captured_image.jpg'.")

        # Apply a scan effect after capturing the image
        from utilities import scan_effect
        scan_effect(frame, frame_size)  # Trigger the scanning effect animation
