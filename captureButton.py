# Import necessary libraries
import cv2

# Global variables to track button state and position
button_pressed = False  # Tracks if button is pressed
button_enabled = False  # Tracks if button is enabled
button_top_left = (0, 0)  # Top-left corner of the button
button_bottom_right = (0, 0)  # Bottom-right corner of the button

def draw_button(frame, enabled):
    """
    Draw a responsive button on the frame. The button appearance changes based on its enabled state.
    
    :param frame: The frame on which the button is drawn.
    :param enabled: Boolean flag indicating whether the button should be active or inactive.
    :return: Frame with the button drawn.
    """
    global button_top_left, button_bottom_right

    # Define button size based on frame dimensions
    button_width = int(frame.shape[1] * 0.16)  # Button width is 40% of frame width
    button_height = int(frame.shape[0] * 0.07)  # Button height is 10% of frame height
    
    # Button colors: Active (Orange), Inactive (Gray)
    button_color = (255, 174, 0) if enabled else (150, 150, 150)  
    text_color = (255, 255, 255) if enabled else (100, 100, 100)

    # Calculate button position (centered horizontally, near the bottom of the frame)
    button_center_x = frame.shape[1] // 2
    button_center_y = frame.shape[0] - int(frame.shape[0] * 0.09)
    
    # Define button boundaries for click detection
    button_top_left = (button_center_x - button_width // 2, button_center_y - button_height // 2)
    button_bottom_right = (button_center_x + button_width // 2, button_center_y + button_height // 2)

    # Draw button with rounded corners
    radius = button_height // 2
    cv2.rectangle(frame, (button_top_left[0] + radius, button_top_left[1]), 
                  (button_bottom_right[0] - radius, button_bottom_right[1]), button_color, -1)
    cv2.circle(frame, (button_top_left[0] + radius, button_center_y), radius, button_color, -1)
    cv2.circle(frame, (button_bottom_right[0] - radius, button_center_y), radius, button_color, -1)

    # Adjust font size and thickness based on frame size
    scaling_factor = frame.shape[0] / 695
    font_scale = 0.6 * scaling_factor
    thickness = int(2 * scaling_factor)

    # Draw button text ('Capture')
    text = "Capture"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = button_center_x - text_size[0] // 2
    text_y = button_center_y + text_size[1] // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    return frame

def button_callback(event, x, y, flags, param):
    """
    Detect button click based on mouse event and position.

    :param event: Mouse event (click, move, etc.).
    :param x: X-coordinate of mouse click.
    :param y: Y-coordinate of mouse click.
    """
    global button_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click is within the button area and button is enabled
        if button_enabled and button_top_left[0] <= x <= button_bottom_right[0] and button_top_left[1] <= y <= button_bottom_right[1]:
            button_pressed = True  # Set button pressed state

def update_button_state(conditions_met):
    """
    Update the button's enabled state based on whether conditions are met.
    
    :param conditions_met: Boolean indicating whether the conditions for enabling the button are met.
    """
    global button_enabled
    button_enabled = conditions_met  # Enable or disable the button based on conditions

def is_button_pressed():
    """
    Check if the button has been pressed and reset its state.
    
    :return: Boolean indicating whether the button was pressed.
    """
    global button_pressed
    was_pressed = button_pressed
    button_pressed = False  # Reset button pressed state
    return was_pressed
