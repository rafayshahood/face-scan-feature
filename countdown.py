import cv2
import time
import requests
from utilities import draw_countdown_text, draw_text_on_frame
from effects import apply_color_effect
import threading

def countdown_sequence(cap, frame, center, axes, process_frame, frame_size):
    """
    Handles the countdown sequence and captures an image when conditions are met.
    Sends the image to the backend for 3D model generation and keeps showing the scan effect until the process is done.
    """

    timer_start = time.time()  # Start time for the countdown
    countdown = 3  # Countdown duration in seconds

    # Countdown loop
    while countdown > 0:
        ret, frame = cap.read()  # Capture a new frame
        if not ret:
            break

        # Process the frame (check conditions)
        display_frame, conditions_met, _, prompt, center, axes = process_frame(frame)

        # If conditions are not met, stop the countdown
        if not conditions_met:
            oval_color = (255, 75, 51)  # Red color for conditions not met
            bgr_color = (oval_color[2], oval_color[1], oval_color[0])
            display_frame = apply_color_effect(display_frame, axes, bgr_color)
            display_frame = draw_text_on_frame(display_frame, prompt, center, axes, t_size=10, fill_color=(255, 75, 51))
            cv2.imshow('Webcam', display_frame)
            cv2.waitKey(1)
            break

        # Display countdown number on the frame
        display_frame = draw_countdown_text(display_frame, text=str(countdown), center=center, t_size=50, fill_color=(0, 139, 183))

        cv2.imshow('Webcam', display_frame)  # Show updated frame
        cv2.waitKey(1)

        # Decrease countdown every second
        if time.time() - timer_start >= 1:
            countdown -= 1
            timer_start = time.time()

    # When countdown reaches 0 and conditions are met, send the image to the backend
    if countdown == 0 and conditions_met:
        # Encode the frame as a JPEG image and send it to the backend
        _, img_encoded = cv2.imencode('.jpg', frame)

        # Send the image to the backend for 3D model generation
        def send_image_to_backend(img_encoded):
            response = requests.post(
                'http://127.0.0.1:8000/generate_model/',
                files={"file": img_encoded.tobytes()}
            )
            if response.status_code == 200:
                print("Image sent to backend successfully.")
                return response.json().get('task_id')
            else:
                print(f"Failed to send image: {response.status_code}")
                return None

        task_id = send_image_to_backend(img_encoded)

        # Start scan effect while polling for the backend to complete 3D model generation
        if task_id:
            model_generation_done = False
            polling_attempts = 0
            max_polling_attempts = 10  # Max number of times to poll

            def poll_backend_for_status(task_id):
                nonlocal model_generation_done, polling_attempts
                while not model_generation_done and polling_attempts < max_polling_attempts:
                    response = requests.get(f'http://127.0.0.1:8000/check_model_status/{task_id}')
                    status = response.json().get("status")
                    if status == "completed":
                        model_generation_done = True
                        print("Model generation completed.")
                    elif status == "failed":
                        model_generation_done = True
                        print("Model generation failed.")
                    else:
                        print("Model generation in progress...")
                    polling_attempts += 1
                    time.sleep(2)  # Poll every 2 seconds

            # Run polling in a separate thread
            status_thread = threading.Thread(target=poll_backend_for_status, args=(task_id,))
            status_thread.start()

            # Continue showing the scan effect until model generation completes or fails
            while not model_generation_done:
                ret, frame = cap.read()
                if not ret:
                    break
                from utilities import scan_effect
                scan_effect(frame, frame_size)
            status_thread.join()

        # Close the face scan window
        cv2.destroyAllWindows()
