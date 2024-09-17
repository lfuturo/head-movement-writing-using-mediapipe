import os
import winsound
from collections import Counter, deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from projection_functions import get_head_rotation
from overlay_functions import detect_quadrant, draw_grid, key_mapping, overlays

current_overlay_index = 0
jaw_open_counter = 0
eye_blink_counter = 0
frame_threshold = 60  
results_buffer = deque(maxlen=120)  
final_prediction = []  


def main():
    """
    This function captures the webcam feed and displays it with an overlay. For each frame, it performs landmark detection, 
    which is used to project the nose, and blendshape detection, where some blendshapes are used for writing rules. 
    The most common quadrant where the nose projection is located is stored in a list along with the overlay index, 
    and it is eventually translated into text.
    """
    cap = cv2.VideoCapture(0)

    model_path = os.path.join("face_landmarker.task")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    global current_overlay_index, jaw_open_counter

    screen_width, screen_height = (
        int(cap.get(3)),
        int(cap.get(4)),
    )  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect_for_video(
            mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC))
        )

        if detection_result.face_landmarks:
            
            blendshape_values = {
                blendshape.category_name: blendshape.score
                for blendshape in detection_result.face_blendshapes[0]
            }

            if blendshape_values.get("jawOpen", 0) > 0.5:
                jaw_open_counter += 1
            else:
                jaw_open_counter = 0

            if jaw_open_counter >= frame_threshold:
                print(
                    "JawOpen greater than 0.5 detected for 60 consecutive frames. Switching overlay."
                )
                jaw_open_counter = 0
                current_overlay_index = (current_overlay_index + 1) % len(overlays)
                results_buffer.clear()

            if (
                blendshape_values.get("eyeBlinkLeft", 0) > 0.5
                and blendshape_values.get("eyeBlinkRight", 0) > 0.5
            ):
                eye_blink_counter += 1
            else:
                eye_blink_counter = 0

            if eye_blink_counter >= frame_threshold:
                final_prediction.append(("Space", current_overlay_index + 1))
                winsound.Beep(800, 200)
                results_buffer.clear()
                eye_blink_counter = 0

                if (
                    len(final_prediction) >= 2
                    and final_prediction[-1][0] == "Space"
                    and final_prediction[-2][0] == "Space"
                ):
                    print("Two Spaces detected. Ending program.")
                    break

            normalized_landmarks = detection_result.face_landmarks[0]

            frame, nose_end_point_2d = get_head_rotation(normalized_landmarks, frame)

            quadrant = detect_quadrant(
                nose_end_point_2d[0][0], screen_width, screen_height
            )

            results_buffer.append((quadrant, current_overlay_index + 1))


            most_common_quadrant, count = Counter(results_buffer).most_common(1)[0]
            if count >= 90:  
                final_prediction.append(most_common_quadrant)
                print(f"Prediction Saved: {most_common_quadrant}")
                winsound.Beep(1000, 200)
                results_buffer.clear()

        frame = draw_grid(
            frame, overlays[current_overlay_index], screen_width, screen_height
        )

        cv2.imshow("Keyboard", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    translated_text = "".join(
        [key_mapping.get(result, "") for result in final_prediction]
    )
    print(f"Final Prediction: {final_prediction}")
    print(f"Final Text Translated: {translated_text}")


if __name__ == "__main__":
    main()
