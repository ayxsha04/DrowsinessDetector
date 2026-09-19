import winsound
import cv2
import mediapipe as mp
import numpy as np
import os

# ==========================
# FACE MESH INITIALIZATION
# ==========================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# EYE LANDMARK INDICES
# ==========================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==========================
# HELPER FUNCTIONS
# ==========================

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(eye_points):
    A = distance(eye_points[1], eye_points[5])
    B = distance(eye_points[2], eye_points[4])
    C = distance(eye_points[0], eye_points[3])

    return (A + B) / (2.0 * C)


# ==========================
# START WEBCAM
# ==========================

cap = cv2.VideoCapture(0)

closed_frames = 0
alarm_played = False
alarm_on = False

# Your open-eye EAR is around 0.30-0.32
EAR_THRESHOLD = 0.15

# Number of consecutive frames before drowsiness alert
DROWSY_FRAMES = 20

print("Face Mesh started. Press 'q' to quit.")

# ==========================
# MAIN LOOP
# ==========================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye_points = []
            right_eye_points = []

            # ==========================
            # LEFT EYE LANDMARKS
            # ==========================

            for idx in LEFT_EYE:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                left_eye_points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )

            # ==========================
            # RIGHT EYE LANDMARKS
            # ==========================

            for idx in RIGHT_EYE:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                right_eye_points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )

            # ==========================
            # CALCULATE EAR
            # ==========================

            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)

            ear = (left_ear + right_ear) / 2

            # ==========================
            # EYE CLOSURE DETECTION
            # ==========================

            if ear < EAR_THRESHOLD:

                closed_frames += 1

                cv2.putText(
                    frame,
                    "EYES CLOSED!",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            else:
                closed_frames = 0

            # ==========================
            # DISPLAY EAR
            # ==========================

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # ==========================
            # DROWSINESS ALERT
            # ==========================

            if closed_frames >= DROWSY_FRAMES:

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # Start alarm only once
                if not alarm_on:
                    sound_path = os.path.join(os.getcwd(),"alarm.wav")

                    winsound.PlaySound(
                        sound_path,
                        winsound.SND_FILENAME |
                        winsound.SND_ASYNC |
                        winsound.SND_LOOP
                    )
                    alarm_on = True

            else:

                # Stop alarm when eyes open
                if alarm_on:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                    alarm_on = False


                
    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# CLEANUP
# ==========================

cap.release()
cv2.destroyAllWindows()