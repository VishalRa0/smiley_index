# import cv2
# import mediapipe as mp
# import numpy as np
# import math
# import streamlit as st
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# # --- Helper functions ---
# def euclidean(p1, p2, w, h):
#     return math.dist((p1.x * w, p1.y * h), (p2.x * w, p2.y * h))

# def compute_smile_metric(landmarks, w, h):
#     # Face width (scale normalization)
#     eye_left, eye_right = landmarks[33], landmarks[263]
#     face_width_px = euclidean(eye_left, eye_right, w, h)
#     scale_factor = face_width_px / 100.0

#     # Mouth width across multiple pairs
#     mouth_widths = [euclidean(landmarks[l], landmarks[r], w, h)
#                     for l, r in zip(LEFT_POINTS, RIGHT_POINTS)]
#     avg_mouth_width = np.mean(mouth_widths)

#     # Corner lift (helps detect subtle smiles)
#     left_lift = (landmarks[37].y - landmarks[61].y) * h
#     right_lift = (landmarks[267].y - landmarks[291].y) * h
#     corner_lift = (left_lift + right_lift) / 2

#     # Smile metric
#     smile_metric = (1.2 * avg_mouth_width / (scale_factor + 1e-6)) + (0.05 * corner_lift)
#     return smile_metric

# def normalize_smile_score(metric):
#     MIN_METRIC = 60
#     MAX_METRIC = 140
#     norm = np.clip((metric - MIN_METRIC) / (MAX_METRIC - MIN_METRIC), 0, 1)
#     score = 1 + norm * 4
#     return round(score, 2)

# # --- Load MediaPipe model ---
# base_options = python.BaseOptions(model_asset_path="C:/Users/vishal.rao.lv/Downloads/face_landmarker (1).task")
# options = vision.FaceLandmarkerOptions(
#     base_options=base_options,
#     output_face_blendshapes=True,
#     output_facial_transformation_matrixes=True,
#     num_faces=1
# )
# detector = vision.FaceLandmarker.create_from_options(options)

# # --- Constants ---
# MOUTH_POINTS = [61, 291, 50, 280, 210, 430, 380, 110]
# LEFT_POINTS = [61, 50, 210, 380]
# RIGHT_POINTS = [291, 280, 430, 110]

# # --- Streamlit UI ---
# st.set_page_config(page_title="Real-Time Smile Detection", layout="centered")
# st.title("😊 Real-Time Smile Detection")
# st.caption("Press **Stop** to end the stream.")

# run = st.checkbox("Start Camera")

# FRAME_WINDOW = st.image([])

# cap = None

# if run:
#     cap = cv2.VideoCapture(0)

# while run:
#     ret, frame = cap.read()
#     if not ret:
#         st.warning("Camera not found!")
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#     detection_result = detector.detect(mp_image)

#     annotated_frame = frame.copy()
#     if detection_result.face_landmarks:
#         landmarks = detection_result.face_landmarks[0]
#         smile_metric = compute_smile_metric(landmarks, w, h)
#         smile_score = normalize_smile_score(smile_metric)
#         status = "😄 Smiling" if smile_metric >= 102 else "😐 Neutral"

#         # Draw landmarks
#         for idx in MOUTH_POINTS:
#             lm = landmarks[idx]
#             x, y = int(lm.x * w), int(lm.y * h)
#             cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)

#         # Display metrics on frame
#         cv2.putText(annotated_frame, f"Smile Metric: {smile_metric:.1f}", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#         cv2.putText(annotated_frame, f"Smile Score: {smile_score:.2f}/5", (20, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#         cv2.putText(annotated_frame, status, (20, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

# if cap:
#     cap.release()




import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks import python

from mediapipe.tasks.python import vision

# --- Utility: Euclidean distance between landmarks ---
def euclidean(p1, p2, w, h):
    return math.dist((p1.x * w, p1.y * h), (p2.x * w, p2.y * h))

# --- Load MediaPipe model ---
base_options = python.BaseOptions(model_asset_path="C:/Users/vishal.rao.lv/Downloads/face_landmarker (1).task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- Mouth landmark indices ---
MOUTH_POINTS = [61, 291, 50, 280, 210, 430, 380, 110]
LEFT_POINTS = [61, 50, 210, 380]
RIGHT_POINTS = [291, 280, 430, 110]

# --- Smile metric computation ---
def compute_smile_metric(landmarks, w, h):
    eye_left, eye_right = landmarks[33], landmarks[263]
    face_width_px = euclidean(eye_left, eye_right, w, h)
    scale_factor = face_width_px / 100.0

    mouth_widths = [euclidean(landmarks[l], landmarks[r], w, h)
                    for l, r in zip(LEFT_POINTS, RIGHT_POINTS)]
    avg_mouth_width = np.mean(mouth_widths)

    left_lift = (landmarks[37].y - landmarks[61].y) * h
    right_lift = (landmarks[267].y - landmarks[291].y) * h
    corner_lift = (left_lift + right_lift) / 2

    smile_metric = (1.2 * avg_mouth_width / (scale_factor + 1e-6)) + (0.05 * corner_lift)
    return smile_metric

# --- Normalize smile metric to score 1–5 ---
def normalize_smile_score(metric):
    MIN_METRIC = 60
    MAX_METRIC = 140
    norm = np.clip((metric - MIN_METRIC) / (MAX_METRIC - MIN_METRIC), 0, 1)
    score = 1 + norm * 4
    return round(score, 2)

# --- Streamlit UI ---
st.set_page_config(page_title="Smile Detection", layout="wide")
st.title("Smile Detection")
st.markdown("**Press 'Start Camera' and 'Stop' to quit. Press 'q' in OpenCV mode.**")

mode = st.radio("Choose camera mode:", ["Streamlit Camera", "OpenCV Live Capture"])

# --- Streamlit Camera Input ---
if mode == "Streamlit Camera":
    img_file_buffer = st.camera_input("Capture a photo")
    if img_file_buffer:
        bytes_data = img_file_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)

        annotated_frame = frame.copy()
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            smile_metric = compute_smile_metric(landmarks, w, h)
            smile_score = normalize_smile_score(smile_metric)
            status = " Smiling" if smile_metric >= 102 else "Not Smiling"

            for idx in MOUTH_POINTS:
                lm = landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)

            cv2.putText(annotated_frame, f"Metric: {smile_metric:.1f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Score: {smile_score}/5", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, status, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=status)
            st.metric("Smile Score (1–5)", smile_score)
            st.metric("Smile Metric", f"{smile_metric:.1f}")

# --- OpenCV Realtime Mode ---
elif mode == "OpenCV Live Capture":
    st.warning("Press 'q' in the OpenCV window to stop.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)

        annotated_frame = frame.copy()
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            smile_metric = compute_smile_metric(landmarks, w, h)
            smile_score = normalize_smile_score(smile_metric)
            status = "Smiling " if smile_metric >= 102 else "Not Smiling "

            for idx in MOUTH_POINTS:
                lm = landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)

            cv2.putText(annotated_frame, f"Smile Metric: {smile_metric:.1f}",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Score: {smile_score}/5",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, status,
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Smile Detection (Press q to quit)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()