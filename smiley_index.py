import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

def euclidean(p1, p2, w, h):
    return math.dist((p1.x * w, p1.y * h), (p2.x * w, p2.y * h))

# --- Load MediaPipe model ---
base_options = python.BaseOptions(model_asset_path="face_landmarker (1).task")
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

def normalize_smile_score(metric):
    MIN_METRIC = 60
    MAX_METRIC = 140
    norm = np.clip((metric - MIN_METRIC) / (MAX_METRIC - MIN_METRIC), 0, 1)
    score = 1 + norm * 4
    return round(score, 2)

# --- Video Processor Class ---
class SmileDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = detector.detect(mp_image)
        
        annotated_frame = img.copy()
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            smile_metric = compute_smile_metric(landmarks, w, h)
            smile_score = normalize_smile_score(smile_metric)
            status = "Smiling" if smile_metric >= 102 else " Not Smiling"
            
            # Draw landmarks
            for idx in MOUTH_POINTS:
                lm = landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
            
            # Add text overlays
            cv2.putText(annotated_frame, f"Metric: {smile_metric:.1f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Score: {smile_score}/5", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, status, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- Streamlit UI ---
st.set_page_config(page_title="Smile Detection", layout="wide")
st.title(" Smile Detection")
st.markdown("**Choose your mode below:**")

mode = st.radio("Choose camera mode:", ["📸 Capture Photo", "🎥 Live Video Stream"])

# --- Photo Capture Mode ---
if mode == "📸 Capture Photo":
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
            status = "Smiling" if smile_metric >= 102 else " Not Smiling"
            
            # Draw landmarks
            for idx in MOUTH_POINTS:
                lm = landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
            
            # Add text overlays
            cv2.putText(annotated_frame, f"Metric: {smile_metric:.1f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Score: {smile_score}/5", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, status, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display image
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=status)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Smile Score (1–5)", smile_score)
            with col2:
                st.metric("Smile Metric", f"{smile_metric:.1f}")
        else:
            st.warning("No face detected. Please try again!")

# --- Live Video Stream Mode ---
elif mode == "🎥 Live Video Stream":
    st.markdown("### Real-time Smile Detection")
    st.info("Click 'START' to begin live detection. Click 'STOP' when done.")
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_streamer(
        key="smile-detection",
        video_processor_factory=SmileDetector,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )
