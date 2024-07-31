import cv2
import numpy as np
import streamlit as st


@st.cache_data
def read_video(video_filename: str,
               video_width: int = 1280,
               video_height: int = 720) -> tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = [
        cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                   (video_width, video_height))
        for success, frame_bgr in iter(lambda: cap.read(), (False, None))
        if success
    ]
    return np.asarray(frames), fps
