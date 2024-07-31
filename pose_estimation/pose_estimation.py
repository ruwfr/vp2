import os

import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

from utils.constants import LANDMARKS_TO_INCLUDE, MODEL_TYPE


def _get_model():
    model_path = os.path.join(
        os.path.dirname(__file__), "model", f"pose_landmarker_{MODEL_TYPE}.task"
    )
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    model = vision.PoseLandmarker.create_from_options(options)
    return model


def _draw_landmarks_on_image(frame, frame_landmarks):
    pose_landmarks_list = frame_landmarks.pose_landmarks
    annotated_image = np.copy(frame)
    excluded_landmarks = set(PoseLandmark).difference(set(LANDMARKS_TO_INCLUDE))
    custom_connections = list(mp.solutions.pose.POSE_CONNECTIONS)
    custom_style = solutions.drawing_styles.get_default_pose_landmarks_style()

    for i in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[i]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )

        for landmark in excluded_landmarks:
            custom_style[landmark] = DrawingSpec(
                color=(255, 255, 0), thickness=None, circle_radius=0
            )
            custom_connections = [
                connection_tuple
                for connection_tuple in custom_connections
                if landmark.value not in connection_tuple
            ]

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            custom_connections,
            custom_style,
        )
    return annotated_image


@st.cache_data
def get_pose_landmarks(
    video: np.ndarray, frames: int
) -> np.ndarray[PoseLandmarkerResult]:
    model = _get_model()
    pose_landmarks = np.empty(frames, dtype=object)
    for i, frame in enumerate(video):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        pose_landmark_result = model.detect(mp_img)
        pose_landmarks[i] = pose_landmark_result
    return pose_landmarks


@st.cache_data
def get_annotated_video(
    video: np.ndarray, pose_landmarks: np.ndarray[PoseLandmarkerResult]
) -> np.ndarray:
    return np.array(
        [
            _draw_landmarks_on_image(frame, frame_landmarks)
            for frame, frame_landmarks in zip(video, pose_landmarks)
        ]
    )
