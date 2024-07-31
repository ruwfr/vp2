import numpy as np
import plotly.graph_objects as go
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from plotly.graph_objects import Figure


def get_depth_verdict_and_fig(
    frames: int,
    pose_landmarks: list[PoseLandmarkerResult],
    frame_nr: int,
    nr_of_reps: int,
) -> tuple[bool, Figure]:
    hips_indices, knees_indices, ankles_indices = [23, 24], [25, 26], [27, 28]
    knee_angles = np.empty(frames)

    for i, landmarks_frame in enumerate(pose_landmarks):
        landmarks_frame = np.array(landmarks_frame.pose_world_landmarks[0])
        hips_vectors = [
            np.array([hip_landmark.x, hip_landmark.y, hip_landmark.z])
            for hip_landmark in landmarks_frame[hips_indices]
        ]
        knees_vectors = [
            np.array([knee_landmark.x, knee_landmark.y, knee_landmark.z])
            for knee_landmark in landmarks_frame[knees_indices]
        ]
        ankles_vectors = [
            np.array([ankle_landmark.x, ankle_landmark.y, ankle_landmark.z])
            for ankle_landmark in landmarks_frame[ankles_indices]
        ]

        mid_hips = hips_vectors[0] + 0.5 * hips_vectors[1]
        mid_knees = knees_vectors[0] + 0.5 * knees_vectors[1]
        mid_ankles = ankles_vectors[0] + 0.5 * ankles_vectors[1]

        mid_hips_to_mid_knees = mid_knees - mid_hips
        mid_knees_to_mid_ankles = mid_ankles - mid_knees

        unit_vector1 = mid_hips_to_mid_knees / np.linalg.norm(mid_hips_to_mid_knees)
        unit_vector2 = mid_knees_to_mid_ankles / np.linalg.norm(mid_knees_to_mid_ankles)

        dot_product = np.dot(unit_vector1, unit_vector2)
        angle = np.arccos(dot_product)

        knee_angles[i] = 180 - np.degrees(angle)

    upper_limit = 90

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=np.array(range(len(knee_angles))) + 1,
            y=knee_angles,
        )
    )

    # fig.update_yaxes(autorange="reversed")

    fig.add_hline(
        upper_limit,
        annotation_text="Upper Limit",
        line_dash="dot",
        line_color="red",
        line_width=1,
    )

    fig.add_vline(
        frame_nr,
        line_width=1,
        line_color="red",
        line_dash="dot",
        annotation_text="Selected Frame",
    )

    fig.update_layout(
        title="Angle formed by Hips and Ankles, centered at Knees",
        hovermode="x",
        xaxis=dict(title="Frame"),
        yaxis=dict(title="Angle [\u00b0]"),
    )

    all_good = (
        np.sum(np.diff((knee_angles < upper_limit).astype(int)) == 1) >= nr_of_reps
    )

    return all_good, fig
