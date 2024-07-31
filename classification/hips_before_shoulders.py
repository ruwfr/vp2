import numpy as np
import plotly.graph_objects as go
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from plotly.graph_objects import Figure


def get_hips_before_shoulders_verdict_and_fig(
    frames: int,
    pose_landmarks: list[PoseLandmarkerResult],
    frame_nr: int,
) -> Figure:
    shoulders_indices, hips_indices = [11, 12], [23, 24]
    shoulders_y, hips_y = np.empty(frames), np.empty(frames)

    for i, landmarks_frame in enumerate(pose_landmarks):
        landmarks_frame = np.array(landmarks_frame.pose_world_landmarks[0])

        shoulders_vectors = [
            np.array([shoulder_landmark.y])
            for shoulder_landmark in landmarks_frame[shoulders_indices]
        ]
        hips_vectors = [
            np.array([hip_landmark.y]) for hip_landmark in landmarks_frame[hips_indices]
        ]

        mid_shoulders = shoulders_vectors[0] + 0.5 * shoulders_vectors[1]
        mid_hips = hips_vectors[0] + 0.5 * hips_vectors[1]

        shoulders_y[i] = mid_shoulders
        hips_y[i] = mid_hips

    shoulders_diff = np.diff(shoulders_y)
    hips_diff = np.diff(hips_y)

    relative_speed = shoulders_diff - hips_diff
    upper_limit = 0.03
    lower_limit = -upper_limit

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            mode="lines",
            showlegend=False,
            x=list(range(len(relative_speed))),
            y=relative_speed,
        )
    )
    fig.add_hline(
        upper_limit,
        annotation_text="Upper Limit",
        line_dash="dot",
        line_color="red",
        line_width=1,
    )
    fig.add_hline(
        lower_limit,
        annotation_text="Lower Limit",
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
        yaxis=dict(title="Relative Speed"),
    )

    all_good = np.all((relative_speed >= lower_limit) & (relative_speed <= upper_limit))

    return all_good, fig
