import numpy as np
import pandas as pd
import plotly.express as px
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from plotly.graph_objects import Figure


def get_knee_distance_verdict_and_fig(
    frames: int,
    pose_landmarks: list[PoseLandmarkerResult],
    active_indices_starts: np.ndarray,
    active_indices_lengths: np.ndarray,
    frame_nr: int,
) -> tuple[bool, Figure]:
    knees_indices, ankles_indices = [25, 26], [27, 28]
    knees_distances, ankles_distances = np.empty(frames), np.empty(frames)

    for i, landmarks_frame in enumerate(pose_landmarks):
        landmarks_frame = np.array(landmarks_frame.pose_world_landmarks[0])
        knees_vectors = [
            np.array([knee_landmark.x, knee_landmark.y])
            for knee_landmark in landmarks_frame[knees_indices]
        ]
        ankles_vectors = [
            np.array([ankle_landmark.x, ankle_landmark.y])
            for ankle_landmark in landmarks_frame[ankles_indices]
        ]

        knees_distances[i] = np.linalg.norm(knees_vectors[0] - knees_vectors[1])
        ankles_distances[i] = np.linalg.norm(ankles_vectors[0] - ankles_vectors[1])

    distances = knees_distances - ankles_distances

    lower_limit = knees_distances.mean() * -0.25

    distances_df = pd.DataFrame(dict(Frame=list(range(frames)), Distance=distances))

    fig = px.line(distances_df, x="Frame", y="Distance")

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

    for i, (start, length) in enumerate(zip(active_indices_starts, active_indices_lengths)):
        fig.add_vrect(
            start,
            start + length,
            fillcolor="red",
            annotation_text="Active" if i == 0 else "",
            annotation_position="bottom left",
            opacity=0.1,
            line_width=0,
        )

    fig.update_layout(
        title="Knee Distance minus Ankle Distance",
        hovermode="x",
        xaxis=dict(title="Frame"),
        yaxis=dict(title="Distance [m]"),
    )

    active_periods_indices = np.concatenate(
        [
            np.arange(start, start + length)
            for start, length in zip(active_indices_starts, active_indices_lengths)
        ]
    )
    all_good = all(distances_df["Distance"][active_periods_indices] > lower_limit)

    return all_good, fig
