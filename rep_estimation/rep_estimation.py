import numpy as np
import plotly.graph_objects as go
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity

from utils.constants import LANDMARKS_TO_INCLUDE


def get_rep_infos(
    pose_landmarks: np.ndarray[PoseLandmarkerResult],
    fps: float,
    frames: int,
    frame_nr: int,
) -> tuple[int, float, np.ndarray, np.ndarray, go.Figure]:
    vid_length_in_s = frames / fps
    similarities = _get_cosine_similarities(pose_landmarks)
    frame_0_similarity = similarities[0]

    frame_0_similarity_filtered = _get_avg_filtered_signal(fps, frame_0_similarity)
    frame_0_peaks, _ = find_peaks(frame_0_similarity_filtered, distance=2 * fps)

    if frame_0_peaks[0] > fps:  # No Peak found within first second
        frame_0_peaks = np.insert(frame_0_peaks, 0, 0)
    if (
        frame_0_peaks[-1] < (vid_length_in_s - 1) * fps
    ):  # No Peak found within last second
        frame_0_peaks = np.append(frame_0_peaks, len(frame_0_similarity) - 1)

    rest_indices_starts, rest_indices_lengths = _get_rest_indices_starts_and_lengths(
        frame_0_similarity_filtered, int(fps / 2)
    )
    reps_fig = _get_frequency_signal_fig(
        frame_0_similarity_filtered,
        frame_nr,
        rest_indices_starts,
        rest_indices_lengths,
    )

    nr_of_reps = len(frame_0_peaks) - 1
    indices_under_tension = frames - rest_indices_lengths.sum()
    time_under_tension = indices_under_tension / fps
    active_indices_starts, active_indices_lengths = (
        _get_active_indices_starts_and_lengths(
            rest_indices_starts, rest_indices_lengths, frames
        )
    )

    return (
        nr_of_reps,
        time_under_tension,
        active_indices_starts,
        active_indices_lengths,
        reps_fig,
    )


def _get_rest_indices_starts_and_lengths(
    frame_similarities, min_consecutive_resting_length
):
    abs_first_derivative = np.abs(np.diff(frame_similarities))
    abs_first_derivative = np.append(abs_first_derivative, abs_first_derivative[-1])
    low_diffs = abs_first_derivative < 2e-4

    consecutive_low_diff_starts, consecutive_low_diff_lengths = (
        _get_consecutive_low_diff_starts_and_lengths(
            low_diffs, min_consecutive_resting_length
        )
    )

    resting_indices_starts, resting_indices_lengths = [], []
    for start, length in zip(consecutive_low_diff_starts, consecutive_low_diff_lengths):
        if all(
            frame_similarities[start : start + length] > 0.985
        ):  # Do not include pauses within Rep
            resting_indices_starts.append(start)
            resting_indices_lengths.append(length)

    return np.array(resting_indices_starts), np.array(resting_indices_lengths)


def _get_active_indices_starts_and_lengths(
    rest_indices_starts, rest_indices_lengths, total_length
):
    rest_indices_ends = [
        start + length
        for start, length in zip(rest_indices_starts, rest_indices_lengths)
    ]

    active_indices_starts, active_indices_lengths = [], []
    if rest_indices_starts[0] > 0:
        active_indices_starts.append(0)
        active_indices_lengths.append(rest_indices_starts[0] - 1)

    for i in range(1, len(rest_indices_starts)):
        active_start = rest_indices_ends[i - 1]
        active_end = rest_indices_starts[i]
        if active_start < active_end:
            active_indices_starts.append(active_start)
            active_indices_lengths.append(active_end - active_start)

    if rest_indices_ends[-1] < total_length:
        active_indices_starts.append(rest_indices_ends[-1])
        active_indices_lengths.append(total_length - rest_indices_ends[-1])

    return np.array(active_indices_starts), np.array(active_indices_lengths)


def _get_consecutive_low_diff_starts_and_lengths(low_diffs, min_consecutive_length):
    n = len(low_diffs)
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(low_diffs[:-1], low_diffs[1:], out=loc_run_start[1:])

    run_starts = np.nonzero(loc_run_start)[0]
    run_lengths = np.diff(np.append(run_starts, n))

    consecutive_low_diff_starts, consecutive_low_diff_lengths = [], []

    for start, length in zip(run_starts, run_lengths):
        if low_diffs[start] and length >= min_consecutive_length:
            consecutive_low_diff_starts.append(start)
            consecutive_low_diff_lengths.append(length)

    return np.array(consecutive_low_diff_starts), np.array(consecutive_low_diff_lengths)


def _get_avg_filtered_signal(fps, signal):
    signal_padded = np.pad(signal, int(fps / 2), "edge")
    avg_filter = np.ones(int(fps)) / fps
    signal_filtered = np.convolve(signal_padded, avg_filter, "valid")
    return signal_filtered


def _get_cosine_similarities(
    pose_landmarks,
):
    frames = len(pose_landmarks)
    landmark_vectors = np.empty((frames, len(LANDMARKS_TO_INCLUDE), 3))

    for i, landmarks_frame in enumerate(pose_landmarks):
        landmarks_frame = np.array(landmarks_frame.pose_landmarks[0])

        landmarks_frame_to_include = landmarks_frame[LANDMARKS_TO_INCLUDE]
        vector_frame = np.array(
            [
                np.array([landmark.x, landmark.y, landmark.z])
                for landmark in landmarks_frame_to_include
            ]
        )

        landmark_vectors[i] = vector_frame

    flattened_vectors = landmark_vectors.reshape(frames, -1)
    similarities = cosine_similarity(flattened_vectors, flattened_vectors)
    return similarities


def _get_frequency_signal_fig(
    frame_similarity,
    frame_nr,
    rest_indices_starts,
    rest_indices_lengths,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            mode="lines",
            name="Similarity",
            showlegend=False,
            x=list(range(len(frame_similarity))),
            y=frame_similarity,
        )
    )

    fig.update_layout(
        hovermode="x",
        title="Repetition Signal",
        xaxis=dict(title="Frames"),
    )
    fig.update_yaxes(visible=False)
    fig.add_vline(
        frame_nr,
        line_width=1,
        line_color="red",
        line_dash="dot",
        annotation_text="Selected Frame",
    )

    for i, (start, length) in enumerate(zip(rest_indices_starts, rest_indices_lengths)):
        fig.add_vrect(
            start,
            start + length,
            fillcolor="green",
            annotation_text="Rest" if i == 0 else "",
            annotation_position="bottom left",
            opacity=0.25,
            line_width=0,
        )

    return fig
