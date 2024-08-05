import glob
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from classification.caved_in_knees import get_knee_distance_verdict_and_fig
from classification.depth import get_depth_verdict_and_fig
from classification.hips_before_shoulders import (
    get_hips_before_shoulders_verdict_and_fig,
)
from pose_estimation.pose_estimation import get_annotated_video, get_pose_landmarks
from rep_estimation.rep_estimation import get_rep_infos
from utils.constants import (
    CAVED_IN_KNEES_BAD_TEXT,
    DEPTH_BAD_TEXT,
    EXPLANATION_TEXT,
    GOOD_TEXT,
    HIPS_BEFORE_SHOULDERS_BAD_TEXT,
)
from utils.repetition_recommendations import get_repetition_recommendation_text
from utils.video import read_video

if __name__ == "__main__":
    st.title("Upload your own Video")
    uploaded_video_io = st.file_uploader(
        "Upload Video",
        type=[".mp4", ".avi"],
        help="Videos from the front with about a 45 degree angle and in landscape mode work best.",
    )
    if uploaded_video_io:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video_io.read())

    st.title("Or choose from Example Videos")
    example_videos = list(glob.glob("example_videos/*.mp4"))

    video_selected = st.selectbox(
        "Select Video",
        example_videos,
        disabled=bool(uploaded_video_io),
        format_func=lambda p: Path(p).stem.replace("_", " ").title(),
    )

    if uploaded_video_io:
        video_selected = temp_file.name

    video, fps = read_video(video_selected)
    frames = len(video)

    pose_landmarks = get_pose_landmarks(video, frames)

    annotated_video = get_annotated_video(video, pose_landmarks)

    frame_nr = st.slider("Select Frame Number", max_value=frames - 1)
    st.image(annotated_video[frame_nr])

    pose_estimation_failed_frames = [
        i for i in range(frames) if len(pose_landmarks[i].pose_landmarks) == 0
    ]

    if pose_estimation_failed_frames:
        st.error(
            f"Pose Estimation did not work for frames {pose_estimation_failed_frames}. Please try a different video."
        )
    else:
        (
            nr_of_reps,
            time_under_tension,
            active_indices_starts,
            active_indices_lengths,
            reps_fig,
        ) = get_rep_infos(pose_landmarks, fps, frames, frame_nr)
        st.title("Repetition Infos")
        repetition_recommendation_text = get_repetition_recommendation_text(
            nr_of_reps, time_under_tension
        )
        st.write(repetition_recommendation_text)

        col1, col2 = st.columns(2)

        col1.metric("Detected Number Of Repetitions", nr_of_reps)
        col2.metric(
            "Total Time under Tension [s]",
            round(time_under_tension, 2),
        )
        with st.expander(EXPLANATION_TEXT):
            st.markdown("""Time under Tension refers to the time a muscle is kept under tension during an exercise. 
                        Higher Time under Tension has been associated with greater increases in rates of muscle protein synthesis than the same movement performed rapidly.)
                        [Source](https://pubmed.ncbi.nlm.nih.gov/22106173/)
                        """)

        st.plotly_chart(reps_fig)
        with st.expander(EXPLANATION_TEXT):
            st.markdown("""The figure shows your repetitions as a signal. 
            The signal is based on the similarity of your body position relative to your starting position and is
            used to estimate the Metrics above.
            The green shaded areas indicate resting between repetitions.""")

        st.title("Caved in Knees")
        all_good_knee_distance, knee_distance_fig = get_knee_distance_verdict_and_fig(
            frames,
            pose_landmarks,
            active_indices_starts,
            active_indices_lengths,
            frame_nr,
        )
        st.write(GOOD_TEXT if all_good_knee_distance else CAVED_IN_KNEES_BAD_TEXT)
        st.plotly_chart(knee_distance_fig)
        with st.expander(EXPLANATION_TEXT):
            st.markdown(
                """Allowing the knees to fall into valgus changes the stress that normally occurs across the knee joint, 
                which can cause pain and increases the risk of injury.
                [Source](https://www.trifectatherapeutics.com/blog/knees-caving-in-when-squattingjumping-heres-your-fix-for-knee-valgus)"""
            )

        st.title("Depth")
        all_good_depth, depth_fig = get_depth_verdict_and_fig(
            frames,
            pose_landmarks,
            frame_nr,
            nr_of_reps,
        )
        st.write(GOOD_TEXT if all_good_depth else DEPTH_BAD_TEXT)
        st.plotly_chart(depth_fig)
        with st.expander(EXPLANATION_TEXT):
            st.markdown(
                """Focusing on range of motion in resistance training has been shown to increase strength and develop the muscles 
                more effectively than when doing partial range of motion.
                [Source](https://pubmed.ncbi.nlm.nih.gov/31230110)
                [Source](https://journals.lww.com/nsca-jscr/fulltext/2012/08000/effect_of_range_of_motion_on_muscle_strength_and.17.aspx)"""
            )

        st.title("Hips before Shoulders")
        all_good_hips_before_shoulders, hips_before_shoulders_fig = (
            get_hips_before_shoulders_verdict_and_fig(
                frames,
                pose_landmarks,
                frame_nr,
            )
        )
        st.write(
            GOOD_TEXT
            if all_good_hips_before_shoulders
            else HIPS_BEFORE_SHOULDERS_BAD_TEXT
        )
        st.plotly_chart(hips_before_shoulders_fig)
        with st.expander(EXPLANATION_TEXT):
            st.markdown(
                """Raising your hips faster than your shoulders when performing the squat is called the good-morning movement because of its resemblance 
                to the this different exercise. The good-morning movement reduces the load through the quads and increases the load through 
                the glutes and hamstrings, which is not the aim of a squat movement.
            [Source](https://www.lbphysiotherapy.co.uk/blog/2020/11/30/the-squat-movement-good-bad-and-the-ugly#:~:text=The%20good%2Dmorning%20movement%20reduces,in%20your%20hamstrings%20and%20quadriceps.)
            """
            )
