from mediapipe.python.solutions.pose import PoseLandmark

MODEL_TYPE = "lite"  # heavy, full or lite

LANDMARKS_TO_INCLUDE = [
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE,
]
EXPLANATION_TEXT = "See Explanation"

GOOD_TEXT = "All good here. Well done!"

CAVED_IN_KNEES_BAD_TEXT = """The analysis shows that your knees caved in during some part of the exercise. Try to keep your knees further apart. The line should be above the lower limit during the active parts of the movement shaded in red."""

DEPTH_BAD_TEXT = """It looks like you have not utilized the full range of motion of the exercise. Try to go deeper next time. The line should be below the upper limit at the lowest points."""

HIPS_BEFORE_SHOULDERS_BAD_TEXT = """You seem to be rising with your hips first when standing back up. Try to rise your hips at the same speed as your shoulders. The line should always be between the limits."""
