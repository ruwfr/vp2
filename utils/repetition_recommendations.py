def get_repetition_recommendation_text(
    nr_of_reps: int, time_under_tension: float
) -> str:
    if nr_of_reps >= 1 and nr_of_reps <=5:
        training_goal = "Strength"
    if nr_of_reps >= 5 and nr_of_reps <=12:
        training_goal = "Hypertrophy"
    if nr_of_reps >= 12:
        training_goal = "Endurance"

    is_optimal = time_under_tension / nr_of_reps >= 2.5

    recommendation_text = f"Your number of repetitions suggest you are training mainly for {training_goal}. "
    if is_optimal:
        recommendation_text += (
            "Your time under tension is in a good range, well done!"
        )
    else:
        recommendation_text += "Your time under tension is too low for optimal muscular development. Try slowing down your repetitions."

    return recommendation_text
