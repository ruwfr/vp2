def get_repetition_recommendation_text(
    nr_of_reps: int, time_under_tension: float
) -> str:
    is_optimal = False

    if 1 <= nr_of_reps < 5:
        training_goal = "Strength"
        if 2 <= time_under_tension <= 20:
            is_optimal = True
        elif time_under_tension < 2:
            too_fast = True
        else:
            too_fast = False

    elif 5 <= nr_of_reps < 8:
        training_goal = "Strength and Hypertrophy"
        if 20 <= time_under_tension <= 40:
            is_optimal = True
        elif time_under_tension < 20:
            too_fast = True
        else:
            too_fast = False

    elif 8 <= nr_of_reps <= 12:
        training_goal = "Hypertrophy"
        if 40 <= time_under_tension <= 70:
            is_optimal = True
        elif time_under_tension < 40:
            too_fast = True
        else:
            too_fast = False

    else:
        return "Number of repetitions out of optimal range. Check the table in the expandable explanation for optimal values."

    recommendation_text = f"Your number of repetitions suggest you are training mainly for {training_goal}. "
    if is_optimal:
        recommendation_text += (
            "Your time under tension is optimal for your goal, well done!"
        )
    else:
        if too_fast:
            recommendation_text += "Your time under tension is too low for your training goal. Try slowing down your repetitions."
        else:
            recommendation_text += "Your time under tension is too high for your training goal. Try speeding up your repetitions."

    return recommendation_text
