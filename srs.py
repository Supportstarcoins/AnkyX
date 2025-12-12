from typing import Dict

learning_steps_min = [10, 60, 1440]
relearning_steps_min = [10, 1440]
ease_start = 250
ease_min = 130


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def schedule_review(card_row: Dict, rating: int, now_ts: int) -> Dict:
    """
    Рассчитать новое состояние карточки.
    rating: 0=Again, 1=Hard, 2=Good, 3=Easy
    """
    state = card_row.get("state") or "new"
    interval = _safe_int(card_row.get("interval"), 0)
    ease = _safe_int(card_row.get("ease"), ease_start) or ease_start
    reps = _safe_int(card_row.get("reps"), 0)
    lapses = _safe_int(card_row.get("lapses"), 0)
    step_index = _safe_int(card_row.get("step_index"), 0)
    phase = _safe_int(card_row.get("phase") or card_row.get("leitner_level"), 1)
    due = _safe_int(card_row.get("due"), now_ts)

    interval_before = interval
    ease_before = ease
    phase_before = phase

    rating = max(0, min(3, rating))

    def minutes_to_seconds(minutes: int) -> int:
        return minutes * 60

    def days_to_seconds(days: int) -> int:
        return days * 24 * 60 * 60

    state_after = state
    due_after = due
    interval_after = interval
    ease_after = ease
    phase_after = phase

    if state in ("new", "learning"):
        state_after = "learning"
        if rating == 0:
            step_index = 0
            due_after = now_ts + minutes_to_seconds(learning_steps_min[0])
            phase_after = max(1, phase - 1)
        else:
            step_index += 1
            if step_index >= len(learning_steps_min):
                state_after = "review"
                interval_after = max(1, interval or 1)
                due_after = now_ts + days_to_seconds(interval_after)
                ease_after = ease_start
                reps += 1
                phase_after = min(10, phase + 1)
            else:
                due_after = now_ts + minutes_to_seconds(learning_steps_min[step_index])
                interval_after = interval
                ease_after = ease
    elif state == "review":
        reps += 1
        if rating == 0:
            state_after = "relearning"
            step_index = 0
            lapses += 1
            due_after = now_ts + minutes_to_seconds(relearning_steps_min[0])
            phase_after = max(1, phase - 2)
        elif rating == 1:
            ease_after = max(ease_min, ease - 20)
            interval_after = max(1, int(interval * 1.2))
            due_after = now_ts + days_to_seconds(interval_after)
            phase_after = phase
        else:
            ease_delta = 15 if rating >= 3 else 0
            ease_after = max(ease_min, ease + ease_delta)
            interval_after = max(1, int(interval * ease_after / 1000))
            if interval_after <= 0:
                interval_after = 1
            if rating >= 3:
                interval_after = int(interval_after * 1.3) + 1
            due_after = now_ts + days_to_seconds(interval_after)
            phase_after = min(10, phase + (2 if rating >= 3 else 1))
    elif state == "relearning":
        if rating == 0:
            step_index = 0
            due_after = now_ts + minutes_to_seconds(relearning_steps_min[0])
            phase_after = max(1, phase - 1)
        else:
            step_index += 1
            if step_index >= len(relearning_steps_min):
                state_after = "review"
                interval_after = max(1, int(max(1, interval) * 0.5))
                due_after = now_ts + days_to_seconds(interval_after)
                phase_after = min(10, phase + 1)
            else:
                due_after = now_ts + minutes_to_seconds(relearning_steps_min[step_index])
    else:
        state_after = "review"
        due_after = now_ts + days_to_seconds(max(1, interval or 1))

    return {
        "state": state_after,
        "due": int(due_after),
        "interval": int(interval_after),
        "ease": int(ease_after),
        "reps": int(reps),
        "lapses": int(lapses),
        "step_index": int(step_index),
        "last_review": int(now_ts),
        "phase": int(phase_after),
        "interval_before": interval_before,
        "interval_after": int(interval_after),
        "ease_before": ease_before,
        "ease_after": int(ease_after),
        "phase_before": phase_before,
        "phase_after": int(phase_after),
    }
