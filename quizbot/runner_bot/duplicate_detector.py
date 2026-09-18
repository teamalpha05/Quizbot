"""Behaviour-based duplicate-account detection for group quizzes.

Telegram does not expose whether two Telegram accounts belong to the same
person. This module therefore detects unusually similar quiz activity only.
It never treats a Telegram user ID as proof of identity.
"""

from __future__ import annotations

from typing import Any


class DuplicateAccountDetector:
    """Compare two participants' answers in the same running quiz session."""

    def __init__(
        self,
        min_common_answers: int = 5,
        min_answer_similarity: float = 0.90,
        max_timing_difference: float = 2.0,
    ) -> None:
        self.min_common_answers = min_common_answers
        self.min_answer_similarity = min_answer_similarity
        self.max_timing_difference = max_timing_difference

    def find_suspicious_pair(
        self,
        session: dict[str, Any],
        current_user_id: int,
    ) -> int | None:
        """Return a likely duplicate user ID, or None.

        A match requires enough questions answered by both accounts, a very
        high same-option ratio, and similar response delays measured from the
        corresponding poll start time. This is deliberately conservative.
        """
        participants = session.get("participants") or {}
        current = participants.get(current_user_id)
        if not current:
            return None

        current_answers = current.get("answers") or {}
        if len(current_answers) < self.min_common_answers:
            return None

        polls = session.get("polls") or {}
        current_by_q = self._by_question(current_answers, polls)
        if len(current_by_q) < self.min_common_answers:
            return None

        for other_id, other in participants.items():
            if other_id == current_user_id:
                continue

            other_answers = other.get("answers") or {}
            other_by_q = self._by_question(other_answers, polls)
            common = sorted(set(current_by_q) & set(other_by_q))

            if len(common) < self.min_common_answers:
                continue

            same_answers = 0
            timing_matches = 0

            for q_index in common:
                a = current_by_q[q_index]
                b = other_by_q[q_index]

                if a.get("option") == b.get("option"):
                    same_answers += 1

                a_delay = self._delay(a)
                b_delay = self._delay(b)
                if a_delay is not None and b_delay is not None:
                    if abs(a_delay - b_delay) <= self.max_timing_difference:
                        timing_matches += 1

            answer_similarity = same_answers / len(common)
            timing_similarity = timing_matches / len(common)

            if (
                answer_similarity >= self.min_answer_similarity
                and timing_similarity >= self.min_answer_similarity
            ):
                return int(other_id)

        return None

    @staticmethod
    def _by_question(
        answers: dict[Any, dict[str, Any]],
        polls: dict[Any, dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for poll_id, answer in answers.items():
            poll = polls.get(poll_id)
            if not poll:
                continue
            try:
                q_index = int(poll.get("question_index", -1))
            except (TypeError, ValueError):
                continue
            if q_index < 0:
                continue
            result[q_index] = answer
        return result

    @staticmethod
    def _delay(answer: dict[str, Any]) -> float | None:
        try:
            sent = float(answer.get("sent_time"))
            answered = float(answer.get("time"))
            return answered - sent
        except (TypeError, ValueError):
            return None
