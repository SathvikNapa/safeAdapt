"""Fixed-core item text for the two validated instruments the topic guide is
built from (§3.5). Vendored as plain strings so this repo has no dependency
on the production SurveyAI platform — this is the *content*, not the scoring
engine, and content never changes per persona/condition.

- PHQ-9: Patient Health Questionnaire (depression). Public domain (Pfizer).
- Climate distress: short item set in the spirit of Latkin et al. (2022)'s
  climate change distress/activism scale; not a verbatim reproduction of the
  published instrument — swap in the licensed item text before using results
  for anything beyond pipeline validation.
"""

from __future__ import annotations

PHQ9_ITEMS: list[tuple[str, str]] = [
    ("phq9_1", "Little interest or pleasure in doing things"),
    ("phq9_2", "Feeling down, depressed, or hopeless"),
    ("phq9_3", "Trouble falling or staying asleep, or sleeping too much"),
    ("phq9_4", "Feeling tired or having little energy"),
    ("phq9_5", "Poor appetite or overeating"),
    (
        "phq9_6",
        "Feeling bad about yourself, or that you are a failure or have let "
        "yourself or your family down",
    ),
    (
        "phq9_7",
        "Trouble concentrating on things, such as reading the newspaper or "
        "watching television",
    ),
    (
        "phq9_8",
        "Moving or speaking so slowly that other people could have noticed? Or "
        "the opposite: being so fidgety or restless that you have been moving "
        "around a lot more than usual",
    ),
    ("phq9_9", "Thoughts that you would be better off dead, or of hurting yourself in some way"),
]

CLIMATE_DISTRESS_ITEMS: list[tuple[str, str]] = [
    ("climate_1", "How concerned are you, personally, about climate change?"),
    ("climate_2", "How much does thinking about climate change affect your day-to-day mood?"),
    ("climate_3", "How often do you feel a sense of hopelessness about the future because of climate change?"),
    ("climate_4", "How much do you feel able to do anything meaningful about climate change?"),
    ("climate_5", "How often do you talk with friends or family about climate change?"),
]
