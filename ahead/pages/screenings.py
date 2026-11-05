"""
ahead/pages/screenings.py
=========================
Condition picker, then the guided predictor for the chosen condition.
"""

import streamlit as st

from ahead.components import condition_cards, current_user, go_to, page_header, stepper
from ahead.config import DISEASES
from ahead.screening import render_predictor


def render_screenings() -> None:
    selected = st.session_state.get("active_screening")

    if selected not in DISEASES:
        if current_user()["is_doctor"]:
            page_header("AHEAD Clinical Screening", "Choose a screening",
                        "Select a condition to score a set of entered values with its screening model.")
            st.info("Screenings run here are not attached to a patient record. A linked patient's saved assessments "
                    "are in Patient Intelligence, and files of patient records are scored on the Clinical Dashboard.")
        else:
            page_header(
                "AHEAD Clinical Screening",
                "Choose a screening",
                "Select a condition to begin a guided early-awareness assessment.",
            )
        condition_cards("begin", button_label="Begin screening")
        return

    if st.button("← Back to all screenings", key="back_to_screenings", type="tertiary"):
        go_to("screenings")

    stepper_slot = st.empty()          # filled once we know whether a result is on screen
    has_result = render_predictor(selected)
    with stepper_slot:
        stepper(3 if has_result else 1)
