# -*- coding: utf-8 -*-
"""Mode Manuel, de bout en bout dans l'app Streamlit."""
from datetime import time
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).with_name("app.py")


def test_ajouter_une_etape_decoupe_la_mission_aux_gares_de_passage():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["gares"] = pd.DataFrame({
        "gare": ["A", "B", "C"], "distance": [0.0, 9.0, 20.0], "infra": ["F", "VE", "F"],
    })
    app.session_state["missions"] = [{
        "origine": "A", "terminus": "C", "frequence": 1.0, "temps_trajet": 30,
        "temps_retournement_A": 5, "temps_retournement_B": 5, "reference_minutes": "0",
        "passing_points": [{"gare": "B", "time_offset_min": 14}],
    }]
    app.session_state["mode_calcul_selector"] = "Standard"
    app.session_state["mode_generation"] = "Manuel"
    app.session_state["heure_debut_service"] = time(8, 0)
    app.session_state["heure_fin_service"] = time(22, 0)
    app.run()
    assert not app.exception

    [ajouter] = [b for b in app.button if b.key == "add_e_1"]
    ajouter.click().run()

    assert not app.exception, app.exception
    etapes = app.session_state["roulement_manuel"][1]
    assert [(e["depart"], e["arrivee"]) for e in etapes] == [("A", "B"), ("B", "C")]
