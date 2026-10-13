from copy import deepcopy
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import optimisation_logic
import robustness_logic


APP_PATH = Path(__file__).with_name("app.py")


def _infrastructure():
    return pd.DataFrame(
        [
            {"gare": "A", "distance": 0.0, "infra": "F", "electrification": "F"},
            {"gare": "X", "distance": 10.0, "infra": "VE", "electrification": "F"},
            {"gare": "B", "distance": 20.0, "infra": "F", "electrification": "F"},
        ]
    )


def _mission():
    return {
        "origine": "A",
        "terminus": "B",
        "frequence": 1.0,
        "temps_trajet": 20,
        "temps_retournement_A": 5,
        "temps_retournement_B": 5,
        "reference_minutes": "0",
        "type_materiel": "diesel",
    }


def _chronologie():
    jour = datetime(2026, 1, 1)
    return {
        "T1": [
            {
                "start": jour.replace(hour=8),
                "end": jour.replace(hour=8, minute=10),
                "origine": "A",
                "terminus": "X",
                "mission": "A → B",
                "is_mission_start": True,
            },
            {
                "start": jour.replace(hour=8, minute=10),
                "end": jour.replace(hour=8, minute=20),
                "origine": "X",
                "terminus": "B",
                "mission": "A → B",
                "is_mission_start": False,
            },
        ]
    }


def _app_test():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["gares"] = _infrastructure()
    app.session_state["missions"] = [_mission()]
    app.session_state["chronologie_calculee"] = _chronologie()
    app.session_state["mode_calcul_selector"] = "Standard"
    app.session_state["mode_generation"] = "Rotation optimisée"
    app.session_state["heure_debut_service"] = time(6, 0)
    app.session_state["heure_fin_service"] = time(22, 0)
    app.session_state["run_calculation"] = False
    app.session_state["warnings_calcul"] = {}
    app.session_state["stats_homogeneite"] = {}
    return app


def _interdire_optimisation(monkeypatch):
    appels = []

    def interdite(*args, **kwargs):
        appels.append((args, kwargs))
        raise AssertionError("Le calcul de robustesse ne doit pas relancer l'optimisation")

    monkeypatch.setattr(optimisation_logic, "optimiser_graphique_horaire", interdite)
    return appels


def _compter_calcul_robustesse(monkeypatch):
    appels = []
    calculer = robustness_logic.evaluer_robustesse

    def compte(*args, **kwargs):
        appels.append((args, kwargs))
        return calculer(*args, **kwargs)

    monkeypatch.setattr(robustness_logic, "evaluer_robustesse", compte)
    return appels


def _cliquer_calcul_robustesse(app):
    boutons = [button for button in app.button if button.key == "btn_calc_robustesse"]
    assert len(boutons) == 1
    boutons[0].click().run(timeout=30)


def test_bouton_robustesse_lance_son_calcul_sans_reoptimiser(monkeypatch):
    appels_optimisation = _interdire_optimisation(monkeypatch)
    appels_robustesse = _compter_calcul_robustesse(monkeypatch)
    app = _app_test().run(timeout=30)

    assert not app.exception
    assert appels_robustesse == []
    _cliquer_calcul_robustesse(app)

    assert not app.exception
    resultat = app.session_state["robustesse_resultats"]
    assert [scenario["delay_min"] for scenario in resultat["scenarios"]] == [5, 10, 20]
    assert len(appels_robustesse) == 1
    assert appels_optimisation == []


def test_chronologie_modifiee_invalide_les_resultats_sans_reoptimiser(monkeypatch):
    appels_optimisation = _interdire_optimisation(monkeypatch)
    appels_robustesse = _compter_calcul_robustesse(monkeypatch)
    app = _app_test().run(timeout=30)
    assert not app.exception
    assert appels_robustesse == []

    _cliquer_calcul_robustesse(app)
    assert app.session_state["robustesse_resultats"] is not None
    assert len(appels_robustesse) == 1

    chronologie = deepcopy(app.session_state["chronologie_calculee"])
    chronologie["T1"][0]["end"] += timedelta(minutes=1)
    app.session_state["chronologie_calculee"] = chronologie
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["robustesse_resultats"] is None
    assert any("invalid" in message.value.lower() for message in app.info)
    assert len(appels_robustesse) == 1
    assert appels_optimisation == []


def test_reserve_active_propose_les_terminus_concrets(monkeypatch):
    appels_optimisation = _interdire_optimisation(monkeypatch)
    appels_robustesse = _compter_calcul_robustesse(monkeypatch)
    app = _app_test()
    app.session_state["robustesse_reserve_active"] = True
    app.session_state["robustesse_reserve_gares"] = ["B"]
    app.run(timeout=30)

    assert not app.exception
    assert appels_robustesse == []
    choix_gares = [item for item in app.multiselect if item.key == "robustesse_reserve_gares"]
    assert len(choix_gares) == 1
    assert choix_gares[0].options == ["A", "B"]

    _cliquer_calcul_robustesse(app)

    assert not app.exception
    assert all(
        scenario["avec_reserve"] is not None
        for scenario in app.session_state["robustesse_resultats"]["scenarios"]
    )
    assert len(appels_robustesse) == 1
    assert appels_optimisation == []

    app.session_state["robustesse_reserve_gares"] = ["A"]
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["robustesse_resultats"] is None
    assert any("invalid" in message.value.lower() for message in app.info)
    assert len(appels_robustesse) == 1


def test_echec_recalcul_ne_reaffiche_pas_les_anciens_resultats(monkeypatch):
    app = _app_test().run(timeout=30)
    _cliquer_calcul_robustesse(app)
    assert app.session_state["robustesse_resultats"] is not None

    def echouer(*args, **kwargs):
        raise ValueError("Échec de rejeu volontaire")

    monkeypatch.setattr(robustness_logic, "evaluer_robustesse", echouer)
    _cliquer_calcul_robustesse(app)
    assert not app.exception
    assert app.session_state["robustesse_resultats"] is None
    assert any("Échec de rejeu volontaire" in item.value for item in app.error)
    assert not any("Incident +" in item.label for item in app.expander)
