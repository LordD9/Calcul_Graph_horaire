# -*- coding: utf-8 -*-
"""Un résultat affiché a toujours été calculé sur les entrées courantes.

Rejoue a862025 : nouvelle grille de même taille, ancien bilan énergie affiché.
"""
from copy import deepcopy
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import etat_session
from energy_logic import calculer_energie_flotte, get_default_energy_params

APP_PATH = Path(__file__).with_name("app.py")


def _gares():
    return pd.DataFrame({"gare": ["A", "B"], "distance": [0.0, 20.0], "infra": ["F", "F"],
                         "electrification": ["F", "F"], "rampe_section_a_venir": [0.0, 0.0]})


def _missions():
    return [{"origine": "A", "terminus": "B", "frequence": 1.0, "temps_trajet": 20,
             "temps_retournement_A": 5, "temps_retournement_B": 5, "reference_minutes": "0",
             "type_materiel": "diesel", "passing_points": []}]


def _chronologie(heure):
    jour = datetime(2026, 1, 1)
    return {1: [{"start": jour.replace(hour=heure), "end": jour.replace(hour=heure, minute=20),
                 "origine": "A", "terminus": "B", "mission": "A → B", "is_mission_start": True}]}


def _app(chronologie):
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["gares"] = _gares()
    app.session_state["missions"] = _missions()
    app.session_state["chronologie_calculee"] = chronologie
    app.session_state["mode_calcul_selector"] = "Calcul Energie"
    app.session_state["mode_generation"] = "Rotation optimisée"
    app.session_state["heure_debut_service"] = time(6, 0)
    app.session_state["heure_fin_service"] = time(22, 0)
    app.session_state["warnings_calcul"] = {}
    app.session_state["stats_homogeneite"] = {}
    return app


def _bilan_affiche(app):
    return app.session_state["energie"].valeur[0]


def _bilan_attendu(chronologie, energy_params):
    return calculer_energie_flotte(chronologie, _missions(), _gares(), energy_params)[0]


def test_bilan_energie_suit_une_nouvelle_grille_de_meme_taille():
    app = _app(_chronologie(8)).run()
    assert not app.exception
    params = deepcopy(app.session_state["energy_params"])

    plus_longue = _chronologie(8)
    plus_longue[1][0]["end"] += timedelta(minutes=25)
    app.session_state["chronologie_calculee"] = plus_longue
    app.run()

    assert not app.exception
    assert _bilan_affiche(app) == _bilan_attendu(plus_longue, params)
    assert _bilan_affiche(app) != _bilan_attendu(_chronologie(8), params)


def test_bilan_energie_suit_le_materiel():
    app = _app(_chronologie(8)).run()
    params = deepcopy(app.session_state["energy_params"])
    params["diesel"] = dict(params["diesel"], masse_tonne=params["diesel"]["masse_tonne"] * 2)
    app.session_state["energy_params"] = params
    app.run()

    assert not app.exception
    assert _bilan_affiche(app) == _bilan_attendu(_chronologie(8), params)


def test_lire_rend_none_et_signale_perime_quand_une_entree_change():
    session = {}
    etat_session.enregistrer(session, "x", (1, {"a": 1}), "valeur")
    assert etat_session.lire(session, "x", (1, {"a": 1})) == ("valeur", False)
    assert etat_session.lire(session, "x", (1, {"a": 2})) == (None, True)
    assert etat_session.lire(session, "x", (1, {"a": 1})) == (None, False)


def test_parametres_par_defaut_sont_lus_par_l_app():
    app = _app(_chronologie(8)).run()
    assert app.session_state["energy_params"]["diesel"] == get_default_energy_params()
