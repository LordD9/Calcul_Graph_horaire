# -*- coding: utf-8 -*-
"""Un cache ne doit jamais rendre un résultat qu'un calcul à froid ne rendrait pas.

Chaque test remplit le cache, modifie UN champ des entrées, puis compare au
calcul à cache vide. Une clé qui oublie ce champ rend l'ancien résultat et
le test échoue (a862025 : bilan énergie réutilisé après une nouvelle grille).
"""
import copy
from datetime import time

import pandas as pd
import pytest

from core_logic import construire_horaire_mission, reset_caches
from optimisation_logic import _baseline_simulation_params, evaluer_params_simulation


def _mission():
    return {
        "origine": "A", "terminus": "C", "temps_trajet": 30, "frequence": 1,
        "reference_minutes": "0", "temps_retournement_A": 5, "temps_retournement_B": 5,
        "type_materiel": "diesel",
        "passing_points": [{"gare": "B", "time_offset_min": 14, "arret_commercial": True,
                            "duree_arret_min": 2}],
    }


def _gares():
    return pd.DataFrame({
        "gare": ["A", "B", "C"], "distance": [0.0, 9.0, 20.0],
        "infra": ["F", "VE", "F"], "electrification": ["F", "F", "F"],
        "rampe_section_a_venir": [0.0, 0.0, 0.0],
    })


MUTATIONS_MISSION = [
    ("temps_trajet", 41),
    ("passing_points", [{"gare": "B", "time_offset_min": 20, "arret_commercial": True,
                         "duree_arret_min": 4}]),
    ("trajet_asymetrique", True),
    ("temps_trajet_retour", 37),
    ("reference_minutes", "15"),
    ("temps_retournement_B", 25),
    ("type_materiel", "batterie"),
]

MUTATIONS_GARES = [
    ("distance", [0.0, 3.0, 20.0]),
    ("infra", ["F", "F", "F"]),
    ("electrification", ["F", "F", "C25"]),
    ("rampe_section_a_venir", [0.0, 12.0, 0.0]),
]


def _muter_mission(mission, champ, valeur):
    mission = copy.deepcopy(mission)
    mission[champ] = valeur
    if champ == "temps_trajet_retour":
        mission["trajet_asymetrique"] = True
    return mission


def _muter_gares(gares, colonne, valeurs):
    gares = gares.copy()
    gares[colonne] = valeurs
    return gares


def _a_froid(fn):
    reset_caches()
    return fn()


@pytest.mark.parametrize("direction", ["aller", "retour"])
@pytest.mark.parametrize("champ,valeur", MUTATIONS_MISSION)
def test_horaire_en_cache_egal_horaire_a_froid_apres_mutation_mission(direction, champ, valeur):
    construire_horaire_mission(_mission(), direction, _gares())
    mutee = _muter_mission(_mission(), champ, valeur)
    chaud = construire_horaire_mission(mutee, direction, _gares())
    assert chaud == _a_froid(lambda: construire_horaire_mission(mutee, direction, _gares()))


@pytest.mark.parametrize("colonne,valeurs", MUTATIONS_GARES)
def test_horaire_en_cache_egal_horaire_a_froid_apres_mutation_gares(colonne, valeurs):
    construire_horaire_mission(_mission(), "aller", _gares())
    mutees = _muter_gares(_gares(), colonne, valeurs)
    chaud = construire_horaire_mission(_mission(), "aller", mutees)
    assert chaud == _a_froid(lambda: construire_horaire_mission(_mission(), "aller", mutees))


def _evaluer(missions, gares):
    return evaluer_params_simulation(
        _baseline_simulation_params(), missions, gares, time(7, 0), time(10, 0))


def _resume(resultat):
    score, chronologie, warnings, _ = resultat
    return score, {k: [(s["start"], s["end"], s["origine"], s["terminus"]) for s in v]
                   for k, v in chronologie.items()}, warnings


@pytest.mark.parametrize("champ,valeur", MUTATIONS_MISSION)
def test_evaluation_en_cache_egale_evaluation_a_froid_apres_mutation_en_place(champ, valeur):
    missions, gares = [_mission()], _gares()
    _evaluer(missions, gares)
    missions[0] = _muter_mission(missions[0], champ, valeur)
    chaud = _evaluer(missions, gares)
    assert _resume(chaud) == _resume(_a_froid(lambda: _evaluer(missions, gares)))


@pytest.mark.parametrize("colonne,valeurs", MUTATIONS_GARES)
def test_evaluation_en_cache_egale_evaluation_a_froid_apres_mutation_gares_en_place(colonne, valeurs):
    missions, gares = [_mission()], _gares()
    _evaluer(missions, gares)
    gares[colonne] = valeurs
    chaud = _evaluer(missions, gares)
    assert _resume(chaud) == _resume(_a_froid(lambda: _evaluer(missions, gares)))


def test_durees_theoriques_suivent_les_temps_de_parcours():
    from optimisation_logic import _construire_durees_theoriques
    missions, gares = [_mission()], _gares()
    assert _construire_durees_theoriques(missions, gares) == {"A → C": 30, "C → A": 30}
    missions[0] = _muter_mission(missions[0], "temps_trajet", 41)
    assert _construire_durees_theoriques(missions, gares)["A → C"] == 41
