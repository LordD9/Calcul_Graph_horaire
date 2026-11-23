# -*- coding: utf-8 -*-
"""Un cache ne doit jamais rendre un résultat qu'un calcul à froid ne rendrait pas.

Chaque test remplit le cache, modifie UN champ des entrées, puis compare au
calcul à cache vide. Une clé qui oublie ce champ rend l'ancien résultat et
le test échoue (a862025 : bilan énergie réutilisé après une nouvelle grille).
"""
import copy
from datetime import datetime, time

import pandas as pd
import pytest

from core_logic import construire_horaire_mission, reset_caches
from energy_logic import fingerprint_energie, get_default_energy_params
from optimisation_logic import _baseline_simulation_params, evaluer_params_simulation
from robustness_logic import fingerprint_robustesse


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


def _chronologie(minute_depart=0):
    jour = datetime(2026, 1, 1)
    return {1: [{"start": jour.replace(hour=8, minute=minute_depart),
                 "end": jour.replace(hour=8, minute=minute_depart + 20),
                 "origine": "A", "terminus": "C", "mission": "A → C",
                 "is_mission_start": True}]}


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


def _entrees_energie():
    return {"diesel": get_default_energy_params()}, [_mission()], _chronologie(), _gares()


MUTATIONS_ENTREES = [
    ("energy_params", lambda e: {"diesel": dict(e["diesel"], masse_tonne=80)}),
    ("missions", lambda m: [_muter_mission(m[0], "type_materiel", "batterie")]),
    ("missions", lambda m: [_muter_mission(m[0], "passing_points", [])]),
    ("chronologie", lambda c: _chronologie(minute_depart=30)),
    ("df_gares", lambda g: _muter_gares(g, "electrification", ["C25", "F", "F"])),
    ("df_gares", lambda g: _muter_gares(g, "rampe_section_a_venir", [5.0, 0.0, 0.0])),
]


@pytest.mark.parametrize("entree,muter", MUTATIONS_ENTREES)
def test_empreinte_energie_change_des_qu_une_entree_lue_change(entree, muter):
    noms = ("energy_params", "missions", "chronologie", "df_gares")
    entrees = dict(zip(noms, _entrees_energie()))
    avant = fingerprint_energie(*entrees.values())
    entrees[entree] = muter(entrees[entree])
    assert fingerprint_energie(*entrees.values()) != avant


@pytest.mark.parametrize("entree,muter", MUTATIONS_ENTREES)
def test_empreinte_robustesse_change_des_qu_une_entree_lue_change(entree, muter):
    noms = ("energy_params", "missions", "chronologie", "df_gares")
    entrees = dict(zip(noms, _entrees_energie()))

    def calcul():
        return fingerprint_robustesse(entrees["chronologie"], entrees["df_gares"],
                                      entrees["missions"], energy_params=entrees["energy_params"])

    avant = calcul()
    entrees[entree] = muter(entrees[entree])
    assert calcul() != avant


def test_empreintes_stables_pour_des_entrees_egales():
    assert fingerprint_energie(*_entrees_energie()) == fingerprint_energie(*_entrees_energie())
    e1, m1, c1, g1 = _entrees_energie()
    e2, m2, c2, g2 = _entrees_energie()
    assert fingerprint_robustesse(c1, g1, m1, e1) == fingerprint_robustesse(c2, g2, m2, e2)


def test_durees_theoriques_suivent_les_temps_de_parcours():
    from optimisation_logic import _construire_durees_theoriques
    missions, gares = [_mission()], _gares()
    assert _construire_durees_theoriques(missions, gares) == {"A → C": 30, "C → A": 30}
    missions[0] = _muter_mission(missions[0], "temps_trajet", 41)
    assert _construire_durees_theoriques(missions, gares)["A → C"] == 41
