# -*- coding: utf-8 -*-
"""Un arrêt n'est compté qu'une fois : entre deux gares, le train roule le temps de marche.

Rejoue a681367 (arrêt de départ de bloc compté deux fois), de27eba (arrêt en
gare sans voie d'évitement ignoré) et l'inversion du retour qui ajoutait
l'arrêt au segment précédent.
"""
import glob
from datetime import time

import pandas as pd
import pytest

from core_logic import construire_horaire_mission, executer_simulation_evenementielle
from scenarios_manager import load_scenario

SCENARIOS = sorted(glob.glob("scenarios/**/*.json", recursive=True))


def _marche(horaire):
    return {(a["gare"], b["gare"]): round(b["time_offset_min"] - a["depart_offset_min"], 1)
            for a, b in zip(horaire, horaire[1:])}


def _charger(chemin):
    scenario = load_scenario(chemin)
    gares = pd.DataFrame(scenario["infrastructure"]["gares"])
    return scenario["missions"], gares.sort_values("distance").reset_index(drop=True)


@pytest.mark.parametrize("chemin", SCENARIOS)
def test_chaque_segment_simule_dure_son_temps_de_marche(chemin):
    missions, gares = _charger(chemin)
    marche = {}
    for mission in missions:
        for sens, (o, t) in (("aller", (mission["origine"], mission["terminus"])),
                             ("retour", (mission["terminus"], mission["origine"]))):
            marche[f"{o} → {t}"] = _marche(construire_horaire_mission(mission, sens, gares))

    chronologie, _, _ = executer_simulation_evenementielle(
        missions, gares, time(6), time(12), turnaround_buffers={})

    ecarts = [
        (row["mission"], row["origine"], row["terminus"], duree, attendu)
        for rows in chronologie.values() for row in rows
        if row["origine"] != row["terminus"]
        for duree in [round((row["end"] - row["start"]).total_seconds() / 60, 1)]
        for attendu in [marche.get(row["mission"], {}).get((row["origine"], row["terminus"]))]
        if attendu is not None and abs(duree - attendu) > 0.05
    ]
    assert ecarts == []


@pytest.mark.parametrize("chemin", SCENARIOS)
def test_retour_symetrique_reprend_les_temps_de_marche_de_l_aller(chemin):
    missions, gares = _charger(chemin)
    for mission in missions:
        if mission.get("trajet_asymetrique"):
            continue
        aller = _marche(construire_horaire_mission(mission, "aller", gares))
        retour = _marche(construire_horaire_mission(mission, "retour", gares))
        assert {(b, a): duree for (a, b), duree in aller.items()} == pytest.approx(retour, abs=0.15)


def test_arret_commercial_intermediaire_decale_le_depart_pas_la_marche():
    gares = pd.DataFrame({"gare": ["A", "B", "C"], "distance": [0.0, 10.0, 20.0],
                          "infra": ["F", "F", "F"]})
    mission = {"origine": "A", "terminus": "C", "temps_trajet": 30, "frequence": 1,
               "reference_minutes": "0", "temps_retournement_A": 5, "temps_retournement_B": 5,
               "passing_points": [{"gare": "B", "time_offset_min": 14, "arret_commercial": True,
                                   "duree_arret_min": 3}]}
    chronologie, _, _ = executer_simulation_evenementielle(
        [mission], gares, time(8), time(9), turnaround_buffers={})
    premier = chronologie[min(chronologie)]
    trace = [(r["origine"], r["terminus"], (r["end"] - r["start"]).total_seconds() / 60)
             for r in premier[:3]]
    assert trace == [("A", "B", 14.0), ("B", "B", 3.0), ("B", "C", 13.0)]
