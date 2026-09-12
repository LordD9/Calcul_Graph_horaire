# -*- coding: utf-8 -*-
"""Tests recalcul energie flotte sans toucher a la grille."""
from datetime import datetime

from energy_logic import (
    associer_mission_au_train,
    calculer_energie_flotte,
    fingerprint_energie,
    get_default_energy_params,
)
from test_energy_aux_terminus import _df_gares, _t


def test_associer_aller_puis_retour():
    missions = [
        {"origine": "A", "terminus": "B", "type_materiel": "diesel"},
    ]
    assert associer_mission_au_train(
        [{"origine": "A", "terminus": "B"}], missions
    )["origine"] == "A"
    assert associer_mission_au_train(
        [{"origine": "B", "terminus": "A"}], missions
    )["terminus"] == "B"
    assert associer_mission_au_train([], missions) is None


def test_flotte_deux_rames_independantes():
    missions = [{"origine": "A", "terminus": "B", "type_materiel": "diesel"}]
    chrono = {
        "T1": [{"start": _t(8, 0), "end": _t(8, 20), "origine": "A", "terminus": "B", "mission": "A → B"}],
        "T2": [{"start": _t(9, 0), "end": _t(9, 20), "origine": "A", "terminus": "B", "mission": "A → B"}],
    }
    params = {"diesel": get_default_energy_params()}
    res, mpt, err = calculer_energie_flotte(chrono, missions, _df_gares(), params)
    assert not err
    assert set(res) == {"T1", "T2"}
    assert mpt["T1"]["origine"] == "A"
    assert res["T1"][1] == "diesel"
    assert res["T1"][0]["total_distance_km"] > 0


def test_fingerprint_change_quand_masse_change():
    missions = [{"type_materiel": "diesel"}]
    chrono = {"T1": [1, 2]}
    p1 = {"diesel": get_default_energy_params()}
    p2 = {"diesel": dict(get_default_energy_params(), masse_tonne=80)}
    assert fingerprint_energie(p1, missions, chrono) != fingerprint_energie(p2, missions, chrono)
    assert fingerprint_energie(p1, missions, chrono) == fingerprint_energie(p1, missions, chrono)
