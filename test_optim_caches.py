# -*- coding: utf-8 -*-
"""Caches d'optimisation : meme resultat, moins d'appels."""
from datetime import time
from unittest.mock import patch

import pandas as pd

from core_logic import (
    construire_horaire_mission,
    reset_caches,
    _construire_horaire_mission_impl,
)
from optimisation_logic import (
    SimulationParams,
    _baseline_simulation_params,
    _construire_durees_theoriques,
    evaluer_params_simulation,
)


def _mission():
    return {
        "origine": "A",
        "terminus": "B",
        "temps_trajet": 20,
        "frequence": 1,
        "reference_minutes": "0",
        "temps_retournement_origine": 5,
        "temps_retournement_terminus": 5,
        "passing_points": [],
    }


def _gares():
    return pd.DataFrame({
        "gare": ["A", "B"],
        "distance": [0.0, 10.0],
        "electrification": ["F", "F"],
        "infra": ["Terminus", "Terminus"],
    })


def test_horaire_cache_meme_resultat_que_impl():
    m, df = _mission(), _gares()
    reset_caches()
    cached = construire_horaire_mission(m, "aller", df)
    direct = _construire_horaire_mission_impl(m, "aller", df)
    assert cached == direct
    assert cached  # pas vide


def test_horaire_cache_n_appelle_impl_qu_une_fois():
    m, df = _mission(), _gares()
    reset_caches()
    with patch(
        "core_logic._construire_horaire_mission_impl",
        wraps=_construire_horaire_mission_impl,
    ) as wrapped:
        a = construire_horaire_mission(m, "aller", df)
        b = construire_horaire_mission(m, "aller", df)
        assert a == b
        assert wrapped.call_count == 1
        reset_caches()
        construire_horaire_mission(m, "aller", df)
        assert wrapped.call_count == 2


def test_durees_theoriques_cache_stable():
    missions, df = [_mission()], _gares()
    d1 = _construire_durees_theoriques(missions, df)
    d2 = _construire_durees_theoriques(missions, df)
    assert d1 == d2
    assert d1["A → B"] == 20


def test_evaluer_params_cache_evite_resimulation():
    missions, df = [_mission()], _gares()
    params = _baseline_simulation_params()
    hd, hf = time(7, 0), time(9, 0)
    reset_caches()
    with patch(
        "core_logic.executer_simulation_evenementielle",
        wraps=__import__("core_logic", fromlist=["executer_simulation_evenementielle"]).executer_simulation_evenementielle,
    ) as wrapped:
        s1, c1, w1, st1 = evaluer_params_simulation(params, missions, df, hd, hf)
        s2, c2, w2, st2 = evaluer_params_simulation(params, missions, df, hd, hf)
        assert s1 == s2
        assert c1 == c2
        assert wrapped.call_count == 1
        other = SimulationParams(
            cadencements={"M1": 10},
            turnaround_buffers={},
            crossing_stop_durations={},
        )
        evaluer_params_simulation(other, missions, df, hd, hf)
        assert wrapped.call_count == 2
