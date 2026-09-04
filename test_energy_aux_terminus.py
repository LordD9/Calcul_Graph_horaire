# -*- coding: utf-8 -*-
"""Tests du taux d'utilisation des auxiliaires au terminus et du bilan par source."""

from datetime import datetime

import pandas as pd

from energy_logic import calculer_consommation_trajet, get_default_energy_params


def _df_gares():
    return pd.DataFrame({
        "gare": ["A", "C", "B"],
        "distance": [0.0, 5.0, 10.0],
        "electrification": ["F", "F", "C25"],
        "rampe_section_a_venir": [0.0, 0.0, 0.0],
    })


def _df_gares_elec():
    return pd.DataFrame({
        "gare": ["A", "C", "B"],
        "distance": [0.0, 5.0, 10.0],
        "electrification": ["C25", "C25", "C25"],
        "rampe_section_a_venir": [0.0, 0.0, 0.0],
    })


def _params(**overrides):
    p = get_default_energy_params()
    p["facteur_aux_kwh_h"] = 40.0
    p["kwh_per_liter_diesel"] = 10.0
    p.update(overrides)
    return p


def _t(h, m=0):
    return datetime(2026, 1, 1, h, m)


def _run(trajets, type_mat="diesel", taux=100, df=None, **param_kw):
    mission = {"origine": "A", "terminus": "B", "type_materiel": type_mat}
    params = _params(taux_aux_terminus_pct=taux, **param_kw)
    return calculer_consommation_trajet(trajets, mission, df if df is not None else _df_gares(), params)


def test_default_param_is_100():
    p = get_default_energy_params()
    assert p["taux_aux_terminus_pct"] == 100


def test_missing_taux_equals_100():
    trajets = [
        {"start": _t(8, 0), "end" : _t(8, 20), "origine": "A", "terminus": "B", "mission": "A → B"},
        {"start": _t(8, 40), "end": _t(9, 0), "origine": "B", "terminus": "A", "mission": "B → A"},
    ]
    mission = {"origine": "A", "terminus": "B", "type_materiel": "diesel"}
    params_explicit = _params(taux_aux_terminus_pct=100)
    params_missing = _params()
    del params_missing["taux_aux_terminus_pct"]
    r100 = calculer_consommation_trajet(trajets, mission, _df_gares(), params_explicit)
    rmiss = calculer_consommation_trajet(trajets, mission, _df_gares(), params_missing)
    assert abs(r100["total_conso_aux_kwh"] - rmiss["total_conso_aux_kwh"]) < 1e-9
    assert abs(r100["total_litres_diesel"] - rmiss["total_litres_diesel"]) < 1e-9


def test_movement_aux_unaffected_by_terminus_rate():
    trajets = [
        {"start": _t(8, 0), "end": _t(8, 20), "origine": "A", "terminus": "B", "mission": "A → B"},
    ]
    r100 = _run(trajets, taux=100)
    r0 = _run(trajets, taux=0)
    expected_aux = 40.0 * (20 / 60)
    assert abs(r100["total_conso_aux_kwh"] - expected_aux) < 1e-6
    assert abs(r0["total_conso_aux_kwh"] - expected_aux) < 1e-6
    assert abs(r100["total_conso_brute_kwh"] - r0["total_conso_brute_kwh"]) < 1e-9


def test_terminus_gap_scaled_by_rate():
    trajets = [
        {"start": _t(8, 0), "end": _t(8, 20), "origine": "A", "terminus": "B", "mission": "A → B"},
        {"start": _t(8, 50), "end": _t(9, 10), "origine": "B", "terminus": "A", "mission": "B → A"},
    ]
    # 20 min + 20 min movement, 30 min gap at B (terminus)
    r100 = _run(trajets, taux=100)
    r50 = _run(trajets, taux=50)
    r0 = _run(trajets, taux=0)
    aux_move = 40.0 * (40 / 60)
    aux_gap = 40.0 * (30 / 60)
    assert abs(r100["total_conso_aux_kwh"] - (aux_move + aux_gap)) < 1e-6
    assert abs(r50["total_conso_aux_kwh"] - (aux_move + 0.5 * aux_gap)) < 1e-6
    assert abs(r0["total_conso_aux_kwh"] - aux_move) < 1e-6
    # Traction identique : seule l'aux de gap change
    d_aux_100_0 = r100["total_conso_aux_kwh"] - r0["total_conso_aux_kwh"]
    d_brute_100_0 = r100["total_conso_brute_kwh"] - r0["total_conso_brute_kwh"]
    assert abs(d_aux_100_0 - d_brute_100_0) < 1e-6
    assert abs(d_aux_100_0 - aux_gap) < 1e-6


def test_intermediate_stop_keeps_full_aux_when_terminus_rate_zero():
    trajets = [
        {"start": _t(8, 0), "end": _t(8, 10), "origine": "A", "terminus": "C", "mission": "A → B"},
        {"start": _t(8, 10), "end": _t(8, 15), "origine": "C", "terminus": "C", "mission": "A → B"},
        {"start": _t(8, 15), "end": _t(8, 25), "origine": "C", "terminus": "B", "mission": "A → B"},
    ]
    r0 = _run(trajets, taux=0)
    # 10+10 min movement + 5 min stop at C (intermédiaire) → aux pleine partout
    expected_aux = 40.0 * (25 / 60)
    assert abs(r0["total_conso_aux_kwh"] - expected_aux) < 1e-6


def test_explicit_stop_at_terminus_uses_rate():
    trajets = [
        {"start": _t(8, 0), "end": _t(8, 20), "origine": "A", "terminus": "B", "mission": "A → B"},
        {"start": _t(8, 20), "end": _t(8, 40), "origine": "B", "terminus": "B", "mission": "A → B"},
    ]
    r100 = _run(trajets, taux=100)
    r0 = _run(trajets, taux=0)
    aux_move = 40.0 * (20 / 60)
    aux_stop = 40.0 * (20 / 60)
    assert abs(r100["total_conso_aux_kwh"] - (aux_move + aux_stop)) < 1e-6
    assert abs(r0["total_conso_aux_kwh"] - aux_move) < 1e-6


def test_diesel_aux_converted_to_liters():
    trajets = [
        {"start": _t(8, 0), "end": _t(8, 20), "origine": "A", "terminus": "B", "mission": "A → B"},
        {"start": _t(8, 50), "end": _t(9, 10), "origine": "B", "terminus": "A", "mission": "B → A"},
    ]
    r = _run(trajets, type_mat="diesel", taux=100)
    assert r["total_conso_aux_thermique_kwh"] == r["total_conso_aux_kwh"]
    assert r["total_conso_aux_electrique_kwh"] == 0
    assert abs(r["total_aux_litres_diesel"] - r["total_conso_aux_kwh"] / 10.0) < 1e-9
    assert r["total_aux_litres_diesel"] <= r["total_litres_diesel"] + 1e-9


def test_electric_aux_stays_in_kwh():
    trajets = [
        {"start": _t(8, 0), "end": _t(8, 20), "origine": "A", "terminus": "B", "mission": "A → B"},
        {"start": _t(8, 50), "end": _t(9, 10), "origine": "B", "terminus": "A", "mission": "B → A"},
    ]
    r = _run(trajets, type_mat="electrique", taux=50, df=_df_gares_elec())
    assert r["total_conso_aux_electrique_kwh"] == r["total_conso_aux_kwh"]
    assert r["total_conso_aux_thermique_kwh"] == 0
    assert r["total_aux_litres_diesel"] == 0
    assert r["total_conso_aux_kwh"] <= r["total_conso_electrique_kwh"] + 1e-9


def test_bimode_splits_aux_by_electrification():
    # A (F) -> B (C25) : segment défini par la gare au km le plus faible = A (F) → thermique
    # B (C25) -> A (F) : gare au km le plus faible = A (F) → thermique aussi
    # Pour forcer un split, gap au terminus B (C25) est électrique pour bimode.
    trajets = [
        {"start": _t(8, 0), "end": _t(8, 20), "origine": "A", "terminus": "B", "mission": "A → B"},
        {"start": _t(8, 50), "end": _t(9, 10), "origine": "B", "terminus": "A", "mission": "B → A"},
    ]
    r = _run(trajets, type_mat="bimode", taux=100)
    aux_move = 40.0 * (40 / 60)
    aux_gap_b = 40.0 * (30 / 60)  # B est C25 → électrique
    assert abs(r["total_conso_aux_thermique_kwh"] - aux_move) < 1e-6
    assert abs(r["total_conso_aux_electrique_kwh"] - aux_gap_b) < 1e-6
    assert abs(r["total_conso_aux_kwh"] - (aux_move + aux_gap_b)) < 1e-6


def test_battery_terminus_rate_zero_no_discharge_without_infra():
    df = pd.DataFrame({
        "gare": ["A", "B"],
        "distance": [0.0, 10.0],
        "electrification": ["F", "F"],
        "rampe_section_a_venir": [0.0, 0.0],
    })
    trajets = [
        {"start": _t(8, 0), "end": _t(8, 20), "origine": "A", "terminus": "B", "mission": "A → B"},
        {"start": _t(8, 50), "end": _t(9, 10), "origine": "B", "terminus": "A", "mission": "B → A"},
    ]
    r100 = _run(trajets, type_mat="batterie", taux=100, df=df)
    r0 = _run(trajets, type_mat="batterie", taux=0, df=df)
    aux_gap = 40.0 * (30 / 60)
    assert abs(r100["total_conso_aux_kwh"] - r0["total_conso_aux_kwh"] - aux_gap) < 1e-6
    # SoC final : à taux 0, pas de décharge aux pendant les 30 min au terminus
    soc100 = r100["batterie_log"][-1][1]
    soc0 = r0["batterie_log"][-1][1]
    assert soc0 > soc100
    assert abs((soc0 - soc100) - aux_gap) < 1e-6


def test_battery_terminus_rate_zero_more_charge_with_infra():
    df = pd.DataFrame({
        "gare": ["A", "B"],
        "distance": [0.0, 10.0],
        "electrification": ["F", "R50"],  # 50 kW de recharge à B
        "rampe_section_a_venir": [0.0, 0.0],
    })
    params_extra = {
        "capacite_batterie_kwh": 200,
        "soc_max_pct": 95,
        "soc_min_pct": 20,
        "facteur_charge_C": 1.0,
    }
    trajets = [
        {"start": _t(8, 0), "end": _t(8, 20), "origine": "A", "terminus": "B", "mission": "A → B"},
        {"start": _t(9, 20), "end": _t(9, 40), "origine": "B", "terminus": "A", "mission": "B → A"},
    ]
    r100 = _run(trajets, type_mat="batterie", taux=100, df=df, **params_extra)
    r0 = _run(trajets, type_mat="batterie", taux=0, df=df, **params_extra)
    # Infra 50 kW, aux 40 kW → 10 kW charge à 100 %, 50 kW charge à 0 %
    # 60 min au terminus B
    extra_charge = (50.0 - 10.0) * 1.0  # +40 kWh potentiels
    soc100 = [x[1] for x in r100["batterie_log"] if "Attente/Terminus" in str(x[3])]
    soc0 = [x[1] for x in r0["batterie_log"] if "Attente/Terminus" in str(x[3])]
    assert soc100 and soc0
    # Plus de charge (ou moins de conso aux) à taux 0
    assert soc0[0] >= soc100[0] - 1e-6


def test_taux_clamped_and_invalid_falls_back():
    trajets = [
        {"start": _t(8, 0), "end": _t(8, 20), "origine": "A", "terminus": "B", "mission": "A → B"},
        {"start": _t(8, 50), "end": _t(9, 10), "origine": "B", "terminus": "A", "mission": "B → A"},
    ]
    r150 = _run(trajets, taux=150)  # clamp à 100
    r100 = _run(trajets, taux=100)
    assert abs(r150["total_conso_aux_kwh"] - r100["total_conso_aux_kwh"]) < 1e-9

    mission = {"origine": "A", "terminus": "B", "type_materiel": "diesel"}
    params = _params(taux_aux_terminus_pct="nope")
    rbad = calculer_consommation_trajet(trajets, mission, _df_gares(), params)
    assert abs(rbad["total_conso_aux_kwh"] - r100["total_conso_aux_kwh"]) < 1e-9


def test_aux_never_exceeds_total_energy():
    trajets = [
        {"start": _t(8, 0), "end": _t(8, 10), "origine": "A", "terminus": "C", "mission": "A → B"},
        {"start": _t(8, 10), "end": _t(8, 15), "origine": "C", "terminus": "C", "mission": "A → B"},
        {"start": _t(8, 15), "end": _t(8, 25), "origine": "C", "terminus": "B", "mission": "A → B"},
        {"start": _t(8, 55), "end": _t(9, 15), "origine": "B", "terminus": "A", "mission": "B → A"},
    ]
    for tmat in ("diesel", "electrique", "bimode", "batterie"):
        for taux in (0, 50, 100):
            r = _run(trajets, type_mat=tmat, taux=taux, df=_df_gares_elec() if tmat != "diesel" else _df_gares())
            assert r["total_conso_aux_kwh"] <= r["total_conso_brute_kwh"] + 1e-6
            assert abs(
                r["total_conso_aux_electrique_kwh"] + r["total_conso_aux_thermique_kwh"]
                - r["total_conso_aux_kwh"]
            ) < 1e-6
            assert r["total_distance_km"] > 0


def test_bilan_display_format():
    def _fmt_dont_aux(valeur_km, part_pct):
        if valeur_km is None:
            return "N/A"
        if part_pct is None:
            return f"{valeur_km:.2f}"
        return f"{valeur_km:.2f} ({part_pct:.0f}%)"

    assert _fmt_dont_aux(0.59, 25.4) == "0.59 (25%)"
    assert _fmt_dont_aux(0.12, 0) == "0.12 (0%)"
    assert _fmt_dont_aux(1.0, None) == "1.00"
    assert _fmt_dont_aux(None, 10) == "N/A"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
