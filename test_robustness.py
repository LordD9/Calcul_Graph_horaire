from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import robustness_logic as robustness_module
from energy_logic import get_default_energy_params
from robustness_logic import (
    _analyser_energie,
    _compiler_services,
    _enumerer_points,
    _evaluuer_avec_reserves,
    _rejouer,
    _retour_normal,
    evaluer_robustesse,
    fingerprint_robustesse,
)


def _time(hour, minute):
    return datetime(2025, 1, 1, hour, minute)


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
        "type_materiel": "diesel",
        "temps_retournement_A": 5,
        "temps_retournement_B": 5,
    }


def _chronologie():
    return {
        "T1": [
            {
                "start": _time(8, 0),
                "end": _time(8, 10),
                "origine": "A",
                "terminus": "X",
                "mission": "A → B",
                "is_mission_start": True,
            },
            {
                "start": _time(8, 10),
                "end": _time(8, 20),
                "origine": "X",
                "terminus": "B",
                "mission": "A → B",
                "is_mission_start": False,
            },
        ]
    }


def _chronologie_batterie_retournement():
    return {
        "T1": [
            {"start": _time(8, 0), "end": _time(8, 15), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(8, 15), "end": _time(8, 30), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
            {"start": _time(8, 50), "end": _time(9, 5), "origine": "B", "terminus": "X",
             "mission": "B → A", "is_mission_start": True},
            {"start": _time(9, 5), "end": _time(9, 20), "origine": "X", "terminus": "A",
             "mission": "B → A", "is_mission_start": False},
        ]
    }


def _chronologie_batterie_rotations():
    chronologie = {"T1": []}
    debut_minute = 8 * 60
    for index in range(4):
        aller = index % 2 == 0
        origine, intermediaire, terminus = (
            ("A", "X", "B") if aller else ("B", "X", "A")
        )
        mission = "A → B" if aller else "B → A"
        chronologie["T1"].extend([
            {"start": _time(debut_minute // 60, debut_minute % 60),
             "end": _time((debut_minute + 15) // 60, (debut_minute + 15) % 60),
             "origine": origine, "terminus": intermediaire, "mission": mission,
             "is_mission_start": True},
            {"start": _time((debut_minute + 15) // 60, (debut_minute + 15) % 60),
             "end": _time((debut_minute + 30) // 60, (debut_minute + 30) % 60),
             "origine": intermediaire, "terminus": terminus, "mission": mission,
             "is_mission_start": False},
        ])
        debut_minute += 50
    return chronologie


def _infrastructure_batterie():
    gares = _infrastructure().copy()
    gares.loc[gares["gare"] == "B", "electrification"] = "C25"
    return gares


def _mission_batterie(**overrides):
    return {
        **_mission(),
        "type_materiel": "batterie",
        "temps_retournement_A": 20,
        "temps_retournement_B": 20,
        **overrides,
    }


def _parametres_batterie_retournement(capacite=19, **overrides):
    params = get_default_energy_params()
    params.update({
        "capacite_batterie_kwh": capacite,
        "soc_min_pct": 20,
        "soc_max_pct": 95,
        "facteur_charge_C": 1,
        "facteur_aux_kwh_h": 0,
    })
    params.update(overrides)
    return {"batterie": params}


def test_evaluer_robustesse_retourne_trois_scenarios_et_compte_une_arrivee():
    chronologie = _chronologie()
    baseline = deepcopy(chronologie)

    resultat = evaluer_robustesse(
        chronologie, _infrastructure(), [_mission()]
    )

    assert [scenario["delay_min"] for scenario in resultat["scenarios"]] == [5, 10, 20]
    assert resultat["candidates_count"] == 2
    assert [scenario["sans_reserve"]["retard_cumule_min"] for scenario in resultat["scenarios"]] == [5, 10, 20]
    assert chronologie == baseline
    assert all(scenario["point"]["gare"] in {"A", "X"} for scenario in resultat["scenarios"])
    replayable = evaluer_robustesse(
        resultat["scenarios"][0]["sans_reserve"]["chronologie"],
        _infrastructure(), [_mission()],
    )
    assert replayable["scenarios"][0]["sans_reserve"]["retard_cumule_min"] == 5


def test_rejouer_prolonge_le_retournement_et_charge_a_la_gare_electrifiee():
    chronologie = _chronologie_batterie_retournement()
    services, gares, infra = _compiler_services(
        chronologie, _infrastructure_batterie(), [_mission_batterie()]
    )
    service = next(item for item in services if item["service_id"] == "str:T1#1")
    incident = next(
        point for point in _enumerer_points(services, infra)
        if point["service_id"] == service["service_id"] and point["kind"] == "depart"
    )

    initial = _rejouer(services, gares, incident, 20)
    prolonge = _rejouer(
        services, gares, incident, 20,
        turnaround_overrides={service["service_id"]: 20},
    )

    retour_initial = next(row for row in initial["services"] if row["service_id"] == "str:T1#2")
    retour_prolonge = next(row for row in prolonge["services"] if row["service_id"] == "str:T1#2")
    stationnement = [
        row for row in prolonge["chronologie"]["T1"]
        if row.get("prolongation_retournement")
    ]
    assert retour_initial["depart_reel"] == _time(8, 55)
    assert retour_prolonge["depart_reel"] == _time(9, 10)
    assert len(stationnement) == 1
    assert (stationnement[0]["start"], stationnement[0]["end"]) == (
        _time(8, 50), _time(9, 5)
    )


def test_rejouer_ignore_la_prolongation_sans_infrastructure_electrique():
    chronologie = _chronologie_batterie_retournement()
    infrastructure = _infrastructure_batterie()
    infrastructure.loc[infrastructure["gare"] == "B", "electrification"] = "F"
    services, gares, infra = _compiler_services(
        chronologie, infrastructure, [_mission_batterie()]
    )
    incident = next(
        point for point in _enumerer_points(services, infra)
        if point["service_id"] == "str:T1#1" and point["kind"] == "depart"
    )

    replay = _rejouer(
        services, gares, incident, 20,
        turnaround_overrides={"str:T1#1": 20},
    )

    retour = next(row for row in replay["services"] if row["service_id"] == "str:T1#2")
    assert retour["depart_reel"] == _time(8, 55)
    assert not any(row.get("prolongation_retournement")
                   for row in replay["chronologie"]["T1"])


def test_nouvelle_alerte_batterie_est_corrigee_par_le_retournement():
    chronologie = _chronologie_batterie_retournement()
    services, gares, infra_map = _compiler_services(
        chronologie, _infrastructure_batterie(), [_mission_batterie()]
    )
    resultat = evaluer_robustesse(
        chronologie,
        _infrastructure_batterie(),
        [_mission_batterie()],
        _parametres_batterie_retournement(),
    )
    scenario_brut = resultat["scenarios"][-1]
    scenario = scenario_brut["sans_reserve"]
    batterie = scenario["battery"]
    recharge = scenario["recharges_prolongees"]
    point = scenario_brut["point"]
    incident = next(
        candidate for candidate in _enumerer_points(services, infra_map)
        if (candidate["service_id"], candidate["gare"], candidate["kind"], candidate["time"])
        == (point["service_id"], point["gare"], point["kind"], point["time"])
    )
    replay_initial = _rejouer(services, gares, incident, 20)
    mission_suivante = next(
        row for row in scenario["arrivals"] if row["service_id"] == "str:T1#2"
    )

    assert batterie["baseline"]["par_train"]["T1"]["autonomie_insuffisante"] is False
    assert batterie["autonomie_insuffisante"] is False
    assert mission_suivante["depart_reel"] == _time(8, 56)
    assert recharge == [{
        "service_id": "str:T1#1",
        "train_id": "T1",
        "gare": "B",
        "retournement_initial_min": 5.0,
        "retournement_retenu_min": 6.0,
        "prolongation_min": 1.0,
        "motif": "suppression d'une nouvelle alerte batterie",
    }]
    assert "retour_normal_sans_recharge_min" in scenario
    assert scenario["retour_normal_sans_recharge_min"] == _retour_normal(
        services, replay_initial["services"], incident, 20,
        replay_initial["infra_violations"],
    )
    assert "_deficit_batterie_kwh" not in batterie["scenario"]["par_train"]["T1"]


def test_recharge_est_bornee_par_horizon_sous_le_retournement_nominal():
    chronologie = _chronologie_batterie_retournement()
    chronologie["T1"][2]["end"] = _time(8, 57)
    chronologie["T1"][3]["start"] = _time(8, 57)
    chronologie["T1"][3]["end"] = _time(9, 5)
    energy_params = _parametres_batterie_retournement(23)
    services, gares, infra = _compiler_services(
        chronologie, _infrastructure_batterie(), [_mission_batterie()]
    )
    service = next(item for item in services if item["service_id"] == "str:T1#1")
    incident = next(
        point for point in _enumerer_points(services, infra)
        if point["service_id"] == service["service_id"] and point["kind"] == "depart"
    )
    nominal = _analyser_energie(chronologie, services, gares, energy_params)
    raw = _rejouer(services, gares, incident, 20)
    raw_energy = _analyser_energie(raw["chronologie"], services, gares, energy_params)

    assert service["turnaround_min"] == 20
    assert robustness_module._retournement_effectif(service, raw) == 5.0
    assert robustness_module._limite_charge_retournement(
        service, raw, services, gares, energy_params
    ) == 15
    assert nominal["autonomie_insuffisante"] is False
    assert raw_energy["autonomie_insuffisante"] is True
    replay_15 = _rejouer(
        services, gares, incident, 20,
        turnaround_overrides={service["service_id"]: 15},
    )
    energy_15 = _analyser_energie(
        replay_15["chronologie"], services, gares, energy_params
    )
    assert energy_15["autonomie_insuffisante"] is False
    replay_12 = _rejouer(
        services, gares, incident, 20,
        turnaround_overrides={service["service_id"]: 12},
    )
    energy_12 = _analyser_energie(
        replay_12["chronologie"], services, gares, energy_params
    )
    assert energy_12["autonomie_insuffisante"] is True

    corrected, corrected_energy, recharges, _ = robustness_module._corriger_retournements(
        raw, services, gares, incident, 20, nominal, energy_params,
        energy_cache={},
    )

    assert corrected_energy["autonomie_insuffisante"] is False
    assert corrected["physical_violations"] == []
    assert corrected["infra_violations"] == []
    assert [
        (row["retournement_initial_min"], row["retournement_retenu_min"],
         row["prolongation_min"])
        for row in recharges
    ] == [(5.0, 13.0, 8.0)]


def test_parametres_batterie_absents_ne_declenchent_pas_de_recharge():
    resultat = evaluer_robustesse(
        _chronologie_batterie_retournement(),
        _infrastructure_batterie(), [_mission_batterie()], None,
    )
    scenario = resultat["scenarios"][-1]["sans_reserve"]

    assert scenario["battery"]["baseline"]["autonomie_non_verifiable"] is True
    assert scenario["battery"]["autonomie_non_verifiable"] is True
    assert scenario["recharges_prolongees"] == []


def test_batterie_deja_insuffisante_nominalement_n_est_pas_corrigee():
    resultat = evaluer_robustesse(
        _chronologie_batterie_retournement(),
        _infrastructure_batterie(), [_mission_batterie()],
        _parametres_batterie_retournement(14),
    )
    scenario = resultat["scenarios"][-1]["sans_reserve"]

    assert scenario["battery"]["baseline"]["autonomie_insuffisante"] is True
    assert scenario["battery"]["autonomie_insuffisante"] is True
    assert scenario["recharges_prolongees"] == []


def test_retournement_non_electrifie_ne_recoit_pas_de_recharge():
    chronologie = {"T1": _chronologie_batterie_retournement()["T1"][:2]}
    gares = _infrastructure_batterie()
    gares.loc[gares["gare"] == "B", "electrification"] = "F"
    resultat = evaluer_robustesse(
        chronologie, gares, [_mission_batterie()],
        _parametres_batterie_retournement(50, facteur_aux_kwh_h=43),
    )
    scenario = resultat["scenarios"][-1]["sans_reserve"]

    assert scenario["battery"]["baseline"]["autonomie_insuffisante"] is False
    assert scenario["battery"]["autonomie_insuffisante"] is True
    assert scenario["recharges_prolongees"] == []


def test_alerte_avant_le_terminus_reste_si_aucune_solution_de_recharge():
    chronologie = {"T1": _chronologie_batterie_retournement()["T1"][:2]}
    resultat = evaluer_robustesse(
        chronologie, _infrastructure_batterie(), [_mission_batterie()],
        _parametres_batterie_retournement(50, facteur_aux_kwh_h=43),
    )
    scenario = resultat["scenarios"][-1]["sans_reserve"]

    assert scenario["battery"]["baseline"]["autonomie_insuffisante"] is False
    assert scenario["battery"]["autonomie_insuffisante"] is True
    assert scenario["recharges_prolongees"] == []


def test_recherche_minimise_le_retournement_sous_le_minimum_nominal():
    chronologie = _chronologie_batterie_rotations()
    energy_params = _parametres_batterie_retournement(40, facteur_aux_kwh_h=10)
    services, gares, infra = _compiler_services(
        chronologie, _infrastructure_batterie(), [_mission_batterie()]
    )
    incident = next(
        point for point in _enumerer_points(services, infra)
        if point["service_id"] == "str:T1#1" and point["kind"] == "depart"
    )
    nominal = _analyser_energie(chronologie, services, gares, energy_params)
    raw = _rejouer(services, gares, incident, 20)
    raw_energy = _analyser_energie(raw["chronologie"], services, gares, energy_params)
    first_only = _rejouer(
        services, gares, incident, 20,
        turnaround_overrides={"str:T1#1": 20},
    )
    first_only_energy = _analyser_energie(
        first_only["chronologie"], services, gares, energy_params
    )

    assert nominal["autonomie_insuffisante"] is False
    assert raw_energy["autonomie_insuffisante"] is True
    assert first_only_energy["autonomie_insuffisante"] is True

    corrected, corrected_energy, recharges, _ = robustness_module._corriger_retournements(
        raw, services, gares, incident, 20, nominal, energy_params,
        energy_cache={},
    )
    assert corrected_energy["autonomie_insuffisante"] is False
    assert corrected["physical_violations"] == []
    assert corrected["infra_violations"] == []
    assert [row["service_id"] for row in recharges] == ["str:T1#1"]
    assert [
        (row["retournement_initial_min"], row["retournement_retenu_min"],
         row["prolongation_min"])
        for row in recharges
    ] == [(5.0, 13.0, 8.0)]

    resultat = evaluer_robustesse(
        chronologie, _infrastructure_batterie(), [_mission_batterie()], energy_params
    )
    scenario = resultat["scenarios"][-1]
    sans_reserve = scenario["sans_reserve"]
    assert sans_reserve["battery"]["autonomie_insuffisante"] is False
    assert all(
        row["depart_reel"] >= row["depart_planifie"]
        for row in sans_reserve["arrivals"] if row["depart_reel"] is not None
    )
    scenario_cinq = resultat["scenarios"][0]["sans_reserve"]
    assert scenario_cinq["battery"]["autonomie_insuffisante"] is False
    assert any(
        item["retournement_retenu_min"] > 20
        for item in scenario_cinq["recharges_prolongees"]
    )


def test_pire_point_est_classe_apres_correction_energetique(monkeypatch):
    chronologie = _chronologie_batterie_rotations()
    missions = [_mission_batterie()]
    gares = _infrastructure_batterie()
    energy_params = _parametres_batterie_retournement(40, facteur_aux_kwh_h=10)
    services, df_gares, infra = _compiler_services(chronologie, gares, missions)
    nombre_points = len(_enumerer_points(services, infra))
    appels = []
    correction_reelle = robustness_module._corriger_retournements

    def tracer(*args, **kwargs):
        replay, energy, recharges, retour = correction_reelle(*args, **kwargs)
        if args[4] == 20:
            point = args[3]
            appels.append((
                (point["service_id"], point["gare"], point["kind"], point["time"]),
                replay["retard_cumule_min"],
            ))
        return replay, energy, recharges, retour

    monkeypatch.setattr(robustness_module, "_corriger_retournements", tracer)
    resultat = evaluer_robustesse(
        chronologie, gares, missions, energy_params
    )["scenarios"][-1]
    public_point = resultat["point"]
    cle = (public_point["service_id"], public_point["gare"],
           public_point["kind"], public_point["time"])

    assert len(appels) == nombre_points
    assert cle in {point for point, _ in appels}
    assert resultat["sans_reserve"]["retard_cumule_min"] == max(
        score for _, score in appels
    )


def test_recherche_de_reserve_evalue_aussi_la_correction_batterie(monkeypatch):
    chronologie = _chronologie_batterie_rotations()
    missions = [_mission_batterie()]
    gares = _infrastructure_batterie()
    energy_params = _parametres_batterie_retournement(40, facteur_aux_kwh_h=10)
    appels_reserve = []
    correction_reelle = robustness_module._corriger_retournements

    def tracer(*args, **kwargs):
        if len(args) > 7 and args[7]:
            appels_reserve.append(args[7])
        return correction_reelle(*args, **kwargs)

    monkeypatch.setattr(robustness_module, "_corriger_retournements", tracer)
    scenario = evaluer_robustesse(
        chronologie, gares, missions, energy_params, reserve_gares=["B"]
    )["scenarios"][-1]["avec_reserve"]

    assert appels_reserve
    assert scenario["battery"]["autonomie_insuffisante"] is False


def _chronologie_croisement():
    return {
        "T1": [
            {"start": _time(8, 0), "end": _time(8, 10), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(8, 10), "end": _time(8, 20), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
        ],
        "T2": [
            {"start": _time(8, 21), "end": _time(8, 31), "origine": "B", "terminus": "X",
             "mission": "B → A", "is_mission_start": True},
            {"start": _time(8, 31), "end": _time(8, 41), "origine": "X", "terminus": "A",
             "mission": "B → A", "is_mission_start": False},
        ],
    }


def test_croisement_interieur_propage_sur_voie_unique_mais_pas_double():
    unique = _infrastructure().copy()
    unique.loc[unique["gare"] == "B", "infra"] = "Terminus"
    chronologie = _chronologie_croisement()
    scenario_unique = evaluer_robustesse(chronologie, unique, [_mission()])["scenarios"][0]

    double = unique.copy()
    double.loc[double["gare"] == "A", "infra"] = "D"
    scenario_double = evaluer_robustesse(chronologie, double, [_mission()])["scenarios"][0]

    assert scenario_unique["sans_reserve"]["retard_cumule_min"] == 10
    assert scenario_double["sans_reserve"]["retard_cumule_min"] == 5
    assert scenario_unique["sans_reserve"]["infra_violations"] == []
    t2_unique = next(row for row in scenario_unique["sans_reserve"]["arrivals"] if row["train_id"] == "T2")
    t2_double = next(row for row in scenario_double["sans_reserve"]["arrivals"] if row["train_id"] == "T2")
    assert t2_unique["retard_arrivee_min"] == 5
    assert t2_double["retard_arrivee_min"] == 0


def test_incident_au_croisement_interieur_preserve_le_prefixe_deja_parcouru():
    chronologie = _chronologie()
    services, gares, infra = _compiler_services(chronologie, _infrastructure(), [_mission()])
    point = next(point for point in _enumerer_points(services, infra)
                 if point["gare"] == "X" and point["kind"] == "croisement")

    replay = _rejouer(services, gares, point, 5)
    movements = [row for row in replay["chronologie"]["T1"]
                 if row["origine"] != row["terminus"]]

    assert movements[0]["start"] == _time(8, 0)
    assert movements[0]["end"] == _time(8, 10)
    assert movements[1]["start"] == _time(8, 15)
    assert movements[1]["end"] == _time(8, 25)


def test_plusieurs_attentes_consecutives_sont_conservees_au_croisement():
    chronologie = {
        "T1": [
            {"start": _time(8, 0), "end": _time(8, 10), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(8, 10), "end": _time(8, 12), "origine": "X", "terminus": "X",
             "mission": "A → B", "is_mission_start": False},
            {"start": _time(8, 12), "end": _time(8, 15), "origine": "X", "terminus": "X",
             "mission": "A → B", "is_mission_start": False, "crossing_extension_min": 3},
            {"start": _time(8, 15), "end": _time(8, 25), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
        ]
    }

    resultat = evaluer_robustesse(chronologie, _infrastructure(), [_mission()])

    assert resultat["candidates_count"] == 2
    assert resultat["scenarios"][0]["sans_reserve"]["retard_cumule_min"] == 5


def test_bloc_sans_evitement_supprime_le_service_et_penalise_l_attente():
    gares = _infrastructure().copy()
    gares.loc[gares["gare"] == "X", "infra"] = "F"
    gares.loc[gares["gare"] == "B", "infra"] = "F"
    chronologie = _chronologie_croisement()
    chronologie["T2"].insert(0, {
        "start": _time(8, 20), "end": _time(8, 21),
        "origine": "B", "terminus": "B", "mission": "B → A",
        "is_mission_start": False,
    })

    evaluation = evaluer_robustesse(
        chronologie, gares, [dict(_mission(), frequence=2.0)]
    )
    scenario = evaluation["scenarios"][0]["sans_reserve"]

    assert scenario["infra_violations"] == []
    assert scenario["physical_violations"] == []
    assert scenario["suppressions_count"] == 1
    suppression, = scenario["suppressions"]
    assert set(("service_id", "motif", "penalite_min")).issubset(suppression)
    assert suppression["penalite_min"] == 40
    assert scenario["retard_trains_min"] == 5
    assert scenario["penalite_suppressions_min"] == 40
    assert scenario["retard_cumule_min"] == 45
    assert evaluation["scenarios"][0]["point"]["train_id"] == "T1"
    assert suppression["hypothese"] == "Fréquence mission 2 train/h"
    suppressed_rows = [row for rows in scenario["chronologie"].values() for row in rows
                       if row.get("service_id") == suppression["service_id"]]
    assert suppressed_rows == []
    suppressed_metric = next(row for row in scenario["arrivals"]
                             if row["service_id"] == suppression["service_id"])
    assert suppressed_metric["complete"] is False


def test_suppressions_multiples_attendent_le_prochain_service_effectivement_assure():
    gares = _infrastructure().copy()
    gares["infra"] = "F"
    chronologie = {
        "T1": [
            {"start": _time(8, 0), "end": _time(8, 10), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(8, 10), "end": _time(8, 20), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
        ],
    }
    for index, minute in enumerate((21, 24, 27, 30, 33, 36, 41), start=2):
        chronologie[f"T{index}"] = [{
            "start": _time(8, minute), "end": _time(8, minute + 2),
            "origine": "B", "terminus": "A", "mission": "B → A",
            "is_mission_start": True,
        }]
    services, infrastructure, _ = _compiler_services(chronologie, gares, [_mission()])
    incident_service = next(service for service in services if service["train_id"] == "T1")
    incident = {
        "service_id": incident_service["service_id"], "gare": "A",
        "time": incident_service["planned_start"], "mission": "A → B",
        "kind": "depart", "block_index": 0,
    }

    replay = _rejouer(services, infrastructure, incident, 20)

    assert replay["infra_violations"] == []
    assert replay["physical_violations"] == []
    assert replay["suppressions_count"] == 6
    assert [item["penalite_min"] for item in replay["suppressions"]] == [
        30, 27, 24, 21, 18, 15,
    ]
    assert {item["service_suivant_assure"] for item in replay["suppressions"]} == {
        "str:T8#1",
    }
    assert replay["retard_trains_min"] == 20
    assert replay["penalite_suppressions_min"] == 135
    assert replay["retard_cumule_min"] == 155
    suppressed_ids = {item["service_id"] for item in replay["suppressions"]}
    assert not any(row.get("service_id") in suppressed_ids
                   for rows in replay["chronologie"].values() for row in rows)
    operated = [row for row in replay["chronologie"]["T8"]
                if row["origine"] != row["terminus"]]
    assert operated[0]["origine"] == "B"


def test_suppression_sans_frequence_explicite_documente_l_hypothese():
    gares = _infrastructure().copy()
    gares.loc[gares["gare"] == "X", "infra"] = "F"
    gares.loc[gares["gare"] == "B", "infra"] = "F"

    scenario = evaluer_robustesse(
        _chronologie_croisement(), gares, [_mission()]
    )["scenarios"][0]["sans_reserve"]

    suppression, = scenario["suppressions"]
    assert suppression["penalite_min"] == 70
    assert suppression["hypothese"] == (
        "Fréquence et intervalle absents : hypothèse explicite de 1 train/h"
    )


def test_conflit_nominal_invalide_leve_value_error_au_lieu_de_supprimer():
    gares = _infrastructure().copy()
    gares.loc[gares["gare"] == "X", "infra"] = "F"
    gares.loc[gares["gare"] == "B", "infra"] = "F"
    chronologie = _chronologie_croisement()
    chronologie["T2"][0]["start"] = _time(8, 10)
    chronologie["T2"][0]["end"] = _time(8, 20)
    chronologie["T2"][1]["start"] = _time(8, 20)
    chronologie["T2"][1]["end"] = _time(8, 30)

    with pytest.raises(ValueError, match="chronologie nominale"):
        evaluer_robustesse(chronologie, gares, [_mission()])


def test_croisements_interieurs_et_pire_depart_plusieurs_heures_apres():
    gares = _infrastructure().copy()
    gares.loc[gares["gare"] == "B", "infra"] = "VE"
    chronologie = {
        "T1": [
            {"start": _time(8, 0), "end": _time(8, 10), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(8, 10), "end": _time(8, 20), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
        ],
        "T3": [
            {"start": _time(12, 0), "end": _time(12, 10), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(12, 10), "end": _time(12, 20), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
        ],
        "T4": [
            {"start": _time(12, 21), "end": _time(12, 31), "origine": "B", "terminus": "X",
             "mission": "B → A", "is_mission_start": True},
            {"start": _time(12, 31), "end": _time(12, 41), "origine": "X", "terminus": "A",
             "mission": "B → A", "is_mission_start": False},
        ],
    }

    result = evaluer_robustesse(chronologie, gares, [_mission()])
    worst = next(scenario for scenario in result["scenarios"] if scenario["delay_min"] == 20)
    services, _, infra = _compiler_services(chronologie, gares, [_mission()])
    points = _enumerer_points(services, infra)

    assert result["candidates_count"] == 8
    assert any(point["train_id"] == "T3" and point["gare"] == "X"
               and point["kind"] == "croisement" and point["time"] == _time(12, 10)
               for point in points)
    assert worst["point"]["train_id"] == "T3"
    assert worst["point"]["gare"] == "A"
    assert worst["point"]["kind"] == "depart"
    assert worst["point"]["time"] == _time(12, 0)
    assert worst["sans_reserve"]["retard_cumule_min"] > 20
    assert worst["sans_reserve"]["infra_violations"] == []


def _chronologie_aller_retour(retour_minute):
    return {
        "T1": [
            {"start": _time(8, 0), "end": _time(8, 10), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(8, 10), "end": _time(8, 20), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
            {"start": _time(8, retour_minute), "end": _time(8, retour_minute + 10),
             "origine": "B", "terminus": "X", "mission": "B → A", "is_mission_start": True},
            {"start": _time(8, retour_minute + 10), "end": _time(8, retour_minute + 20),
             "origine": "X", "terminus": "A", "mission": "B → A", "is_mission_start": False},
        ]
    }


def _scenario_recuperation_suppression(suppression_future=False, delay_min=5):
    gares = _infrastructure().copy()
    gares.loc[gares["gare"] == "X", "infra"] = "F"
    chronologie = {
        "T1": [
            {"start": _time(8, 0), "end": _time(8, 10), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(8, 10), "end": _time(8, 20), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
        ],
        "T2": [
            {"start": _time(8, 20), "end": _time(8, 25), "origine": "B", "terminus": "X",
             "mission": "B → A", "is_mission_start": True},
            {"start": _time(8, 25), "end": _time(8, 30), "origine": "X", "terminus": "A",
             "mission": "B → A", "is_mission_start": False},
        ],
        "T3": [
            {"start": _time(8, 30), "end": _time(8, 40), "origine": "B", "terminus": "X",
             "mission": "B → A", "is_mission_start": True},
            {"start": _time(8, 40), "end": _time(8, 50), "origine": "X", "terminus": "A",
             "mission": "B → A", "is_mission_start": False},
        ],
    }
    if suppression_future:
        chronologie["T2"].extend([
            {"start": _time(8, 50), "end": _time(9, 0), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(9, 0), "end": _time(9, 10), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
        ])
    services, gares, infra_map = _compiler_services(
        chronologie, gares, [dict(_mission(), frequence=6.0)]
    )
    incident = next(
        point for point in _enumerer_points(services, infra_map)
        if point["service_id"] == "str:T1#1" and point["kind"] == "depart"
    )
    replay = _rejouer(services, gares, incident, delay_min)
    return services, incident, replay, chronologie, gares


def _chronologie_retournements(turnaround_a, turnaround_b):
    retour_depart = _time(8, 20) + pd.Timedelta(minutes=max(5, turnaround_b))
    retour_arrivee = retour_depart + pd.Timedelta(minutes=20)
    aller_suivant = retour_arrivee + pd.Timedelta(minutes=max(5, turnaround_a))
    return {
        "T1": [
            {"start": _time(8, 0), "end": _time(8, 10), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(8, 10), "end": _time(8, 20), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
            {"start": retour_depart, "end": retour_depart + pd.Timedelta(minutes=10),
             "origine": "B", "terminus": "X", "mission": "B → A", "is_mission_start": True},
            {"start": retour_depart + pd.Timedelta(minutes=10), "end": retour_arrivee,
             "origine": "X", "terminus": "A", "mission": "B → A", "is_mission_start": False},
            {"start": aller_suivant, "end": aller_suivant + pd.Timedelta(minutes=10),
             "origine": "A", "terminus": "X", "mission": "A → B", "is_mission_start": True},
            {"start": aller_suivant + pd.Timedelta(minutes=10),
             "end": aller_suivant + pd.Timedelta(minutes=20),
             "origine": "X", "terminus": "B", "mission": "A → B", "is_mission_start": False},
        ],
        "T2": [
            {"start": _time(10, 0), "end": _time(10, 10), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(10, 10), "end": _time(10, 20), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
        ],
    }


def test_retournement_retarde_garde_cinq_minutes_aux_deux_terminus():
    for terminus in ("A", "B"):
        for turnaround in (0, 2, 5, 10):
            turnaround_a = turnaround if terminus == "A" else 5
            turnaround_b = turnaround if terminus == "B" else 5
            mission = dict(
                _mission(),
                temps_retournement_A=turnaround_a,
                temps_retournement_B=turnaround_b,
            )
            services, gares, _ = _compiler_services(
                _chronologie_retournements(turnaround_a, turnaround_b),
                _infrastructure(), [mission],
            )
            nominal = _rejouer(services, gares)
            assert all(row["depart_reel"] == row["depart_planifie"]
                       for row in nominal["services"])

            incident_start = _time(8, 0 if terminus == "B" else 25)
            incident_service = next(
                service for service in services
                if service["planned_start"] == incident_start
            )
            next_service = next(
                service for service in services
                if service["train_id"] == "T1"
                and service["origine"] == terminus
                and service["planned_start"] > incident_start
            )
            incident = {
                "service_id": incident_service["service_id"],
                "gare": incident_service["origine"],
                "time": incident_start,
                "mission": incident_service["mission"],
                "kind": "depart",
                "block_index": 0,
            }
            replay = _rejouer(services, gares, incident, 5)
            actual_incident = next(
                row for row in replay["services"]
                if row["service_id"] == incident_service["service_id"]
            )
            actual_next = next(
                row for row in replay["services"]
                if row["service_id"] == next_service["service_id"]
            )
            expected_departure = max(
                next_service["planned_start"],
                actual_incident["arrivee_reelle"] + pd.Timedelta(
                    minutes=5
                ),
            )
            assert actual_next["depart_reel"] == expected_departure

            unaffected = next(row for row in replay["services"] if row["train_id"] == "T2")
            assert unaffected["depart_reel"] == unaffected["depart_planifie"]
            earlier = [row for row in replay["services"]
                       if row["depart_planifie"] < incident_start]
            assert all(row["depart_reel"] == row["depart_planifie"] for row in earlier)


def test_retournement_minimum_cinq_minutes_et_recuperation_verifiee():
    mission = dict(_mission(), temps_retournement_B=5, temps_retournement_A=5)
    nominal_dix = evaluer_robustesse(
        _chronologie_aller_retour(30), _infrastructure(), [mission]
    )["scenarios"][0]["sans_reserve"]
    nominal_cinq = evaluer_robustesse(
        _chronologie_aller_retour(25), _infrastructure(), [mission]
    )["scenarios"][0]["sans_reserve"]

    assert nominal_dix["retard_cumule_min"] == 5
    assert nominal_dix["retour_normal_min"] == 30
    assert nominal_cinq["retard_cumule_min"] == 10
    assert nominal_cinq["retour_normal_min"] is None
    movements = [row for row in nominal_dix["chronologie"]["T1"]
                 if row["origine"] != row["terminus"]]
    assert movements[0]["start"] == _time(8, 5)
    assert movements[2]["start"] == _time(8, 30)


def test_retour_normal_apres_une_suppression_est_prouve_par_le_service_suivant():
    services, incident, replay, _, _ = _scenario_recuperation_suppression()
    service_assure = next(row for row in replay["services"] if row["train_id"] == "T3")

    assert replay["suppressions_count"] == 1
    assert service_assure["complete"] is True
    assert service_assure["depart_reel"] == _time(8, 30)
    assert _retour_normal(
        services, replay["services"], incident, 5, replay["infra_violations"]
    ) == 30


def test_suppression_future_empeche_de_prouver_un_retour_anterieur():
    services, incident, replay, _, _ = _scenario_recuperation_suppression(
        suppression_future=True
    )
    service_assure = next(row for row in replay["services"] if row["train_id"] == "T3")
    service_supprime_plus_tard = next(
        row for row in replay["services"]
        if row["train_id"] == "T2" and row["depart_planifie"] == _time(8, 50)
    )

    assert service_assure["depart_reel"] == _time(8, 30)
    assert service_supprime_plus_tard["supprime"] is True
    assert _retour_normal(
        services, replay["services"], incident, 5, replay["infra_violations"]
    ) is None


def test_reserve_remplace_un_service_supprime_malgre_une_suppression_restante():
    services, incident, no_reserve, chronologie, gares = (
        _scenario_recuperation_suppression(suppression_future=True, delay_min=10)
    )
    baseline_energy = _analyser_energie(chronologie, services, gares, None)

    resultat = _evaluuer_avec_reserves(
        services, gares, incident, 10, ["A"], no_reserve, baseline_energy, None
    )

    assert resultat is not None
    assert no_reserve["retard_cumule_min"] == 50
    assert no_reserve["suppressions_count"] == 2
    assert resultat["retard_cumule_min"] == 30
    assert [item["service_id"] for item in resultat["suppressions"]] == ["str:T2#1"]
    assert len(resultat["reserve_activations"]) == 1
    activation, = resultat["reserve_activations"]
    assert activation["service_id"] == "str:T2#2"
    assert activation["gain_cumule_min"] == 20
    assert activation["retard_evite_min"] == 20


def test_retournements_repetees_reduisent_le_retard_jusqua_reprise_du_plan():
    mission = dict(_mission(), temps_retournement_A=10, temps_retournement_B=10)
    chronologie = {"T1": []}
    segments = [
        ("A", "X", 8, 0, 8, 10, "A → B", True),
        ("X", "B", 8, 10, 8, 20, "A → B", False),
        ("B", "X", 8, 30, 8, 40, "B → A", True),
        ("X", "A", 8, 40, 8, 50, "B → A", False),
        ("A", "X", 9, 0, 9, 10, "A → B", True),
        ("X", "B", 9, 10, 9, 20, "A → B", False),
        ("B", "X", 9, 30, 9, 40, "B → A", True),
        ("X", "A", 9, 40, 9, 50, "B → A", False),
        ("A", "X", 10, 0, 10, 10, "A → B", True),
        ("X", "B", 10, 10, 10, 20, "A → B", False),
        ("B", "X", 10, 30, 10, 40, "B → A", True),
        ("X", "A", 10, 40, 10, 50, "B → A", False),
    ]
    chronologie["T1"] = [
        {"start": _time(start_h, start_m), "end": _time(end_h, end_m),
         "origine": origin, "terminus": terminus, "mission": label,
         "is_mission_start": is_start}
        for origin, terminus, start_h, start_m, end_h, end_m, label, is_start in segments
    ]
    services, gares, _ = _compiler_services(chronologie, _infrastructure(), [mission])
    incident_service = services[1]
    incident = {
        "service_id": incident_service["service_id"], "gare": "B",
        "time": incident_service["planned_start"], "mission": "B → A",
        "kind": "depart", "block_index": 0,
    }

    replay = _rejouer(services, gares, incident, 20)

    assert [row["depart_reel"] for row in replay["services"]] == [
        _time(8, 0), _time(8, 50), _time(9, 15),
        _time(9, 40), _time(10, 5), _time(10, 30),
    ]
    assert [row["retard_arrivee_min"] for row in replay["services"]] == [
        0, 20, 15, 10, 5, 0,
    ]
    assert all(row["depart_reel"] >= row["depart_planifie"] for row in replay["services"])


def test_reserve_remplace_une_rotation_aval_seulement_si_gain_strict():
    chronologie = _chronologie_aller_retour(25)
    mission = dict(_mission(), temps_retournement_B=5, temps_retournement_A=5)

    reserve_a = evaluer_robustesse(
        chronologie, _infrastructure(), [mission], reserve_gares=("A",)
    )["scenarios"][0]["avec_reserve"]
    reserve_b = evaluer_robustesse(
        chronologie, _infrastructure(), [mission], reserve_gares=("B",)
    )["scenarios"][0]["avec_reserve"]
    reserves_ab = evaluer_robustesse(
        chronologie, _infrastructure(), [mission], reserve_gares=("A", "B")
    )["scenarios"][0]["avec_reserve"]

    assert reserve_a["reserve_activations"] == []
    assert any("physiquement" in reason
               for item in reserve_a["reserve_refus"] for reason in item["raisons"])
    assert len(reserve_b["reserve_activations"]) == 1
    assert reserve_b["reserve_activations"][0]["mission"] == "B → A"
    assert reserve_b["retard_cumule_min"] == 5
    assert reserve_b["reserve_activations"][0]["gain_cumule_min"] == 5
    return_service_id = reserve_b["reserve_activations"][0]["service_id"]
    reserve_rows = next(rows for train_id, rows in reserve_b["chronologie"].items()
                        if str(train_id).startswith("Réserve B"))
    reserve_first_movement = next(row for row in reserve_rows if row["origine"] != row["terminus"])
    assert reserve_first_movement["service_id"] == return_service_id
    assert reserve_first_movement["is_mission_start"] is True
    assert len(reserves_ab["reserve_activations"]) == 1
    assert reserves_ab["reserve_activations"][0]["mission"] == "B → A"


def test_reserve_physique_ne_peut_pas_refaire_un_trajet_sans_revenir():
    chronologie = {
        "T1": [
            {"start": _time(8, 0), "end": _time(8, 10), "origine": "B", "terminus": "X",
             "mission": "B → A", "is_mission_start": True},
            {"start": _time(8, 10), "end": _time(8, 20), "origine": "X", "terminus": "A",
             "mission": "B → A", "is_mission_start": False},
        ],
        "T2": [
            {"start": _time(9, 0), "end": _time(9, 10), "origine": "B", "terminus": "X",
             "mission": "B → A", "is_mission_start": True},
            {"start": _time(9, 10), "end": _time(9, 20), "origine": "X", "terminus": "A",
             "mission": "B → A", "is_mission_start": False},
        ],
    }
    services, gares, _ = _compiler_services(chronologie, _infrastructure(), [_mission()])
    reserve_map = {service["service_id"]: "B" for service in services}

    replay = _rejouer(services, gares, reserve_map=reserve_map, reserve_stations=("B",))

    assert len(replay["physical_violations"]) == 1
    assert replay["services"][1]["complete"] is False


def test_reserve_peut_enchainer_une_rotation_si_son_parcours_la_repositionne():
    chronologie = _chronologie_aller_retour(25)
    chronologie["T1"].extend([
        {"start": _time(8, 50), "end": _time(9, 0), "origine": "A", "terminus": "X",
         "mission": "A → B", "is_mission_start": True},
        {"start": _time(9, 0), "end": _time(9, 10), "origine": "X", "terminus": "B",
         "mission": "A → B", "is_mission_start": False},
    ])
    mission = dict(_mission(), temps_retournement_B=5, temps_retournement_A=5)

    reserve = evaluer_robustesse(
        chronologie, _infrastructure(), [mission], reserve_gares=("B",)
    )["scenarios"][0]["avec_reserve"]

    assert [item["mission"] for item in reserve["reserve_activations"]] == ["B → A", "A → B"]
    reserve_id = reserve["reserve_activations"][0]["reserve_train_id"]
    reserve_missions = [row["service_id"] for row in reserve["chronologie"][reserve_id]
                        if row.get("is_mission_start")]
    assert reserve_missions == [item["service_id"] for item in reserve["reserve_activations"]]
    assert reserve["physical_violations"] == []


def test_reserve_batterie_demarre_pleine_et_autonomie_insuffisante_exclut():
    chronologie = _chronologie_aller_retour(25)
    mission = dict(_mission(), type_materiel="batterie", temps_retournement_B=5)
    params = get_default_energy_params()
    resultat = evaluer_robustesse(
        chronologie, _infrastructure(), [mission],
        energy_params={"batterie": params}, reserve_gares=("B",),
    )["scenarios"][0]["avec_reserve"]

    assert len(resultat["reserve_activations"]) == 1
    reserve_id = resultat["reserve_activations"][0]["reserve_train_id"]
    assert resultat["battery"]["scenario"]["par_train"][reserve_id]["soc_initial_pct"] == 95

    batterie_limitee = dict(params, capacite_batterie_kwh=1, soc_min_pct=20, soc_max_pct=95)
    limite = evaluer_robustesse(
        chronologie, _infrastructure(), [mission],
        energy_params={"batterie": batterie_limitee}, reserve_gares=("B",),
    )["scenarios"][0]["avec_reserve"]
    assert limite["reserve_activations"] == []
    assert any("Batterie au seuil minimum" in error for error in limite["reserve_refus"][0]["raisons"])
    assert limite["battery"]["autonomie_insuffisante"] is True
    assert limite["battery"]["autonomie_non_verifiable"] is False
    assert limite["battery"]["alertes_autonomie"]
    assert any("20" in alert for alert in limite["battery"]["alertes_autonomie"])


def _bilan_batterie(energy_params):
    chronologie = _chronologie()
    mission = dict(_mission(), type_materiel="batterie")
    services, gares, _ = _compiler_services(
        chronologie, _infrastructure(), [mission]
    )
    return _analyser_energie(chronologie, services, gares, energy_params)


@pytest.mark.parametrize(
    "energy_params",
    [
        pytest.param(None, id="params-none"),
        pytest.param({}, id="params-vides"),
        pytest.param({"diesel": get_default_energy_params()}, id="batterie-absente"),
    ],
)
def test_parametres_batterie_absents_sont_non_verifiables_pas_insuffisants(energy_params):
    bilan = _bilan_batterie(energy_params)

    assert bilan["autonomie_non_verifiable"] is True
    assert bilan["alertes_autonomie_non_verifiable"]
    assert bilan["autonomie_insuffisante"] is False
    train = bilan["par_train"]["T1"]
    assert train["type_materiel"] == "batterie"
    assert train["autonomie_non_verifiable"] is True
    assert train["erreurs"]
    for field in (
        "soc_initial_pct", "min_soc_pct", "soc_final_pct", "marge_kwh",
        "marge_soc_pct", "energie_utilisable_au_dessus_seuil_kwh",
    ):
        assert train[field] is None


def test_absence_parametres_conserve_le_defaut_pour_materiel_non_batterie():
    chronologie = _chronologie()
    services, gares, _ = _compiler_services(
        chronologie, _infrastructure(), [_mission()]
    )

    bilan = _analyser_energie(chronologie, services, gares, None)

    assert bilan["par_train"]["T1"]["consommation_nette_kwh"] is not None
    assert bilan["autonomie_non_verifiable"] is False
    assert bilan["autonomie_insuffisante"] is False


@pytest.mark.parametrize(
    "invalid_case",
    [
        "capacite_absente", "capacite_nulle", "seuil_absent", "seuil_invalide",
        "configuration_mal_formee",
    ],
)
def test_parametres_batterie_invalides_sont_non_verifiables_sans_calcul(
    invalid_case, monkeypatch
):
    if invalid_case == "configuration_mal_formee":
        energy_params = {"batterie": "invalide"}
    else:
        params = get_default_energy_params()
        if invalid_case == "capacite_absente":
            params.pop("capacite_batterie_kwh")
        elif invalid_case == "capacite_nulle":
            params["capacite_batterie_kwh"] = 0
        elif invalid_case == "seuil_absent":
            params.pop("soc_min_pct")
        else:
            params["soc_min_pct"] = "invalide"
        energy_params = {"batterie": params}
    monkeypatch.setattr(
        robustness_module,
        "calculer_consommation_trajet",
        lambda *args, **kwargs: pytest.fail("calcul lancé avec des paramètres invalides"),
    )

    bilan = _bilan_batterie(energy_params)

    assert bilan["autonomie_non_verifiable"] is True
    assert bilan["autonomie_insuffisante"] is False
    assert bilan["alertes_autonomie_non_verifiable"]
    train = bilan["par_train"]["T1"]
    assert train["type_materiel"] == "batterie"
    assert train["erreurs"]
    assert train["soc_initial_pct"] is None
    assert train["marge_kwh"] is None


def test_log_batterie_absent_rend_le_bilan_non_verifiable(monkeypatch):
    monkeypatch.setattr(
        robustness_module,
        "calculer_consommation_trajet",
        lambda *args, **kwargs: {
            "total_conso_nette_kwh": 10,
            "total_conso_brute_kwh": 10,
            "total_distance_km": 20,
            "erreurs": [],
        },
    )

    bilan = _bilan_batterie({"batterie": get_default_energy_params()})

    assert bilan["autonomie_non_verifiable"] is True
    assert bilan["autonomie_insuffisante"] is False
    assert bilan["alertes_autonomie_non_verifiable"]
    train = bilan["par_train"]["T1"]
    assert train["soc_initial_pct"] is None
    assert train["marge_kwh"] is None
    assert train["erreurs"]


def test_exception_calcul_batterie_conserve_un_train_non_verifiable(monkeypatch):
    def lever_exception(*args, **kwargs):
        raise RuntimeError("calcul interrompu")

    monkeypatch.setattr(robustness_module, "calculer_consommation_trajet", lever_exception)

    bilan = _bilan_batterie({"batterie": get_default_energy_params()})

    assert bilan["autonomie_non_verifiable"] is True
    assert bilan["autonomie_insuffisante"] is False
    assert bilan["alertes_autonomie_non_verifiable"]
    train = bilan["par_train"]["T1"]
    assert train["type_materiel"] == "batterie"
    assert train["soc_initial_pct"] is None
    assert train["marge_kwh"] is None
    assert any("calcul interrompu" in error for error in train["erreurs"])


def test_reserve_refuse_si_une_autre_rame_batterie_atteint_le_seuil():
    gares = pd.concat([
        _infrastructure(),
        pd.DataFrame([{"gare": "C", "distance": 30.0, "infra": "F", "electrification": "F"}]),
    ], ignore_index=True)
    chronologie = _chronologie_aller_retour(25)
    chronologie["T2"] = [
        {"start": _time(10, 0), "end": _time(10, 10), "origine": "C", "terminus": "B",
         "mission": "C → A", "is_mission_start": True},
        {"start": _time(10, 10), "end": _time(10, 20), "origine": "B", "terminus": "X",
         "mission": "C → A", "is_mission_start": False},
        {"start": _time(10, 20), "end": _time(10, 30), "origine": "X", "terminus": "A",
         "mission": "C → A", "is_mission_start": False},
    ]
    mission_diesel = _mission()
    mission_batterie = {
        "origine": "C",
        "terminus": "A",
        "type_materiel": "batterie",
        "temps_retournement_A": 5,
        "temps_retournement_B": 5,
    }
    params_diesel = get_default_energy_params()
    params_batterie = dict(
        get_default_energy_params(), capacite_batterie_kwh=1, soc_min_pct=20, soc_max_pct=95
    )

    scenario = evaluer_robustesse(
        chronologie, gares, [mission_diesel, mission_batterie],
        energy_params={"diesel": params_diesel, "batterie": params_batterie},
        reserve_gares=("B",),
    )["scenarios"][0]["avec_reserve"]

    assert any(
        "Batterie au seuil minimum à A!" in error
        for error in scenario["battery"]["scenario"]["erreurs"]
    )
    assert scenario["reserve_activations"] == []
    assert any(
        "Batterie au seuil minimum à A!" in reason
        for refusal in scenario["reserve_refus"]
        for reason in refusal["raisons"]
    )


def test_bilan_batterie_expose_distance_marge_physique_et_delta_nominal():
    mission = dict(_mission(), type_materiel="batterie")
    params = get_default_energy_params()
    scenario = evaluer_robustesse(
        _chronologie_aller_retour(25), _infrastructure(), [mission],
        energy_params={"batterie": params},
    )["scenarios"][0]["sans_reserve"]
    baseline_train = scenario["battery"]["baseline"]["par_train"]["T1"]
    scenario_train = scenario["battery"]["scenario"]["par_train"]["T1"]

    assert baseline_train["distance_km"] == 40.0
    assert scenario_train["distance_km"] == 40.0
    assert scenario["battery"]["autonomie_insuffisante"] is False
    assert scenario["battery"]["autonomie_non_verifiable"] is False
    assert scenario["battery"]["alertes_autonomie"] == []
    assert scenario["battery"]["alertes_autonomie_non_verifiable"] == []
    assert scenario_train["energie_utilisable_au_dessus_seuil_kwh"] == scenario_train["marge_kwh"]
    assert scenario_train["marge_soc_pct"] is not None
    delta = scenario_train["delta_vs_nominal"]
    assert delta["distance_km"] == 0.0
    assert delta["marge_kwh"] == round(
        scenario_train["marge_kwh"] - baseline_train["marge_kwh"], 6
    )
    assert delta["marge_soc_pct"] == round(
        scenario_train["marge_soc_pct"] - baseline_train["marge_soc_pct"], 6
    )


def test_export_batterie_expose_les_donnees_non_verifiables_et_ses_deltas_absents():
    mission = dict(_mission(), type_materiel="batterie")
    scenario = evaluer_robustesse(
        _chronologie(), _infrastructure(), [mission], energy_params=None,
    )["scenarios"][0]["sans_reserve"]
    battery = scenario["battery"]

    assert battery["autonomie_non_verifiable"] is True
    assert battery["alertes_autonomie_non_verifiable"]
    assert battery["baseline"]["autonomie_non_verifiable"] is True
    assert battery["baseline"]["alertes_autonomie_non_verifiable"]
    assert battery["baseline_autonomie_non_verifiable"] is True
    assert battery["alertes_autonomie_non_verifiable_baseline"]
    assert battery["scenario"]["autonomie_non_verifiable"] is True
    assert battery["scenario"]["alertes_autonomie_non_verifiable"]
    assert battery["delta_conso_nette_kwh"] is None
    assert battery["delta_kwh"] is None
    train = battery["scenario"]["par_train"]["T1"]
    assert train["soc_initial_pct"] is None
    assert train["marge_kwh"] is None
    assert train["delta_vs_nominal"]["marge_kwh"] is None


def test_reserve_refuse_un_bilan_batterie_non_verifiable():
    mission = dict(_mission(), type_materiel="batterie", temps_retournement_B=5)
    scenario = evaluer_robustesse(
        _chronologie_aller_retour(25), _infrastructure(), [mission],
        energy_params=None, reserve_gares=("B",),
    )["scenarios"][0]["avec_reserve"]

    assert scenario["reserve_activations"] == []
    assert scenario["battery"]["autonomie_non_verifiable"] is True
    assert any(scenario["battery"]["alertes_autonomie_non_verifiable"])
    assert scenario["reserve_refus"]


def test_fingerprint_robustesse_est_stable_et_sensible_aux_reserves():
    chronologie = _chronologie()
    missions = [_mission()]
    key = fingerprint_robustesse(chronologie, _infrastructure(), missions)

    assert key == fingerprint_robustesse(chronologie, _infrastructure(), missions)
    assert key != fingerprint_robustesse(
        chronologie, _infrastructure(), missions, reserve_gares=("B",)
    )


def _chronologie_derniere_mission_tronquee():
    """A → B complète, puis B → A coupée à X par la fin de service."""
    return {
        "T1": [
            {"start": _time(8, 0), "end": _time(8, 10), "origine": "A", "terminus": "X",
             "mission": "A → B", "is_mission_start": True},
            {"start": _time(8, 10), "end": _time(8, 20), "origine": "X", "terminus": "B",
             "mission": "A → B", "is_mission_start": False},
            {"start": _time(8, 30), "end": _time(8, 40), "origine": "B", "terminus": "X",
             "mission": "B → A", "is_mission_start": True},
            {"start": _time(8, 40), "end": _time(8, 42), "origine": "X", "terminus": "X",
             "mission": "B → A", "is_mission_start": False},
        ]
    }


def test_derniere_mission_tronquee_par_la_fin_de_service_est_evaluee():
    chronologie = _chronologie_derniere_mission_tronquee()
    services, _, _ = _compiler_services(chronologie, _infrastructure(), [_mission()])

    assert [(s["mission"], s["terminus"], s["tronquee"]) for s in services] == [
        ("A → B", "B", False),
        ("B → A", "X", True),
    ]
    assert services[1]["planned_arrival"] == _time(8, 40)

    resultat = evaluer_robustesse(chronologie, _infrastructure(), [_mission()])
    assert len(resultat["scenarios"]) == 3
    assert all(s["sans_reserve"]["retard_cumule_min"] > 0 for s in resultat["scenarios"])


def test_mission_tronquee_hors_fin_de_rame_reste_refusee():
    chronologie = _chronologie_derniere_mission_tronquee()
    rows = chronologie["T1"]
    # La mission B → A s'arrête à X puis la rame repart pour un A → B : incohérent.
    chronologie["T1"] = [rows[0], rows[1], rows[2], {
        "start": _time(9, 0), "end": _time(9, 20), "origine": "X", "terminus": "B",
        "mission": "A → B", "is_mission_start": True,
    }]
    with pytest.raises(ValueError):
        _compiler_services(chronologie, _infrastructure(), [_mission()])


def test_robustesse_sur_chronologie_reelle_avec_missions_tronquees():
    from core_logic import generer_tous_trajets_optimises
    from scenarios_manager import load_scenario

    chemin = Path(__file__).parent / "scenarios" / "bayonne_sjpp_2032+_croisement_itxassou.json"
    scenario = load_scenario(chemin)
    gares = pd.DataFrame(scenario["infrastructure"]["gares"])
    debut = datetime.strptime(scenario["service"]["heure_debut"], "%H:%M").time()
    fin = datetime.strptime(scenario["service"]["heure_fin"], "%H:%M").time()
    chronologie, _, _ = generer_tous_trajets_optimises(
        scenario["missions"], gares, debut, fin, allow_sharing=True, search_strategy="simple",
    )
    services, _, _ = _compiler_services(chronologie, gares, scenario["missions"])
    assert any(s["tronquee"] for s in services)

    resultat = evaluer_robustesse(
        chronologie, gares, scenario["missions"], energy_params=scenario.get("energy_params"),
    )
    assert [s["delay_min"] for s in resultat["scenarios"]] == [5, 10, 20]
    assert all(s["point"] is not None for s in resultat["scenarios"])
    for item in resultat["scenarios"]:
        replay = item["sans_reserve"]
        assert replay["infra_violations"] == []
        assert replay["physical_violations"] == []
        assert replay["suppressions_count"] == len(replay["suppressions"])
        assert all({"service_id", "motif", "penalite_min"}.issubset(suppression)
                   for suppression in replay["suppressions"])
        assert replay["retard_cumule_min"] == round(
            replay["retard_trains_min"] + replay["penalite_suppressions_min"], 6
        )
