from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from energy_logic import get_default_energy_params
from robustness_logic import (
    _compiler_services,
    _enumerer_points,
    _rejouer,
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


def test_bloc_sans_evitement_signale_un_conflit_sans_attendre_en_gare_f():
    gares = _infrastructure().copy()
    gares.loc[gares["gare"] == "X", "infra"] = "F"
    gares.loc[gares["gare"] == "B", "infra"] = "F"
    chronologie = _chronologie_croisement()

    scenario = evaluer_robustesse(chronologie, gares, [_mission()])["scenarios"][0]

    assert scenario["sans_reserve"]["infra_violations"]
    assert any(item["gare"] == "B" for item in scenario["sans_reserve"]["infra_violations"])


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
                    minutes=max(5, turnaround)
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
