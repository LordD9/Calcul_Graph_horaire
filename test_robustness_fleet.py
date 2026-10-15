from datetime import datetime

import pandas as pd
import pytest

from energy_logic import get_default_energy_params
from robustness_logic import evaluer_robustesse, fingerprint_robustesse


def _infrastructure():
    return pd.DataFrame([
        {"gare": "A", "distance": 0, "infra": "F"},
        {"gare": "X", "distance": 10, "infra": "VE"},
        {"gare": "B", "distance": 20, "infra": "F"},
    ])


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
            {"start": datetime(2025, 1, 1, 8, 0), "end": datetime(2025, 1, 1, 8, 10),
             "origine": "A", "terminus": "X", "mission": "A → B", "is_mission_start": True},
            {"start": datetime(2025, 1, 1, 8, 10), "end": datetime(2025, 1, 1, 8, 20),
             "origine": "X", "terminus": "B", "mission": "A → B", "is_mission_start": False},
        ]
    }


@pytest.mark.parametrize("invalid", [True, False, -1, 1.0, "1"])
def test_evaluer_rejette_un_plafond_de_rames_non_entier_ou_negatif(invalid):
    with pytest.raises(ValueError):
        evaluer_robustesse(
            _chronologie(), _infrastructure(), [_mission()], max_reserve_rames=invalid
        )


@pytest.mark.parametrize("invalid", [True, False, -1, 1.0, "1"])
def test_fingerprint_rejette_un_plafond_de_rames_non_entier_ou_negatif(invalid):
    with pytest.raises(ValueError):
        fingerprint_robustesse(
            _chronologie(), _infrastructure(), [_mission()], max_reserve_rames=invalid
        )


def test_fingerprint_inclut_le_plafond_de_rames_physiques():
    args = (_chronologie(), _infrastructure(), [_mission()])

    assert fingerprint_robustesse(*args) != fingerprint_robustesse(
        *args, max_reserve_rames=0
    )


def _chronologie_avec_rotation_disponible():
    return {
        "T1": [
            {"start": datetime(2025, 1, 1, 8, 0), "end": datetime(2025, 1, 1, 8, 10),
             "origine": "A", "terminus": "X", "mission": "A → B", "is_mission_start": True},
            {"start": datetime(2025, 1, 1, 8, 10), "end": datetime(2025, 1, 1, 8, 20),
             "origine": "X", "terminus": "B", "mission": "A → B", "is_mission_start": False},
            {"start": datetime(2025, 1, 1, 8, 25), "end": datetime(2025, 1, 1, 8, 35),
             "origine": "B", "terminus": "X", "mission": "B → A", "is_mission_start": True},
            {"start": datetime(2025, 1, 1, 8, 35), "end": datetime(2025, 1, 1, 8, 45),
             "origine": "X", "terminus": "A", "mission": "B → A", "is_mission_start": False},
            {"start": datetime(2025, 1, 1, 8, 50), "end": datetime(2025, 1, 1, 9, 0),
             "origine": "A", "terminus": "X", "mission": "A → B", "is_mission_start": True},
            {"start": datetime(2025, 1, 1, 9, 0), "end": datetime(2025, 1, 1, 9, 10),
             "origine": "X", "terminus": "B", "mission": "A → B", "is_mission_start": False},
            {"start": datetime(2025, 1, 1, 9, 40), "end": datetime(2025, 1, 1, 9, 50),
             "origine": "B", "terminus": "X", "mission": "B → A", "is_mission_start": True},
            {"start": datetime(2025, 1, 1, 9, 50), "end": datetime(2025, 1, 1, 10, 0),
             "origine": "X", "terminus": "A", "mission": "B → A", "is_mission_start": False},
        ],
        "T2": [
            {"start": datetime(2025, 1, 1, 7, 0), "end": datetime(2025, 1, 1, 7, 10),
             "origine": "A", "terminus": "X", "mission": "A → B", "is_mission_start": True},
            {"start": datetime(2025, 1, 1, 7, 10), "end": datetime(2025, 1, 1, 7, 20),
             "origine": "X", "terminus": "B", "mission": "A → B", "is_mission_start": False},
            {"start": datetime(2025, 1, 1, 9, 15), "end": datetime(2025, 1, 1, 9, 25),
             "origine": "B", "terminus": "X", "mission": "B → A", "is_mission_start": True},
            {"start": datetime(2025, 1, 1, 9, 25), "end": datetime(2025, 1, 1, 9, 35),
             "origine": "X", "terminus": "A", "mission": "B → A", "is_mission_start": False},
        ],
    }


@pytest.mark.parametrize("plafond", [0, 1, 2])
def test_rotation_physique_existante_est_preferee_sans_rame_de_reserve(plafond):
    resultat = evaluer_robustesse(
        _chronologie_avec_rotation_disponible(), _infrastructure(), [_mission()],
        reserve_gares=("A", "B"), max_reserve_rames=plafond,
    )

    scenarios = [scenario["avec_reserve"] for scenario in resultat["scenarios"]]
    scenario = next(
        value for value in scenarios
        if value is not None and value["redeploiements_rotation"]
    )
    replacements = scenario["redeploiements_rotation"]
    target = next(item for item in replacements if item["service_id"] == "str:T1#2")

    assert target["train_id"] == "T2"
    assert target["service_label"] == "08:25 B → A"
    assert target["depart_planifie"] == datetime(2025, 1, 1, 8, 25)
    assert scenario["reserve_rames_utilisees"] == 0
    assert scenario["reserve_activations"] == []
    assert scenario["infra_violations"] == []
    assert scenario["physical_violations"] == []
    target_row = next(row for row in scenario["arrivals"]
                      if row["service_id"] == "str:T1#2")
    assert target_row["train_reel_id"] == "T2"
    assert target_row["depart_reel"] == target_row["depart_planifie"]
    t2_next = next(row for row in scenario["arrivals"] if row["service_id"] == "str:T2#2")
    assert t2_next["depart_reel"] == t2_next["depart_planifie"]


def test_une_reserve_est_reutilisee_sur_une_rotation_sans_compter_deux_fois():
    chronologie = {"T1": _chronologie_avec_rotation_disponible()["T1"]}

    resultat = evaluer_robustesse(
        chronologie, _infrastructure(), [_mission()], reserve_gares=("A", "B"),
        max_reserve_rames=1,
    )
    scenarios = [scenario["avec_reserve"] for scenario in resultat["scenarios"]]
    scenario = next(
        value for value in scenarios
        if value is not None and len(value["reserve_activations"]) >= 2
    )

    assert scenario["reserve_rames_utilisees"] == 1
    assert len({item["reserve_train_id"] for item in scenario["reserve_activations"]}) == 1
    assert [item["service_id"] for item in scenario["reserve_activations"]] == [
        "str:T1#2", "str:T1#3"
    ]
    assert all(item["service_label"].startswith(("08:25", "08:50"))
               for item in scenario["reserve_activations"])


def test_plafond_zero_refuse_une_nouvelle_identite_physique():
    chronologie = {"T1": _chronologie_avec_rotation_disponible()["T1"]}

    scenario = evaluer_robustesse(
        chronologie, _infrastructure(), [_mission()], reserve_gares=("B",),
        max_reserve_rames=0,
    )["scenarios"][0]["avec_reserve"]

    assert scenario["reserve_rames_utilisees"] == 0
    assert scenario["reserve_activations"] == []
    refusal = next(item for item in scenario["reserve_refus"]
                   if item["service_id"] == "str:T1#2")
    assert refusal["service_label"] == "08:25 B → A"
    assert any("Plafond de 0" in reason for reason in refusal["raisons"])


def test_refus_rotation_documente_position_et_retournement_sans_fausse_incompatibilite():
    chronologie = _chronologie_avec_rotation_disponible()
    chronologie["T2"] = [
        {"start": datetime(2025, 1, 1, 7, 0), "end": datetime(2025, 1, 1, 7, 10),
         "origine": "B", "terminus": "X", "mission": "B → A", "is_mission_start": True},
        {"start": datetime(2025, 1, 1, 7, 10), "end": datetime(2025, 1, 1, 7, 20),
         "origine": "X", "terminus": "A", "mission": "B → A", "is_mission_start": False},
        {"start": datetime(2025, 1, 1, 9, 15), "end": datetime(2025, 1, 1, 9, 25),
         "origine": "A", "terminus": "X", "mission": "A → B", "is_mission_start": True},
        {"start": datetime(2025, 1, 1, 9, 25), "end": datetime(2025, 1, 1, 9, 35),
         "origine": "X", "terminus": "B", "mission": "A → B", "is_mission_start": False},
    ]

    results = evaluer_robustesse(
        chronologie, _infrastructure(), [_mission()], max_reserve_rames=0,
    )["scenarios"]
    refusal = next(
        item
        for scenario in results
        for item in (scenario["avec_reserve"] or {}).get("rotation_refus", [])
        if item["service_id"] == "str:T1#2" and item["train_id"] == "T2"
    )

    assert refusal["service_label"] == "08:25 B → A"
    assert refusal["localisation"] == "A"
    assert refusal["retour_minimum_min"] == 5
    assert refusal["disponible_a"] == datetime(2025, 1, 1, 7, 25)
    assert refusal["marge_disponibilite_min"] == 60
    assert any("localisée à 'A'" in reason for reason in refusal["raisons"])
    assert not any("Matériel incompatible" in reason for reason in refusal["raisons"])


def test_rotation_batterie_non_autonome_est_refusee():
    mission = dict(_mission(), type_materiel="batterie")
    battery = get_default_energy_params()
    battery.update({
        "capacite_batterie_kwh": 1,
        "soc_min_pct": 20,
        "soc_max_pct": 95,
        "facteur_charge_C": 1,
        "facteur_aux_kwh_h": 0,
    })

    resultats = evaluer_robustesse(
        _chronologie_avec_rotation_disponible(), _infrastructure(), [mission],
        energy_params={"batterie": battery}, reserve_gares=("B",),
        max_reserve_rames=0,
    )["scenarios"]
    refusals = [
        refusal
        for scenario in resultats
        for refusal in (scenario["avec_reserve"] or {}).get("rotation_refus", [])
        if refusal["service_id"] == "str:T1#2" and refusal["train_id"] == "T2"
    ]

    assert refusals
    assert all(any("Batterie" in reason or "autonomie" in reason.lower()
                   for reason in refusal["raisons"]) for refusal in refusals)
    assert all(not (scenario["avec_reserve"] or {}).get("redeploiements_rotation")
               for scenario in resultats)


def test_rotation_libre_moins_de_cinq_minutes_est_refusee():
    chronologie = _chronologie_avec_rotation_disponible()
    chronologie["T2"][0]["start"] = datetime(2025, 1, 1, 8, 0)
    chronologie["T2"][0]["end"] = datetime(2025, 1, 1, 8, 10)
    chronologie["T2"][1]["start"] = datetime(2025, 1, 1, 8, 10)
    chronologie["T2"][1]["end"] = datetime(2025, 1, 1, 8, 20)
    gares = _infrastructure()
    gares.loc[gares["gare"] == "A", "infra"] = "D"
    gares.loc[gares["gare"] == "X", "infra"] = "F"

    scenarios = evaluer_robustesse(
        chronologie, gares, [_mission()], reserve_gares=("B",),
        max_reserve_rames=0,
    )["scenarios"]
    refusal = next(
        refusal
        for scenario in scenarios
        for refusal in (scenario["avec_reserve"] or {}).get("rotation_refus", [])
        if refusal["service_id"] == "str:T1#2" and refusal["train_id"] == "T2"
    )

    assert refusal["disponible_a"] == datetime(2025, 1, 1, 8, 25)
    assert refusal["marge_disponibilite_min"] == 0
    assert any("retournement minimum de 5 min" in reason
               for reason in refusal["raisons"])


def test_deux_identites_de_reserve_peuvent_partager_un_lieu_initial():
    from copy import deepcopy
    from datetime import timedelta
    from robustness_logic import _compiler_services, _rejouer
    first = deepcopy(_chronologie()["T1"])
    second = deepcopy(first)
    for row in second:
        row['start'] += timedelta(minutes=30)
        row['end'] += timedelta(minutes=30)
    services, gares, _ = _compiler_services({1: first, 2: second}, _infrastructure(), [_mission()])
    assignments = {services[0]['service_id']: ('A', 1), services[1]['service_id']: ('A', 2)}
    replay = _rejouer(services, gares, reserve_map=assignments, reserve_stations=('A',))
    assert set(replay['chronologie']) == {'Réserve A #1', 'Réserve A #2'}
    assert not replay['physical_violations']
    assert not replay['infra_violations']
    assert all(row['depart_reel'] == row['depart_planifie'] for row in replay['services'])


def test_desactivation_renforts_invalide_empreinte_et_empeche_redéploiements():
    args = (_chronologie_avec_rotation_disponible(), _infrastructure(), [_mission()])
    assert fingerprint_robustesse(*args, activer_renforts=False) != fingerprint_robustesse(*args)
    result = evaluer_robustesse(*args, reserve_gares=('A', 'B'), activer_renforts=False)
    assert all(scenario['avec_reserve'] is None for scenario in result['scenarios'])


def test_rotation_diesel_ne_masque_pas_une_alerte_batterie_independante():
    gares = pd.concat([_infrastructure(), pd.DataFrame([
        {'gare': 'C', 'distance': 30, 'infra': 'D'},
        {'gare': 'D', 'distance': 50, 'infra': 'F'},
    ])], ignore_index=True)
    chronology = _chronologie_avec_rotation_disponible()
    chronology['BATT'] = [dict(
        start=datetime(2025, 1, 1, 6), end=datetime(2025, 1, 1, 6, 20),
        origine='C', terminus='D', mission='C → D', is_mission_start=True,
    )]
    battery = get_default_energy_params()
    battery['capacite_batterie_kwh'] = 1
    result = evaluer_robustesse(
        chronology, gares, [_mission(), dict(_mission(), origine='C', terminus='D', type_materiel='batterie')],
        energy_params={'diesel': get_default_energy_params(), 'batterie': battery},
        max_reserve_rames=0,
    )
    for scenario in result['scenarios']:
        reinforcement = scenario['avec_reserve']
        assert not reinforcement['redeploiements_rotation']
        assert reinforcement['reserve_rames_utilisees'] == 0
        assert reinforcement['battery']['baseline']['par_train']['BATT']['autonomie_insuffisante']
        assert reinforcement['battery']['scenario']['par_train']['BATT']['autonomie_insuffisante']


def test_localisation_et_disponibilite_apres_une_suppression_sans_teleportation():
    from test_robustness import _chronologie_croisement
    from robustness_logic import _compiler_services, _enumerer_points, _rejouer, _disponibilite_rame_rotation
    chronology = _chronologie_croisement()
    gares = _infrastructure()
    gares['infra'] = 'F'
    services, gares, infra = _compiler_services(chronology, gares, [_mission()])
    replay = _rejouer(services, gares, _enumerer_points(services, infra)[0], 5)
    assert replay['suppressions_count'] == 1
    departure = services[0]['planned_start'].replace(hour=9)
    available = _disponibilite_rame_rotation(replay, services, 'T2', departure)
    assert available['localisation'] == 'B'
    assert available['disponible_a'] is not None
    assert available['disponible_a'] < departure
    assert available['mission_precedente_terminee']
    from datetime import timedelta
    cancelled = next(service for service in services if service['train_id'] == 'T2')
    assert available['disponible_a'] >= cancelled['planned_start'] + timedelta(minutes=5)
    too_early = _disponibilite_rame_rotation(
        replay, services, 'T2', cancelled['planned_start'] + timedelta(minutes=1),
    )
    assert too_early['marge_disponibilite_min'] < 0
