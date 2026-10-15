from copy import deepcopy
from datetime import datetime

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import plotting

from core_logic import generer_tous_trajets_optimises
from robustness_logic import evaluer_robustesse
from scenarios_manager import load_scenario


SCENARIO_PATH = (
    Path(__file__).parent
    / "scenarios"
    / "Bretagne"
    / "Auray-Quiberon"
    / "auray_quiberon_freq_heure_omnibus.json"
)


class _EnregistreurIdentifiants:
    def __init__(self):
        self.identifiants = []

    def get(self, identifiant, default=None):
        self.identifiants.append(identifiant)
        return default


@pytest.fixture(scope="module")
def scenario_auray_quiberon_avec_reserves():
    scenario = load_scenario(SCENARIO_PATH)
    gares = pd.DataFrame(scenario["infrastructure"]["gares"])
    heure_debut = datetime.strptime(
        scenario["service"]["heure_debut"], "%H:%M"
    ).time()
    heure_fin = datetime.strptime(
        scenario["service"]["heure_fin"], "%H:%M"
    ).time()
    chronologie, _, _ = generer_tous_trajets_optimises(
        scenario["missions"],
        gares,
        heure_debut,
        heure_fin,
        allow_sharing=True,
        search_strategy="simple",
    )
    gares_reserve = list(dict.fromkeys(
        gare
        for mission in scenario["missions"]
        for gare in (mission.get("origine"), mission.get("terminus"))
        if gare
    ))
    resultats = evaluer_robustesse(
        chronologie, gares, scenario["missions"],
        energy_params=scenario.get("energy_params"), reserve_gares=gares_reserve,
    )
    return scenario, gares, heure_debut, resultats


def test_auray_audit_couvre_toute_la_journee_et_signale_les_vrais_ex_aequo(
    scenario_auray_quiberon_avec_reserves,
):
    from collections import Counter
    _, _, _, result = scenario_auray_quiberon_avec_reserves
    assert result['candidates_count'] == 66
    for scenario, score, ties, crossing_max in zip(result['scenarios'], [10, 25, 75], [16, 15, 14], [5, 15, 45]):
        candidates = scenario['candidates_evaluated']
        assert len(candidates) == 66
        assert Counter(row['kind'] for row in candidates) == {'depart': 33, 'croisement': 33}
        assert min(row['time'].strftime('%H:%M') for row in candidates) == '06:00'
        assert max(row['time'].strftime('%H:%M') for row in candidates) == '22:22'
        assert {row['time'].hour for row in candidates} == set(range(6, 23))
        assert {row['service_id'] for row in candidates if row['kind'] == 'depart'} == {
            row['service_id'] for row in scenario['sans_reserve']['arrivals']
        }
        assert scenario['sans_reserve']['retard_cumule_min'] == score
        assert max(row['score_min'] for row in candidates if row['kind'] == 'croisement') == crossing_max
        assert scenario['point']['time'].strftime('%H:%M') == '07:00'
        assert scenario['tie_count'] == ties == len(scenario['points_ex_aequo'])
        assert all(row['score_min'] == score for row in scenario['points_ex_aequo'])
        assert scenario['sans_reserve']['suppressions_count'] == 0
        assert scenario['sans_reserve']['battery']['baseline']['par_train']['1']['autonomie_insuffisante']


def test_graphique_robustesse_accepte_ids_int_et_str_sans_les_coercer(
    scenario_auray_quiberon_avec_reserves,
):
    _, gares, heure_debut, resultats = scenario_auray_quiberon_avec_reserves
    chronologie = dict(resultats["scenarios"][0]["avec_reserve"]["chronologie"])
    identifiant_reserve = "Réserve Quiberon #1"
    id_train = next(identifiant for identifiant in chronologie if type(identifiant) is int)
    # Le scénario réel ne mobilise pas de rame de réserve ; réutilise un segment
    # réel pour exercer aussi le chemin de tracé d'un ID réserve.
    chronologie[identifiant_reserve] = [deepcopy(chronologie[id_train][0])]
    assert any(type(identifiant) is int for identifiant in chronologie)
    assert all(type(identifiant) in (int, str) for identifiant in chronologie)

    enregistreur = _EnregistreurIdentifiants()
    figure = plotting.creer_graphique_horaire(
        chronologie,
        gares,
        heure_debut,
        {"decalage_heure": 0, "duree_fenetre": 16.5},
        mode_calcul="Calcul Energie",
        missions_par_train=enregistreur,
        all_energy_params={},
    )

    identifiants_attendus = sorted(
        (identifiant for identifiant in chronologie if type(identifiant) is int)
    ) + sorted(
        (identifiant for identifiant in chronologie if type(identifiant) is str)
    )
    assert enregistreur.identifiants == identifiants_attendus
    assert any(type(identifiant) is int for identifiant in enregistreur.identifiants)
    assert enregistreur.identifiants[-1] == identifiant_reserve
    figure.canvas.draw()
    plt.close(figure)


def test_figures_reelles_avec_reserve_pour_les_retards_5_10_20(
    scenario_auray_quiberon_avec_reserves,
):
    _, gares, heure_debut, resultats = scenario_auray_quiberon_avec_reserves

    assert [scenario["delay_min"] for scenario in resultats["scenarios"]] == [5, 10, 20]
    assert all(scenario["avec_reserve"] is not None for scenario in resultats["scenarios"])

    figures = []
    try:
        for scenario in resultats["scenarios"]:
            figure = plotting.creer_graphique_horaire(
                scenario["avec_reserve"]["chronologie"],
                gares,
                heure_debut,
                {"decalage_heure": 0, "duree_fenetre": 16.5},
            )
            figures.append(figure)
            figure.canvas.draw()
            assert len(figure.axes) == 2
            assert figure.axes[1].lines
    finally:
        for figure in figures:
            plt.close(figure)


def test_figure_avec_reserve_reellement_deployee_et_identifiants_mixtes(tmp_path):
    from test_robustness_fleet import _chronologie_avec_rotation_disponible, _infrastructure, _mission
    gares = _infrastructure()
    gares.loc[gares['gare'] == 'A', 'infra'] = 'D'
    result = evaluer_robustesse(
        {1: _chronologie_avec_rotation_disponible()['T1']}, gares, [_mission()],
        reserve_gares=('B',), max_reserve_rames=1,
    )
    for row in result['scenarios']:
        scenario = row['avec_reserve']
        assert scenario['reserve_rames_utilisees'] == 1
        assert not scenario['infra_violations']
        assert not scenario['physical_violations']
        assert {type(key) for key in scenario['chronologie']} == {int, str}
        figure = plotting.creer_graphique_horaire(
            scenario['chronologie'], gares, datetime(2025, 1, 1, 7).time(),
            {'decalage_heure': 0, 'duree_fenetre': 5},
        )
        try:
            figure.canvas.draw()
            image = tmp_path / f"reserve-{row['delay_min']}.png"
            figure.savefig(image)
            assert image.stat().st_size > 1000
        finally:
            plt.close(figure)
