from copy import deepcopy
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import optimisation_logic
import robustness_logic


APP_PATH = Path(__file__).with_name("app.py")


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
        "frequence": 1.0,
        "temps_trajet": 20,
        "temps_retournement_A": 5,
        "temps_retournement_B": 5,
        "reference_minutes": "0",
        "type_materiel": "diesel",
    }


def _chronologie():
    jour = datetime(2026, 1, 1)
    return {
        "T1": [
            {
                "start": jour.replace(hour=8),
                "end": jour.replace(hour=8, minute=10),
                "origine": "A",
                "terminus": "X",
                "mission": "A → B",
                "is_mission_start": True,
            },
            {
                "start": jour.replace(hour=8, minute=10),
                "end": jour.replace(hour=8, minute=20),
                "origine": "X",
                "terminus": "B",
                "mission": "A → B",
                "is_mission_start": False,
            },
        ]
    }


def _app_test():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.session_state["gares"] = _infrastructure()
    app.session_state["missions"] = [_mission()]
    app.session_state["chronologie_calculee"] = _chronologie()
    app.session_state["mode_calcul_selector"] = "Standard"
    app.session_state["mode_generation"] = "Rotation optimisée"
    app.session_state["heure_debut_service"] = time(6, 0)
    app.session_state["heure_fin_service"] = time(22, 0)
    app.session_state["run_calculation"] = False
    app.session_state["warnings_calcul"] = {}
    app.session_state["stats_homogeneite"] = {}
    return app


def _interdire_optimisation(monkeypatch):
    appels = []

    def interdite(*args, **kwargs):
        appels.append((args, kwargs))
        raise AssertionError("Le calcul de robustesse ne doit pas relancer l'optimisation")

    monkeypatch.setattr(optimisation_logic, "optimiser_graphique_horaire", interdite)
    return appels


def _compter_calcul_robustesse(monkeypatch):
    appels = []
    calculer = robustness_logic.evaluer_robustesse

    def compte(*args, **kwargs):
        appels.append((args, kwargs))
        return calculer(*args, **kwargs)

    monkeypatch.setattr(robustness_logic, "evaluer_robustesse", compte)
    return appels


def _cliquer_calcul_robustesse(app):
    boutons = [button for button in app.button if button.key == "btn_calc_robustesse"]
    assert len(boutons) == 1
    boutons[0].click().run(timeout=30)


def test_robustesse_est_dans_une_rubrique_dediee_apres_le_calcul():
    app = _app_test().run(timeout=30)

    assert not app.exception
    libelles = [expander.label for expander in app.expander]
    assert "6. Calcul et Affichage" in libelles
    assert "7. Évaluation de la robustesse" in libelles
    assert libelles.index("7. Évaluation de la robustesse") > libelles.index(
        "6. Calcul et Affichage"
    )


def test_graphique_perturbe_surligne_le_service_incident_et_centre_sa_fenetre(monkeypatch):
    import matplotlib.dates as mdates
    import plotting

    jour = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    scenarios = []
    for delay, heure in ((5, 12), (10, 15), (20, 18)):
        incident_time = jour.replace(hour=heure)
        service_id = "service-incident"
        scenario_chronologie = {
            "T1": [
                {
                    "start": incident_time + timedelta(minutes=delay),
                    "end": incident_time + timedelta(minutes=delay + 20),
                    "origine": "A",
                    "terminus": "X",
                    "mission": "A → B",
                    "is_mission_start": True,
                    "service_id": service_id,
                }
            ],
            "T2": [
                {
                    "start": incident_time + timedelta(minutes=30),
                    "end": incident_time + timedelta(minutes=50),
                    "origine": "X",
                    "terminus": "B",
                    "mission": "A → B",
                    "is_mission_start": True,
                    "service_id": "service-contexte",
                }
            ],
        }
        chronologie_reserve = {
            "Réserve B #1": [deepcopy(scenario_chronologie["T1"][0])],
            "T2": [deepcopy(scenario_chronologie["T2"][0])],
        }
        scenarios.append(
            {
                "delay_min": delay,
                "point": {
                    "train_id": "T1",
                    "service_id": service_id,
                    "gare": "A",
                    "time": incident_time,
                    "kind": "depart",
                },
                "sans_reserve": {
                    "chronologie": scenario_chronologie,
                    "retard_cumule_min": delay,
                    "retour_normal_min": None,
                    "battery": {},
                },
                "avec_reserve": None,
            }
        )
        scenarios[-1]["avec_reserve"] = {
            "chronologie": chronologie_reserve,
            "retard_cumule_min": delay,
            "retour_normal_min": None,
            "battery": {},
        }
    resultats = {"scenarios": scenarios, "candidates_count": 6}

    figures = []
    render_original = plotting.creer_graphique_horaire

    def render_capture(*args, **kwargs):
        figure = render_original(*args, **kwargs)
        figures.append(figure)
        return figure

    monkeypatch.setattr(plotting, "creer_graphique_horaire", render_capture)
    monkeypatch.setattr(robustness_logic, "evaluer_robustesse", lambda *a, **k: resultats)
    app = _app_test()
    app.session_state["robustesse_reserve_active"] = True
    app.session_state["robustesse_reserve_gares"] = ["B"]
    app.run(timeout=30)
    assert not app.exception
    figures.clear()

    _cliquer_calcul_robustesse(app)

    assert not app.exception
    assert len(figures) == 7  # le graphe nominal et les deux variantes par amplitude
    assert not [line for line in figures[0].axes[1].lines if line.get_color() == "#d62728"]
    debut_nominal = mdates.num2date(figures[0].axes[1].get_xlim()[0]).replace(tzinfo=None)
    date_zero = datetime.combine(jour.date(), time())
    assert (debut_nominal - date_zero).total_seconds() / 3600 == 6
    for index, heure_attendue in enumerate((9.5, 12.5, 15.5)):
        for figure in figures[1 + index * 2 : 3 + index * 2]:
            lignes_incident = [
                line for line in figure.axes[1].lines if line.get_color() == "#d62728"
            ]
            assert len(lignes_incident) == 1
            debut_fenetre, fin_fenetre = figure.axes[1].get_xlim()
            debut_fenetre = mdates.num2date(debut_fenetre).replace(tzinfo=None)
            decalage_heures = (debut_fenetre - date_zero).total_seconds() / 3600
            assert decalage_heures == heure_attendue
            assert abs((fin_fenetre - mdates.date2num(debut_fenetre)) * 24 - 5) < 1e-8


def test_service_supprime_est_trace_en_rouge_pointille_sans_circulation_reelle(monkeypatch):
    import plotting

    service_id = "str:T1#1"
    resultat = {
        "chronologie": {
            "T2": [
                {
                    "start": datetime(2026, 1, 1, 8, 30),
                    "end": datetime(2026, 1, 1, 8, 40),
                    "origine": "X",
                    "terminus": "B",
                    "mission": "A → B",
                    "is_mission_start": True,
                    "service_id": "str:T2#1",
                }
            ]
        },
        "suppressions_count": 1,
        "suppressions": [{"service_id": service_id, "motif": "Conflit"}],
        "retard_cumule_min": 0,
        "retour_normal_min": None,
        "battery": {},
    }
    resultats = {
        "scenarios": [
            {
                "delay_min": 5,
                "point": {
                    "train_id": "T1",
                    "service_id": service_id,
                    "gare": "A",
                    "time": datetime(2026, 1, 1, 8),
                    "kind": "depart",
                },
                "sans_reserve": resultat,
                "avec_reserve": None,
            }
        ],
        "candidates_count": 1,
    }
    figures = []
    render_original = plotting.creer_graphique_horaire

    def render_capture(*args, **kwargs):
        figure = render_original(*args, **kwargs)
        figures.append(figure)
        return figure

    monkeypatch.setattr(plotting, "creer_graphique_horaire", render_capture)
    monkeypatch.setattr(robustness_logic, "evaluer_robustesse", lambda *a, **k: resultats)
    app = _app_test().run(timeout=30)
    figures.clear()

    _cliquer_calcul_robustesse(app)

    assert not app.exception
    assert len(figures) == 2
    assert not [line for line in figures[0].axes[1].lines if line.get_color() == "#d62728"]
    graphique_incident = figures[1].axes[1]
    lignes_incident = [
        line for line in graphique_incident.lines if line.get_color() == "#d62728"
    ]
    assert len(lignes_incident) == 2
    assert all(line.get_linestyle() == "--" for line in lignes_incident)
    assert {tuple(line.get_xdata()) for line in lignes_incident} == {
        (datetime(2026, 1, 1, 8), datetime(2026, 1, 1, 8, 10)),
        (datetime(2026, 1, 1, 8, 10), datetime(2026, 1, 1, 8, 20)),
    }
    assert "Service supprimé — tracé prévu" in [
        texte.get_text() for texte in graphique_incident.get_legend().get_texts()
    ]


def test_suppressions_penalites_et_autonomie_insuffisante_sont_visibles(monkeypatch):
    jour = datetime.today().replace(hour=12, minute=0, second=0, microsecond=0)
    chronologie = deepcopy(_chronologie())
    for trajet in chronologie["T1"]:
        trajet["service_id"] = "service-T1"
    resultat = {
        "chronologie": chronologie,
        "retard_trains_min": 8,
        "penalite_suppressions_min": 14,
        "suppressions_count": 1,
        "suppressions": [
            {
                "service_id": "service-T2",
                "motif": "Autonomie insuffisante",
                "attente_min": 4,
                "penalite_min": 14,
            }
        ],
        "retard_cumule_min": 22,
        "retour_normal_min": None,
        "battery": {
            "autonomie_insuffisante": True,
            "alertes_autonomie": [
                "Train T1: SoC minimal 15.00% au seuil soc_min_pct 20.00%"
            ],
            "scenario": {
                "par_train": {
                    "T1": {"min_soc_pct": 15, "seuil_soc_min_pct": 20}
                }
            },
            "baseline": {"par_train": {"T1": {"min_soc_pct": 30}}},
        },
    }
    resultats = {
        "scenarios": [
            {
                "delay_min": 5,
                "point": {
                    "train_id": "T1",
                    "service_id": "service-T1",
                    "gare": "A",
                    "time": jour,
                    "kind": "depart",
                },
                "sans_reserve": resultat,
                "avec_reserve": None,
            }
        ],
        "candidates_count": 6,
    }
    monkeypatch.setattr(robustness_logic, "evaluer_robustesse", lambda *a, **k: resultats)
    app = _app_test().run(timeout=30)
    _cliquer_calcul_robustesse(app)

    assert not app.exception
    assert any(
        "Score = retards des trains" in item.value
        and "services opérés" in item.value
        and "attente du prochain service assuré" in item.value
        and "10 min" in item.value
        and "Points candidats évalués : 6" in item.value
        and "tous les départs et points de croisement" in item.value
        for item in app.caption
    )
    assert any(
        "Autonomie insuffisante" in item.value
        and "T1" in item.value
        and "SoC minimal 15.0 %" in item.value
        and "seuil 20.0 %" in item.value
        and "vs nominal -15.0 points" in item.value
        for item in app.error
    )
    tables = [item.value for item in app.dataframe]
    assert any(
        isinstance(table, pd.DataFrame)
        and "Retards des trains sans réserve" in table.columns
        and table.iloc[0]["Retards des trains sans réserve"] == "8.0 min"
        and table.iloc[0]["Pénalités suppressions sans réserve"] == "14.0 min"
        for table in tables
    )
    assert any(
        isinstance(table, pd.DataFrame)
        and "Pénalité de suppression" in table.columns
        and table.iloc[0]["Service"] == "service-T2"
        and table.iloc[0]["Pénalité de suppression"] == "14.0 min"
        for table in tables
    )


def test_recharge_terminus_affiche_prolongation_et_impact_sur_retour_normal(monkeypatch):
    chronologie = deepcopy(_chronologie())
    for trajet in chronologie["T1"]:
        trajet["service_id"] = "service-T1"
    motif = "Sécurité de charge prioritaire sur le délai de récupération"
    resultat = {
        "chronologie": chronologie,
        "retard_cumule_min": 0,
        "retour_normal_min": 37,
        "retour_normal_sans_recharge_min": 25,
        "recharges_prolongees": [
            {
                "service_id": "service-T1",
                "train_id": "T1",
                "gare": "B",
                "retournement_initial_min": 5,
                "retournement_retenu_min": 17,
                "prolongation_min": 12,
                "motif": motif,
            }
        ],
        "battery": {},
    }
    resultats = {
        "scenarios": [{
            "delay_min": 5,
            "point": {
                "train_id": "T1",
                "service_id": "service-T1",
                "gare": "A",
                "time": datetime(2026, 1, 1, 8),
                "kind": "depart",
            },
            "sans_reserve": resultat,
            "avec_reserve": None,
        }],
        "candidates_count": 1,
    }
    monkeypatch.setattr(robustness_logic, "evaluer_robustesse", lambda *a, **k: resultats)
    app = _app_test().run(timeout=30)
    _cliquer_calcul_robustesse(app)

    assert not app.exception
    assert not any("Autonomie insuffisante" in item.value for item in app.error)
    tableaux = [item.value for item in app.dataframe if isinstance(item.value, pd.DataFrame)]
    tableau = next(table for table in tableaux if "Prolongation pour recharge" in table.columns)
    ligne = tableau.iloc[0]
    assert ligne["Service"] == "service-T1"
    assert ligne["Train"] == "T1"
    assert ligne["Gare"] == "B"
    assert ligne["Retournement initial"] == "5.0 min"
    assert ligne["Retournement retenu"] == "17.0 min"
    assert ligne["Prolongation pour recharge"] == "12.0 min"
    assert ligne["Motif"] == motif
    assert any(
        "sécurité de charge est prioritaire sur le délai de récupération" in item.value
        for item in app.caption
    )
    textes = [item.value for item in app.get("text")]
    textes.extend(item.value for item in app.markdown)
    assert any("Retour normal avec prolongation de recharge : 37.0 min" in texte for texte in textes)
    assert any("Retour normal sans recharge : 25.0 min" in texte for texte in textes)


def test_bouton_robustesse_lance_son_calcul_sans_reoptimiser(monkeypatch):
    appels_optimisation = _interdire_optimisation(monkeypatch)
    appels_robustesse = _compter_calcul_robustesse(monkeypatch)
    app = _app_test().run(timeout=30)

    assert not app.exception
    assert appels_robustesse == []
    _cliquer_calcul_robustesse(app)

    assert not app.exception
    resultat = app.session_state["robustesse_resultats"]
    assert [scenario["delay_min"] for scenario in resultat["scenarios"]] == [5, 10, 20]
    assert len(appels_robustesse) == 1
    assert appels_optimisation == []


def test_chronologie_modifiee_invalide_les_resultats_sans_reoptimiser(monkeypatch):
    appels_optimisation = _interdire_optimisation(monkeypatch)
    appels_robustesse = _compter_calcul_robustesse(monkeypatch)
    app = _app_test().run(timeout=30)
    assert not app.exception
    assert appels_robustesse == []

    _cliquer_calcul_robustesse(app)
    assert app.session_state["robustesse_resultats"] is not None
    assert len(appels_robustesse) == 1

    chronologie = deepcopy(app.session_state["chronologie_calculee"])
    chronologie["T1"][0]["end"] += timedelta(minutes=1)
    app.session_state["chronologie_calculee"] = chronologie
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["robustesse_resultats"] is None
    assert any("invalid" in message.value.lower() for message in app.info)
    assert len(appels_robustesse) == 1
    assert appels_optimisation == []


def test_reserve_active_propose_les_terminus_concrets(monkeypatch):
    appels_optimisation = _interdire_optimisation(monkeypatch)
    appels_robustesse = _compter_calcul_robustesse(monkeypatch)
    app = _app_test()
    app.session_state["robustesse_reserve_active"] = True
    app.session_state["robustesse_reserve_gares"] = ["B"]
    app.run(timeout=30)

    assert not app.exception
    assert appels_robustesse == []
    choix_gares = [item for item in app.multiselect if item.key == "robustesse_reserve_gares"]
    assert len(choix_gares) == 1
    assert choix_gares[0].options == ["A", "B"]

    _cliquer_calcul_robustesse(app)

    assert not app.exception
    assert all(
        scenario["avec_reserve"] is not None
        for scenario in app.session_state["robustesse_resultats"]["scenarios"]
    )
    assert len(appels_robustesse) == 1
    assert appels_optimisation == []

    app.session_state["robustesse_reserve_gares"] = ["A"]
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["robustesse_resultats"] is None
    assert any("invalid" in message.value.lower() for message in app.info)
    assert len(appels_robustesse) == 1


def test_echec_recalcul_ne_reaffiche_pas_les_anciens_resultats(monkeypatch):
    app = _app_test().run(timeout=30)
    _cliquer_calcul_robustesse(app)
    assert app.session_state["robustesse_resultats"] is not None

    def echouer(*args, **kwargs):
        raise ValueError("Échec de rejeu volontaire")

    monkeypatch.setattr(robustness_logic, "evaluer_robustesse", echouer)
    _cliquer_calcul_robustesse(app)
    assert not app.exception
    assert app.session_state["robustesse_resultats"] is None
    assert any("Échec de rejeu volontaire" in item.value for item in app.error)
    assert not any("Incident +" in item.label for item in app.expander)


def test_autonomie_non_verifiable_affiche_le_motif_sans_chiffres_rassurants(monkeypatch):
    chronologie = deepcopy(_chronologie())
    for trajet in chronologie["T1"]:
        trajet["service_id"] = "service-T1"
    motif = "paramètres absents pour batterie; autonomie non vérifiable avec les données fournies"
    resultat = {
        "chronologie": chronologie,
        "retard_cumule_min": 0,
        "retour_normal_min": None,
        "battery": {
            "autonomie_insuffisante": True,
            "autonomie_non_verifiable": True,
            "alertes_autonomie": [],
            "alertes_autonomie_non_verifiable": [
                {"train_id": 7, "raison": motif},
            ],
            "scenario": {
                "par_train": {
                    "7": {
                        "min_soc_pct": 91.3,
                        "seuil_soc_min_pct": 20,
                        "autonomie_non_verifiable": True,
                    },
                },
            },
        },
    }
    resultats = {
        "scenarios": [{
            "delay_min": 5,
            "point": {
                "train_id": "T1",
                "service_id": "service-T1",
                "gare": "A",
                "time": datetime(2026, 1, 1, 8),
                "kind": "depart",
            },
            "sans_reserve": resultat,
            "avec_reserve": None,
        }],
        "candidates_count": 1,
    }
    monkeypatch.setattr(robustness_logic, "evaluer_robustesse", lambda *a, **k: resultats)
    app = _app_test().run(timeout=30)
    _cliquer_calcul_robustesse(app)

    assert not app.exception
    avertissements = [item.value for item in app.warning]
    assert any("Autonomie non vérifiable" in texte and motif in texte and "7" in texte
               for texte in avertissements)
    assert not any("Autonomie insuffisante" in item.value for item in app.error)
    assert not any("91.3" in texte or "20.0" in texte for texte in avertissements)


def test_hypothese_penalite_suppression_distingue_observee_et_extrapolee(monkeypatch):
    chronologie = deepcopy(_chronologie())
    for trajet in chronologie["T1"]:
        trajet["service_id"] = "service-T1"
    resultat = {
        "chronologie": chronologie,
        "retard_cumule_min": 40,
        "suppressions_count": 2,
        "suppressions": [
            {
                "service_id": "service-extrapole",
                "motif": "Conflit",
                "attente_min": 30,
                "penalite_min": 40,
                "service_suivant_assure": None,
                "hypothese": "Fréquence mission 2 train/h",
            },
            {
                "service_id": "service-observe",
                "motif": "Conflit",
                "attente_min": 12,
                "penalite_min": 22,
                "service_suivant_assure": "str:T3#1",
            },
        ],
        "retour_normal_min": None,
        "battery": {},
    }
    resultats = {
        "scenarios": [{
            "delay_min": 5,
            "point": {
                "train_id": "T1",
                "service_id": "service-T1",
                "gare": "A",
                "time": datetime(2026, 1, 1, 8),
                "kind": "depart",
            },
            "sans_reserve": resultat,
            "avec_reserve": None,
        }],
        "candidates_count": 1,
    }
    monkeypatch.setattr(robustness_logic, "evaluer_robustesse", lambda *a, **k: resultats)
    app = _app_test().run(timeout=30)
    _cliquer_calcul_robustesse(app)

    assert not app.exception
    tableaux = [item.value for item in app.dataframe if isinstance(item.value, pd.DataFrame)]
    tableau = next(
        table for table in tableaux if "Hypothèse de pénalité" in table.columns
    )
    extrapolee = tableau.set_index("Service").loc["service-extrapole"]
    observee = tableau.set_index("Service").loc["service-observe"]
    assert extrapolee["Hypothèse de pénalité"] == "Fréquence mission 2 train/h"
    assert extrapolee["Origine"] == "Extrapolée"
    assert "hors horizon" in extrapolee["Détail de l'hypothèse"]
    assert "non observé" in extrapolee["Détail de l'hypothèse"]
    assert "+ 10 min" in extrapolee["Détail de l'hypothèse"]
    assert observee["Origine"] == "Observée"
    assert "str:T3#1" in observee["Détail de l'hypothèse"]


def test_alerte_autonomie_non_verifiable_legacy_ne_devient_pas_insuffisance(monkeypatch):
    chronologie = deepcopy(_chronologie())
    for trajet in chronologie["T1"]:
        trajet["service_id"] = "service-T1"
    motif = "Train 7: paramètres absents pour batterie; autonomie non vérifiable"
    resultat = {
        "chronologie": chronologie,
        "retard_cumule_min": 0,
        "retour_normal_min": None,
        "battery": {
            "autonomie_insuffisante": True,
            "alertes_autonomie": [motif],
            "scenario": {
                "par_train": {
                    7: {"min_soc_pct": 91.3, "seuil_soc_min_pct": 20},
                },
            },
        },
    }
    resultats = {
        "scenarios": [{
            "delay_min": 5,
            "point": {
                "train_id": "T1",
                "service_id": "service-T1",
                "gare": "A",
                "time": datetime(2026, 1, 1, 8),
                "kind": "depart",
            },
            "sans_reserve": resultat,
            "avec_reserve": None,
        }],
        "candidates_count": 1,
    }
    monkeypatch.setattr(robustness_logic, "evaluer_robustesse", lambda *a, **k: resultats)
    app = _app_test().run(timeout=30)
    _cliquer_calcul_robustesse(app)

    assert not app.exception
    assert any(
        "Autonomie non vérifiable" in item.value and motif in item.value
        for item in app.warning
    )
    assert not any("Autonomie insuffisante" in item.value for item in app.error)
    assert not any("91.3" in item.value or "20.0" in item.value for item in app.warning)
