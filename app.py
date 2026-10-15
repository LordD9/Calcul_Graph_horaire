# -*- coding: utf-8 -*-
"""
app.py
======

Fichier principal de l'application Chronofer (Streamlit).

Ce module gère l'interface utilisateur (UI), la configuration de la session,
et l'orchestration des différents modules fonctionnels :
- `core_logic` : Moteur de simulation et gestion des horaires.
- `optimisation_logic` : Algorithmes d'optimisation (génétique, etc.).
- `energy_logic` : Calculs de consommation énergétique.
- `plotting` : Visualisation graphique (grilles horaires, batteries).

Fonctionnalités principales :
- Définition de l'infrastructure (gares, distances, types de voies).
- Configuration des missions (origines, terminus, fréquences).
- Génération d'horaires (mode automatique optimisé ou manuel).
- Analyse de performance (nombre de rames, régularité).
- Simulation énergétique détaillée (profils de vitesse, consommation, batterie).

Usage :
    Lancer avec : `streamlit run app.py`
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import defaultdict
import time


# Import des modules locaux
from utils import trouver_mission_pour_od, obtenir_temps_trajet_defaut_etape_manuelle, obtenir_temps_retournement_defaut
from core_logic import (
    generer_tous_trajets_optimises,
    preparer_roulement_manuel,
    importer_roulements_fichier,
    analyser_frequences_manuelles,
    generer_exports,
    construire_horaire_mission
)
from plotting import creer_graphique_horaire, creer_graphique_batterie
from energy_logic import (
    get_default_energy_params,
    calculer_energie_flotte,
    fingerprint_energie,
)

from optimisation_logic import (
    OptimizationConfig,
    CrossingOptimization,
    optimiser_graphique_horaire)

import scenarios_manager
from robustness_logic import evaluer_robustesse, fingerprint_robustesse


def _rendre_robustesse(
    chronologie,
    df_gares,
    missions,
    energy_params,
    heure_debut_service,
    params_affichage,
    mode_calcul,
    missions_par_train,
):
    """Affiche et calcule explicitement les scénarios de robustesse du graphe courant."""
    import matplotlib.pyplot as plt

    st.subheader("Analyse de robustesse")
    st.caption(
        "Retard cumulé = retards des services opérés + attente du prochain service assuré "
        "+ 10 min par suppression. "
        "L'analyse rejoue le graphe existant, sans relancer l'optimisation."
    )

    terminus_options = []
    for mission in missions or []:
        for gare in (mission.get("origine"), mission.get("terminus")):
            if gare and gare not in terminus_options:
                terminus_options.append(gare)

    reserve_active = st.checkbox(
        "Activer les rames de réserve",
        key="robustesse_reserve_active",
        help="Les gares sont des lieux disponibles, dans la limite du plafond global de rames supplémentaires.",
    )
    reserve_gares = []
    max_reserve_rames = 0
    if reserve_active:
        max_reserve_rames = st.number_input(
            "Maximum global de rames de réserve supplémentaires",
            min_value=0, value=1, step=1, key="robustesse_max_reserve_rames",
            help="Identités physiques distinctes sur tout le scénario. 0 autorise uniquement les redéploiements de rotation.",
        )
        selection = st.multiselect(
            "Gares terminus de réserve (A, B ou les deux)",
            options=terminus_options,
            key="robustesse_reserve_gares",
            help="Pour plusieurs missions, les terminus concrets de toutes les missions sont proposés.",
        )
        reserve_gares = [gare for gare in terminus_options if gare in selection]
        params_batterie = (energy_params or {}).get("batterie", {})
        soc_max = float(params_batterie.get("soc_max_pct", 95) or 95)
        st.caption(
            "Le plafond est global, pas par gare. Une rame réutilisée ne compte qu'une fois. "
            "Une rame de rotation disponible est privilégiée. La réserve reprend le matériel du service remplacé "
            "et ne peut couvrir qu'un matériel compatible. Pour une rame batterie, le SoC initial "
            f"est le SoC maximal opérationnel configuré ({soc_max:g} %)."
        )

    try:
        empreinte = fingerprint_robustesse(
            chronologie,
            df_gares,
            missions,
            energy_params=energy_params,
            reserve_gares=reserve_gares,
            max_reserve_rames=max_reserve_rames,
            activer_renforts=reserve_active,
        )
    except Exception as exc:
        st.session_state["robustesse_resultats"] = None
        st.session_state["robustesse_fingerprint"] = None
        st.error(f"Empreinte de robustesse impossible : {type(exc).__name__} — {exc}")
        return

    if "robustesse_resultats" not in st.session_state:
        st.session_state["robustesse_resultats"] = None
    if "robustesse_fingerprint" not in st.session_state:
        st.session_state["robustesse_fingerprint"] = None

    resultats_enregistres = st.session_state.get("robustesse_resultats")
    empreinte_enregistree = st.session_state.get("robustesse_fingerprint")
    if resultats_enregistres is not None and empreinte_enregistree != empreinte:
        st.session_state["robustesse_resultats"] = None
        st.session_state["robustesse_fingerprint"] = None
        resultats_enregistres = None
        st.info("Les résultats de robustesse ont été invalidés : les entrées ont changé. Relancez le calcul.")

    if st.button("Calculer la robustesse", key="btn_calc_robustesse", type="primary"):
        st.session_state["robustesse_resultats"] = None
        st.session_state["robustesse_fingerprint"] = None
        resultats_enregistres = None
        try:
            resultats = evaluer_robustesse(
                chronologie,
                df_gares,
                missions,
                energy_params=energy_params,
                reserve_gares=reserve_gares,
                max_reserve_rames=max_reserve_rames,
                activer_renforts=reserve_active,
            )
        except Exception as exc:
            st.error(f"Calcul de robustesse impossible : {type(exc).__name__} — {exc}")
        else:
            st.session_state["robustesse_resultats"] = resultats
            st.session_state["robustesse_fingerprint"] = empreinte
            resultats_enregistres = resultats

    if resultats_enregistres is None:
        st.caption("Aucun résultat à jour. Le calcul ne démarre qu'au clic sur le bouton dédié.")
        return

    scenarios = resultats_enregistres.get("scenarios") or []
    if not scenarios:
        st.warning("Le moteur n'a retourné aucun scénario exploitable.")
        return

    def _heure(value):
        if value is None:
            return "—"
        if isinstance(value, datetime):
            return value.strftime("%d/%m %H:%M")
        if hasattr(value, "strftime"):
            return value.strftime("%H:%M")
        return str(value)

    services_labels = {}
    for scenario in scenarios:
        for mode in ("sans_reserve", "avec_reserve"):
            for row in (scenario.get(mode) or {}).get("arrivals", []):
                departure = row.get("depart_planifie")
                if row.get("service_id") and departure is not None:
                    services_labels[row["service_id"]] = (
                        f"{departure:%H:%M} {row.get('mission', '')}"
                    )
        for row in scenario.get("candidates_evaluated", []):
            if row.get("kind") == "depart":
                services_labels[row["service_id"]] = f"{row['time']:%H:%M} {row.get('mission', '')}"

    def _service_humain(value):
        if value is None:
            return "—"
        texte = str(value)
        for identifiant in sorted(services_labels, key=lambda key: -len(str(key))):
            texte = texte.replace(str(identifiant), services_labels[identifiant])
        return texte

    def _minutes(value):
        return "—" if value is None else f"{float(value):.1f} min"

    def _kwh(value):
        return "—" if value is None else f"{float(value):.1f} kWh"

    def _soc(value):
        return "—" if value is None else f"{float(value):.1f} %"

    def _liste_valeurs(value):
        if value is None:
            return []
        return list(value) if isinstance(value, (list, tuple)) else [value]

    def _texte_alerte_autonomie(alerte):
        if isinstance(alerte, dict):
            for champ in ("raison", "reason", "message", "motif", "details", "texte"):
                if alerte.get(champ) is not None:
                    return str(alerte[champ])
            return ""
        return "" if alerte is None else str(alerte)

    def _train_id_alerte(alerte):
        if isinstance(alerte, dict):
            train_id = next(
                (alerte.get(champ) for champ in ("train_id", "train")
                 if alerte.get(champ) is not None),
                None,
            )
            if train_id is not None:
                return train_id
        texte = _texte_alerte_autonomie(alerte)
        return texte[6:].split(":", 1)[0] if texte.startswith("Train ") else None

    def _alerte_autonomie_non_verifiable(alerte):
        if isinstance(alerte, dict) and alerte.get("autonomie_non_verifiable"):
            return True
        texte = _texte_alerte_autonomie(alerte).casefold().replace("-", " ")
        return any(marqueur in texte for marqueur in (
            "non vérifiable", "non verifiable", "unverifiable", "unverified",
        ))

    def _autonomie_reellement_insuffisante(batterie):
        alertes = _liste_valeurs(batterie.get("alertes_autonomie"))
        if any(not _alerte_autonomie_non_verifiable(alerte) for alerte in alertes):
            return True
        scenario = batterie.get("scenario") or {}
        par_train = scenario.get("par_train") or {}
        statuts = [statut for statut in par_train.values() if isinstance(statut, dict)]
        if any(
            statut.get("autonomie_insuffisante")
            and not statut.get("autonomie_non_verifiable")
            for statut in statuts
        ):
            return True
        donnees_non_verifiables = (
            batterie.get("autonomie_non_verifiable")
            or batterie.get("alertes_autonomie_non_verifiable")
            or any(_alerte_autonomie_non_verifiable(alerte) for alerte in alertes)
            or any(statut.get("autonomie_non_verifiable") for statut in statuts)
        )
        return bool(batterie.get("autonomie_insuffisante")) and not donnees_non_verifiables

    def _retour_normal(resultat):
        if resultat is None:
            return "Indisponible"
        retour = resultat.get("retour_normal_min")
        return _minutes(retour) if retour is not None else "Non rétabli sur la période"

    def _etat_reserve_absente():
        if not reserve_active:
            return "Réserve non activée"
        if not reserve_gares:
            return "Aucune gare de réserve sélectionnée"
        return "Résultat avec réserve indisponible"

    def _diagnostics(resultat, label):
        if resultat is None:
            st.caption(f"{label} : aucun résultat.")
            return
        batterie = resultat.get("battery") or {}
        diagnostics = (
            ("infra_violations", "Violations d'infrastructure"),
            ("physical_violations", "Violations physiques"),
        )
        for champ, titre in diagnostics:
            valeurs = resultat.get(champ) or []
            if valeurs:
                st.error(f"{titre} — {label}")
                for valeur in valeurs:
                    st.write(_service_humain(valeur))
            else:
                st.caption(f"{titre} — {label} : aucune signalée.")
        for champ, titre in (
            ("erreurs", "Erreurs énergétiques"),
            ("erreurs_reserve", "Erreurs énergétiques de réserve"),
        ):
            valeurs = batterie.get(champ) or []
            if valeurs:
                st.error(f"{titre} — {label}")
                for valeur in valeurs:
                    st.write(valeur)

    def _resume_diagnostics(resultat):
        if resultat is None:
            return "Indisponible"
        compteurs = []
        for champ, nom in (
            ("infra_violations", "infra"),
            ("physical_violations", "physique"),
        ):
            nombre = len(resultat.get(champ) or [])
            if nombre:
                compteurs.append(f"{nom} : {nombre}")
        batterie = resultat.get("battery") or {}
        erreurs_energie = len(batterie.get("erreurs") or [])
        erreurs_reserve = len(batterie.get("erreurs_reserve") or [])
        refus = len(resultat.get("reserve_refus") or [])
        if erreurs_energie:
            compteurs.append(f"énergie : {erreurs_energie}")
        if erreurs_reserve:
            compteurs.append(f"énergie réserve : {erreurs_reserve}")
        if refus:
            compteurs.append(f"refus réserve : {refus}")
        if _autonomie_reellement_insuffisante(batterie):
            compteurs.append("autonomie insuffisante")
        return " ; ".join(compteurs) if compteurs else "Aucune violation signalée"

    def _afficher_suppressions(resultat, label):
        suppressions = resultat.get("suppressions") or []
        nombre = resultat.get("suppressions_count")
        nombre = len(suppressions) if nombre is None else int(nombre)
        if nombre == 0:
            st.caption(f"Aucune suppression de service — {label}.")
            return
        st.markdown(f"**Services supprimés — {label} ({nombre})**")
        if not suppressions:
            st.error(f"{nombre} suppression(s) signalée(s), mais les détails sont absents.")
            return

        def _detail_hypothese(suppression):
            service_suivant = suppression.get("service_suivant_assure")
            hypothese = suppression.get("hypothese")
            if service_suivant is not None:
                return f"Départ observé du service {_service_humain(service_suivant)}"
            if hypothese:
                return (
                    f"{hypothese} ; prochain départ hors horizon, non observé ; + 10 min"
                )
            return "Aucun prochain départ observé ni hypothèse renseignée"

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Service": _service_humain(suppression.get("service_id")),
                        "Motif": _service_humain(suppression.get("motif", "Non précisé")),
                        "Attente du prochain train": _minutes(suppression.get("attente_min")),
                        "Pénalité de suppression": _minutes(suppression.get("penalite_min")),
                        "Hypothèse de pénalité": suppression.get("hypothese", "—"),
                        "Origine": (
                            "Observée"
                            if suppression.get("service_suivant_assure") is not None
                            else "Extrapolée"
                            if suppression.get("hypothese")
                            else "Non précisée"
                        ),
                        "Détail de l'hypothèse": _detail_hypothese(suppression),
                    }
                    for suppression in suppressions
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    def _afficher_recharges_prolongees(resultat, label):
        recharges = resultat.get("recharges_prolongees") or []
        if not recharges:
            return
        st.caption(
            f"{label} — La sécurité de charge est prioritaire sur le délai de récupération : "
            "le retournement est prolongé quand une recharge l'exige."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Service": _service_humain(recharge.get("service_id")),
                        "Train": recharge.get("train_id", "—"),
                        "Gare": recharge.get("gare", "—"),
                        "Retournement initial": _minutes(
                            recharge.get("retournement_initial_min")
                        ),
                        "Retournement retenu": _minutes(
                            recharge.get("retournement_retenu_min")
                        ),
                        "Prolongation pour recharge": _minutes(
                            recharge.get("prolongation_min")
                        ),
                        "Motif": _service_humain(recharge.get("motif", "Non précisé")),
                    }
                    for recharge in recharges
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    def _nombre_suppressions(resultat):
        if resultat is None:
            return "Indisponible"
        nombre = resultat.get("suppressions_count")
        return len(resultat.get("suppressions") or []) if nombre is None else int(nombre)

    def _retard_trains(resultat):
        if resultat is None:
            return "Indisponible"
        retard = resultat.get("retard_trains_min")
        if retard is None and _nombre_suppressions(resultat) == 0:
            retard = resultat.get("retard_cumule_min")
        return _minutes(retard)

    def _afficher_alerte_autonomie(resultat, label):
        batterie = resultat.get("battery") or {}
        alertes = _liste_valeurs(batterie.get("alertes_autonomie"))
        alertes_non_verifiables = _liste_valeurs(
            batterie.get("alertes_autonomie_non_verifiable")
        )
        alertes_non_verifiables.extend(
            alerte for alerte in alertes
            if _alerte_autonomie_non_verifiable(alerte)
        )

        scenario = batterie.get("scenario") or {}
        nominal = batterie.get("baseline") or {}
        scenario_par_train = scenario.get("par_train") or {}
        nominal_par_train = nominal.get("par_train") or {}

        def _ligne_par_train(source, train_id):
            if not isinstance(source, dict) or train_id is None:
                return {}
            ligne = source.get(train_id)
            if ligne is None:
                ligne = next(
                    (valeur for identifiant, valeur in source.items()
                     if str(identifiant) == str(train_id)),
                    None,
                )
            return ligne if isinstance(ligne, dict) else {}

        def _raison_statut(statut):
            raison = _texte_alerte_autonomie(statut)
            if raison:
                return raison
            erreurs = _liste_valeurs(statut.get("erreurs")) if isinstance(statut, dict) else []
            return str(erreurs[0]) if erreurs else (
                "Les données énergétiques ne permettent pas de vérifier l'autonomie."
            )

        alertes_uniques = []
        cles_alertes = set()
        for alerte in alertes_non_verifiables:
            train_id = _train_id_alerte(alerte)
            raison = _texte_alerte_autonomie(alerte) or (
                "Les données énergétiques ne permettent pas de vérifier l'autonomie."
            )
            cle = (str(train_id), raison)
            if cle not in cles_alertes:
                cles_alertes.add(cle)
                alertes_uniques.append({"train_id": train_id, "raison": raison})

        statuts_non_verifiables = [
            (train_id, statut)
            for train_id, statut in scenario_par_train.items()
            if isinstance(statut, dict) and statut.get("autonomie_non_verifiable")
        ]
        for train_id, statut in statuts_non_verifiables:
            cle_train = str(train_id)
            if not any(str(alerte.get("train_id")) == cle_train for alerte in alertes_uniques):
                alertes_uniques.append({
                    "train_id": train_id,
                    "raison": _raison_statut(statut),
                })

        if batterie.get("autonomie_non_verifiable") and not alertes_uniques:
            alertes_uniques.append({
                "train_id": None,
                "raison": "Les données énergétiques ne permettent pas de vérifier l'autonomie.",
            })
        for alerte in alertes_uniques:
            train_id = alerte["train_id"]
            prefixe = f"Train {train_id} : " if train_id is not None else ""
            st.warning(
                f"Autonomie non vérifiable — {label} — {prefixe}{alerte['raison']}."
            )

        alertes_reelles = [
            alerte for alerte in alertes
            if not _alerte_autonomie_non_verifiable(alerte)
        ]
        for train_id, statut in scenario_par_train.items():
            if (
                not isinstance(statut, dict)
                or not statut.get("autonomie_insuffisante")
                or statut.get("autonomie_non_verifiable")
            ):
                continue
            if not any(str(_train_id_alerte(alerte)) == str(train_id) for alerte in alertes_reelles):
                alerte_statut = dict(statut)
                alerte_statut["train_id"] = train_id
                alertes_reelles.append(alerte_statut)

        if not alertes_reelles and _autonomie_reellement_insuffisante(batterie):
            alertes_reelles = [None]
        if not alertes_uniques and not alertes_reelles:
            return

        def _premiere_valeur(source, champs):
            if not isinstance(source, dict):
                return None
            return next((source.get(champ) for champ in champs if source.get(champ) is not None), None)

        for alerte in alertes_reelles:
            texte = _texte_alerte_autonomie(alerte)
            train_id = _train_id_alerte(alerte)
            ligne_scenario = _ligne_par_train(scenario_par_train, train_id)
            ligne_nominale = _ligne_par_train(nominal_par_train, train_id)
            soc = _premiere_valeur(
                alerte,
                ("min_soc_pct", "soc_min_pct", "soc_pct", "soc_reel_pct"),
            )
            if soc is None:
                soc = ligne_scenario.get("min_soc_pct")
            seuil = _premiere_valeur(
                alerte,
                ("seuil_soc_min_pct", "seuil_soc_pct", "threshold_soc_pct", "seuil_pct"),
            )
            if seuil is None:
                seuil = ligne_scenario.get("seuil_soc_min_pct")
            delta_nominal = _premiere_valeur(
                alerte,
                ("delta_soc_nominal_pct", "delta_vs_nominal_pct", "ecart_nominal_pct"),
            )
            if delta_nominal is None and soc is not None:
                soc_nominal = ligne_nominale.get("min_soc_pct")
                if soc_nominal is not None:
                    delta_nominal = float(soc) - float(soc_nominal)

            details = f"Train {train_id} : " if train_id else ""
            if soc is not None and seuil is not None:
                details += f"SoC minimal {float(soc):.1f} %, seuil {float(seuil):.1f} %"
                if delta_nominal is not None:
                    details += f"; écart vs nominal {float(delta_nominal):+.1f} points"
            else:
                details += texte or "SoC minimal et seuil indisponibles"
            st.error(f"Autonomie insuffisante — {label} — {details}.")

    def _fenetre_incident(point):
        params_scenario = dict(params_affichage or {})
        incident_time = (point or {}).get("time")
        if incident_time is None:
            return params_scenario

        try:
            if isinstance(incident_time, datetime):
                evenement = incident_time
                debut_service = datetime.combine(evenement.date(), heure_debut_service)
                if evenement < debut_service:
                    debut_service -= timedelta(days=1)
            elif hasattr(incident_time, "hour"):
                debut_service = datetime.combine(datetime.today().date(), heure_debut_service)
                evenement = datetime.combine(debut_service.date(), incident_time)
                if evenement < debut_service:
                    evenement += timedelta(days=1)
            else:
                return params_scenario
            decalage_incident = (evenement - debut_service).total_seconds() / 3600
            duree_fenetre = float(params_scenario.get("duree_fenetre") or 0)
            duree_service = float(params_scenario.get("duree_service") or 24)
            decalage_max = max(0.0, duree_service - duree_fenetre)
            params_scenario["decalage_heure"] = min(
                decalage_max,
                max(0.0, decalage_incident - duree_fenetre / 2),
            )
        except (TypeError, ValueError):
            return params_scenario
        return params_scenario

    def _trajets_nominal_service(point):
        train_id = (point or {}).get("train_id")
        service_id = (point or {}).get("service_id")
        if train_id is None or service_id is None:
            return []
        trajets_train = (chronologie or {}).get(train_id)
        if trajets_train is None:
            trajets_train = next(
                (
                    trajets
                    for identifiant, trajets in (chronologie or {}).items()
                    if str(identifiant) == str(train_id)
                ),
                [],
            )
        if any(str(trajet.get("service_id")) == str(service_id) for trajet in trajets_train):
            return [
                trajet
                for trajet in trajets_train
                if str(trajet.get("service_id")) == str(service_id)
            ]

        prefixe = f"{type(train_id).__name__}:{train_id}#"
        service_id_texte = str(service_id)
        if not service_id_texte.startswith(prefixe):
            return []
        try:
            ordinal = int(service_id_texte[len(prefixe):])
        except ValueError:
            return []
        if ordinal < 1:
            return []

        entrees = sorted(
            enumerate(trajets_train or []),
            key=lambda item: (
                item[1].get("start"),
                item[1].get("end"),
                item[0],
            ),
        )
        services = []
        service_courant = None
        attente_prefixe = []
        for _, trajet in entrees:
            if trajet.get("is_mission_start", False):
                if service_courant is not None:
                    services.append(service_courant)
                service_courant = attente_prefixe + [trajet]
                attente_prefixe = []
            elif service_courant is None:
                if trajet.get("origine") == trajet.get("terminus"):
                    attente_prefixe.append(trajet)
            else:
                service_courant.append(trajet)
        if service_courant is not None:
            services.append(service_courant)
        return services[ordinal - 1] if ordinal <= len(services) else []

    def _surligner_service_incident(figure, chronologie_perturbee, point, resultat):
        service_id = (point or {}).get("service_id")
        if service_id is None or len(figure.axes) < 2:
            return
        distances = df_gares.set_index("gare")["distance"].to_dict()
        axe_horaire = figure.axes[1]
        trajets_reels = [
            trajet
            for trajets in chronologie_perturbee.values()
            for trajet in trajets
            if str(trajet.get("service_id")) == str(service_id)
        ]
        segments_reels = set()
        for trajet in trajets_reels:
            origine = trajet.get("origine")
            terminus = trajet.get("terminus")
            if origine not in distances or terminus not in distances:
                continue
            if origine != terminus:
                segments_reels.add((origine, terminus))
            axe_horaire.plot(
                [trajet.get("start"), trajet.get("end")],
                [distances[origine], distances[terminus]],
                color="#d62728",
                linewidth=3.0,
                zorder=10,
                solid_capstyle="round",
            )

        suppressions = (resultat or {}).get("suppressions") or []
        service_supprime = any(
            str(suppression.get("service_id")) == str(service_id)
            for suppression in suppressions
        )
        if not service_supprime:
            return

        legende_ajoutee = False
        for trajet in _trajets_nominal_service(point):
            origine = trajet.get("origine")
            terminus = trajet.get("terminus")
            if (
                origine == terminus
                or (origine, terminus) in segments_reels
                or origine not in distances
                or terminus not in distances
            ):
                continue
            axe_horaire.plot(
                [trajet.get("start"), trajet.get("end")],
                [distances[origine], distances[terminus]],
                color="#d62728",
                linestyle="--",
                linewidth=3.0,
                zorder=10,
                label=("Service supprimé — tracé prévu" if not legende_ajoutee else None),
            )
            legende_ajoutee = True

        if legende_ajoutee:
            handles, labels = axe_horaire.get_legend_handles_labels()
            labels_uniques = {}
            for handle, label in zip(handles, labels):
                if label and label not in labels_uniques:
                    labels_uniques[label] = handle
            if labels_uniques:
                axe_horaire.legend(
                    labels_uniques.values(),
                    labels_uniques.keys(),
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1),
                    title="Trains",
                )

    def _graphiques(resultat, label, point):
        if resultat is None:
            return
        chronologie_perturbee = resultat.get("chronologie") or {}
        figure = None
        try:
            figure = creer_graphique_horaire(
                chronologie_perturbee,
                df_gares,
                heure_debut_service,
                _fenetre_incident(point),
                mode_calcul=mode_calcul,
                missions_par_train=missions_par_train,
                all_energy_params=energy_params,
            )
            _surligner_service_incident(figure, chronologie_perturbee, point, resultat)
            st.pyplot(figure)
        except Exception as exc:
            st.error(f"Graphique horaire indisponible ({label}) : {type(exc).__name__} — {exc}")
        finally:
            if figure is not None:
                plt.close(figure)

        energie_scenario = ((resultat.get("battery") or {}).get("scenario") or {})
        params_par_train = energie_scenario.get("par_train") or {}
        terminus_valides = set(terminus_options)
        for train_id, bilan in params_par_train.items():
            log = bilan.get("batterie_log") or bilan.get("battery_log")
            if not log:
                continue
            type_materiel = bilan.get("type_materiel", "batterie")
            params = (energy_params or {}).get(type_materiel, {})
            fig_batterie = None
            try:
                fig_batterie = creer_graphique_batterie(
                    log,
                    train_id,
                    soc_min_pct=params.get("soc_min_pct", 20),
                    soc_max_pct=params.get("soc_max_pct", 95),
                    terminaux_valides=terminus_valides,
                    max_c_rate=params.get("facteur_charge_C", 4.0),
                    df_gares=df_gares,
                )
                if fig_batterie is not None:
                    st.pyplot(fig_batterie)
            except Exception as exc:
                st.error(f"Graphique batterie indisponible ({label}, {train_id}) : {exc}")
            finally:
                if fig_batterie is not None:
                    plt.close(fig_batterie)

    lignes_comparaison = []
    for scenario in scenarios:
        point = scenario.get("point") or {}
        sans_reserve = scenario.get("sans_reserve")
        avec_reserve = scenario.get("avec_reserve")
        if point:
            kind = str(point.get("kind") or "incident")
            type_point = {"depart": "départ", "croisement": "croisement"}.get(
                kind, kind
            )
            point_critique = (
                f"{point.get('train_id', '—')} — {point.get('gare', '—')} "
                f"({_heure(point.get('time'))}, {type_point})"
            )
        else:
            point_critique = "Non déterminé"
        batterie_sans = (sans_reserve or {}).get("battery") or {}
        batterie_avec = (avec_reserve or {}).get("battery") or {}
        lignes_comparaison.append(
            {
                "Retard injecté": f"{scenario.get('delay_min', '—')} min",
                "Point critique": point_critique,
                "Points ex æquo au maximum": scenario.get("tie_count", 1),
                "Retards des trains sans réserve": _retard_trains(sans_reserve),
                "Suppressions sans réserve": _nombre_suppressions(sans_reserve),
                "Pénalités suppressions sans réserve": _minutes(
                    (sans_reserve or {}).get("penalite_suppressions_min")
                ),
                "Retard cumulé — Sans réserve": _minutes((sans_reserve or {}).get("retard_cumule_min")),
                "Retour normal sans réserve": _retour_normal(sans_reserve),
                "SoC min. sans réserve": _soc(batterie_sans.get("min_soc_pct")),
                "Δ batterie sans réserve": _kwh(batterie_sans.get("delta_kwh")),
                "Diagnostics sans réserve": _resume_diagnostics(sans_reserve),
                "Retards des trains avec réserve": _retard_trains(avec_reserve),
                "Suppressions avec réserve": _nombre_suppressions(avec_reserve),
                "Pénalités suppressions avec réserve": _minutes(
                    (avec_reserve or {}).get("penalite_suppressions_min")
                ),
                "Retard cumulé — Avec réserve": (
                    _minutes(avec_reserve.get("retard_cumule_min"))
                    if avec_reserve is not None
                    else _etat_reserve_absente()
                ),
                "Retour normal avec réserve": _retour_normal(avec_reserve)
                if avec_reserve is not None
                else _etat_reserve_absente(),
                "SoC min. avec réserve": _soc(batterie_avec.get("min_soc_pct")),
                "Δ batterie avec réserve": _kwh(batterie_avec.get("delta_kwh")),
                "Diagnostics avec réserve": _resume_diagnostics(avec_reserve),
            }
        )

    st.caption(
        "Score = retards des trains (services opérés) + attente du prochain service assuré "
        "+ 10 min par suppression. "
        f"Points candidats évalués : {resultats_enregistres.get('candidates_count', 0)} "
        "(tous les départs et points de croisement recensés sur toute la journée). "
        "Un retour normal absent signifie que l'horaire n'est pas rétabli sur la période analysée."
    )
    st.dataframe(pd.DataFrame(lignes_comparaison), hide_index=True, width="stretch")

    for scenario in scenarios:
        point = scenario.get("point") or {}
        delay = scenario.get("delay_min", "—")
        description_point = (
            f"{point.get('train_id', '—')} à {point.get('gare', '—')} — {_heure(point.get('time'))}"
            if point
            else "point critique non déterminé"
        )
        with st.expander(f"Incident +{delay} min — {description_point}", expanded=False):
            if point.get("service_id") in services_labels:
                st.caption(f"Service : {_service_humain(point['service_id'])}")
            if scenario.get("tie_count", 1) > 1:
                st.caption(
                    f"{scenario['tie_count']} points ex æquo au score maximal ; "
                    "le premier dans l'ordre déterministe est présenté, pas un pire point unique."
                )
            candidats = scenario.get("candidates_evaluated") or []
            if candidats:
                with st.expander("Audit des départs et croisements de la journée"):
                    st.dataframe(pd.DataFrame([
                        {"Service": _service_humain(item.get("service_id")),
                         "Point": item.get("gare", "—"),
                         "Type": item.get("kind", "—"),
                         "Heure incident": _heure(item.get("time")),
                         "Retard cumulé — Sans réserve": _minutes(item.get("score_min"))}
                        for item in candidats
                    ]), hide_index=True, width="stretch")
            sans_reserve = scenario.get("sans_reserve")
            avec_reserve = scenario.get("avec_reserve")
            colonnes = st.columns(2)
            for colonne, resultat, label in (
                (colonnes[0], sans_reserve, "Sans réserve"),
                (colonnes[1], avec_reserve, "Avec réserve"),
            ):
                with colonne:
                    st.markdown(f"**{label}**")
                    if resultat is None:
                        st.caption(_etat_reserve_absente())
                        continue
                    batterie = resultat.get("battery") or {}
                    st.write(f"Retard cumulé — {label} : {_minutes(resultat.get('retard_cumule_min'))}")
                    st.write(f"Retards des trains : {_retard_trains(resultat)}")
                    st.write(f"Pénalités de suppressions : {_minutes(resultat.get('penalite_suppressions_min'))}")
                    _afficher_suppressions(resultat, label)
                    _afficher_recharges_prolongees(resultat, label)
                    _afficher_alerte_autonomie(resultat, label)
                    retour = resultat.get("retour_normal_min")
                    libelle_retour = (
                        "Retour normal avec prolongation de recharge"
                        if resultat.get("recharges_prolongees")
                        else "Retour normal"
                    )
                    st.write(
                        f"{libelle_retour} : "
                        + (_minutes(retour) if retour is not None else "Non rétabli sur la période")
                    )
                    if "retour_normal_sans_recharge_min" in resultat:
                        retour_sans_recharge = resultat.get("retour_normal_sans_recharge_min")
                        st.write(
                            "Retour normal sans recharge : "
                            + (
                                _minutes(retour_sans_recharge)
                                if retour_sans_recharge is not None
                                else "Non rétabli sur la période"
                            )
                        )
                    st.write(f"SoC minimal : {_soc(batterie.get('min_soc_pct'))}")
                    st.write(f"Variation batterie : {_kwh(batterie.get('delta_kwh'))}")
                    _diagnostics(resultat, label)
                    _graphiques(resultat, f"{delay} min — {label}", point)

            if avec_reserve is not None:
                st.caption(
                    f"Rames supplémentaires utilisées : {avec_reserve.get('reserve_rames_utilisees', 0)} "
                    f"/ {max_reserve_rames}."
                )
                redeploiements = avec_reserve.get("redeploiements_rotation") or []
                if redeploiements:
                    st.markdown("**Redéploiements de rotation**")
                    st.dataframe(pd.DataFrame([
                        {"Service": item.get("service_label") or _service_humain(item.get("service_id")),
                         "Rame": item.get("train_id", "—")}
                        for item in redeploiements
                    ]), hide_index=True, width="stretch")
                activations = avec_reserve.get("reserve_activations") or []
                st.markdown("**Activations de réserve**")
                if activations:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Gare": item.get("gare", "—"),
                                    "Départ planifié": _heure(item.get("depart_planifie")),
                                    "Service": item.get("service_label") or _service_humain(item.get("service_id")),
                                    "Départ réel": _heure(item.get("depart_reel")),
                                    "Rame remplacée": item.get("train_id", "—"),
                                    "Rame de réserve": item.get("reserve_train_id", "—"),
                                    "Mission": item.get("mission", "—"),
                                    "Gain cumulé": _minutes(item.get("gain_cumule_min")),
                                }
                                for item in activations
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.caption("Aucune activation de réserve.")

                refus = (avec_reserve.get("reserve_refus") or []) + (avec_reserve.get("rotation_refus") or [])
                st.markdown("**Refus de réserve**")
                if refus:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Gare": item.get("gare", "—"),
                                    "Service": item.get("service_label") or _service_humain(item.get("service_id")),
                                    "Rame": item.get("train_id", "Réserve supplémentaire"),
                                    "Localisation": item.get("localisation", item.get("gare", "—")),
                                    "Disponible à": _heure(item.get("disponible_a")),
                                    "Minimum de préparation": _minutes(item.get("retournement_minimum_min")),
                                    "Marge avant départ": _minutes(item.get("marge_disponibilite_min")),
                                    "Raisons": "; ".join(_service_humain(reason) for reason in item.get("raisons") or []),
                                }
                                for item in refus
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.caption("Aucun refus de réserve signalé.")


def _library_scenarios():
    """Liste les scénarios en normalisant region / ligne / rel_path.

    Tolère un `scenarios_manager` plus ancien (déploiement Streamlit Cloud
    qui n'aurait pas encore rechargé le module).
    """
    raw = scenarios_manager.list_scenarios()
    out = []
    scenarios_dir = getattr(scenarios_manager, "SCENARIOS_DIR", Path(__file__).parent / "scenarios")
    classify = getattr(scenarios_manager, "classify_scenario_path", None)
    for s in raw:
        md = dict(s.get("metadata") or {})
        path = s.get("path")
        rel = s.get("rel_path")
        if not rel and path is not None:
            try:
                rel = Path(path).resolve().relative_to(Path(scenarios_dir).resolve()).as_posix()
            except Exception:
                rel = Path(path).name
        region = s.get("region")
        ligne = s.get("ligne")
        if not region or not ligne:
            if callable(classify) and path is not None:
                region, ligne = classify(Path(path), md)
            else:
                region = (md.get("region") or "").strip() or "Non classé"
                ligne = (md.get("ligne") or "").strip() or "Général"
        if "nom" not in md:
            md["nom"] = Path(path).stem if path is not None else "scénario"
        entry = dict(s)
        entry["metadata"] = md
        entry["rel_path"] = rel or md["nom"]
        entry["region"] = region
        entry["ligne"] = ligne
        out.append(entry)
    return out


def _group_scenarios_library(scenarios):
    """Regroupe {région: {ligne: [scénarios]}}, avec repli si le module est ancien."""
    fn = getattr(scenarios_manager, "group_scenarios_by_region_line", None)
    if callable(fn):
        try:
            tree = fn(scenarios)
            if tree:
                return tree
        except (TypeError, KeyError, AttributeError):
            pass
    tree = {}
    for s in scenarios:
        tree.setdefault(s["region"], {}).setdefault(s["ligne"], []).append(s)

    def _sort_label(label):
        return (1 if label == "Non classé" else 0, str(label).casefold())

    ordered = {}
    for region in sorted(tree, key=_sort_label):
        ordered[region] = {
            ligne: tree[region][ligne]
            for ligne in sorted(tree[region], key=_sort_label)
        }
    return ordered


def _parse_bulk_pp_for_endpoints(raw_text, terminus, origine, mode_calcul):
    """
    Extrait depuis le texte de saisie en lot des points de passage :
    - le temps trajet aller (ligne au nom du terminus, parts[1])
    - le temps trajet retour (ligne au nom de l'origine, parts[3] en mode Énergie,
      parts[2] en mode Standard)

    Retourne (t_aller, t_retour) — chaque champ vaut None si non détecté.
    """
    if not raw_text:
        return (None, None)
    t_aller, t_retour = None, None
    idx_retour = 3 if mode_calcul == "Calcul Energie" else 2
    for line in raw_text.strip().split('\n'):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(';')]
        if len(parts) < 2:
            continue
        gare = parts[0]
        try:
            t_a = int(parts[1])
        except (ValueError, IndexError):
            t_a = None
        try:
            t_r = int(parts[idx_retour]) if len(parts) > idx_retour else None
        except (ValueError, IndexError):
            t_r = None
        if gare == terminus and t_a is not None:
            t_aller = t_a
        if gare == origine and t_r is not None:
            t_retour = t_r
    return (t_aller, t_retour)


class ProgressTracker:
    """Tracker pour estimation dynamique du temps de calcul."""

    def __init__(self, total_work):
        self.total_work = total_work
        self.start_time = time.time()
        self.last_update = self.start_time
        self.work_done = 0
        self.speed_samples = []
        self.max_samples = 10

    def update(self, work_increment=1):
        """Met à jour la progression et calcule le temps restant."""
        self.work_done += work_increment
        current_time = time.time()

        # Calculer la vitesse actuelle
        time_diff = current_time - self.last_update
        if time_diff > 0:
            current_speed = work_increment / time_diff
            self.speed_samples.append(current_speed)

            # Garder seulement les N derniers échantillons
            if len(self.speed_samples) > self.max_samples:
                self.speed_samples.pop(0)

        self.last_update = current_time

    def get_eta(self):
        """Calcule le temps estimé restant."""
        if not self.speed_samples or self.work_done == 0:
            return "Calcul en cours..."

        # Vitesse moyenne récente
        avg_speed = sum(self.speed_samples) / len(self.speed_samples)

        if avg_speed <= 0:
            return "Calcul en cours..."

        remaining_work = self.total_work - self.work_done
        eta_seconds = remaining_work / avg_speed

        if eta_seconds < 2:
            return "< 2 secondes"
        elif eta_seconds < 60:
            return f"~{int(eta_seconds)} secondes"
        else:
            minutes = int(eta_seconds / 60)
            return f"~{minutes} minute{'s' if minutes > 1 else ''}"

    def get_progress_percent(self):
        """Retourne le pourcentage de progression (limité à 0-100)."""
        if self.total_work == 0:
            return 0
        percent = int((self.work_done / self.total_work) * 100)
        return min(100, max(0, percent))  # Limite strictement à [0, 100]

    def get_elapsed_time(self):
        """Retourne le temps écoulé."""
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        else:
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"


def _calculer_temps_parcours_table(chronologie, gares_info=None):
    """
    Retourne (trips, turnarounds) :
      trips       : {label: [{'dur_total': min, 'dist_km': km, 'dur_mouvement': min}]}
      turnarounds : {label: [durées retournement en min]}
    gares_info : dict {nom_gare: {'distance': km}} pour le calcul des distances.
    """
    trips = defaultdict(list)
    turnarounds = defaultdict(list)

    def _dist_segment(orig, term):
        if not gares_info:
            return 0.0
        d_o = gares_info.get(orig, {}).get('distance', None)
        d_t = gares_info.get(term, {}).get('distance', None)
        if d_o is None or d_t is None:
            return 0.0
        return abs(d_t - d_o)

    for trajets in chronologie.values():
        if not trajets:
            continue
        trajets_tries = sorted(trajets, key=lambda x: x['start'])

        if any('mission' in t for t in trajets_tries):
            # Mode optimisé : is_mission_start délimite chaque trajet complet
            current_mission = None
            trip_start = None
            last_travel_end = None
            dist_acc = 0.0
            dur_mouvement_acc = 0.0

            for t in trajets_tries:
                if t.get('is_mission_start', False):
                    if current_mission is not None and last_travel_end is not None:
                        dur_total = (last_travel_end - trip_start).total_seconds() / 60
                        trips[current_mission].append({
                            'dur_total': dur_total,
                            'dist_km': dist_acc,
                            'dur_mouvement': dur_mouvement_acc,
                        })
                        ret = (t['start'] - last_travel_end).total_seconds() / 60
                        if 0 <= ret <= 240:
                            turnarounds[current_mission].append(ret)
                    current_mission = t['mission']
                    trip_start = t['start']
                    last_travel_end = None
                    dist_acc = 0.0
                    dur_mouvement_acc = 0.0

                if t.get('origine') != t.get('terminus'):
                    last_travel_end = t['end']
                    dist_acc += _dist_segment(t['origine'], t['terminus'])
                    dur_mouvement_acc += (t['end'] - t['start']).total_seconds() / 60

            if current_mission is not None and last_travel_end is not None:
                dur_total = (last_travel_end - trip_start).total_seconds() / 60
                trips[current_mission].append({
                    'dur_total': dur_total,
                    'dist_km': dist_acc,
                    'dur_mouvement': dur_mouvement_acc,
                })
        else:
            # Mode manuel
            cs = trajets_tries[0]
            ce = trajets_tries[0]
            dist_acc = _dist_segment(cs['origine'], cs['terminus']) if cs['origine'] != cs['terminus'] else 0.0
            dur_mouvement_acc = (cs['end'] - cs['start']).total_seconds() / 60 if cs['origine'] != cs['terminus'] else 0.0

            for seg in trajets_tries[1:]:
                if (seg['origine'] == ce['terminus'] and
                        (seg['start'] - ce['end']).total_seconds() / 60 < 20):
                    ce = seg
                    if seg['origine'] != seg['terminus']:
                        dist_acc += _dist_segment(seg['origine'], seg['terminus'])
                        dur_mouvement_acc += (seg['end'] - seg['start']).total_seconds() / 60
                else:
                    key = f"{cs['origine']} → {ce['terminus']}"
                    trips[key].append({
                        'dur_total': (ce['end'] - cs['start']).total_seconds() / 60,
                        'dist_km': dist_acc,
                        'dur_mouvement': dur_mouvement_acc,
                    })
                    ret = (seg['start'] - ce['end']).total_seconds() / 60
                    if 0 <= ret <= 240:
                        turnarounds[key].append(ret)
                    cs = seg
                    ce = seg
                    dist_acc = _dist_segment(seg['origine'], seg['terminus']) if seg['origine'] != seg['terminus'] else 0.0
                    dur_mouvement_acc = (seg['end'] - seg['start']).total_seconds() / 60 if seg['origine'] != seg['terminus'] else 0.0

            key = f"{cs['origine']} → {ce['terminus']}"
            trips[key].append({
                'dur_total': (ce['end'] - cs['start']).total_seconds() / 60,
                'dist_km': dist_acc,
                'dur_mouvement': dur_mouvement_acc,
            })

    return trips, turnarounds


def _afficher_tableau_temps_parcours(chronologie, temps_theoriques=None, df_gares=None):
    """Affiche le tableau Streamlit des temps de parcours par mission."""
    gares_info = df_gares.set_index('gare').to_dict('index') if df_gares is not None else None
    trips, turnarounds = _calculer_temps_parcours_table(chronologie, gares_info=gares_info)
    if not trips:
        return
    if temps_theoriques is None:
        temps_theoriques = {}

    def _fmt(val_min, ref_min):
        if ref_min:
            delta = val_min - ref_min
            sign = "+" if delta >= 0 else ""
            return f"{val_min:.1f} ({sign}{delta:.0f})"
        return f"{val_min:.1f}"

    def _vitesse(dist_km, dur_min):
        if dur_min and dur_min > 0 and dist_km and dist_km > 0:
            return f"{dist_km / (dur_min / 60):.1f}"
        return "—"

    st.subheader("Temps de parcours par mission")
    rows = []
    for mission_label, data in sorted(trips.items()):
        ref = temps_theoriques.get(mission_label)
        durs = [d['dur_total'] for d in data]
        moy = sum(durs) / len(durs)
        dist_moy = sum(d['dist_km'] for d in data) / len(data)
        dur_mouv_moy = sum(d['dur_mouvement'] for d in data) / len(data)
        ret_list = turnarounds.get(mission_label, [])
        ret_str = f"{sum(ret_list)/len(ret_list):.0f}" if ret_list else "—"
        rows.append({
            "Mission": mission_label,
            "Nb trajets": len(data),
            "Tps moyen (min)": _fmt(moy, ref),
            "Tps min (min)": _fmt(min(durs), ref),
            "Tps max (min)": _fmt(max(durs), ref),
            "V. commerciale (km/h)": _vitesse(dist_moy, moy),
            "V. de marche (km/h)": _vitesse(dist_moy, dur_mouv_moy),
            "Retournement moy. (min)": ret_str,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# --- Configuration de la page et initialisation de l'état de session ---
st.set_page_config(layout="wide")

# --- AJOUT DU LOGO (Haut gauche dans la sidebar) ---
logo_url = "logo.png"
st.image(logo_url, width=500)


st.title("Graphique horaire ferroviaire - Prototype")


# État de session (variables globales de l'application)
if "gares" not in st.session_state: st.session_state.gares = None
if "missions" not in st.session_state: st.session_state.missions = []
if "roulement_manuel" not in st.session_state: st.session_state.roulement_manuel = {}
if "mode_calcul" not in st.session_state: st.session_state.mode_calcul = "Calcul Energie"
if "mode_calcul_selector" not in st.session_state: st.session_state.mode_calcul_selector = "Calcul Energie"
if "mode_generation" not in st.session_state: st.session_state.mode_generation = "Rotation optimisée"
if "chronologie_calculee" not in st.session_state: st.session_state.chronologie_calculee = None
if "stats_homogeneite" not in st.session_state: st.session_state.stats_homogeneite = {}
if "warnings_calcul" not in st.session_state: st.session_state.warnings_calcul = {}
if "run_calculation" not in st.session_state: st.session_state.run_calculation = False

# Initialise avec les paramètres par défaut si nécessaire
default_params = get_default_energy_params()
if "energy_params" not in st.session_state: st.session_state.energy_params = {
    "electrique": default_params.copy(),
    "bimode": default_params.copy(),
    "diesel": default_params.copy(),
    "batterie": default_params.copy()
}


if "energy_errors" not in st.session_state: st.session_state.energy_errors = []
if "resultats_energie_par_train" not in st.session_state: st.session_state.resultats_energie_par_train = {}
if "missions_par_train" not in st.session_state: st.session_state.missions_par_train = {}
if "energy_cache_key" not in st.session_state: st.session_state.energy_cache_key = None
if "run_energy_only" not in st.session_state: st.session_state.run_energy_only = False

# Heures de service (centralisées dans session_state pour permettre le chargement
# d'un scénario qui les redéfinit).
if "heure_debut_service" not in st.session_state:
    st.session_state.heure_debut_service = datetime.strptime("06:00", "%H:%M").time()
if "heure_fin_service" not in st.session_state:
    st.session_state.heure_fin_service = datetime.strptime("22:00", "%H:%M").time()


_SCENARIO_PLACEHOLDER = "— Sélectionner —"


def _apply_pending_scenario():
    """Lit et applique un scénario en attente AVANT le rendu des widgets.

    Les boutons de chargement se contentent de déposer une source dans
    `_pending_scenario_source` (callback Streamlit). On relit toujours
    le fichier ici pour que le 2e chargement écrase réellement le 1er.
    """
    pending = st.session_state.pop("_pending_scenario_source", None)
    if not pending:
        return
    try:
        kind = pending.get("type")
        if kind == "path":
            data = scenarios_manager.load_scenario(pending["path"])
        elif kind == "bytes":
            data = scenarios_manager.load_scenario(pending["bytes"])
        else:
            raise ValueError(f"Source de scénario inconnue : {kind!r}")
        scenarios_manager.apply_scenario_to_session(data, st.session_state)
        nom = (
            data.get("metadata", {}).get("nom")
            or pending.get("name")
            or "scénario"
        )
        st.session_state["_scenario_loaded_msg"] = f"Scénario « {nom} » chargé."
    except Exception as e:
        st.session_state["_scenario_load_error"] = f"Impossible de charger le scénario : {e}"


def _on_scenario_region_change():
    for k in ("scenario_ligne", "scenario_selector"):
        st.session_state.pop(k, None)


def _on_scenario_ligne_change():
    st.session_state.pop("scenario_selector", None)


def _queue_library_scenario():
    """Callback : mémorise le chemin à relire au prochain passage en tête de script."""
    rel = st.session_state.get("scenario_selector")
    if not rel or rel == _SCENARIO_PLACEHOLDER:
        return
    sc = next(
        (s for s in _library_scenarios() if s.get("rel_path") == rel),
        None,
    )
    if sc is None:
        st.session_state["_scenario_load_error"] = "Scénario introuvable dans la bibliothèque."
        return
    st.session_state["_pending_scenario_source"] = {
        "type": "path",
        "path": str(sc["path"]),
        "name": sc["metadata"].get("nom") or sc["path"].stem,
    }


def _queue_uploaded_scenario():
    """Callback : copie les bytes du fichier importé pour une relecture sûre."""
    epoch = st.session_state.get("_scenario_ui_epoch", 0)
    uploaded_file = st.session_state.get(f"scenario_upload_{epoch}")
    if uploaded_file is None:
        return
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if not raw:
        st.session_state["_scenario_load_error"] = "Le fichier importé est vide."
        return
    st.session_state["_pending_scenario_source"] = {
        "type": "bytes",
        "bytes": raw,
        "name": getattr(uploaded_file, "name", None),
    }


_apply_pending_scenario()


def _compter_trains_initiaux(missions, heure_debut, heure_fin):
    """Compte le nombre de trains aller initiaux sur la plage de service."""
    total = 0
    dt_debut = datetime.combine(datetime.today(), heure_debut)
    dt_fin = datetime.combine(datetime.today(), heure_fin)
    if dt_fin <= dt_debut:
        dt_fin += timedelta(days=1)
    for mission in missions:
        if mission.get("frequence", 0) <= 0:
            continue
        try:
            minutes_ref = [int(m.strip()) for m in mission.get("reference_minutes", "0").split(',') if m.strip().isdigit()]
            if not minutes_ref:
                minutes_ref = [0]
        except Exception:
            minutes_ref = [0]
        intervalle = timedelta(hours=1 / mission["frequence"])
        for minute_ref in minutes_ref:
            curseur = dt_debut.replace(minute=minute_ref % 60, second=0, microsecond=0)
            curseur += timedelta(hours=minute_ref // 60)
            while curseur < dt_debut:
                curseur += timedelta(hours=1)
            while curseur < dt_fin:
                total += 1
                curseur += intervalle
    return total


def _format_duree(secondes):
    """Formate une durée en secondes en chaîne lisible."""
    if secondes < 10:
        return "< 10 secondes"
    elif secondes < 60:
        return f"~{int(secondes)} secondes"
    elif secondes < 120:
        return "~1 minute"
    elif secondes < 3600:
        minutes = int(secondes / 60)
        return f"~{minutes} minutes"
    else:
        heures = secondes / 3600
        return f"~{heures:.1f}h"


def estimer_temps_calcul(missions, heure_debut, heure_fin, mode="simple", optimization_params=None):
    """Estime le temps de calcul en secondes selon la complexité du scénario et le mode d'optimisation.

    Utilise la dernière durée réelle observée (session_state['last_calcul_info']) pour calibrer
    l'estimation si elle correspond au même mode.
    """
    n = _compter_trains_initiaux(missions, heure_debut, heure_fin)
    if n == 0:
        return "N/A"

    # Calibrage adaptatif depuis le dernier calcul observé
    last = st.session_state.get("last_calcul_info")
    if last and last.get("mode") == mode and last.get("n_events", 0) > 0 and last.get("elapsed_s", 0) > 0:
        # Mise à l'échelle par rapport au dernier run connu (complexité sous-quadratique)
        base_s = last["elapsed_s"] * (n / last["n_events"]) ** 1.5
    else:
        # Heuristique par défaut : N^1.5 / 5  (calibrée sur l'expérience réelle)
        base_s = (n ** 1.5) / 5.0

    # Multiplicateur selon le mode d'optimisation
    params = optimization_params or {}
    if mode == "genetic":
        import os
        pop = params.get("population_size", 100)
        gen = params.get("generations", 150)
        workers = max(1, (os.cpu_count() or 4) - 1)
        estimated_s = base_s * pop * gen / workers
    elif mode == "smart_progressive":
        estimated_s = base_s * 10
    elif mode == "exhaustif":
        estimated_s = base_s * 20
    else:  # simple, fast
        estimated_s = base_s

    return _format_duree(estimated_s)

# --- SECTION 1: Définition des gares ---
with st.expander("1. Gares et Infrastructure", expanded=True):

    # Sélection du mode de calcul qui conditionne l'affichage
    st.session_state.mode_calcul = st.radio(
        "Mode de calcul",
        ["Calcul Energie", "Standard"],
        horizontal=True,
        key="mode_calcul_selector",
    )
    mode_calcul = st.session_state.mode_calcul

    # En mode Énergie, proposer de charger un scénario pré-enregistré
    if mode_calcul == "Calcul Energie":
        _scenario_loaded_flag = st.session_state.pop("_scenario_loaded_msg", None)
        if _scenario_loaded_flag:
            st.success(_scenario_loaded_flag)
        _scenario_load_error = st.session_state.pop("_scenario_load_error", None)
        if _scenario_load_error:
            st.error(_scenario_load_error)

        with st.expander("📚 Charger un scénario", expanded=(st.session_state.gares is None)):
            tab_biblio, tab_fichier = st.tabs(["Bibliothèque", "Importer un fichier"])

            with tab_biblio:
                scenarios = _library_scenarios()
                if not scenarios:
                    st.caption("Aucun scénario dans `scenarios/`. Téléchargez votre configuration "
                               "actuelle depuis le menu d'export du graphe horaire, puis déposez-la "
                               "dans `scenarios/<région>/<ligne>/`.")
                else:
                    tree = _group_scenarios_library(scenarios)
                    regions = list(tree.keys())
                    placeholder = _SCENARIO_PLACEHOLDER
                    st.caption("Naviguez par région puis par ligne. Fichiers : `scenarios/<région>/<ligne>/`.")

                    col_r, col_l = st.columns(2)
                    with col_r:
                        region = st.selectbox(
                            "Région",
                            regions,
                            key="scenario_region",
                            on_change=_on_scenario_region_change,
                        )
                    lignes = list(tree.get(region, {}).keys())
                    if st.session_state.get("scenario_ligne") not in lignes and lignes:
                        st.session_state["scenario_ligne"] = lignes[0]
                    with col_l:
                        ligne = st.selectbox(
                            "Ligne",
                            lignes,
                            key="scenario_ligne",
                            on_change=_on_scenario_ligne_change,
                        )

                    line_scenarios = tree.get(region, {}).get(ligne, [])
                    rel_options = [placeholder] + [s["rel_path"] for s in line_scenarios]
                    labels = {s["rel_path"]: s["metadata"]["nom"] for s in line_scenarios}

                    def _fmt_scenario(rel):
                        if rel == placeholder:
                            return placeholder
                        return labels.get(rel, rel)

                    if st.session_state.get("scenario_selector") not in rel_options:
                        st.session_state["scenario_selector"] = placeholder

                    col_sel, col_btn = st.columns([4, 1])
                    with col_sel:
                        choix = st.selectbox(
                            "Scénario",
                            rel_options,
                            format_func=_fmt_scenario,
                            key="scenario_selector",
                        )
                    with col_btn:
                        st.markdown("&nbsp;")  # alignement vertical
                        load_clicked = st.button(
                            "Charger",
                            disabled=(choix == placeholder),
                            key="scenario_load_btn",
                            on_click=_queue_library_scenario,
                        )
                    # Filet de sécurité : si le callback n'a pas tourné, on
                    # file la source et on relance pour l'appliquer en tête.
                    if load_clicked and "_scenario_loaded_msg" not in st.session_state:
                        _queue_library_scenario()
                        st.rerun()

                    if choix != placeholder:
                        sc_meta = next((s for s in line_scenarios if s["rel_path"] == choix), None)
                        if sc_meta:
                            md = sc_meta["metadata"]
                            tags_str = ", ".join(md.get("tags", []) or [])
                            st.caption(
                                f"**{md['nom']}** — {sc_meta['region']} / {sc_meta['ligne']}  \n"
                                f"{md.get('description', '')}  \n"
                                f"Auteur : {md.get('auteur') or '?'} · "
                                f"Modifié : {md.get('date_modification') or '?'}  \n"
                                f"Tags : {tags_str or '—'}"
                            )

            with tab_fichier:
                upload_key = f"scenario_upload_{st.session_state.get('_scenario_ui_epoch', 0)}"

                uploaded = st.file_uploader(
                    "Fichier scénario (.json)", type=["json"], key=upload_key
                )
                if uploaded is not None:
                    try:
                        data = scenarios_manager.load_scenario(
                            uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
                        )
                        nom = data.get("metadata", {}).get("nom") or uploaded.name
                        st.caption(f"Prêt à importer : **{nom}**")
                        apply_clicked = st.button(
                            "Appliquer ce fichier",
                            key="scenario_apply_upload_btn",
                            on_click=_queue_uploaded_scenario,
                        )
                        if apply_clicked and "_scenario_loaded_msg" not in st.session_state:
                            _queue_uploaded_scenario()
                            st.rerun()
                    except Exception as e:
                        st.error(f"Fichier invalide : {e}")

    with st.form("formulaire_gares"):

        # Aide contextuelle basée sur le mode
        help_text = "Format: nom;distance_km;[infra]\nInfra (optionnel): VE=Voie d'Évitement (croisement), F=Pas de croisement, D=Début/Fin de voie double."
        default_text = "Nîmes;0;VE\nVauvert;20;VE\nLe Grau-du-Roi;50;VE"

        if mode_calcul == "Calcul Energie":
            help_text = "Format: nom;km;infra (VE/F/D);electrification;rampe_section_a_venir\n" \
                        "- electrification: RXXXX (recharge kW), C1500, C25, F (non électrifié)\n" \
                        "- rampe_section_a_venir: Pente en ‰ (ex: -8 ou 8)"
            default_text = "Nîmes;0;VE;C1500;5\nVauvert;20;D;F;-3\nLe Grau-du-Roi;50;VE;R500;0"

        # Initialisation unique du buffer texte ; le chargement d'un scénario
        # réécrit cette clé via scenarios_manager.apply_scenario_to_session.
        if "gares_texte_input" not in st.session_state:
            st.session_state["gares_texte_input"] = default_text

        gares_texte = st.text_area(
            "Liste des gares (une par ligne)",
            key="gares_texte_input",
            help=help_text,
        )

        if st.form_submit_button("Valider les gares et l'infrastructure"):
            try:
                lignes = [ligne.strip().split(";") for ligne in gares_texte.strip().split("\n")]
                donnees_gares = []
                colonnes = ["gare", "distance", "infra"]

                if mode_calcul == "Calcul Energie":
                    colonnes.extend(["electrification", "rampe_section_a_venir"])

                for i, ligne in enumerate(lignes):
                    if not ligne or not ligne[0]: continue

                    gare_data = {"gare": ligne[0]}

                    if mode_calcul == "Standard":
                        if not (2 <= len(ligne) <= 3):
                            st.error(f"Format de ligne incorrect : '{' '.join(ligne)}'. Utilisez 'nom;distance_km;[infra]'.")
                            continue
                        gare_data["distance"] = float(ligne[1])
                        if len(ligne) == 3:
                            infra_val = ligne[2].upper() if ligne[2] else None
                            if infra_val not in ['VE', 'F', 'D']: # MODIFIÉ: 'T' -> 'VE'
                                st.warning(f"Type d'infrastructure '{ligne[2]}' non reconnu à la ligne {i+1}. Ignoré.")
                                infra_val = None
                            gare_data["infra"] = infra_val
                        else:
                            gare_data["infra"] = None

                    elif mode_calcul == "Calcul Energie":
                        if len(ligne) != 5:
                            st.error(f"Format de ligne incorrect : '{' '.join(ligne)}'. Utilisez 'nom;km;infra;electrification;rampe'.")
                            continue
                        gare_data["distance"] = float(ligne[1])
                        infra_val = ligne[2].upper() if ligne[2] else None
                        if infra_val not in ['VE', 'F', 'D']: # MODIFIÉ: 'T' -> 'VE'
                            st.warning(f"Type d'infrastructure '{ligne[2]}' non reconnu à la ligne {i+1}. Ignoré.")
                            infra_val = 'F'
                        gare_data["infra"] = infra_val
                        gare_data["electrification"] = ligne[3].upper() if ligne[3] else "F"
                        gare_data["rampe_section_a_venir"] = float(ligne[4])

                    donnees_gares.append(gare_data)

                df = pd.DataFrame(donnees_gares, columns=colonnes)
                df = df.sort_values("distance").reset_index(drop=True)

                # Logique par défaut pour l'infrastructure (colonne 'infra')
                for i, row in df.iterrows():
                    if (i == 0 or i == len(df) - 1) and pd.isna(row['infra']):
                        df.loc[i, 'infra'] = 'VE' # MODIFIÉ: 'T' -> 'VE'
                    elif pd.isna(row['infra']):
                        df.loc[i, 'infra'] = 'F'

                if mode_calcul == "Calcul Energie":
                     if "electrification" not in df.columns: df["electrification"] = "F"
                     if "rampe_section_a_venir" not in df.columns: df["rampe_section_a_venir"] = 0

                st.session_state.gares = df
                st.success("Gares et infrastructure enregistrées !")
                st.session_state.chronologie_calculee = None
                st.session_state.run_calculation = False # Pour éviter le recalcul automatique

                # Affichage du récapitulatif
                df_display = df.copy()
                infra_map = {'VE': "Voie d'Évitement (VE)", 'D': 'Voie double (D)', 'F': ''} # MODIFIÉ: 'T' -> 'VE'
                df_display['Description'] = df_display['infra'].map(infra_map).fillna('')

                cols_to_show = ['gare', 'distance', 'Description']
                if mode_calcul == "Calcul Energie":
                    # Formatage amélioré pour l'affichage
                    def format_electrification(e):
                        e_upper = str(e).upper()
                        if e_upper == "C1500": return "Section électrifiée 1500V"
                        if e_upper == "C25": return "Section électrifiée 25kV"
                        if e_upper == "F": return "Non électrifié"
                        if e_upper.startswith("R"):
                            try:
                                kw = int(e_upper[1:])
                                return f"Point de recharge {kw} kW"
                            except ValueError:
                                return e # Retourne la valeur brute en cas d'erreur
                        return e

                    df_display["Electrification"] = df_display['electrification'].apply(format_electrification)
                    df_display["Rampe sur la section"] = df_display['rampe_section_a_venir'].apply(lambda x: f"{x} ‰")

                    cols_to_show.extend(["Electrification", "Rampe sur la section"])

                st.dataframe(df_display[cols_to_show], width="stretch")

            except Exception as e:
                st.error(f"Erreur de format. Vérifiez vos données. Détails : {e}")


# --- La suite de l'application ne s'affiche que si les gares sont définies ---
if st.session_state.get('gares') is not None:
    dataframe_gares = st.session_state.gares
    gares_list = dataframe_gares["gare"].tolist()

    # --- SECTION 2: Paramètres généraux du service ---
    with st.expander("2. Paramètres de service", expanded=True):
        # Les valeurs initiales sont dans session_state ; on utilise key= pour le binding.
        heure_debut_service = st.time_input(
            "Début de service",
            key="heure_debut_service",
        )
        heure_fin_service = st.time_input(
            "Fin de service",
            key="heure_fin_service",
            help="Heure limite pour le DÉPART des trains. Un train qui part juste "
                 "avant cette heure est tracé jusqu'au bout, même si l'arrivée est après.",
        )

        # Rotation optimisée = défaut. Le mode Manuel est une option : les
        # roulements sont saisis (ou importés), et l'énergie est tout de même
        # évaluée si on est en Calcul Énergie.
        mode_generation = st.radio(
            "Mode de génération des trains",
            ["Rotation optimisée", "Manuel"],
            horizontal=True,
            key="mode_generation",
            help="Rotation optimisée : l'algorithme construit les roulements. "
                 "Manuel : vous les définissez (saisie ou import Excel). "
                 "En Calcul Énergie, la consommation est évaluée dans les deux cas.",
        )
        if mode_calcul == "Calcul Energie":
            if mode_generation == "Manuel":
                st.info("Les roulements saisis manuellement seront simulés énergétiquement (consommation, batterie, etc.).")
            else:
                st.info("Le moteur événementiel simule les trajets et la consommation énergétique.")

    # --- SECTION 3: Définition des missions ---

    # Map pour capitaliser les options de matériel roulant
    options_materiel_map = {
        "diesel": "Diesel",
        "electrique": "Electrique",
        "bimode": "Bimode",
        "batterie": "Batterie"
    }
    options_materiel_list = list(options_materiel_map.keys())

    with st.expander("3. Missions", expanded=True):
        if "nombre_missions" not in st.session_state:
            st.session_state.nombre_missions = len(st.session_state.missions) or 1
        nombre_missions = st.number_input("Nombre de types de missions", 1, 10, key="nombre_missions")

        while len(st.session_state.missions) < nombre_missions:
            st.session_state.missions.append({})
        while len(st.session_state.missions) > nombre_missions:
            st.session_state.missions.pop()

        for i in range(nombre_missions):
            with st.container(border=True):
                st.subheader(f"Mission {i+1} (trajet Aller)")
                mission = st.session_state.missions[i]
                if f"saisie_pp_{i}" not in st.session_state:
                    st.session_state[f"saisie_pp_{i}"] = "Saisie manuelle par lot"

                cols = st.columns([2, 2, 3])
                origine = cols[0].selectbox(f"Origine M{i+1}", gares_list, index=gares_list.index(mission.get("origine", gares_list[0])) if mission.get("origine") in gares_list else 0, key=f"orig{i}")
                terminus = cols[0].selectbox(f"Terminus M{i+1}", gares_list, index=gares_list.index(mission.get("terminus", gares_list[-1])) if mission.get("terminus") in gares_list else len(gares_list)-1, key=f"term{i}")


                frequence = cols[1].number_input(f"Fréquence (train/h) M{i+1}", 0.1, 10.0, mission.get("frequence", 1.0), 0.1, key=f"freq{i}")

                # Auto-update du temps trajet planifié si la saisie en lot contient
                # le terminus (aller) ou l'origine (retour). On détecte un changement
                # du texte brut pour éviter d'écraser une modif manuelle ultérieure.
                _is_bulk = st.session_state.get(f"saisie_pp_{i}", "Saisie manuelle par lot") == "Saisie manuelle par lot"
                if _is_bulk:
                    _cur_bulk = st.session_state.get(f"pp_raw_{i}", "")
                    _prev_bulk = st.session_state.get(f"_prev_pp_raw_{i}")
                    if _cur_bulk != _prev_bulk:
                        _t_term, _t_orig = _parse_bulk_pp_for_endpoints(
                            _cur_bulk, terminus, origine, mode_calcul
                        )
                        if _t_term is not None and 1 <= _t_term <= 720:
                            st.session_state[f"tt{i}"] = _t_term
                        if _t_orig is not None and 1 <= _t_orig <= 720:
                            st.session_state[f"tt_retour_{i}"] = _t_orig
                        st.session_state[f"_prev_pp_raw_{i}"] = _cur_bulk

                # Ce temps est utilisé par core_logic pour le planning
                # energy_logic l'utilisera comme contrainte pour déduire la vitesse
                temps_trajet = cols[1].number_input(f"Temps trajet PLANIFIÉ (min) M{i+1}", 1, 720, mission.get("temps_trajet", 45), key=f"tt{i}", help="Temps utilisé pour le planning. Le simulateur d'énergie en déduira la vitesse. En saisie en lot, ajouter une ligne au nom du terminus met cette valeur à jour automatiquement.")

                retournement_A = cols[2].number_input(f"Retournement MINIMUM à {origine} (min)", 0, 120, mission.get("temps_retournement_A", 10), key=f"tr_a_{i}")
                retournement_B = cols[2].number_input(f"Retournement MINIMUM à {terminus} (min)", 0, 120, mission.get("temps_retournement_B", 10), key=f"tr_b_{i}")

                # Ajout pour le mode Énergie
                type_materiel = "diesel"
                if mode_calcul == "Calcul Energie":
                    type_materiel = cols[0].selectbox(
                        f"Type de matériel M{i+1}",
                        options=options_materiel_list,
                        format_func=lambda x: options_materiel_map.get(x, x), # Affiche la version capitalisée
                        index=options_materiel_list.index(mission.get("type_materiel", "diesel")),
                        key=f"type_mat_{i}"
                    )

                ref_minutes = "0"
                if mode_generation == "Rotation optimisée" or mode_calcul == "Calcul Energie":
                    ref_minutes = cols[2].text_input(
                        f"Minute(s) de réf. M{i+1}",
                        mission.get("reference_minutes", "0"),
                        key=f"ref_mins{i}",
                        help="Minutes de départ après le début de chaque heure (ex: '15,45'). Peut être > 59 pour décaler (ex: '75' pour un départ à H+1h15)."
                    )

                # --- NOUVEAUTÉ : INJECTION TERMINUS 2 ---
                inject_t2 = st.checkbox(
                    f"Autoriser l'injection de nouvelles rames depuis {terminus} (Terminus 2)",
                    value=mission.get("inject_from_terminus_2", False),
                    key=f"inj_t2_{i}",
                    help="Cochez cette case pour autoriser le système à injecter des trains au départ du terminus retour si aucune rame n'est disponible depuis l'aller."
                )

                st.markdown("**Points de passage optionnels :**")
                trajet_asymetrique = st.checkbox("Saisir un temps/parcours différent pour le retour", mission.get("trajet_asymetrique", False), key=f"asym_{i}")

                saisie_pp_mode = st.radio(
                    "Méthode de saisie des points de passage",
                    ["Saisie manuelle par lot", "Interface Guidée"],
                    key=f"saisie_pp_{i}",
                    horizontal=True,
                )

                passing_points = []
                passing_points_retour = []

                if saisie_pp_mode == "Interface Guidée":
                    st.markdown("**Aller :**")
                    gares_passage_dispo = [g for g in gares_list if g not in [origine, terminus]]
                    nb_pp = st.number_input(f"Nombre de points de passage (Aller M{i+1})", 0, 10, len(mission.get("passing_points", [])), key=f"n_pass_{i}")

                    if gares_passage_dispo and nb_pp > 0:
                        dernier_temps = 0
                        for j in range(nb_pp):
                            pp_cols = st.columns([2, 2, 1, 1]) if mode_calcul == "Calcul Energie" else st.columns(2)

                            pp_gare = pp_cols[0].selectbox(f"Gare PP {j+1}", gares_passage_dispo, key=f"pp_gare_{i}_{j}")
                            pp_temps = pp_cols[1].number_input(f"Temps depuis {origine} (min)", min_value=dernier_temps + 1, max_value=temps_trajet - 1, value=min(dernier_temps + 15, temps_trajet - 1), key=f"pp_tps_{i}_{j}")

                            pp_arret_commercial = False
                            pp_temps_arret = 0
                            if mode_calcul == "Calcul Energie":
                                with pp_cols[2]:
                                    st.write(" &nbsp; ")
                                    pp_arret_commercial = st.checkbox("Arrêt", key=f"pp_arret_{i}_{j}", help="Arrêt commercial à ce point de passage")

                                pp_temps_arret = pp_cols[3].number_input(
                                    "Durée",
                                    min_value=0, # min_value doit être 0 pour autoriser la valeur 0
                                    max_value=60,
                                    value=2 if pp_arret_commercial else 0, # Défaut à 2 si coché, 0 sinon
                                    key=f"pp_duree_arret_{i}_{j}",
                                    disabled=not pp_arret_commercial, # Désactivé si la case n'est pas cochée
                                    help="Durée de l'arrêt en minutes"
                                )

                            passing_points.append({
                                "gare": pp_gare,
                                "time_offset_min": pp_temps,
                                "arret_commercial": pp_arret_commercial,
                                "duree_arret_min": int(pp_temps_arret)
                            })
                            dernier_temps = pp_temps

                    if trajet_asymetrique:
                        st.markdown("**Retour :**")
                        temps_trajet_retour = st.number_input(f"Temps trajet RETOUR (min)", 1, 720, mission.get("temps_trajet_retour", temps_trajet), key=f"tt_retour_{i}")
                        nb_pp_retour = st.number_input(f"Nombre de PP (Retour M{i+1})", 0, 10, len(mission.get("passing_points_retour", [])), key=f"n_pass_retour_{i}")

                        if gares_passage_dispo and nb_pp_retour > 0:
                            dernier_temps_retour = 0
                            for j in range(nb_pp_retour):
                                pp_cols_r = st.columns([2, 2, 1, 1]) if mode_calcul == "Calcul Energie" else st.columns(2)

                                pp_gare_r = pp_cols_r[0].selectbox(f"Gare PP {j+1} (Retour)", gares_passage_dispo, key=f"pp_gare_retour_{i}_{j}")
                                pp_temps_r = pp_cols_r[1].number_input(f"Temps depuis {terminus} (min)", dernier_temps_retour + 1, temps_trajet_retour - 1, min(dernier_temps_retour + 15, temps_trajet_retour - 1), key=f"pp_tps_retour_{i}_{j}")

                                pp_arret_commercial_r = False
                                pp_temps_arret_r = 0
                                if mode_calcul == "Calcul Energie":
                                    with pp_cols_r[2]:
                                        st.write(" &nbsp; ")
                                        pp_arret_commercial_r = st.checkbox("Arrêt", key=f"pp_arret_r_{i}_{j}", help="Arrêt commercial à ce point de passage")

                                    pp_temps_arret_r = pp_cols_r[3].number_input(
                                        "Durée",
                                        min_value=0,
                                        max_value=60,
                                        value=2 if pp_arret_commercial_r else 0,
                                        key=f"pp_duree_arret_r_{i}_{j}",
                                        disabled=not pp_arret_commercial_r,
                                        help="Durée de l'arrêt en minutes"
                                    )

                                passing_points_retour.append({
                                    "gare": pp_gare_r,
                                    "time_offset_min": pp_temps_r,
                                    "arret_commercial": pp_arret_commercial_r,
                                    "duree_arret_min": int(pp_temps_arret_r)
                                })
                                dernier_temps_retour = pp_temps_r
                    else:
                        temps_trajet_retour = temps_trajet

                else: # Saisie manuelle par lot
                    placeholder_text = "Vauvert;20\n...ou si case cochée...\nVauvert;20;30"
                    help_text = (
                        f"Format: Gare;Temps depuis {origine} (min)[;Temps depuis {terminus} (min)]\n"
                        f"Astuce : une ligne au nom du terminus ({terminus}) met automatiquement à jour "
                        f"le champ 'Temps trajet PLANIFIÉ'. Idem pour l'origine ({origine}) en retour."
                    )
                    if mode_calcul == "Calcul Energie":
                         help_text = (
                            "Format: Gare;Temps_Aller;[Arrêt_Aller_min];[Temps_Retour];[Arrêt_Retour_min]\n"
                            "Ex: Vauvert;20;2;30;2 (2min d'arrêt à l'aller et au retour)\n"
                            "Ex: Vauvert;20;0;30;0 (pas d'arrêt)\n"
                            f"Astuce : une ligne au nom du terminus ({terminus}) met automatiquement "
                            f"à jour le champ 'Temps trajet PLANIFIÉ'. Idem pour l'origine ({origine}) en retour."
                         )
                         placeholder_text = "Vauvert;20;2;30;2"


                    pp_raw_text = st.text_area("Points de passage (un par ligne)", value=mission.get("pp_raw_text", ""), placeholder=placeholder_text, help=help_text, key=f"pp_raw_{i}")

                    mission["pp_raw_text"] = pp_raw_text

                    temps_trajet_retour = temps_trajet
                    if trajet_asymetrique:
                        temps_trajet_retour = st.number_input(f"Temps trajet RETOUR (min)", 1, 720, mission.get("temps_trajet_retour", temps_trajet), key=f"tt_retour_{i}")

                    try:
                        for line in pp_raw_text.strip().split('\n'):
                            if not line: continue
                            parts = [p.strip() for p in line.split(';')]

                            gare = parts[0]
                            if gare not in gares_list:
                                st.warning(f"Gare '{gare}' non reconnue. Elle doit être dans la liste principale.")
                                continue

                            # Les lignes correspondant aux terminus (aller/retour) ne sont
                            # pas des points de passage : leur temps met à jour le champ
                            # "Temps trajet PLANIFIÉ" via l'auto-update plus haut.
                            if gare in (origine, terminus):
                                continue

                            if mode_calcul == "Standard":
                                if not (2 <= len(parts) <= 3):
                                    st.warning(f"Ligne ignorée (format incorrect): '{line}'")
                                    continue
                                t_aller = int(parts[1])
                                passing_points.append({"gare": gare, "time_offset_min": t_aller, "arret_commercial": False, "duree_arret_min": 0})
                                if trajet_asymetrique and len(parts) == 3:
                                    t_retour = int(parts[2])
                                    passing_points_retour.append({"gare": gare, "time_offset_min": t_retour, "arret_commercial": False, "duree_arret_min": 0})

                            else: # Mode Calcul Energie
                                if not (3 <= len(parts) <= 5):
                                    st.warning(f"Ligne ignorée (format incorrect Energie): '{line}'")
                                    continue
                                t_aller = int(parts[1])
                                arret_aller = int(parts[2])
                                passing_points.append({"gare": gare, "time_offset_min": t_aller, "arret_commercial": arret_aller > 0, "duree_arret_min": arret_aller})

                                if trajet_asymetrique and len(parts) == 5:
                                    t_retour = int(parts[3])
                                    arret_retour = int(parts[4])
                                    passing_points_retour.append({"gare": gare, "time_offset_min": t_retour, "arret_commercial": arret_retour > 0, "duree_arret_min": arret_retour})

                    except (ValueError, IndexError) as e:
                        st.error(f"Erreur de parsing dans la saisie manuelle. Détails : {e}")

                # Enregistrement de la mission dans l'état de session
                st.session_state.missions[i] = {
                    "origine": origine, "terminus": terminus, "frequence": frequence,
                    "temps_trajet": temps_trajet, # Nommé "temps_trajet" pour compatibilité core_logic
                    "temps_retournement_A": retournement_A, "temps_retournement_B": retournement_B, "reference_minutes": ref_minutes,
                    "passing_points": sorted(passing_points, key=lambda x: x['time_offset_min']),
                    "trajet_asymetrique": trajet_asymetrique,
                    "temps_trajet_retour": temps_trajet_retour,
                    "passing_points_retour": sorted(passing_points_retour, key=lambda x: x['time_offset_min']),
                    "pp_raw_text": mission.get("pp_raw_text", ""),
                    "type_materiel": type_materiel,
                    "inject_from_terminus_2": inject_t2
                }

    # --- SECTION 4/5: Mode Manuel (option, y compris en Calcul Énergie) ---
    if mode_generation == "Manuel":
        _manual_num = "5" if mode_calcul == "Calcul Energie" else "4"
        with st.expander(f"{_manual_num}. Construction manuelle des roulements", expanded=True):
            nombre_trains = st.number_input("Nombre de trains", 1, 30, len(st.session_state.roulement_manuel) or 1)

            current_ids = list(st.session_state.roulement_manuel.keys())
            for i in range(1, nombre_trains + 1):
                if i not in current_ids: st.session_state.roulement_manuel[i] = []
            for train_id in current_ids:
                if train_id > nombre_trains: del st.session_state.roulement_manuel[train_id]

            tab1, tab2 = st.tabs(["Édition des roulements", "Vue d'ensemble"])
            with tab1:
                st.subheader("Importer des roulements")
                uploaded_file = st.file_uploader(
                    "Sélectionner un fichier Excel de roulement",
                    type=['xlsx', 'xls'],
                    help="Format: Train | Début | Fin | Origine | Terminus"
                )
                if uploaded_file is not None:
                    if st.button("Importer et remplacer les roulements actuels"):
                        roulement, err = importer_roulements_fichier(uploaded_file, dataframe_gares)
                        if err:
                            st.error(err)
                        else:
                            st.session_state.roulement_manuel = roulement
                            st.success(f"Importation réussie. {len(roulement)} trains chargés.")
                            st.rerun()
                st.divider()

                for id_train in sorted(st.session_state.roulement_manuel.keys()):
                    with st.expander(f"Train {id_train}"):
                        if st.session_state.roulement_manuel[id_train]:
                            st.markdown("**Roulement actuel :**")
                            df_original = pd.DataFrame([{"Étape": i + 1, "Départ": e["depart"], "Heure départ": datetime.strptime(e["heure_depart"], "%H:%M").time(), "Arrivée": e["arrivee"], "Heure arrivée": e["heure_arrivee"], "Temps trajet (min)": e["temps_trajet"]} for i, e in enumerate(st.session_state.roulement_manuel[id_train])])

                            edited_df = st.data_editor(df_original, key=f"editor_{id_train}", hide_index=True, width='stretch',
                                column_config={"Heure départ": st.column_config.TimeColumn("Heure départ", format="HH:mm", step=60), "Étape": st.column_config.NumberColumn(disabled=True), "Départ": st.column_config.TextColumn(disabled=True), "Arrivée": st.column_config.TextColumn(disabled=True), "Heure arrivée": st.column_config.TextColumn(disabled=True), "Temps trajet (min)": st.column_config.NumberColumn(disabled=True)})

                            if st.button(f"Appliquer les modifications pour le Train {id_train}", key=f"apply_{id_train}"):
                                for i, row in edited_df.iterrows():
                                    if row["Heure départ"] != df_original.iloc[i]["Heure départ"]:
                                        roulement = st.session_state.roulement_manuel[id_train]
                                        dt_depart = datetime.combine(datetime.today(), row["Heure départ"])
                                        for j in range(i, len(roulement)):
                                            etape = roulement[j]
                                            if j > i:
                                                dt_arrivee_prec = datetime.strptime(roulement[j-1]["heure_arrivee"], "%H:%M")
                                                temps_ret = obtenir_temps_retournement_defaut(etape["depart"], st.session_state.missions)
                                                dt_depart = dt_arrivee_prec + timedelta(minutes=temps_ret)
                                            dt_arrivee = dt_depart + timedelta(minutes=etape["temps_trajet"])
                                            etape["heure_depart"] = dt_depart.strftime("%H:%M")
                                            etape["heure_arrivee"] = dt_arrivee.strftime("%H:%M")
                                        st.success(f"Horaires du train {id_train} mis à jour.")
                                        st.rerun()

                        st.markdown("**Ajouter une nouvelle étape :**")
                        derniere_etape = st.session_state.roulement_manuel[id_train][-1] if st.session_state.roulement_manuel[id_train] else None
                        if derniere_etape: st.info(f"Dernière position : {derniere_etape['arrivee']} à {derniere_etape['heure_arrivee']}")

                        add_cols = st.columns(2)

                        if derniere_etape:
                            idx_dep_defaut = gares_list.index(derniere_etape['arrivee'])
                            mission_precedente = trouver_mission_pour_od(derniere_etape['depart'], derniere_etape['arrivee'], st.session_state.missions)
                            if mission_precedente:
                                idx_arr_defaut = gares_list.index(mission_precedente['origine'])
                            else:
                                idx_arr_defaut = 0 if idx_dep_defaut + 1 >= len(gares_list) else idx_dep_defaut + 1
                        else:
                            idx_dep_defaut = gares_list.index(st.session_state.missions[0]['origine']) if st.session_state.missions else 0
                            idx_arr_defaut = gares_list.index(st.session_state.missions[0]['terminus']) if st.session_state.missions else 1

                        gare_dep = add_cols[0].selectbox(f"Gare départ (T{id_train})", gares_list, index=idx_dep_defaut, key=f"dep_g_{id_train}")
                        gare_arr = add_cols[0].selectbox(f"Gare arrivée (T{id_train})", gares_list, index=idx_arr_defaut, key=f"arr_g_{id_train}")

                        heure_dep_defaut = (datetime.strptime(derniere_etape['heure_arrivee'], "%H:%M") + timedelta(minutes=obtenir_temps_retournement_defaut(gare_dep, st.session_state.missions))).time() if derniere_etape and derniere_etape['arrivee'] == gare_dep else heure_debut_service
                        heure_dep = add_cols[1].time_input(f"Heure départ (T{id_train})", heure_dep_defaut, key=f"dep_t_{id_train}", step=60)
                        temps_traj_defaut = obtenir_temps_trajet_defaut_etape_manuelle(gare_dep, gare_arr, st.session_state.missions)
                        temps_traj = add_cols[1].number_input(f"Temps trajet (min) (T{id_train})", 1, 720, temps_traj_defaut, key=f"tt_m_{id_train}")

                        if st.button(f"Ajouter étape au train {id_train}", key=f"add_e_{id_train}"):
                            mission_associee = trouver_mission_pour_od(gare_dep, gare_arr, st.session_state.missions)
                            mission_key = json.dumps(mission_associee, sort_keys=True) if mission_associee else None

                            if mission_associee:
                                df_gares_json = st.session_state.gares.to_json(orient='records')
                                horaire_complet = construire_horaire_mission(mission_key, "aller", df_gares_json)
                            else:
                                horaire_complet = []

                            if horaire_complet and len(horaire_complet) > 1 and horaire_complet[-1]["time_offset_min"] > 0:
                                temps_total_mission = horaire_complet[-1]["time_offset_min"]

                                for k in range(len(horaire_complet) - 1):
                                    p1 = horaire_complet[k]
                                    p2 = horaire_complet[k+1]

                                    tps_p1_scaled = p1["time_offset_min"] * (temps_traj / temps_total_mission)
                                    tps_p2_scaled = p2["time_offset_min"] * (temps_traj / temps_total_mission)
                                    duree_segment_scaled = tps_p2_scaled - tps_p1_scaled

                                    dt_depart_etape = datetime.combine(datetime.today(), heure_dep) + timedelta(minutes=tps_p1_scaled)

                                    duree_arret = p1.get("duree_arret_min", 0)
                                    dt_depart_etape += timedelta(minutes=duree_arret)

                                    dt_arrivee_etape = dt_depart_etape + timedelta(minutes=duree_segment_scaled)

                                    st.session_state.roulement_manuel[id_train].append({
                                        "depart": p1["gare"],
                                        "heure_depart": dt_depart_etape.strftime("%H:%M"),
                                        "arrivee": p2["gare"],
                                        "heure_arrivee": dt_arrivee_etape.strftime("%H:%M"),
                                        "temps_trajet": int(duree_segment_scaled)
                                    })
                            else:
                                dt_depart = datetime.combine(datetime.today(), heure_dep)
                                dt_arrivee = dt_depart + timedelta(minutes=temps_traj)
                                st.session_state.roulement_manuel[id_train].append({"depart": gare_dep, "heure_depart": dt_depart.strftime("%H:%M"), "arrivee": gare_arr, "heure_arrivee": dt_arrivee.strftime("%H:%M"), "temps_trajet": temps_traj})

                            st.session_state.roulement_manuel[id_train].sort(key=lambda x: datetime.strptime(x['heure_depart'], "%H:%M"))
                            st.rerun()

                        if st.session_state.roulement_manuel[id_train]:
                            if st.button(f"Supprimer dernière étape du train {id_train}", key=f"del_e_{id_train}", type="secondary"):
                                st.session_state.roulement_manuel[id_train].pop()
                                st.rerun()

            with tab2:
                all_etapes = [{"Train": id_t, "Étape": i+1, "Départ": e['depart'], "Heure départ": e['heure_depart'], "Arrivée": e['arrivee'], "Heure arrivée": e['heure_arrivee'], "Temps trajet (min)": e['temps_trajet']} for id_t, etapes in st.session_state.roulement_manuel.items() for i, e in enumerate(etapes)]
                if all_etapes:
                    st.dataframe(pd.DataFrame(all_etapes), width="stretch")
                    st.download_button("Télécharger (CSV)", pd.DataFrame(all_etapes).to_csv(index=False, sep=';').encode('utf-8-sig'), "config_roulements.csv", "text/csv")


    # --- SECTION 4 (Alternative): Paramètres Énergétiques (uniquement en mode Energie) ---
    if mode_calcul == "Calcul Energie":
        with st.expander("4. Paramètres Énergétiques", expanded=True):

            # Récupérer les types de matériel uniques utilisés dans les missions définies
            types_materiel_utilises = set(
                m.get("type_materiel", "diesel") for m in st.session_state.missions if m.get("type_materiel")
            )

            if not types_materiel_utilises:
                st.info("Définissez des missions à la section 3 pour configurer les paramètres énergétiques associés.")
            else:
                st.info("Ajustez les caractéristiques physiques pour les types de matériel sélectionnés dans vos missions.")

                types_materiel_tries = sorted(list(types_materiel_utilises))
                # Utilise la map pour afficher les noms capitalisés dans les onglets
                tabs = st.tabs([options_materiel_map.get(t, t).capitalize() for t in types_materiel_tries])

                for i, type_mat in enumerate(types_materiel_tries):
                    with tabs[i]:
                        # Utiliser get pour éviter KeyError si type_mat n'existe pas encore
                        # Utiliser default_params.copy() pour éviter modification par référence
                        params = st.session_state.energy_params.setdefault(type_mat, get_default_energy_params().copy())

                        st.markdown(f"**Caractéristiques Générales [{type_mat}]**")
                        c1, c2 = st.columns(2)
                        params["masse_tonne"] = c1.number_input(
                            f"Masse (tonnes)", 10, 1000,
                            value=params.get("masse_tonne", 100),
                            key=f"masse_{type_mat}"
                        )
                        params["facteur_aux_kwh_h"] = c1.number_input(
                            f"Puissance Auxiliaires (kW)", 0.0, 500.0,
                            value=float(params.get("facteur_aux_kwh_h", 50.0)),
                            step=1.0, key=f"f_aux_{type_mat}"
                        )
                        params["taux_aux_terminus_pct"] = c1.slider(
                            "Taux d'utilisation des auxiliaires au terminus",
                            min_value=0,
                            max_value=100,
                            value=int(params.get("taux_aux_terminus_pct", 100)),
                            key=f"taux_aux_term_{type_mat}",
                            help="Pourcentage de la puissance auxiliaires consommé lorsque le train est immobilisé au terminus (sans voyageurs). 100 % = même consommation qu'en ligne ; 0 % = auxiliaires coupés au terminus.",
                        )

                        # Affichage conditionnel des paramètres batterie dans la 2e colonne
                        if type_mat == "batterie":
                            params["capacite_batterie_kwh"] = c2.number_input(
                                f"Capacité nominale (kWh)", 100, 10000,
                                value=params.get("capacite_batterie_kwh", 600),
                                key=f"cap_batt_{type_mat}"
                            )
                            params["facteur_charge_C"] = c2.number_input(
                                f"Charge (XC)", 1.0, 10.0,
                                value=float(params.get("facteur_charge_C", 4.0)),
                                step=0.1, key=f"f_charge_c_{type_mat}",
                                help="Puissance de charge max = XC * Capacité kWh"
                            )

                            st.markdown(f"**Plage d'utilisation & Vieillissement [{type_mat}]**")
                            cb1, cb2 = st.columns(2)
                            params["simuler_fin_de_vie"] = cb1.checkbox(
                                "Simuler batterie fin de vie",
                                value=params.get("simuler_fin_de_vie", False),
                                key=f"eol_check_{type_mat}",
                                help="Réduit la capacité effective selon le pourcentage de fin de vie"
                            )
                            if params["simuler_fin_de_vie"]:
                                params["capacite_eol_pct"] = cb2.slider(
                                    "Capacité fin de vie (%)",
                                    min_value=50, max_value=99,
                                    value=params.get("capacite_eol_pct", 80),
                                    key=f"eol_pct_{type_mat}",
                                    help="Pourcentage de la capacité nominale conservée en fin de vie"
                                )
                                cap_eol = params["capacite_batterie_kwh"] * params["capacite_eol_pct"] / 100
                                cb2.caption(f"Capacité effective : {cap_eol:.0f} kWh")
                            else:
                                params["capacite_eol_pct"] = params.get("capacite_eol_pct", 80)

                            cs1, cs2 = st.columns(2)
                            params["soc_min_pct"] = cs1.slider(
                                "SoC minimum (%)", min_value=0, max_value=40,
                                value=params.get("soc_min_pct", 20),
                                key=f"soc_min_{type_mat}",
                                help="Niveau de décharge minimum autorisé"
                            )
                            params["soc_max_pct"] = cs2.slider(
                                "SoC maximum (%)", min_value=60, max_value=100,
                                value=params.get("soc_max_pct", 95),
                                key=f"soc_max_{type_mat}",
                                help="Niveau de charge maximum (limiter prolonge la durée de vie)"
                            )


                        st.markdown(f"**Performance Physique [{type_mat}]**")
                        c3, c4 = st.columns(2)
                        params["accel_ms2"] = c3.number_input(
                            f"Accélération (m/s²)", 0.1, 2.0,
                            value=float(params.get("accel_ms2", 0.5)),
                            step=0.05, format="%.2f", key=f"accel_{type_mat}"
                        )
                        params["decel_ms2"] = c4.number_input(
                            f"Décélération (m/s²)", 0.1, 2.5,
                            value=float(params.get("decel_ms2", 0.8)),
                            step=0.05, format="%.2f", key=f"decel_{type_mat}"
                        )

                        st.markdown(f"**Équation de Davis (Résistance en N par tonne) [{type_mat}]**")
                        c5, c6, c7 = st.columns(3)
                        params["davis_A_N_t"] = c5.number_input(
                            "Coeff. A (N/t)", 0.0, 200.0,
                            value=float(params.get("davis_A_N_t", 20.0)),
                            format="%.2f", key=f"f_davis_a_{type_mat}",
                            help="Résistance mécanique"
                        )
                        params["davis_B_N_t_kph"] = c6.number_input(
                            "Coeff. B (N/t/kph)", 0.0, 5.0,
                            value=float(params.get("davis_B_N_t_kph", 0.5)),
                            format="%.3f", key=f"f_davis_b_{type_mat}",
                            help="Résistance roulements"
                        )
                        params["davis_C_N_t_kph2"] = c7.number_input(
                            "Coeff. C (N/t/kph²)", 0.0, 0.1,
                            value=float(params.get("davis_C_N_t_kph2", 0.005)),
                            format="%.4f", key=f"f_davis_c_{type_mat}",
                            help="Résistance aéro."
                        )

                        st.markdown(f"**Rendements et Spécificités [{type_mat}]**")

                        # Ligne Thermique (Conditionnelle)
                        if type_mat in ["diesel", "bimode"]:
                            c8_thermique, c9_thermique = st.columns(2)
                            with c8_thermique:
                                params["rendement_thermique_pct"] = st.slider(
                                    f"Rdt. Thermique (%)", 10, 60,
                                    value=params.get("rendement_thermique_pct", 38),
                                    key=f"rend_therm_{type_mat}"
                                )
                            with c9_thermique:
                                 params["kwh_per_liter_diesel"] = st.number_input(
                                     f"Équiv. Carburant (kWh/L)", 5.0, 15.0,
                                     value=float(params.get("kwh_per_liter_diesel", 10.0)),
                                     step=0.1, key=f"f_kwh_l_{type_mat}"
                                 )

                        # Ligne Électrique (Conditionnelle)
                        if type_mat in ["electrique", "bimode", "batterie"]:
                            c8_electrique, c9_electrique = st.columns(2)
                            with c8_electrique:
                                 params["rendement_electrique_pct"] = st.slider(
                                    f"Rdt. Électrique (%)", 50, 100,
                                    value=params.get("rendement_electrique_pct", 88),
                                    key=f"rend_elec_{type_mat}"
                                )
                            with c9_electrique:
                                 params["recuperation_pct"] = st.slider(
                                    f"Efficacité récupération (%)", 0, 100,
                                    value=params.get("recuperation_pct", 65),
                                    key=f"recup_{type_mat}"
                                )

                        # Sauvegarde directe dans l'état de session
                        st.session_state.energy_params[type_mat] = params


    # --- SECTION 5: Vérification de Fréquence (mode Manuel) ---
    if mode_generation == "Manuel" and any(st.session_state.roulement_manuel.values()):
        _freq_num = "5b" if mode_calcul == "Calcul Energie" else "5"
        with st.expander(f"{_freq_num}. Vérification de cohérence des fréquences", expanded=True):
            analyses = analyser_frequences_manuelles(st.session_state.roulement_manuel, st.session_state.missions, heure_debut_service, heure_fin_service)
            for mission_key, resultat in analyses.items():
                st.subheader(f"Analyse pour: {mission_key}")
                if resultat["df"] is not None and not resultat["df"].empty:
                    st.dataframe(resultat["df"], width="stretch")
                    if resultat["conformite"] == 100: st.success(f"✅ Objectif respecté à 100%.")
                    elif resultat["conformite"] >= 75: st.warning(f"⚠️ Objectif respecté à {resultat['conformite']:.1f}%")
                    else: st.error(f"❌ Objectif respecté seulement à {resultat['conformite']:.1f}%")


# =============================================================================
# SECTION : PARAMÈTRES D'OPTIMISATION AVANCÉE
# =============================================================================

if st.session_state.gares is not None and st.session_state.missions:
    st.markdown("---")
    if mode_generation != "Manuel":
      with st.expander("5. ⚙️ Configuration de l'Optimisation", expanded=True):

        st.markdown("""
        **Sélectionnez un mode d'optimisation** pour générer les horaires de manière intelligente.
        Le choix d'un mode est maintenant **obligatoire** pour garantir des résultats optimaux.
        """)

        # Le mode d'optimisation est maintenant OBLIGATOIRE (pas de checkbox)
        use_advanced_optimization = True  # Toujours activé

        if True:  # Toujours vrai
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Mode d'optimisation")

                optimization_mode = st.selectbox(
                    "Algorithme",
                    ["simple", "fast", "smart_progressive", "exhaustif", "genetic"],
                    index=0,  # Simple par défaut
                    format_func=lambda x: {
                        "simple": "🎯 Simple - Simulation directe (respect strict paramètres)",
                        "fast": "⚡ Fast - Ultra rapide avec logique optimisée (pas de 10 min)",
                        "smart_progressive": "🎯🔍 Smart Progressive - Affinement intelligent",
                        "exhaustif": "🔍 Exhaustif",
                        "genetic": "🧬 Génétique - Évolutionnaire"
                    }[x],
                    help="""
                    • Simple : Simulation directe (fort respect temps retournement - TRÈS EFFICACE)
                    • Fast : Recherche rapide intégrant les forces du mode simple
                    • Smart Progressive : Recherche progressive optimisée (10min → 1min) - RECOMMANDÉ
                    • Exhaustif
                    • Génétique : Algorithme amélioré pour grandes instances

                    """
                )

                # Afficher des informations sur le mode sélectionné
                if optimization_mode == "simple":
                    st.success("""
                    🎯 **Mode Simple** :

                    - Utilise directement les temps de retournement configurés
                    - **TRÈS EFFICACE** : Produit souvent de meilleurs graphiques que les algos complexes
                    - Contrôle total sur les paramètres
                    - Exécution ultra-rapide (< 1 seconde)

                    **Forces du mode Simple** :
                    - ✅ Respect strict des contraintes de retournement
                    - ✅ Gestion intelligente des conflits
                    - ✅ Graphiques propres et réguliers
                    """)
                elif optimization_mode == "smart_progressive":
                    st.info("""
                    🎯 **Mode Smart Progressive**  :

                    1. **Phase 1** : Recherche grossière (pas de 10 min) → Identifier la zone prometteuse
                    2. **Phase 2** : Affinement moyen (pas de 5 min) → Resserrer la recherche
                    3. **Phase 3** : Affinement fin (pas de 2 min) → Préciser
                    4. **Phase 4** : Recherche fine (pas de 1 min) → Optimum local
                    """)
                elif optimization_mode == "fast":
                    st.warning("""
                    ⚡ **Mode Fast** :
                    - Pas de recherche : 10 minutes
                    - Recommandé pour : tests rapides, grandes instances (7+ missions)
                    """)
                elif optimization_mode == "exhaustif":
                    st.warning("""
                    🔍 **Mode Exhaustif**:
                    - Garantit la solution optimale
                    - Intègre la logique optimisée du mode Simple
                    - **ATTENTION** : Très lent pour > 3-4 missions
                    - Temps de calcul exponentiel
                    - Recommandé uniquement pour petites instances
                    """)

                # Paramètres spécifiques au mode génétique
                elif optimization_mode == "genetic":
                    st.info("Paramètres de l'algorithme génétique (AMÉLIORÉS)")

                    col1a, col1b = st.columns(2)
                    with col1a:
                        population_size = st.number_input(
                            "Taille population",
                            min_value=10, max_value=200, value=50, step=10,
                            help="Nombre d'individus par génération"
                        )
                    with col1b:
                        generations = st.number_input(
                            "Générations",
                            min_value=10, max_value=500, value=100, step=10,
                            help="Nombre d'itérations"
                        )

                    col1c, col1d = st.columns(2)
                    with col1c:
                        mutation_rate = st.slider(
                            "Taux mutation",
                            min_value=0.0, max_value=0.5, value=0.20, step=0.05,
                            help="Valeur optimisée : 0.20"
                        )
                    with col1d:
                        crossover_rate = st.slider(
                            "Taux croisement",
                            min_value=0.3, max_value=1.0, value=0.85, step=0.05,
                            help="Valeur optimisée : 0.85"
                        )

            with col2:
                st.subheader("Optimisation des croisements")

                is_genetic = optimization_mode == "genetic"
                enable_crossing_opt = st.checkbox(
                    "Activer l'optimisation des croisements",
                    value=True if is_genetic else False,
                    disabled=is_genetic,
                    help=(
                        "Verrouillé en mode génétique : l'algorithme optimise désormais "
                        "les croisements par paire de trains."
                        if is_genetic else
                        "Prolonge stratégiquement les arrêts pour améliorer les croisements"
                    ),
                )

                if enable_crossing_opt:
                    st.info("Paramètres des croisements")

                    max_delay = st.number_input(
                        "Délai maximum (minutes)",
                        min_value=1, max_value=30, value=5, step=1,
                        help="Durée maximale de prolongement d'un arrêt à une gare de croisement (VE)"
                    )

                    delay_penalty = st.slider(
                        "Pénalité par minute de retard",
                        min_value=1.0, max_value=10.0, value=2.0, step=0.5,
                        help="Poids de la pénalité dans le score"
                    )

                    st.caption("""
                    💡 **Fonctionnement** :
                    L'algorithme peut prolonger un arrêt à une voie d'évitement pour :
                    • Éviter un conflit avec un autre train
                    • Créer un croisement plus efficace
                    • Optimiser le flux global

                    ⚠️ **Garantie** : Aucune violation d'infrastructure ne sera créée
                    """)

            st.markdown("---")
            st.subheader("Partage des rames")
            col_share1, col_share2 = st.columns(2)
            with col_share1:
                allow_sharing = st.checkbox(
                    "Autoriser le partage des rames entre missions",
                    value=True,
                    key="allow_sharing_checkbox",
                    help="""
                    Si activé, les rames peuvent être réutilisées entre missions du MÊME type de matériel :
                    - ✅ Diesel avec Diesel
                    - ✅ Batterie avec Batterie
                    - ✅ Électrique avec Électrique
                    - ✅ Bimode avec Bimode
                    - ❌ JAMAIS entre types différents (ex: Diesel + Batterie)

                    Cela réduit le nombre total de rames nécessaires.
                    """
                )
            with col_share2:
                if st.session_state.missions:
                    nb_missions = len([m for m in st.session_state.missions if m.get("frequence", 0) > 0])
                    if nb_missions > 0:
                        if allow_sharing:
                            st.info("📊 Partage activé : nombre de rames optimisé")
                        else:
                            st.warning(f"⚠️ Sans partage : environ {nb_missions * 2}+ rames nécessaires")

            # Récapitulatif
            st.markdown("---")
            st.subheader("📋 Récapitulatif de la configuration")

            col_recap1, col_recap2, col_recap3, col_recap4 = st.columns(4)

            with col_recap1:
                st.metric("Mode", optimization_mode.upper())
                if optimization_mode == "genetic":
                    st.caption(f"Pop: {population_size}, Gen: {generations}")

            with col_recap2:
                if enable_crossing_opt:
                    st.metric("Croisements", "✅ Activé")
                    st.caption(f"Max: {max_delay} min, Pénalité: {delay_penalty}")
                else:
                    st.metric("Croisements", "❌ Désactivé")

            with col_recap3:
                st.metric("Partage rames", "✅ Activé" if allow_sharing else "❌ Désactivé")

            with col_recap4:
                _opt_params_est = {}
                if optimization_mode == "genetic":
                    _opt_params_est = {"population_size": population_size, "generations": generations}
                time_est = estimer_temps_calcul(
                    st.session_state.missions,
                    heure_debut_service,
                    heure_fin_service,
                    mode=optimization_mode,
                    optimization_params=_opt_params_est,
                )
                st.metric("Temps estimé", time_est)

            # Sauvegarder dans session_state
            st.session_state.optimization_mode = optimization_mode
            st.session_state.use_advanced_optimization = True  # Toujours True maintenant

            if optimization_mode == "genetic":
                st.session_state.genetic_params = {
                    'population_size': population_size,
                    'generations': generations,
                    'mutation_rate': mutation_rate,
                    'crossover_rate': crossover_rate
                }

            if enable_crossing_opt:
                st.session_state.crossing_params = {
                    'max_delay': max_delay,
                    'penalty': delay_penalty
                }
            else:
                st.session_state.crossing_params = None

    # L'optimisation est maintenant toujours active - pas besoin de "else"

    # --- SECTION 6: Calcul et Affichage ---
    with st.expander("6. Calcul et Affichage", expanded=True):
        dt_debut_s = datetime.combine(datetime.min, heure_debut_service)
        dt_fin_s = datetime.combine(datetime.min, heure_fin_service)
        duree_heures_s = (dt_fin_s - dt_debut_s).total_seconds() / 3600
        if duree_heures_s <= 0: duree_heures_s += 24
        fenetre_heures = st.number_input("Durée de la fenêtre (h)", 1.0, duree_heures_s, min(5.0, duree_heures_s))
        decalage_heures = st.slider("Début de la fenêtre (h)", 0.0, max(0.0, duree_heures_s - fenetre_heures), 0.0, 0.5)
        params_affichage = {
            "duree_fenetre": fenetre_heures,
            "decalage_heure": decalage_heures,
            "duree_service": duree_heures_s,
        }

        allow_sharing = st.session_state.get("allow_sharing_checkbox", True)
        if mode_generation == "Manuel":
            if not any(st.session_state.roulement_manuel.values()):
                st.warning("Définissez au moins une étape de roulement dans la section « Construction manuelle des roulements ».")

        estimation = "N/A"
        if mode_generation != "Manuel" and (
            mode_generation == "Rotation optimisée" or mode_calcul == "Calcul Energie"
        ):
            _mode_est = st.session_state.get("optimization_mode", "simple")
            _params_est = st.session_state.get("genetic_params", {}) if _mode_est == "genetic" else {}
            estimation = estimer_temps_calcul(
                st.session_state.missions,
                heure_debut_service,
                heure_fin_service,
                mode=_mode_est,
                optimization_params=_params_est,
            )

        if estimation != "N/A":
            st.caption(f"⏱️ Temps de calcul estimé : **{estimation}**")

        # Map pour lier les trains à leurs missions (nécessaire pour le plotting physique)
        missions_par_train = {}

        if "run_calculation" not in st.session_state:
            st.session_state.run_calculation = False

        if st.button("Generer le graphique horaire", type="primary", key="btn_gen_graph"):
            st.session_state.run_calculation = True
            st.session_state.energy_cache_key = None

        if st.session_state.get("chronologie_calculee") and mode_calcul == "Calcul Energie":
            if st.button("Recalculer l energie uniquement", help="Garde la grille horaire, recalcule conso / batterie avec le materiel actuel.", key="btn_recalc_energie"):
                st.session_state.run_energy_only = True
                st.session_state.energy_cache_key = None

        if st.session_state.run_calculation:
            st.session_state.run_calculation = False

            # Déterminer si on utilise l'optimisation avancée
            use_advanced = st.session_state.get('use_advanced_optimization', False)

            if mode_generation == "Manuel":
                with st.spinner("Génération du graphique à partir des roulements manuels..."):
                    try:
                        if not any(st.session_state.roulement_manuel.values()):
                            st.error("Aucun roulement manuel défini. Ajoutez des étapes (ou importez un Excel) avant de générer.")
                        else:
                            chronologie = preparer_roulement_manuel(st.session_state.roulement_manuel)
                            warnings = {}
                            from core_logic import _calculer_stats_homogeneite
                            stats_homogeneite = _calculer_stats_homogeneite(chronologie)
                            st.session_state.chronologie_calculee = chronologie
                            st.session_state.warnings_calcul = warnings
                            st.session_state.stats_homogeneite = stats_homogeneite
                            st.success("✅ Graphique généré à partir des roulements manuels.")
                    except Exception as e:
                        st.error(f"Une erreur est survenue lors de la génération du graphique : {e}")
                        st.session_state.chronologie_calculee = None
                        st.session_state.warnings_calcul = {}
                        st.session_state.stats_homogeneite = {}
                        st.stop()

            elif use_advanced:
                # =====================================================================
                # MODE OPTIMISATION AVANCÉE
                # =====================================================================

                st.info(f"Mode d'optimisation : **{optimization_mode.upper()}**")

                # Configuration de l'optimisation
                crossing_params = st.session_state.get('crossing_params') or {}
                ui_max_delay = int(crossing_params.get('max_delay', 5)) if enable_crossing_opt else 15
                ui_penalty = float(crossing_params.get('penalty', 2.0)) if enable_crossing_opt else 2.0

                config = OptimizationConfig(
                    mode=optimization_mode,
                    crossing_optimization=CrossingOptimization(
                        enabled=enable_crossing_opt,
                        max_delay_minutes=ui_max_delay,
                        penalty_per_minute=ui_penalty,
                    ),
                    population_size=100 if optimization_mode == "genetic" else 50,
                    generations=150 if optimization_mode == "genetic" else 100,
                    use_parallel=True,  # Activer la parallélisation
                    num_workers=None,  # Auto-detect
                    turnaround_max_buffer=max(30, ui_max_delay + 10),
                )

                # Estimation initiale
                total_events_estimate = 0
                for mission in st.session_state.missions:
                    if mission.get("frequence", 0) > 0:
                        freq_per_hour = mission["frequence"]
                        total_events_estimate += int((heure_fin_service.hour - heure_debut_service.hour) * freq_per_hour)

                if optimization_mode == "genetic":
                    total_work = config.population_size * config.generations
                elif optimization_mode == "exhaustif":
                    # Compter les combinaisons possibles
                    mission_retours = []
                    for mission in st.session_state.missions:
                        mission_retour_id = f"{mission['terminus']}→{mission['origine']}"
                        has_return = any(
                            f"{m['origine']}→{m['terminus']}" == mission_retour_id
                            for m in st.session_state.missions
                        )
                        if has_return:
                            mission_retours.append(12)  # 12 options par pas de 5 minutes

                    total_work = 1
                    for count in mission_retours:
                        total_work *= count
                else:
                    total_work = total_events_estimate

                # Interface de progression
                progress_container = st.empty()
                stats_container = st.empty()

                with progress_container.container():
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    eta_text = st.empty()
                    status_text.text("🔄 Initialisation de l'optimisation...")

                # Tracker de progression
                tracker = ProgressTracker(total_work)

                def progress_callback(current, total, best_score, num_rames=0, delay=0):
                    """Callback pour mise à jour dynamique."""
                    tracker.update(1)

                    # Synchronise le total réel communiqué par l'optimiseur (peut être
                    # très différent de l'estimation initiale, notamment en smart_progressive).
                    if total > 0 and total != tracker.total_work:
                        tracker.total_work = total
                        tracker.work_done = current  # aligner work_done sur le vrai compteur

                    if total > 0:
                        progress = min(100, int((current / total) * 100))
                    else:
                        progress = tracker.get_progress_percent()
                    progress_bar.progress(progress)

                    # Message de statut
                    if best_score == float('inf'):
                        status_msg = f"⏳ Recherche en cours ({current}/{total})..."
                    else:
                        if optimization_mode == "genetic":
                            status_msg = f"🧬 Génération {current}/{total} | Score: {best_score:.0f} | Rames: {num_rames}"
                        elif optimization_mode == "exhaustif":
                            status_msg = f"🔍 Test {current}/{total} | Score: {best_score:.0f} | Rames: {num_rames}"
                        else:
                            status_msg = f"⚡ Optimisation en cours ({current}/{total})"

                    status_text.text(status_msg)

                    # Estimation temps restant
                    eta = tracker.get_eta()
                    elapsed = tracker.get_elapsed_time()
                    eta_text.text(f"⏱️ Écoulé: {elapsed} | Restant: {eta}")

                # Lancement de l'optimisation
                try:
                    start_time = time.time()

                    chronologie, warnings, optim_stats = optimiser_graphique_horaire(
                        st.session_state.missions,
                        st.session_state.gares,
                        heure_debut_service,
                        heure_fin_service,
                        config=config,
                        allow_sharing=allow_sharing,
                        progress_callback=progress_callback
                    )

                    elapsed_time = time.time() - start_time

                    # Calibration adaptative : mémorise le temps réel pour la prochaine estimation
                    st.session_state["last_calcul_info"] = {
                        "mode": optimization_mode,
                        "n_events": total_events_estimate,
                        "elapsed_s": elapsed_time,
                    }

                    progress_bar.progress(100)  # Garanti d'être dans [0, 100] grâce à get_progress_percent
                    status_text.text(f"✅ Optimisation terminée !")
                    eta_text.text(f"⏱️ Temps total: {elapsed_time:.1f} secondes")

                    # Vérification de sécurité
                    infra_violations = warnings.get("infra_violations", [])
                    if infra_violations:
                        st.error("❌ ERREUR CRITIQUE : Des violations d'infrastructure ont été détectées !")
                        st.error("Cela ne devrait JAMAIS arriver. Veuillez signaler ce bug.")
                        for violation in infra_violations:
                            st.write(f"  - {violation}")
                    else:
                        num_trains = sum(len(trajets) for trajets in chronologie.values())
                        st.success(f"✨ Solution optimale trouvée ! {len(chronologie)} rames, {num_trains} trajets, SANS violation.")

                    # Afficher les statistiques d'optimisation (UNE SEULE FOIS)
                    with stats_container.container():
                        st.subheader("📊 Statistiques d'optimisation")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Mode", optimization_mode.upper())

                        with col2:
                            if optimization_mode == "genetic":
                                gens_completed = optim_stats.get('generations', 0)
                                st.metric("Générations", f"{gens_completed}/{config.generations}")
                            elif optimization_mode == "exhaustif":
                                tested = optim_stats.get('combinations_tested', 0)
                                st.metric("Combinaisons testées", f"{tested:,}")
                            else:
                                st.metric("Stratégie", "Heuristique rapide")

                        with col3:
                            final_score = optim_stats.get('final_score', optim_stats.get('best_score', 'N/A'))
                            if final_score != 'N/A' and final_score != float('inf'):
                                st.metric("Score final", f"{final_score:.0f}")

                        # Graphique d'évolution pour génétique
                        if optimization_mode == "genetic" and 'best_score_history' in optim_stats:
                            with st.expander("📈 Évolution de l'optimisation", expanded=False):
                                import matplotlib.pyplot as plt

                                history = optim_stats['best_score_history']

                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.plot(range(len(history)), history, linewidth=2, color='green', marker='o', markersize=3)
                                ax.set_xlabel('Génération')
                                ax.set_ylabel('Meilleur Score')
                                ax.set_title('Convergence de l\'algorithme génétique')
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig)

                        # Stats pour exhaustif
                        elif optimization_mode == "exhaustif":
                            valid_count = optim_stats.get('valid_combinations', 0)
                            tested_count = optim_stats.get('combinations_tested', 0)
                            if tested_count > 0:
                                success_rate = (valid_count / tested_count) * 100
                                st.info(f"Taux de solutions valides : {success_rate:.1f}% ({valid_count}/{tested_count})")

                    # Sauvegarder les résultats
                    st.session_state.chronologie_calculee = chronologie
                    st.session_state.warnings_calcul = warnings

                    # Calculer les vraies stats d'homogénéité par mission et sens
                    from core_logic import _calculer_stats_homogeneite
                    stats_gini = _calculer_stats_homogeneite(chronologie)
                    st.session_state.stats_homogeneite = stats_gini

                except Exception as e:
                    st.error(f"❌ Erreur lors de l'optimisation : {e}")
                    import traceback
                    with st.expander("Détails de l'erreur"):
                        st.code(traceback.format_exc())

            else:
                # =====================================================================
                # MODE STANDARD (code existant)
                # =====================================================================

                with st.spinner("Génération du graphique en cours..."):
                    try:
                        _t0_standard = time.time()
                        chronologie, warnings, stats_homogeneite = generer_tous_trajets_optimises(
                            st.session_state.missions,
                            st.session_state.gares,
                            heure_debut_service,
                            heure_fin_service,
                            allow_sharing=allow_sharing,
                            search_strategy='smart'
                        )

                        _elapsed_standard = time.time() - _t0_standard
                        _n_std = _compter_trains_initiaux(st.session_state.missions, heure_debut_service, heure_fin_service)
                        st.session_state["last_calcul_info"] = {
                            "mode": "simple",
                            "n_events": _n_std,
                            "elapsed_s": _elapsed_standard,
                        }

                        st.session_state.chronologie_calculee = chronologie
                        st.session_state.warnings_calcul = warnings
                        st.session_state.stats_homogeneite = stats_homogeneite

                        st.success(f"✅ Graphique généré avec succès ! ({_elapsed_standard:.1f}s)")

                    except Exception as e:
                        st.error(f"Une erreur est survenue lors de la génération du graphique : {e}")
                        st.session_state.chronologie_calculee = None
                        st.session_state.warnings_calcul = {}
                        st.session_state.stats_homogeneite = {}
                        st.stop()

        # Affichage des résultats si un calcul a été fait
        if st.session_state.chronologie_calculee:
            chronologie = st.session_state.chronologie_calculee
            warnings = st.session_state.warnings_calcul

            # Calcul des statistiques sur les rames utilisées
            if chronologie:
                nb_rames_total = len(set(chronologie.keys()))

                st.subheader("Statistiques d'utilisation des rames")

                # Préparation des données pour les stats
                gare_dist_map = {row['gare']: row['distance'] for _, row in st.session_state.gares.iterrows()}

                nb_trajets_od = 0
                total_km_parcourus = 0

                for train_id, trajets in chronologie.items():
                    for t in trajets:
                        # 1. Calcul des trajets OD (un trajet = une mission complète)
                        # Dans le mode optimisé, is_mission_start est True uniquement au début de la mission
                        # Dans le mode manuel, on considère chaque étape saisie comme un trajet (is_mission_start par défaut True)
                        # On ignore aussi les arrêts techniques (origine == terminus)
                        if t.get('is_mission_start', True) and t['origine'] != t['terminus']:
                            nb_trajets_od += 1

                        # 2. Calcul des km (tous les segments comptent, même si ce n'est pas un début de mission)
                        if t['origine'] != t['terminus'] and t['origine'] in gare_dist_map and t['terminus'] in gare_dist_map:
                            dist = abs(gare_dist_map[t['terminus']] - gare_dist_map[t['origine']])
                            total_km_parcourus += dist

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric(
                        label="Rames totales utilisées",
                        value=nb_rames_total,
                        help="Nombre total de rames différentes mobilisées"
                    )

                with col_stat2:
                    st.metric(
                        label="Trajets (OD) réalisés",
                        value=nb_trajets_od,
                        help="Nombre total de trajets (missions) réalisés"
                    )

                with col_stat3:
                    if nb_rames_total > 0:
                        km_moyen = total_km_parcourus / nb_rames_total
                        st.metric(
                            label="Km moyen / rame",
                            value=f"{km_moyen:.1f} km",
                            help="Kilométrage moyen parcouru par rame sur la journée"
                        )

            # --- Calcul énergétique (fait ici pour pouvoir remonter les erreurs) ---
            all_energy_errors = []
            resultats_energie_par_train = st.session_state.get("resultats_energie_par_train") or {}
            missions_par_train = st.session_state.get("missions_par_train") or {}

            if mode_calcul == "Calcul Energie":
                energy_key = fingerprint_energie(
                    st.session_state.energy_params,
                    st.session_state.missions,
                    chronologie,
                    dataframe_gares,
                )
                must_recompute = (
                    st.session_state.get("run_energy_only")
                    or st.session_state.get("energy_cache_key") != energy_key
                )
                if must_recompute:
                    with st.spinner("Calcul de la consommation energetique..."):
                        resultats_energie_par_train, missions_par_train, all_energy_errors = (
                            calculer_energie_flotte(
                                chronologie,
                                st.session_state.missions,
                                dataframe_gares,
                                st.session_state.energy_params,
                            )
                        )
                    st.session_state.resultats_energie_par_train = resultats_energie_par_train
                    st.session_state.missions_par_train = missions_par_train
                    st.session_state.energy_errors = all_energy_errors
                    st.session_state.energy_cache_key = energy_key
                    st.session_state.run_energy_only = False
                    for msg in all_energy_errors:
                        if "Impossible de trouver la mission" in msg:
                            st.warning(msg)
                else:
                    all_energy_errors = st.session_state.get("energy_errors") or []
                    resultats_energie_par_train = st.session_state.resultats_energie_par_train
                    missions_par_train = st.session_state.missions_par_train

            # --- Affichage des Avertissements (y compris les erreurs d'énergie) ---
            infra_violations = warnings.get("infra_violations", [])
            other_warns = warnings.get("other", [])
            energy_errs = st.session_state.get("energy_errors", [])

            total_warnings = len(infra_violations) + len(other_warns) + len(energy_errs)

            if total_warnings > 0:
                with st.expander(f"⚠️ {total_warnings} avertissement(s) généré(s) - Cliquez pour voir le détail"):
                    if infra_violations:
                        st.error(f"**{len(infra_violations)} VIOLATION(S) DE CONTRAINTE D'INFRASTRUCTURE**")
                        for w in infra_violations:
                            st.write(f"- {w}")
                        st.markdown("---")

                    if energy_errs:
                        st.error(f"**{len(energy_errs)} ERREUR(S) DE SIMULATION ÉNERGÉTIQUE**")
                        for err in energy_errs:
                            st.write(f"- {err}")
                        st.markdown("---")

                    if other_warns:
                        st.warning(f"**{len(other_warns)} autre(s) avertissement(s)**")
                        for w in other_warns:
                            st.write(f"- {w}")


            # --- Affichage Homogénéité (PAR MISSION ET SENS) ---
            if st.session_state.stats_homogeneite:
                # Vérifier que c'est bien un dict de stats Gini et pas les stats d'optim
                stats = st.session_state.stats_homogeneite

                # Filtrer pour n'avoir que les missions (pas les stats d'optim)
                mission_stats = {}
                for key, val in stats.items():
                    if isinstance(val, (int, float)) and '→' in str(key):
                        mission_stats[key] = val

                if mission_stats:
                    st.subheader("📈 Qualité du Cadencement (par mission et sens)")
                    st.info("Indice d'homogénéité (Gini inverse) : 1.00 = Intervalles parfaitement réguliers.")

                    # Organiser par couple OD (Origine-Destination sans ordre)
                    missions_organisees = {}
                    for mission_key, gini_val in mission_stats.items():
                        # Extraire origine et terminus
                        parts = mission_key.split(' → ')
                        if len(parts) == 2:
                            origine, terminus = parts
                            # Clé unique pour le couple de gares (ordre alphabétique pour regrouper A->B et B->A)
                            base_key = f"{min(origine, terminus)} ↔ {max(origine, terminus)}"

                            if base_key not in missions_organisees:
                                missions_organisees[base_key] = []

                            missions_organisees[base_key].append({
                                'label': mission_key, # Le vrai sens : A → B
                                'value': gini_val
                            })

                    # Afficher par groupe
                    for base_key, directions in missions_organisees.items():
                        st.markdown(f"**Liaison {base_key}**")

                        # Trier pour afficher A->B puis B->A (ou autre ordre logique)
                        directions.sort(key=lambda x: x['label'])

                        cols = st.columns(len(directions)) if len(directions) > 0 else [st.container()]

                        for idx, info in enumerate(directions):
                            with cols[idx]:
                                val = info['value']
                                delta = None
                                if val > 0.9:
                                    delta = "Excellent"
                                    delta_color = "normal"
                                elif val < 0.8:
                                    delta = "À améliorer"
                                    delta_color = "inverse"
                                else:
                                    delta = "Correct"
                                    delta_color = "off"

                                st.metric(
                                    label=info['label'], # Affiche "Nîmes → Le Grau du Roi" explicitement
                                    value=f"{val:.2f}",
                                    delta=delta
                                )
                        st.markdown("---")
            # --- Affichage du Graphique ---
            st.subheader("Graphique horaire")
            if not chronologie or all(not t for t in chronologie.values()):
                st.warning("Aucun train à afficher.")
            else:
                # Passe les infos de calcul au module de plotting
                figure = creer_graphique_horaire(
                    chronologie,
                    dataframe_gares,
                    heure_debut_service,
                    params_affichage,
                    mode_calcul=mode_calcul,
                    missions_par_train=missions_par_train,
                    all_energy_params=st.session_state.energy_params
                )
                st.pyplot(figure)

            if chronologie and any(chronologie.values()):

                # Pré-construction des figures batterie pour l'export PDF
                figures_batterie_pdf = []
                bat_figs_by_train = {}
                if mode_calcul == "Calcul Energie":
                    terminaux_valides = set()
                    for m in st.session_state.get('missions', []):
                        if "origine" in m: terminaux_valides.add(m["origine"])
                        if "terminus" in m: terminaux_valides.add(m["terminus"])
                        
                    bat_params_pdf = st.session_state.energy_params.get("batterie", {})
                    max_c = bat_params_pdf.get("facteur_charge_C", 4.0)

                    for _tid, (_res, _tmat) in resultats_energie_par_train.items():
                        if _tmat == "batterie" and _res.get("batterie_log"):
                            _fig_b = creer_graphique_batterie(
                                _res["batterie_log"], _tid,
                                soc_min_pct=bat_params_pdf.get("soc_min_pct", 20),
                                soc_max_pct=bat_params_pdf.get("soc_max_pct", 95),
                                terminaux_valides=terminaux_valides,
                                max_c_rate=max_c,
                                df_gares=dataframe_gares
                            )
                            if _fig_b:
                                figures_batterie_pdf.append(_fig_b)
                                bat_figs_by_train[_tid] = _fig_b

                excel_buffer, pdf_buffer = generer_exports(
                    chronologie, figure,
                    figures_batterie=figures_batterie_pdf,
                    logo_path="logo.png"
                )
                st.download_button("Télécharger roulements (Excel)", excel_buffer, "roulements.xlsx")
                st.download_button("Télécharger graphique (PDF)", pdf_buffer, "graphique.pdf")

                # Téléchargement du scénario (mode Énergie uniquement) — round-trip
                # avec l'encart "Charger un scénario" en haut de page.
                if mode_calcul == "Calcul Energie":
                    with st.expander("💾 Télécharger ce scénario (.json)", expanded=False):
                        col_n, col_t = st.columns(2)
                        nom_dl = col_n.text_input(
                            "Nom (optionnel)",
                            key="scenario_dl_name",
                            help="Apparaîtra dans la bibliothèque. Vide → nom horodaté.",
                        )
                        tags_dl = col_t.text_input(
                            "Tags (séparés par virgules, optionnel)",
                            key="scenario_dl_tags",
                        )
                        col_reg, col_lig = st.columns(2)
                        region_dl = col_reg.text_input(
                            "Région (optionnel)",
                            key="scenario_dl_region",
                            help="Pour classer le fichier : `scenarios/<région>/<ligne>/`.",
                        )
                        ligne_dl = col_lig.text_input(
                            "Ligne (optionnel)",
                            key="scenario_dl_ligne",
                            help="Sous-dossier de la région dans la bibliothèque.",
                        )
                        desc_dl = st.text_area(
                            "Description (optionnel)",
                            key="scenario_dl_desc",
                            height=68,
                        )

                        try:
                            scenario_dict = scenarios_manager.build_scenario_from_session(st.session_state)
                            if nom_dl:
                                scenario_dict["metadata"]["nom"] = nom_dl
                            if tags_dl:
                                scenario_dict["metadata"]["tags"] = [
                                    t.strip() for t in tags_dl.split(",") if t.strip()
                                ]
                            if desc_dl:
                                scenario_dict["metadata"]["description"] = desc_dl
                            if region_dl:
                                scenario_dict["metadata"]["region"] = region_dl.strip()
                            if ligne_dl:
                                scenario_dict["metadata"]["ligne"] = ligne_dl.strip()

                            scenario_bytes = scenarios_manager.serialize_scenario(scenario_dict)
                            default_name = scenarios_manager.default_download_filename(scenario_dict)

                            st.download_button(
                                "📥 Télécharger le scénario (.json)",
                                data=scenario_bytes,
                                file_name=default_name,
                                mime="application/json",
                                key="dl_scenario_json",
                                help="Le fichier peut être rechargé via l'onglet « Importer un fichier » "
                                     "ou déposé dans `scenarios/<région>/<ligne>/` pour la bibliothèque.",
                            )
                        except Exception as e:
                            st.error(f"Impossible de préparer le scénario : {e}")

                temps_theoriques = {}
                for m in st.session_state.missions:
                    orig, term = m.get('origine', ''), m.get('terminus', '')
                    t_aller = m.get('temps_trajet') or 0
                    t_retour = m.get('temps_trajet_retour') or t_aller if m.get('trajet_asymetrique') else t_aller
                    if orig and term:
                        temps_theoriques[f"{orig} → {term}"] = t_aller
                        temps_theoriques[f"{term} → {orig}"] = t_retour
                _afficher_tableau_temps_parcours(chronologie, temps_theoriques, df_gares=dataframe_gares)

            # --- Affichage des Résultats Énergétiques (si mode Energie) ---
            if mode_calcul == "Calcul Energie":
                with st.expander("Résultats de la simulation énergétique", expanded=True):

                    resultats_globaux = []
                    has_diesel = False
                    has_elec = False
                    can_recup = False # Au moins un train peut-il récupérer ?

                    def _fmt_dont_aux(valeur_km, part_pct):
                        if valeur_km is None:
                            return "N/A"
                        if part_pct is None:
                            return f"{valeur_km:.2f}"
                        return f"{valeur_km:.2f} ({part_pct:.0f}%)"

                    for id_train, (resultat_train, type_mat) in resultats_energie_par_train.items():

                        # Vérifier si ce type de matériel peut récupérer
                        mat_can_recup = type_mat in ["electrique", "bimode", "batterie"]
                        if mat_can_recup: can_recup = True # Si un seul le peut, on affiche la colonne

                        total_km = resultat_train.get("total_distance_km", 0)
                        conso_elec_kwh = resultat_train.get("total_conso_electrique_kwh", 0)
                        total_litres = resultat_train.get("total_litres_diesel", 0)
                        recup_kwh = resultat_train.get("total_recup_kwh", 0)
                        aux_elec_kwh = resultat_train.get("total_conso_aux_electrique_kwh", 0)
                        aux_litres = resultat_train.get("total_aux_litres_diesel", 0)

                        conso_kwh_km_str = "N/A"
                        conso_L_km_str = "N/A"
                        recup_kwh_km_str = "N/A"
                        aux_kwh_km_str = "N/A"
                        aux_L_km_str = "N/A"

                        if total_km > 0:
                            # kwh/km - Afficher si le train a un rendement elec
                            if type_mat in ["electrique", "bimode", "batterie"]:
                                conso_kwh_km_str = f"{conso_elec_kwh / total_km:.2f}"
                                pct_aux_elec = (100.0 * aux_elec_kwh / conso_elec_kwh) if conso_elec_kwh > 0 else None
                                aux_kwh_km_str = _fmt_dont_aux(aux_elec_kwh / total_km, pct_aux_elec)
                                has_elec = True
                            # L/km - Afficher si le train a un rendement thermique
                            if type_mat in ["diesel", "bimode"]:
                                conso_L_km_str = f"{total_litres / total_km:.2f}"
                                pct_aux_diesel = (100.0 * aux_litres / total_litres) if total_litres > 0 else None
                                aux_L_km_str = _fmt_dont_aux(aux_litres / total_km, pct_aux_diesel)
                                has_diesel = True
                            # recup/km - Afficher si le train peut récupérer
                            if mat_can_recup:
                                recup_kwh_km_str = f"{recup_kwh / total_km:.2f}" if recup_kwh > 0 else "0.00"

                        resultats_globaux.append({
                            "Train": id_train,
                            "Type": options_materiel_map.get(type_mat, type_mat),
                            "Dist. (km)": f"{total_km:.1f}",
                            "Conso. Électrique (kWh/km)": conso_kwh_km_str,
                            "Dont consommation des auxiliaires (kWh/km)": aux_kwh_km_str,
                            "Conso. Diesel (L/km)": conso_L_km_str,
                            "Dont consommation des auxiliaires (L/km)": aux_L_km_str,
                            "Économie Récupération (kWh/km)": recup_kwh_km_str,
                        })

                        if type_mat == "batterie" and resultat_train["batterie_log"]:
                            found_bat = True
                            with st.expander(f"Train {id_train} (Batterie)"):
                                # Tableau
                                raw_log = resultat_train["batterie_log"]
                                times = [x[0] for x in raw_log]
                                heures_debut = [times[0]] + times[:-1]

                                def _fmt_dur(td):
                                    total_min = int(td.total_seconds() // 60)
                                    h, m = divmod(total_min, 60)
                                    return f"{h}h{m:02d}" if h else f"{m} min"

                                durees = ["—"] + [_fmt_dur(times[i] - times[i-1]) for i in range(1, len(times))]

                                df_log = pd.DataFrame(raw_log, columns=["_heure_fin", "Niveau kWh", "SoC", "Événement", "P Charge (C)", "Is Stat", "Gare"])
                                df_log.insert(0, "Heure début", [t.strftime("%H:%M") if isinstance(t, datetime) else str(t) for t in heures_debut])
                                df_log.insert(1, "Durée", durees)
                                df_log.drop(columns=["_heure_fin", "P Charge (C)", "Is Stat", "Gare"], inplace=True)
                                df_log["Niveau kWh"] = df_log["Niveau kWh"].apply(lambda x: f"{x:.1f}")

                                st.dataframe(df_log, width="stretch")

                                # Graphique SoC (réutilise la figure déjà construite pour le PDF)
                                fig_bat = bat_figs_by_train.get(id_train)
                                if fig_bat:
                                    st.pyplot(fig_bat)

                    st.subheader("Bilan énergétique global")
                    if resultats_globaux:
                        df_bilan = pd.DataFrame(resultats_globaux)

                        # Construction dynamique des colonnes : conso puis « dont auxiliaires »
                        cols_to_display = ["Train", "Type", "Dist. (km)"]
                        if has_elec:
                            cols_to_display.append("Conso. Électrique (kWh/km)")
                            cols_to_display.append("Dont consommation des auxiliaires (kWh/km)")
                        if has_diesel:
                            cols_to_display.append("Conso. Diesel (L/km)")
                            cols_to_display.append("Dont consommation des auxiliaires (L/km)")
                        if can_recup:
                             cols_to_display.append("Économie Récupération (kWh/km)")

                        st.dataframe(df_bilan[cols_to_display])
                    else:
                        st.warning("Aucun résultat énergétique à afficher.")

    with st.expander("7. Évaluation de la robustesse", expanded=False):
        chronologie_robustesse = st.session_state.get("chronologie_calculee") or {}
        if not chronologie_robustesse or all(not train for train in chronologie_robustesse.values()):
            st.info("Générez d'abord un graphique horaire pour analyser sa robustesse.")
        else:
            _rendre_robustesse(
                chronologie_robustesse,
                dataframe_gares,
                st.session_state.missions,
                st.session_state.energy_params,
                heure_debut_service,
                params_affichage,
                mode_calcul,
                st.session_state.get("missions_par_train") or {},
            )

else:
    if st.session_state.get('gares') is None:
        st.warning("Veuillez d'abord définir et valider les gares à la section 1.")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption("""
🚄 **Chronofer** - Développé par le Cerema.

Modes :
- **Calcul Énergie** (défaut) : simulation de consommation sur les roulements générés ou saisis.
- **Rotation optimisée** : algorithmes Simple / Fast / Smart / Exhaustif / Génétique.
- **Manuel** (option) : roulements saisis ou importés Excel, avec évaluation énergétique.
""")
