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
import json
import re # Import re, bien que les helpers soient supprimés
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
from energy_logic import get_default_energy_params, calculer_consommation_trajet

from optimisation_logic import (
    OptimizationConfig,
    CrossingOptimization,
    optimiser_graphique_horaire)

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
if "mode_calcul" not in st.session_state: st.session_state.mode_calcul = "Standard"
if "chronologie_calculee" not in st.session_state: st.session_state.chronologie_calculee = None
if "stats_homogeneite" not in st.session_state: st.session_state.stats_homogeneite = {}
if "run_calculation" not in st.session_state: st.session_state.run_calculation = False

# Initialise avec les paramètres par défaut si nécessaire
default_params = get_default_energy_params()
if "energy_params" not in st.session_state: st.session_state.energy_params = {
    "electrique": default_params.copy(),
    "bimode": default_params.copy(),
    "diesel": default_params.copy(),
    "batterie": default_params.copy()
}


if "chronologie_calculee" not in st.session_state: st.session_state.chronologie_calculee = None
if "warnings_calcul" not in st.session_state: st.session_state.warnings_calcul = {}
if "energy_errors" not in st.session_state: st.session_state.energy_errors = []


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
st.header("1. Gares et Infrastructure")

# Sélection du mode de calcul qui conditionne l'affichage
st.session_state.mode_calcul = st.radio("Mode de calcul", ["Standard", "Calcul Energie"], horizontal=True, key="mode_calcul_selector")
mode_calcul = st.session_state.mode_calcul

with st.form("formulaire_gares"):

    # Aide contextuelle basée sur le mode
    help_text = "Format: nom;distance_km;[infra]\nInfra (optionnel): VE=Voie d'Évitement (croisement), F=Pas de croisement, D=Début/Fin de voie double."
    default_text = "Nîmes;0;VE\nVauvert;20;VE\nLe Grau-du-Roi;50;VE"

    if mode_calcul == "Calcul Energie":
        help_text = "Format: nom;km;infra (VE/F/D);electrification;rampe_section_a_venir\n" \
                    "- electrification: RXXXX (recharge kW), C1500, C25, F (non électrifié)\n" \
                    "- rampe_section_a_venir: Pente en ‰ (ex: -8 ou 8)"
        default_text = "Nîmes;0;VE;C1500;5\nVauvert;20;D;F;-3\nLe Grau-du-Roi;50;VE;R500;0"

    gares_texte = st.text_area(
        "Liste des gares (une par ligne)",
        default_text,
        help=help_text
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
    st.header("2. Paramètres de service")
    heure_debut_service = st.time_input("Début de service", value=datetime.strptime("06:00", "%H:%M").time())
    heure_fin_service = st.time_input("Fin de service", value=datetime.strptime("22:00", "%H:%M").time())

    # Le mode de génération n'est affiché qu'en mode "Standard"
    mode_generation = "Rotation optimisée" # Par défaut pour le mode Énergie
    if mode_calcul == "Standard":
        mode_generation = st.radio("Mode de génération des trains", ["Manuel", "Rotation optimisée"],index=1)
    else:
        st.info("Le mode 'Calcul Energie' utilise le moteur événementiel pour simuler les trajets et la consommation énergétique.")

    # --- SECTION 3: Définition des missions ---

    # Map pour capitaliser les options de matériel roulant
    options_materiel_map = {
        "diesel": "Diesel",
        "electrique": "Electrique",
        "bimode": "Bimode",
        "batterie": "Batterie"
    }
    options_materiel_list = list(options_materiel_map.keys())

    st.header("3. Missions")
    nombre_missions = st.number_input("Nombre de types de missions", 1, 10, len(st.session_state.missions) or 1)

    while len(st.session_state.missions) < nombre_missions:
        st.session_state.missions.append({})
    while len(st.session_state.missions) > nombre_missions:
        st.session_state.missions.pop()

    for i in range(nombre_missions):
        with st.container(border=True):
            st.subheader(f"Mission {i+1} (trajet Aller)")
            mission = st.session_state.missions[i]

            cols = st.columns([2, 2, 3])
            origine = cols[0].selectbox(f"Origine M{i+1}", gares_list, index=gares_list.index(mission.get("origine", gares_list[0])) if mission.get("origine") in gares_list else 0, key=f"orig{i}")
            terminus = cols[0].selectbox(f"Terminus M{i+1}", gares_list, index=gares_list.index(mission.get("terminus", gares_list[-1])) if mission.get("terminus") in gares_list else len(gares_list)-1, key=f"term{i}")


            frequence = cols[1].number_input(f"Fréquence (train/h) M{i+1}", 0.1, 10.0, mission.get("frequence", 1.0), 0.1, key=f"freq{i}")

            # Ce temps est utilisé par core_logic pour le planning
            # energy_logic l'utilisera comme contrainte pour déduire la vitesse
            temps_trajet = cols[1].number_input(f"Temps trajet PLANIFIÉ (min) M{i+1}", 1, 720, mission.get("temps_trajet", 45), key=f"tt{i}", help="Temps utilisé pour le planning. Le simulateur d'énergie en déduira la vitesse.")

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

            saisie_pp_mode = st.radio("Méthode de saisie des points de passage", ["Interface Guidée", "Saisie manuelle par lot"], key=f"saisie_pp_{i}", horizontal=True)

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
                help_text = f"Format: Gare;Temps depuis {origine} (min)[;Temps depuis {terminus} (min)]"
                if mode_calcul == "Calcul Energie":
                     help_text = f"Format: Gare;Temps_Aller;[Arrêt_Aller_min];[Temps_Retour];[Arrêt_Retour_min]\nEx: Vauvert;20;2;30;2 (2min d'arrêt à l'aller et au retour)\nEx: Vauvert;20;0;30;0 (pas d'arrêt)"
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

    # --- SECTION 4: Mode Manuel (uniquement en mode Standard) ---
    if mode_calcul == "Standard" and mode_generation == "Manuel":
        st.header("4. Construction manuelle des roulements")
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
        st.header("4. Paramètres Énergétiques")

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
                        f"Conso. Auxiliaires (kWh / h)", 0.0, 500.0,
                        value=float(params.get("facteur_aux_kwh_h", 50.0)),
                        step=1.0, key=f"f_aux_{type_mat}"
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


    # --- SECTION 5: Vérification de Fréquence (uniquement en mode Standard/Manuel) ---
    if mode_calcul == "Standard" and mode_generation == "Manuel" and any(st.session_state.roulement_manuel.values()):
        st.header("5. Vérification de cohérence des fréquences")
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

if st.session_state.gares is not None and st.session_state.missions and not (mode_calcul == "Standard" and mode_generation == "Manuel"):
    st.markdown("---")
    st.header("⚙️ Configuration de l'Optimisation (OBLIGATOIRE)")

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
                index=2,  # Smart progressive par défaut
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

            enable_crossing_opt = st.checkbox(
                "Activer l'optimisation des croisements",
                value=False,
                help="Prolonge stratégiquement les arrêts pour améliorer les croisements"
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

        # Récapitulatif
        st.markdown("---")
        st.subheader("📋 Récapitulatif de la configuration")

        col_recap1, col_recap2, col_recap3 = st.columns(3)

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
    st.header("6. Calcul et Affichage")
    dt_debut_s = datetime.combine(datetime.min, heure_debut_service)
    dt_fin_s = datetime.combine(datetime.min, heure_fin_service)
    duree_heures_s = (dt_fin_s - dt_debut_s).total_seconds() / 3600
    if duree_heures_s <= 0: duree_heures_s += 24
    fenetre_heures = st.number_input("Durée de la fenêtre (h)", 1.0, duree_heures_s, min(5.0, duree_heures_s))
    decalage_heures = st.slider("Début de la fenêtre (h)", 0.0, max(0.0, duree_heures_s - fenetre_heures), 0.0, 0.5)

    st.subheader("Options d'optimisation")

    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
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

    with col_opt2:
        # Affichage du nombre attendu de rames (estimation)
        if st.session_state.missions:
            # Calcul approximatif du nombre de rames si pas de partage
            nb_missions = len([m for m in st.session_state.missions if m.get("frequence", 0) > 0])
            if nb_missions > 0:
                if allow_sharing:
                    st.info(f"📊 Partage activé : nombre de rames optimisé")
                else:
                    st.warning(f"⚠️ Sans partage : environ {nb_missions * 2}+ rames nécessaires")


    estimation = "N/A"
    if mode_generation == "Rotation optimisée" or mode_calcul == "Calcul Energie":
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

    if st.button("🚀 Générer le graphique horaire", type="primary"):
        st.session_state.run_calculation = True

    if st.session_state.run_calculation:
        st.session_state.run_calculation = False

        # Déterminer si on utilise l'optimisation avancée
        use_advanced = st.session_state.get('use_advanced_optimization', False)

        if use_advanced:
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
                    if mode_generation == "Manuel":
                        chronologie = preparer_roulement_manuel(st.session_state.roulement_manuel)
                        warnings = {} # Pas de warnings en mode manuel pour l'instant

                        # Calculer les stats d'homogénéité
                        from core_logic import _calculer_stats_homogeneite
                        stats_homogeneite = _calculer_stats_homogeneite(chronologie)
                    else:
                        chronologie, warnings, stats_homogeneite = generer_tous_trajets_optimises(
                            st.session_state.missions,
                            st.session_state.gares,
                            heure_debut_service,
                            heure_fin_service,
                            allow_sharing=allow_sharing,
                            search_strategy='smart'
                        )

                    _elapsed_standard = time.time() - _t0_standard
                    # Calibration adaptative
                    if mode_generation != "Manuel":
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

            # Comptage par type de matériel
            rames_par_type = defaultdict(set)
            for train_id in chronologie.keys():
                # Trouver le type de matériel de ce train
                for mission in st.session_state.missions:
                    # Chercher dans les trajets pour identifier la mission
                    # (cette partie nécessite d'enrichir les données retournées)
                    pass

            st.subheader("📊 Statistiques d'utilisation des rames")

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
        resultats_energie_par_train = {}

        # Doit être calculé ici pour être passé au plotting ET au bilan
        missions_par_train = {}
        if mode_calcul == "Calcul Energie":
            for id_train, trajets in chronologie.items():
                if not trajets: continue
                premier_trajet = trajets[0]
                # Associer ce train à la première mission correspondante
                for mission in st.session_state.missions:
                    if mission["origine"] == premier_trajet["origine"]:
                         missions_par_train[id_train] = mission
                         break

        if mode_calcul == "Calcul Energie":
            with st.spinner("Calcul de la consommation énergétique..."):
                for id_train, trajets in chronologie.items():
                    mission = missions_par_train.get(id_train)
                    if not mission:
                        # Essayer de trouver une mission retour si le premier trajet n'est pas un aller
                        # (peut arriver si le train commence en milieu de journée)
                        found_mission = False
                        premier_trajet = trajets[0] # Assurer que premier_trajet est défini
                        for m_ret in st.session_state.missions:
                            if m_ret["terminus"] == premier_trajet["origine"]:
                                missions_par_train[id_train] = m_ret
                                mission = m_ret
                                found_mission = True
                                break
                        if not found_mission:
                            st.warning(f"Impossible de trouver la mission pour le Train {id_train} (démarrant à {premier_trajet['origine']}). Calcul énergétique ignoré.")
                            continue

                    type_mat = mission.get("type_materiel", "diesel")
                    params_mat = st.session_state.energy_params.get(type_mat)

                    if not params_mat:
                         st.error(f"Paramètres matériels non trouvés pour le type '{type_mat}'. Utilisation des valeurs par défaut.")
                         params_mat = get_default_energy_params()

                    resultat_train = calculer_consommation_trajet(trajets, mission, dataframe_gares, params_mat)
                    resultats_energie_par_train[id_train] = (resultat_train, type_mat)

                    if resultat_train["erreurs"]:
                        for err in resultat_train["erreurs"]:
                            all_energy_errors.append(f"Train {id_train}: {err}")

            st.session_state.energy_errors = all_energy_errors


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
            params_affichage = {'duree_fenetre': fenetre_heures, 'decalage_heure': decalage_heures}

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

            excel_buffer, pdf_buffer = generer_exports(chronologie, figure)
            st.download_button("Télécharger roulements (Excel)", excel_buffer, "roulements.xlsx")
            st.download_button("Télécharger graphique (PDF)", pdf_buffer, "graphique.pdf")

        # --- Affichage des Résultats Énergétiques (si mode Energie) ---
        if mode_calcul == "Calcul Energie":
            st.header("7. Résultats de la simulation énergétique")

            resultats_globaux = []
            has_diesel = False
            has_elec = False
            can_recup = False # Au moins un train peut-il récupérer ?

            for id_train, (resultat_train, type_mat) in resultats_energie_par_train.items():

                # Vérifier si ce type de matériel peut récupérer
                mat_can_recup = type_mat in ["electrique", "bimode", "batterie"]
                if mat_can_recup: can_recup = True # Si un seul le peut, on affiche la colonne

                total_km = resultat_train.get("total_distance_km", 0)
                conso_elec_kwh = resultat_train.get("total_conso_electrique_kwh", 0)
                total_litres = resultat_train.get("total_litres_diesel", 0)
                recup_kwh = resultat_train.get("total_recup_kwh", 0)

                conso_kwh_km_str = "N/A"
                conso_L_km_str = "N/A"
                recup_kwh_km_str = "N/A"

                if total_km > 0:
                    # kwh/km - Afficher si le train a un rendement elec
                    if type_mat in ["electrique", "bimode", "batterie"]:
                        conso_kwh_km_str = f"{conso_elec_kwh / total_km:.2f}"
                        has_elec = True
                    # L/km - Afficher si le train a un rendement thermique
                    if type_mat in ["diesel", "bimode"]:
                        conso_L_km_str = f"{total_litres / total_km:.2f}"
                        has_diesel = True
                    # recup/km - Afficher si le train peut récupérer
                    if mat_can_recup:
                        recup_kwh_km_str = f"{recup_kwh / total_km:.2f}" if recup_kwh > 0 else "0.00"


                resultats_globaux.append({
                    "Train": id_train,
                    "Type": options_materiel_map.get(type_mat, type_mat),
                    "Dist. (km)": f"{total_km:.1f}",
                    "Conso. Électrique (kWh/km)": conso_kwh_km_str,
                    "Conso. Diesel (L/km)": conso_L_km_str,
                    "Économie Récupération (kWh/km)": recup_kwh_km_str,
                })

                if type_mat == "batterie" and resultat_train["batterie_log"]:
                    found_bat = True
                    with st.expander(f"Train {id_train} (Batterie)"):
                        # Tableau
                        df_log = pd.DataFrame(resultat_train["batterie_log"], columns=["Heure", "Niveau kWh", "SoC", "Événement"])
                        # Formatage
                        df_log["Heure"] = df_log["Heure"].apply(lambda x: x.strftime("%H:%M") if isinstance(x, datetime) else str(x))
                        df_log["Niveau kWh"] = df_log["Niveau kWh"].apply(lambda x: f"{x:.1f}")

                        st.dataframe(df_log, width="stretch")

                        # Graphique SoC
                        bat_params = st.session_state.energy_params.get("batterie", {})
                        fig_bat = creer_graphique_batterie(
                            resultat_train["batterie_log"], id_train,
                            soc_min_pct=bat_params.get("soc_min_pct", 20),
                            soc_max_pct=bat_params.get("soc_max_pct", 95),
                        )
                        if fig_bat:
                            st.pyplot(fig_bat)

            st.subheader("Bilan énergétique global")
            if resultats_globaux:
                df_bilan = pd.DataFrame(resultats_globaux)

                # Construction dynamique des colonnes
                cols_to_display = ["Train", "Type", "Dist. (km)"]
                if has_elec:
                    cols_to_display.append("Conso. Électrique (kWh/km)")
                if has_diesel:
                    cols_to_display.append("Conso. Diesel (L/km)")
                if can_recup:
                     cols_to_display.append("Économie Récupération (kWh/km)")

                st.dataframe(df_bilan[cols_to_display])
            else:
                st.warning("Aucun résultat énergétique à afficher.")

else:
    st.warning("Veuillez d'abord définir et valider les gares à la section 1.")

# Bloc séparé pour le mode Manuel : bouton de calcul + affichage des résultats
if (st.session_state.get('gares') is not None and st.session_state.missions
        and mode_calcul == "Standard" and mode_generation == "Manuel"):

    st.header("6. Calcul et Affichage")
    dt_debut_s = datetime.combine(datetime.min, heure_debut_service)
    dt_fin_s = datetime.combine(datetime.min, heure_fin_service)
    duree_heures_s = (dt_fin_s - dt_debut_s).total_seconds() / 3600
    if duree_heures_s <= 0: duree_heures_s += 24
    fenetre_heures = st.number_input("Durée de la fenêtre (h)", 1.0, duree_heures_s, min(5.0, duree_heures_s), key="fenetre_manuel")
    decalage_heures = st.slider("Début de la fenêtre (h)", 0.0, max(0.0, duree_heures_s - fenetre_heures), 0.0, 0.5, key="decalage_manuel")

    if "run_calculation_manuel" not in st.session_state:
        st.session_state.run_calculation_manuel = False

    if st.button("🚀 Générer le graphique horaire", type="primary", key="btn_generate_manuel"):
        st.session_state.run_calculation_manuel = True

    if st.session_state.run_calculation_manuel:
        st.session_state.run_calculation_manuel = False

        with st.spinner("Génération du graphique en cours..."):
            try:
                chronologie = preparer_roulement_manuel(st.session_state.roulement_manuel)
                warnings = {}
                from core_logic import _calculer_stats_homogeneite
                stats_homogeneite = _calculer_stats_homogeneite(chronologie)

                st.session_state.chronologie_calculee = chronologie
                st.session_state.warnings_calcul = warnings
                st.session_state.stats_homogeneite = stats_homogeneite
                st.success("✅ Graphique généré avec succès !")

            except Exception as e:
                st.error(f"Une erreur est survenue lors de la génération du graphique : {e}")
                st.session_state.chronologie_calculee = None
                st.session_state.warnings_calcul = {}
                st.session_state.stats_homogeneite = {}
                st.stop()

    if st.session_state.chronologie_calculee:
        chronologie = st.session_state.chronologie_calculee
        params_affichage = {'duree_fenetre': fenetre_heures, 'decalage_heure': decalage_heures}
        dataframe_gares = st.session_state.gares
        figure = creer_graphique_horaire(
            chronologie,
            dataframe_gares,
            heure_debut_service,
            params_affichage,
            mode_calcul=mode_calcul,
            missions_par_train={},
            all_energy_params=st.session_state.energy_params
        )
        st.subheader("Graphique horaire")
        st.pyplot(figure)

        excel_buffer, pdf_buffer = generer_exports(chronologie, figure)
        st.download_button("Télécharger roulements (Excel)", excel_buffer, "roulements.xlsx", key="dl_excel_manuel")
        st.download_button("Télécharger graphique (PDF)", pdf_buffer, "graphique.pdf", key="dl_pdf_manuel")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption("""
🚄 **Chronofer** - Développé par le Cerema.

Modes d'optimisation :
- **Simple** : Simulation directe basée sur les temps de retournement.
- **Fast / Smart** : Heuristiques de recherche progressive.
- **Exhaustif** : Exploration complète pour solutions optimales (petites instances).
- **Génétique** : Algorithme évolutionnaire parallèle pour grandes instances.
""")
