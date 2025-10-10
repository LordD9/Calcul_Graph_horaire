# -*- coding: utf-8 -*-
"""
core_logic.py

Ce module contient le "cerveau" de l'application : les fonctions de calcul complexes
qui ne dépendent pas de l'interface utilisateur.
- Génération des horaires en mode optimisé.
- Traitement des données pour le mode manuel.
- Analyse des fréquences.
- Import/Export de fichiers.
"""
from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO

def generer_tous_trajets_optimises(missions, heure_debut, heure_fin):
    """
    Génère la chronologie complète des trajets en mode "Rotation optimisée"
    en se basant sur un système d'événements (départs planifiés, trains disponibles).
    """
    chronologie_optimale = {}
    id_train_counter = 1
    warnings = []

    dt_debut_service = datetime.combine(datetime.today(), heure_debut)
    dt_fin_service = datetime.combine(datetime.today(), heure_fin)
    if dt_fin_service <= dt_debut_service:
        dt_fin_service += timedelta(days=1)

    trains_disponibles = []
    evenements = []

    def preparer_horaire_mission(mission_config):
        """Crée une liste ordonnée de points de passage (gare, temps) pour une mission."""
        horaire = [{"gare": mission_config["origine"], "time_offset_min": 0}]
        horaire.extend(mission_config.get("passing_points", []))
        horaire.append({"gare": mission_config["terminus"], "time_offset_min": mission_config["temps_trajet"]})
        horaire_unique = [dict(t) for t in {tuple(d.items()) for d in horaire}]
        return sorted(horaire_unique, key=lambda x: x["time_offset_min"])

    for i, mission in enumerate(missions):
        try:
            minutes_ref = sorted(list(set([int(m.strip()) for m in mission.get("reference_minutes", "0").split(',') if m.strip().isdigit() and 0 <= int(m.strip()) <= 59])))
            if not minutes_ref: minutes_ref = [0]
        except:
            warnings.append(f"Format des minutes de référence invalide pour M{i+1}. Utilisation de :00.")
            minutes_ref = [0]

        if mission["frequence"] <= 0: continue
        intervalle = timedelta(hours=1 / mission["frequence"])

        for minute in minutes_ref:
            curseur_temps = dt_debut_service.replace(minute=minute, second=0, microsecond=0)
            if curseur_temps < dt_debut_service:
                curseur_temps += timedelta(hours=1)

            while curseur_temps < dt_fin_service:
                if (curseur_temps + timedelta(minutes=mission["temps_trajet"])) <= dt_fin_service:
                    evenements.append({"type": "depart_aller", "heure": curseur_temps, "details": {"mission": mission}})
                curseur_temps += intervalle

    while evenements:
        evenements.sort(key=lambda x: x["heure"])
        event = evenements.pop(0)

        if event["type"] == "depart_aller":
            mission_cfg = event["details"]["mission"]
            train_assigne = None
            for i, train in enumerate(sorted(trains_disponibles, key=lambda t: t["disponible_a"])):
                if train["gare"] == mission_cfg["origine"] and train["disponible_a"] <= event["heure"]:
                    train_assigne = trains_disponibles.pop(i)
                    break
            if not train_assigne:
                train_assigne = {"id": id_train_counter, "gare": mission_cfg["origine"]}
                chronologie_optimale[train_assigne["id"]] = []
                id_train_counter += 1

            horaire = preparer_horaire_mission(mission_cfg)
            heure_arrivee_finale = event["heure"]
            for i in range(len(horaire) - 1):
                p_dep, p_arr = horaire[i], horaire[i+1]
                t_dep = event["heure"] + timedelta(minutes=p_dep["time_offset_min"])
                t_arr = event["heure"] + timedelta(minutes=p_arr["time_offset_min"])
                chronologie_optimale[train_assigne["id"]].append({"start": t_dep, "end": t_arr, "origine": p_dep["gare"], "terminus": p_arr["gare"]})
                heure_arrivee_finale = t_arr

            temps_retournement = mission_cfg.get("temps_retournement_B", 10)
            heure_dispo_retour = heure_arrivee_finale + timedelta(minutes=temps_retournement)
            if heure_dispo_retour < dt_fin_service:
                evenements.append({"type": "disponible_pour_retour", "heure": heure_dispo_retour, "details": {"train": train_assigne, "gare": mission_cfg["terminus"], "mission_aller": mission_cfg}})

        elif event["type"] == "disponible_pour_retour":
            train = event["details"]["train"]
            mission_aller = event["details"]["mission_aller"]

            if mission_aller.get("trajet_asymetrique"):
                temps_trajet_retour = mission_aller.get("temps_trajet_retour", mission_aller["temps_trajet"])
                mission_retour_cfg = {
                    "origine": mission_aller["terminus"], "terminus": mission_aller["origine"],
                    "temps_trajet": temps_trajet_retour,
                    "passing_points": mission_aller.get("passing_points_retour", [])
                }
            else:
                 temps_trajet_retour = mission_aller["temps_trajet"]
                 pp_inverses = []
                 if mission_aller.get("passing_points"):
                     pp_inverses = sorted([
                         {"gare": pp["gare"], "time_offset_min": temps_trajet_retour - pp["time_offset_min"]}
                         for pp in mission_aller["passing_points"]
                     ], key=lambda x: x["time_offset_min"])
                 mission_retour_cfg = {
                    "origine": mission_aller["terminus"], "terminus": mission_aller["origine"],
                    "temps_trajet": temps_trajet_retour, "passing_points": pp_inverses
                 }

            horaire_retour = preparer_horaire_mission(mission_retour_cfg)
            heure_arrivee_finale = event["heure"]
            for i in range(len(horaire_retour) - 1):
                 p_dep, p_arr = horaire_retour[i], horaire_retour[i+1]
                 t_dep = event["heure"] + timedelta(minutes=p_dep["time_offset_min"])
                 t_arr = event["heure"] + timedelta(minutes=p_arr["time_offset_min"])
                 if t_arr > dt_fin_service: break
                 chronologie_optimale[train["id"]].append({"start": t_dep, "end": t_arr, "origine": p_dep["gare"], "terminus": p_arr["gare"]})
                 heure_arrivee_finale = t_arr

            temps_retournement = mission_aller.get("temps_retournement_A", 10)
            heure_dispo_finale = heure_arrivee_finale + timedelta(minutes=temps_retournement)
            if heure_dispo_finale < dt_fin_service:
                trains_disponibles.append({"id": train["id"], "gare": mission_aller["origine"], "disponible_a": heure_dispo_finale})

    return chronologie_optimale, warnings


def preparer_roulement_manuel(roulement_manuel):
    """Convertit le dictionnaire de roulement manuel en chronologie de trajets pour le graphique."""
    chronologie_trajets = {}
    for id_train, etapes_train in roulement_manuel.items():
        if not etapes_train: continue
        chronologie_trajets[id_train] = []
        last_arrival_time = None
        for etape in etapes_train:
            try:
                dt_debut = datetime.combine(datetime.today(), datetime.strptime(etape["heure_depart"], "%H:%M").time())
                dt_fin = datetime.combine(datetime.today(), datetime.strptime(etape["heure_arrivee"], "%H:%M").time())
                if last_arrival_time and dt_debut < last_arrival_time: dt_debut += timedelta(days=1)
                if dt_fin < dt_debut: dt_fin += timedelta(days=1)
                chronologie_trajets[id_train].append({"start": dt_debut, "end": dt_fin, "origine": etape["depart"], "terminus": etape["arrivee"]})
                last_arrival_time = dt_fin
            except (ValueError, KeyError) as e:
                print(f"Skipping invalid step for train {id_train}: {etape}. Error: {e}")
                continue
    return chronologie_trajets

def importer_roulements_fichier(uploaded_file, dataframe_gares):
    """
    CORRECTION: Reprise de la logique de votre fichier monolithique, plus directe et robuste.
    Traite un fichier CSV ou Excel importé et le convertit en structure de roulement.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df_import = pd.read_csv(uploaded_file, sep=';')
            # Renommage pour le format CSV simple
            df_import = df_import.rename(columns={
                "Départ": "origine", "Arrivée": "terminus",
                "Heure départ": "heure_depart", "Heure arrivée": "heure_arrivee",
                "Temps trajet (min)": "temps_trajet", "Train": "train_id"
            })
        else:  # .xlsx
            df_import = pd.read_excel(uploaded_file)
            # Renommage et calcul pour le format d'export de l'application
            df_import = df_import.rename(columns={
                "Origine": "origine", "Terminus": "terminus",
                "Train": "train_id"
            })
            df_import['heure_depart'] = pd.to_datetime(df_import['Début']).dt.strftime('%H:%M')
            df_import['heure_arrivee'] = pd.to_datetime(df_import['Fin']).dt.strftime('%H:%M')
            start_times = pd.to_datetime(df_import['Début'])
            end_times = pd.to_datetime(df_import['Fin'])
            df_import['temps_trajet'] = (end_times - start_times).dt.total_seconds() / 60
            df_import['temps_trajet'] = df_import['temps_trajet'].astype(int)

        required_cols = ["train_id", "origine", "heure_depart", "terminus", "heure_arrivee", "temps_trajet"]
        if not all(col in df_import.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_import.columns]
            return None, f"Fichier invalide. Colonnes manquantes: {', '.join(missing)}"

        new_roulement = {}
        for _, row in df_import.iterrows():
            try:
                train_id = int(row['train_id'])
                # S'assurer que les heures sont bien des chaînes de caractères au format HH:MM
                heure_depart_str = str(row['heure_depart'])
                heure_arrivee_str = str(row['heure_arrivee'])
                # Valider le format de l'heure
                datetime.strptime(heure_depart_str, "%H:%M")
                datetime.strptime(heure_arrivee_str, "%H:%M")

                etape = {
                    "depart": row['origine'],
                    "heure_depart": heure_depart_str,
                    "arrivee": row['terminus'],
                    "heure_arrivee": heure_arrivee_str,
                    "temps_trajet": int(row['temps_trajet'])
                }
                new_roulement.setdefault(train_id, []).append(etape)
            except (ValueError, TypeError) as e:
                # Ignore les lignes avec des données mal formatées
                print(f"Ligne ignorée lors de l'import : {row}. Erreur : {e}")
                continue

        # Tri final des étapes pour chaque train
        for train_id in new_roulement:
            new_roulement[train_id].sort(key=lambda x: datetime.strptime(x['heure_depart'], "%H:%M").time())

        return new_roulement, None

    except Exception as e:
        return None, f"Erreur critique lors de la lecture du fichier : {e}"


def analyser_frequences_manuelles(roulement_manuel, missions, heure_debut_service, heure_fin_service):
    """Analyse la conformité des fréquences du roulement manuel par rapport aux missions."""
    comptes_par_heure = {}
    for etapes in roulement_manuel.values():
        for etape in etapes:
            cle_segment = f"{etape['depart']} → {etape['arrivee']}"
            heure = datetime.strptime(etape['heure_depart'], "%H:%M").hour
            comptes_par_heure.setdefault(cle_segment, {}).setdefault(heure, 0)
            comptes_par_heure[cle_segment][heure] += 1
    resultats_analyse = {}
    heures_service = list(range(heure_debut_service.hour, heure_fin_service.hour))
    for mission in missions:
        cle_mission = f"{mission['origine']} → {mission['terminus']}"
        objectif = mission['frequence']
        donnees_analyse = []
        heures_respectees = 0
        trains_reels_segment = comptes_par_heure.get(cle_mission, {})
        for heure in heures_service:
            trains_reels = trains_reels_segment.get(heure, 0)
            statut = "✓" if trains_reels >= objectif else "❌"
            if statut == "✓": heures_respectees += 1
            donnees_analyse.append({"Heure": f"{heure:02d}:00", "Trains sur segment": trains_reels, "Objectif": f"≥ {objectif}", "Statut": statut})
        df_analyse = pd.DataFrame(donnees_analyse)
        conformite = (heures_respectees / len(heures_service)) * 100 if heures_service else 100
        resultats_analyse[cle_mission] = {"df": df_analyse, "conformite": conformite}
    return resultats_analyse

def generer_exports(chronologie_trajets, figure):
    """Génère les fichiers Excel et PDF pour le téléchargement."""
    lignes_export = [{"Train": id_t, "Début": t["start"], "Fin": t["end"], "Origine": t["origine"], "Terminus": t["terminus"]} for id_t, trajets in chronologie_trajets.items() for t in trajets]
    df_export = pd.DataFrame(lignes_export)
    excel_buffer = BytesIO()
    df_export.to_excel(excel_buffer, index=False, sheet_name="Roulements")
    excel_buffer.seek(0)
    pdf_buffer = BytesIO()
    figure.savefig(pdf_buffer, format="pdf", bbox_inches='tight')
    pdf_buffer.seek(0)
    return excel_buffer, pdf_buffer

