# -*- coding: utf-8 -*-
"""
app.py

Fichier principal de l'application Streamlit.
Ce fichier gère l'ensemble de l'interface utilisateur (UI) et orchestre les appels
aux modules de logique (`core_logic`) et de visualisation (`plotting`).
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Import des modules locaux
from utils import trouver_mission_pour_od, obtenir_temps_trajet_defaut_etape_manuelle, obtenir_temps_retournement_defaut
from core_logic import generer_tous_trajets_optimises, preparer_roulement_manuel, importer_roulements_fichier, analyser_frequences_manuelles, generer_exports
from plotting import creer_graphique_horaire

# --- Configuration de la page et initialisation de l'état de session ---
st.set_page_config(layout="wide")
st.title("Graphique horaire ferroviaire - Prototype")

if "gares" not in st.session_state: st.session_state.gares = None
if "missions" not in st.session_state: st.session_state.missions = []
if "roulement_manuel" not in st.session_state: st.session_state.roulement_manuel = {}

# --- SECTION 1: Définition des gares ---
st.header("1. Gares et distances")
with st.form("formulaire_gares"):
    gares_texte = st.text_area("Liste des gares (une par ligne, format: nom;distance_km)", "Nîmes;0\nVauvert;20\nLe Grau-du-Roi;50")
    if st.form_submit_button("Valider les gares"):
        try:
            lignes = [ligne.split(";") for ligne in gares_texte.strip().split("\n") if len(ligne.split(";")) == 2]
            df = pd.DataFrame(lignes, columns=["gare", "distance"])
            df["distance"] = pd.to_numeric(df["distance"])
            st.session_state.gares = df.sort_values("distance").reset_index(drop=True)
            st.success("Gares enregistrées !")
        except Exception as e:
            st.error(f"Erreur de format. Utilisez 'nom;distance_km'. Détails : {e}")

# --- La suite de l'application ne s'affiche que si les gares sont définies ---
if st.session_state.get('gares') is not None:
    dataframe_gares = st.session_state.gares
    gares_list = dataframe_gares["gare"].tolist()

    # --- SECTION 2: Paramètres généraux du service ---
    st.header("2. Paramètres de service et mode de génération")
    heure_debut_service = st.time_input("Début de service", value=datetime.strptime("06:00", "%H:%M").time())
    heure_fin_service = st.time_input("Fin de service", value=datetime.strptime("22:00", "%H:%M").time())
    mode_generation = st.radio("Mode de génération des trains", ["Manuel", "Rotation optimisée"])

    # --- SECTION 3: Définition des missions ---
    st.header("3. Missions")
    nombre_missions = st.number_input("Nombre de types de missions", 1, 10, len(st.session_state.missions) or 1)

    # Ajustement de la liste des missions dans l'état de session
    while len(st.session_state.missions) < nombre_missions:
        st.session_state.missions.append({})
    while len(st.session_state.missions) > nombre_missions:
        st.session_state.missions.pop()

    for i in range(nombre_missions):
        with st.container(border=True):
            st.subheader(f"Mission {i+1} (trajet Aller)")
            mission = st.session_state.missions[i]

            # Formulaire principal
            cols = st.columns([2, 2, 3] if mode_generation == "Rotation optimisée" else [2, 2])
            origine = cols[0].selectbox(f"Origine M{i+1}", gares_list, index=gares_list.index(mission.get("origine", gares_list[0])) if mission.get("origine") in gares_list else 0, key=f"orig{i}")
            terminus = cols[0].selectbox(f"Terminus M{i+1}", gares_list, index=gares_list.index(mission.get("terminus", gares_list[-1])) if mission.get("terminus") in gares_list else len(gares_list)-1, key=f"term{i}")
            frequence = cols[1].number_input(f"Fréquence (train/h) M{i+1}", 0.1, 10.0, mission.get("frequence", 1.0), 0.1, key=f"freq{i}")
            temps_trajet = cols[1].number_input(f"Temps trajet (min) M{i+1}", 1, 720, mission.get("temps_trajet", 45), key=f"tt{i}")
            if mode_generation == "Rotation optimisée":
                retournement_A = cols[2].number_input(f"Retournement à {origine} (min)", 0, 120, mission.get("temps_retournement_A", 10), key=f"tr_a_opt_{i}")
                retournement_B = cols[2].number_input(f"Retournement à {terminus} (min)", 0, 120, mission.get("temps_retournement_B", 10), key=f"tr_b_opt_{i}")
                ref_minutes = cols[2].text_input(f"Minute(s) de réf. M{i+1}", mission.get("reference_minutes", "0,30"), key=f"ref_mins{i}")
            else:
                retournement_A = cols[1].number_input(f"Retournement à {origine} (min)", 0, 120, mission.get("temps_retournement_A", 10), key=f"tr_a_man_{i}")
                retournement_B = cols[1].number_input(f"Retournement à {terminus} (min)", 0, 120, mission.get("temps_retournement_B", 10), key=f"tr_b_man_{i}")
                ref_minutes = "0"

            # Points de passage Aller
            st.markdown("**Points de passage optionnels (Aller) :**")
            gares_passage_dispo = [g for g in gares_list if g not in [origine, terminus]]
            nb_pp = st.number_input(f"Nombre de points de passage (M{i+1})", 0, 10, len(mission.get("passing_points", [])), key=f"n_pass_{i}")

            passing_points = []
            if gares_passage_dispo and nb_pp > 0:
                dernier_temps = 0
                for j in range(nb_pp):
                    pp_cols = st.columns(2)
                    pp_gare = pp_cols[0].selectbox(f"Gare PP {j+1}", gares_passage_dispo, key=f"pp_gare_{i}_{j}")
                    pp_temps = pp_cols[1].number_input(f"Temps depuis {origine} (min)", min_value=dernier_temps + 1, max_value=temps_trajet - 1, value=min(dernier_temps + 15, temps_trajet - 1), key=f"pp_tps_{i}_{j}")
                    passing_points.append({"gare": pp_gare, "time_offset_min": pp_temps})
                    dernier_temps = pp_temps

            # Trajet asymétrique
            st.markdown("**Options pour le trajet Retour :**")
            trajet_asymetrique = st.checkbox("Saisir un temps/parcours différent pour le retour", mission.get("trajet_asymetrique", False), key=f"asym_{i}")
            temps_trajet_retour, passing_points_retour = temps_trajet, []
            if trajet_asymetrique:
                temps_trajet_retour = st.number_input(f"Temps trajet RETOUR (min)", 1, 720, mission.get("temps_trajet_retour", temps_trajet), key=f"tt_retour_{i}")

                # Pré-calcul des points de passage inversés pour les valeurs par défaut
                pp_inverses_defaut = []
                if passing_points:
                    pp_inverses_defaut = sorted([
                        {"gare": pp["gare"], "time_offset_min": temps_trajet - pp["time_offset_min"]} for pp in passing_points
                    ], key=lambda x: x["time_offset_min"])

                pp_retour_existants = mission.get("passing_points_retour", pp_inverses_defaut)
                nb_pp_retour = st.number_input(f"Nombre de PP (Retour M{i+1})", 0, 10, len(pp_retour_existants), key=f"n_pass_retour_{i}")

                if gares_passage_dispo and nb_pp_retour > 0:
                    dernier_temps_retour = 0
                    for j in range(nb_pp_retour):
                        default_gare = pp_retour_existants[j]['gare'] if j < len(pp_retour_existants) else gares_passage_dispo[0]
                        default_temps = pp_retour_existants[j]['time_offset_min'] if j < len(pp_retour_existants) else dernier_temps_retour + 15

                        pp_cols_r = st.columns(2)
                        pp_gare_r = pp_cols_r[0].selectbox(f"Gare PP {j+1} (Retour)", gares_passage_dispo, index=gares_passage_dispo.index(default_gare) if default_gare in gares_passage_dispo else 0, key=f"pp_gare_retour_{i}_{j}")
                        pp_temps_r = pp_cols_r[1].number_input(f"Temps depuis {terminus} (min)", dernier_temps_retour + 1, temps_trajet_retour - 1, min(max(dernier_temps_retour + 1, default_temps), temps_trajet_retour - 1), key=f"pp_tps_retour_{i}_{j}")
                        passing_points_retour.append({"gare": pp_gare_r, "time_offset_min": pp_temps_r})
                        dernier_temps_retour = pp_temps_r

            # Sauvegarde de la mission complète dans l'état de session
            st.session_state.missions[i] = {
                "origine": origine, "terminus": terminus, "frequence": frequence, "temps_trajet": temps_trajet,
                "temps_retournement_A": retournement_A, "temps_retournement_B": retournement_B, "reference_minutes": ref_minutes,
                "passing_points": sorted(passing_points, key=lambda x: x['time_offset_min']),
                "trajet_asymetrique": trajet_asymetrique, "temps_trajet_retour": temps_trajet_retour,
                "passing_points_retour": sorted(passing_points_retour, key=lambda x: x['time_offset_min'])
            }

    # --- SECTION 4: Mode Manuel ---
    if mode_generation == "Manuel":
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
            uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=['csv', 'xlsx'])
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

                        edited_df = st.data_editor(df_original, key=f"editor_{id_train}", hide_index=True, use_container_width=True,
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

                    # Logique de la gare de départ par défaut
                    if derniere_etape:
                        idx_dep_defaut = gares_list.index(derniere_etape['arrivee'])
                        mission_precedente = trouver_mission_pour_od(derniere_etape['depart'], derniere_etape['arrivee'], st.session_state.missions)
                        if mission_precedente:
                            idx_arr_defaut = gares_list.index(mission_precedente['origine'])
                        else:
                            idx_arr_defaut = 0 if idx_dep_defaut + 1 >= len(gares_list) else idx_dep_defaut + 1
                    else: # Premier trajet du train
                        idx_dep_defaut = gares_list.index(st.session_state.missions[0]['origine']) if st.session_state.missions else 0
                        idx_arr_defaut = gares_list.index(st.session_state.missions[0]['terminus']) if st.session_state.missions else 1

                    gare_dep = add_cols[0].selectbox(f"Gare départ (T{id_train})", gares_list, index=idx_dep_defaut, key=f"dep_g_{id_train}")
                    gare_arr = add_cols[0].selectbox(f"Gare arrivée (T{id_train})", gares_list, index=idx_arr_defaut, key=f"arr_g_{id_train}")

                    heure_dep_defaut = (datetime.strptime(derniere_etape['heure_arrivee'], "%H:%M") + timedelta(minutes=obtenir_temps_retournement_defaut(gare_dep, st.session_state.missions))).time() if derniere_etape and derniere_etape['arrivee'] == gare_dep else heure_debut_service
                    heure_dep = add_cols[1].time_input(f"Heure départ (T{id_train})", heure_dep_defaut, key=f"dep_t_{id_train}")
                    temps_traj_defaut = obtenir_temps_trajet_defaut_etape_manuelle(gare_dep, gare_arr, st.session_state.missions)
                    temps_traj = add_cols[1].number_input(f"Temps trajet (min) (T{id_train})", 1, 720, temps_traj_defaut, key=f"tt_m_{id_train}")

                    if st.button(f"Ajouter étape au train {id_train}", key=f"add_e_{id_train}"):
                        mission_associee = trouver_mission_pour_od(gare_dep, gare_arr, st.session_state.missions)
                        if mission_associee and mission_associee.get("passing_points"):
                            horaire_complet = [{"gare": mission_associee["origine"], "time_offset_min": 0}] + mission_associee["passing_points"] + [{"gare": mission_associee["terminus"], "time_offset_min": mission_associee["temps_trajet"]}]
                            horaire_complet.sort(key=lambda x: x["time_offset_min"])
                            for k in range(len(horaire_complet) - 1):
                                p1, p2 = horaire_complet[k], horaire_complet[k+1]
                                dt_depart_etape = datetime.combine(datetime.today(), heure_dep) + timedelta(minutes=p1["time_offset_min"])
                                dt_arrivee_etape = datetime.combine(datetime.today(), heure_dep) + timedelta(minutes=p2["time_offset_min"])
                                st.session_state.roulement_manuel[id_train].append({"depart": p1["gare"], "heure_depart": dt_depart_etape.strftime("%H:%M"), "arrivee": p2["gare"], "heure_arrivee": dt_arrivee_etape.strftime("%H:%M"), "temps_trajet": int((dt_arrivee_etape - dt_depart_etape).total_seconds() / 60)})
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
                st.dataframe(pd.DataFrame(all_etapes), use_container_width=True)
                st.download_button("Télécharger (CSV)", pd.DataFrame(all_etapes).to_csv(index=False, sep=';').encode('utf-8-sig'), "config_roulements.csv", "text/csv")

    # --- SECTION 5: Vérification de Fréquence ---
    if mode_generation == "Manuel" and any(st.session_state.roulement_manuel.values()):
        st.header("5. Vérification de cohérence des fréquences")
        analyses = analyser_frequences_manuelles(st.session_state.roulement_manuel, st.session_state.missions, heure_debut_service, heure_fin_service)
        for mission_key, resultat in analyses.items():
            st.subheader(f"Analyse pour: {mission_key}")
            if resultat["df"] is not None and not resultat["df"].empty:
                st.dataframe(resultat["df"], use_container_width=True)
                if resultat["conformite"] == 100: st.success(f"✅ Objectif respecté à 100%.")
                elif resultat["conformite"] >= 75: st.warning(f"⚠️ Objectif respecté à {resultat['conformite']:.1f}%.")
                else: st.error(f"❌ Objectif respecté seulement à {resultat['conformite']:.1f}%.")

    # --- SECTION 6: Calcul et Affichage ---
    st.header("6. Calcul et Affichage")
    dt_debut_s = datetime.combine(datetime.min, heure_debut_service)
    dt_fin_s = datetime.combine(datetime.min, heure_fin_service)
    duree_heures_s = (dt_fin_s - dt_debut_s).total_seconds() / 3600
    if duree_heures_s <= 0: duree_heures_s += 24
    fenetre_heures = st.number_input("Durée de la fenêtre (h)", 1.0, duree_heures_s, min(5.0, duree_heures_s))
    decalage_heures = st.slider("Début de la fenêtre (h)", 0.0, max(0.0, duree_heures_s - fenetre_heures), 0.0, 0.5)

    if st.button("Lancer le calcul et afficher le graphique", type="primary"):
        chronologie = {}
        try:
            with st.spinner("Calcul des horaires..."):
                if mode_generation == "Rotation optimisée":
                    chronologie, warnings = generer_tous_trajets_optimises(st.session_state.missions, heure_debut_service, heure_fin_service)
                    for w in warnings: st.warning(w)
                else:
                    chronologie = preparer_roulement_manuel(st.session_state.roulement_manuel)

            st.subheader("Graphique horaire")
            if not chronologie or all(not t for t in chronologie.values()):
                st.warning("Aucun train à afficher.")
            else:
                params = {'duree_fenetre': fenetre_heures, 'decalage_heure': decalage_heures}
                figure = creer_graphique_horaire(chronologie, dataframe_gares, heure_debut_service, params)
                st.pyplot(figure)
                excel_buffer, pdf_buffer = generer_exports(chronologie, figure)
                st.download_button("Télécharger roulements (Excel)", excel_buffer, "roulements.xlsx")
                st.download_button("Télécharger graphique (PDF)", pdf_buffer, "graphique.pdf")
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la génération du graphique : {e}")

else:
    st.warning("Veuillez d'abord définir et valider les gares à la section 1.")

