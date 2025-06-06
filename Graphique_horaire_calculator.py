import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates # Pour la gestion des dates sur les axes Matplotlib
from io import BytesIO # Pour la manipulation de flux binaires (export Excel/PDF)
import numpy as np # Pas utilisé explicitement, mais souvent une dépendance de Pandas/Matplotlib

# Configuration de la page Streamlit en mode large
st.set_page_config(layout="wide")
st.title("Graphique horaire ferroviaire - Prototype")

# --- Initialisation de session_state (état de la session Streamlit) ---
if "gares" not in st.session_state:
    st.session_state.gares = None
if "missions" not in st.session_state:
    st.session_state.missions = []
if "roulement_manuel" not in st.session_state:
    st.session_state.roulement_manuel = {}
if "verification_freq" not in st.session_state: # Non utilisé activement
    st.session_state.verification_freq = {}
if "mode_generation_precedent" not in st.session_state: # Pour gérer le changement de mode
    st.session_state.mode_generation_precedent = "Manuel"


# --- Définition des gares et distances ---
st.header("1. Gares et distances")
with st.form("formulaire_gares"):
    gares_texte = st.text_area("Liste des gares (une par ligne, format: nom;distance_km)",
                             "Nîmes;0\nVauvert;20\nLe Grau-du-Roi;50")
    formulaire_gares_soumis = st.form_submit_button("Valider les gares")

if formulaire_gares_soumis:
    lignes_gares = gares_texte.strip().split("\n")
    donnees_gares = []
    erreur_format_gare = False
    for ligne in lignes_gares:
        elements = ligne.split(";")
        if len(elements) == 2:
            donnees_gares.append(elements)
        else:
            st.error(f"Format incorrect pour la ligne : '{ligne}'. Utilisez 'nom;distance_km'.")
            erreur_format_gare = True
            break

    if not erreur_format_gare:
        try:
            df_gares_initiales = pd.DataFrame(donnees_gares, columns=["gare", "distance"])
            df_gares_initiales["distance"] = pd.to_numeric(df_gares_initiales["distance"], errors='coerce')
            if df_gares_initiales["distance"].isnull().any():
                st.error("Certaines distances n'ont pas pu être converties en nombres. Vérifiez les valeurs.")
            else:
                df_gares_initiales = df_gares_initiales.sort_values("distance").reset_index(drop=True)
                st.session_state.gares = df_gares_initiales
                st.success("Gares enregistrées !")
        except Exception as e:
            st.error(f"Erreur lors de la création du DataFrame des gares : {e}")

if st.session_state.gares is not None:
    dataframe_gares = st.session_state.gares
else:
    st.warning("Veuillez d'abord définir les gares.")
    st.stop()

# --- Paramètres de génération (Mode et Horaires globaux) ---
st.header("3. Paramètres de service et mode de génération")
heure_debut_service = st.time_input("Début de service", value=datetime.strptime("06:00", "%H:%M").time(), key="heure_debut_service_global")
heure_fin_service = st.time_input("Fin de service", value=datetime.strptime("22:00", "%H:%M").time(), key="heure_fin_service_global")

mode_generation = st.radio("Mode de génération des trains", ["Manuel", "Rotation optimisée"], key="mode_generation_choix")

if mode_generation != st.session_state.mode_generation_precedent:
    st.session_state.mode_generation_precedent = mode_generation


# --- Definition des missions ---
st.header("2. Missions")
nombre_missions = st.number_input("Nombre de missions (chaque mission est un aller, le retour sera généré)", min_value=0, max_value=10, value=len(st.session_state.missions) or 1, key="nombre_missions_input", help="Chaque mission définie ici correspond à un trajet 'aller'. Le système tentera de générer un trajet 'retour' en mode optimisé.")

while len(st.session_state.missions) < nombre_missions:
    st.session_state.missions.append({"passing_points": [], "reference_minutes": "0"})
while len(st.session_state.missions) > nombre_missions:
    st.session_state.missions.pop()

for i in range(nombre_missions):
    st.subheader(f"Mission {i+1} (trajet Aller)")
    if i >= len(st.session_state.missions) or not isinstance(st.session_state.missions[i], dict):
        st.session_state.missions[i] = {"passing_points": [], "reference_minutes": "0"}

    mission_actuelle = st.session_state.missions[i]

    origine_mission_saisie = mission_actuelle.get("origine")
    terminus_mission_saisie = mission_actuelle.get("terminus")
    frequence_saisie = mission_actuelle.get("frequence", 1.0)
    temps_trajet_saisi = mission_actuelle.get("temps_trajet", 45)
    temps_retournement_saisi = mission_actuelle.get("temps_retournement", 10)
    minutes_reference_texte_saisi = mission_actuelle.get("reference_minutes", "0")

    if mode_generation == "Rotation optimisée":
        col_mission_1, col_mission_2, col_mission_3 = st.columns([2,2,3])
    else:
        col_mission_1, col_mission_2 = st.columns(2)

    with col_mission_1:
        default_orig_index = 0
        if mission_actuelle.get("origine") and mission_actuelle.get("origine") in dataframe_gares["gare"].tolist():
            default_orig_index = dataframe_gares["gare"].tolist().index(mission_actuelle.get("origine"))
        origine_mission_saisie = st.selectbox(f"Origine mission {i+1}", dataframe_gares["gare"], key=f"orig{i}", index=default_orig_index)

        default_term_index = len(dataframe_gares["gare"]) - 1 if len(dataframe_gares["gare"]) > 0 else 0
        if mission_actuelle.get("terminus") and mission_actuelle.get("terminus") in dataframe_gares["gare"].tolist():
            default_term_index = dataframe_gares["gare"].tolist().index(mission_actuelle.get("terminus"))
        terminus_mission_saisie = st.selectbox(f"Terminus mission {i+1}", dataframe_gares["gare"], key=f"term{i}", index=default_term_index)

    with col_mission_2:
        frequence_saisie = st.number_input(f"Fréquence (train/h) mission {i+1}", min_value=0.01, max_value=10.0,
                               value=mission_actuelle.get("frequence", 1.0), step=0.1, key=f"freq{i}")
        temps_trajet_saisi = st.number_input(f"Temps trajet (min) mission {i+1} (un sens)", min_value=1, max_value=720,
                                     value=mission_actuelle.get("temps_trajet", 45), key=f"tt{i}", help="Temps total pour un sens de la mission (ex: aller).")
        if mode_generation != "Rotation optimisée":
            temps_retournement_saisi = st.number_input(f"Temps de retournement (min) M{i+1}", min_value=0, max_value=120,
                                            value=mission_actuelle.get("temps_retournement", 10), key=f"tr_manual_{i}")

    if mode_generation == "Rotation optimisée":
        with col_mission_3:
            temps_retournement_saisi = st.number_input(f"Temps de retournement (min) M{i+1}", min_value=0, max_value=120,
                                            value=mission_actuelle.get("temps_retournement", 10), key=f"tr_optim_{i}", help="Temps d'arrêt au terminus avant de pouvoir repartir (pour l'aller et le retour).")
            minutes_reference_texte_saisi = st.text_input(f"Minute(s) de réf. M{i+1} (ex: 0,15,30)",
                                            value=mission_actuelle.get("reference_minutes", "0"),
                                            key=f"ref_mins{i}",
                                            help="Pour Rotation Optimisée: Minutes de départ dans l'heure pour le trajet ALLER (depuis l'origine de la mission).")

    st.markdown("**Points de passage optionnels (pour le trajet Aller défini ci-dessus):**")
    if "passing_points" not in mission_actuelle:
        mission_actuelle["passing_points"] = []

    nombre_points_passage = st.number_input(f"Nombre de points de passage (Mission {i+1})", min_value=0, max_value=10,
                             value=len(mission_actuelle.get("passing_points", [])), key=f"n_pass_{i}")

    points_passage_temporaires = []
    if nombre_points_passage > 0:
        st.caption(f"Définir les points de passage pour la mission {origine_mission_saisie} → {terminus_mission_saisie}:")
        dernier_temps_pp_depuis_origine = 0
        gares_passage_disponibles = [g for g in dataframe_gares["gare"].tolist() if g != origine_mission_saisie and g != terminus_mission_saisie]

        for j in range(nombre_points_passage):
            existing_pp_list = mission_actuelle.get("passing_points", [])
            default_pp_gare_val = None
            default_pp_tps_val = dernier_temps_pp_depuis_origine + 15

            if j < len(existing_pp_list) and isinstance(existing_pp_list[j], dict) and existing_pp_list[j].get("gare") in gares_passage_disponibles:
                default_pp_gare_val = existing_pp_list[j]["gare"]
                default_pp_tps_val = existing_pp_list[j]["temps_depuis_origine"]

            if not gares_passage_disponibles:
                st.warning("Pas assez de gares intermédiaires disponibles pour définir des points de passage.")
                break

            cols_pp = st.columns(2)
            with cols_pp[0]:
                default_pp_gare_idx = 0
                if default_pp_gare_val and default_pp_gare_val in gares_passage_disponibles:
                    default_pp_gare_idx = gares_passage_disponibles.index(default_pp_gare_val)
                pp_gare = st.selectbox(f"Gare de passage {j+1} (M{i+1})", options=gares_passage_disponibles,
                                       key=f"pp_gare_{i}_{j}", index=default_pp_gare_idx,
                                       help="Sélectionnez une gare intermédiaire.")
            with cols_pp[1]:
                val_for_pp_tps = min(max(default_pp_tps_val, dernier_temps_pp_depuis_origine + 1), temps_trajet_saisi -1 if temps_trajet_saisi > dernier_temps_pp_depuis_origine + 1 else dernier_temps_pp_depuis_origine + 1)
                if temps_trajet_saisi <= dernier_temps_pp_depuis_origine +1 :
                     st.warning(f"Temps total de trajet ({temps_trajet_saisi}min) insuffisant pour ajouter un point de passage après {dernier_temps_pp_depuis_origine}min.")
                     pp_tps_depuis_origine = dernier_temps_pp_depuis_origine +1
                else:
                    pp_tps_depuis_origine = st.number_input(f"Temps depuis {origine_mission_saisie} (min) PP{j+1}",
                                                            min_value=dernier_temps_pp_depuis_origine + 1,
                                                            max_value=temps_trajet_saisi - 1,
                                                            value=val_for_pp_tps,
                                                            key=f"pp_tps_{i}_{j}",
                                                            help=f"Temps total pour atteindre {pp_gare} depuis l'origine de la mission ({origine_mission_saisie}). Doit être < temps total mission.")

            if pp_gare and pp_tps_depuis_origine:
                if pp_tps_depuis_origine > dernier_temps_pp_depuis_origine and pp_tps_depuis_origine < temps_trajet_saisi :
                    points_passage_temporaires.append({"gare": pp_gare, "temps_depuis_origine": pp_tps_depuis_origine})
                    dernier_temps_pp_depuis_origine = pp_tps_depuis_origine
                else:
                    st.error(f"Temps pour le point de passage {j+1} ({pp_tps_depuis_origine} min) invalide. Doit être > {dernier_temps_pp_depuis_origine} et < {temps_trajet_saisi}.")

    st.session_state.missions[i] = {
        "origine": origine_mission_saisie, "terminus": terminus_mission_saisie, "frequence": frequence_saisie,
        "temps_trajet": temps_trajet_saisi, "temps_retournement": temps_retournement_saisi,
        "passing_points": sorted(points_passage_temporaires, key=lambda x: x["temps_depuis_origine"]),
        "reference_minutes": minutes_reference_texte_saisi
    }
    st.markdown("---")

if mode_generation == "Rotation optimisée":
    st.subheader("Options pour la Rotation Optimisée")
    decalage_par_gare_missions = st.number_input("Décalage de base entre séries de missions 'Aller' (min) depuis une même gare", min_value=0, max_value=120, value=0, step=5, key="decalage_missions_opt", help="Optionnel. Si plusieurs types de missions 'Aller' partent de la même gare, ce décalage s'appliquera successivement à leur 'heure de début de calcul de série' avant l'application de leurs minutes de référence spécifiques.")

def trouver_mission_pour_od(origine_selectionnee, terminus_selectionne, toutes_les_missions):
    for donnees_mission in toutes_les_missions:
        if donnees_mission["origine"] == origine_selectionnee and donnees_mission["terminus"] == terminus_selectionne:
            return donnees_mission

    for donnees_mission in toutes_les_missions:
        if donnees_mission["origine"] == terminus_selectionne and donnees_mission["terminus"] == origine_selectionnee:
            mission_inversee = {
                "origine": origine_selectionnee,
                "terminus": terminus_selectionne,
                "frequence": donnees_mission["frequence"],
                "temps_trajet": donnees_mission["temps_trajet"],
                "temps_retournement": donnees_mission["temps_retournement"],
                "passing_points": []
            }
            temps_total_mission_aller = donnees_mission["temps_trajet"]
            points_passage_inverses = []
            points_passage_actuels = donnees_mission.get("passing_points", [])
            if isinstance(points_passage_actuels, list):
                for pp in reversed(points_passage_actuels):
                    if isinstance(pp, dict) and "gare" in pp and "temps_depuis_origine" in pp:
                         points_passage_inverses.append({
                            "gare": pp["gare"],
                            "temps_depuis_origine": temps_total_mission_aller - pp["temps_depuis_origine"]
                        })
            mission_inversee["passing_points"] = sorted(points_passage_inverses, key=lambda x: x["temps_depuis_origine"])
            return mission_inversee
    return None

def obtenir_temps_trajet_defaut_etape_manuelle(gare_depart_etape, gare_arrivee_etape, toutes_les_missions, df_gares_local_fct):
    mission_correspondante = trouver_mission_pour_od(gare_depart_etape, gare_arrivee_etape, toutes_les_missions)
    if mission_correspondante:
        return mission_correspondante["temps_trajet"]

    for donnees_mission in toutes_les_missions:
        chemin_mission = [(donnees_mission["origine"], 0)]
        points_passage_actuels = donnees_mission.get("passing_points", [])
        if isinstance(points_passage_actuels, list):
            for pp in points_passage_actuels:
                 if isinstance(pp, dict) and "gare" in pp and "temps_depuis_origine" in pp:
                    chemin_mission.append((pp["gare"], pp["temps_depuis_origine"]))
        chemin_mission.append((donnees_mission["terminus"], donnees_mission["temps_trajet"]))
        chemin_mission.sort(key=lambda x: x[1])

        for indice_chemin in range(len(chemin_mission) - 1):
            gare_arret1, temps_arret1 = chemin_mission[indice_chemin]
            gare_arret2, temps_arret2 = chemin_mission[indice_chemin+1]
            if gare_arret1 == gare_depart_etape and gare_arret2 == gare_arrivee_etape:
                duree_segment = temps_arret2 - temps_arret1
                if duree_segment > 0:
                    return int(duree_segment)
    return 30

def obtenir_temps_retournement_defaut(nom_gare_retournement, toutes_les_missions):
    for donnees_mission in toutes_les_missions:
        if nom_gare_retournement == donnees_mission["origine"] or nom_gare_retournement == donnees_mission["terminus"]:
            return donnees_mission.get("temps_retournement", 10)
    return 10

if mode_generation == "Manuel":
    st.header("4. Construction manuelle des roulements")
    if "nombre_trains_manuel_saisi" not in st.session_state:
        st.session_state.nombre_trains_manuel_saisi = 2

    nombre_trains_manuel_interface = st.number_input("Nombre de trains", min_value=1, max_value=30,
                             value=st.session_state.nombre_trains_manuel_saisi, key="num_manual_trains_input")
    st.session_state.nombre_trains_manuel_saisi = nombre_trains_manuel_interface

    if not isinstance(st.session_state.get("roulement_manuel"), dict):
        st.session_state.roulement_manuel = {}

    ids_trains_existants = list(st.session_state.roulement_manuel.keys())
    for id_train_init in range(1, nombre_trains_manuel_interface + 1):
        if id_train_init not in ids_trains_existants:
            st.session_state.roulement_manuel[id_train_init] = []
    for id_train_suppr in ids_trains_existants:
        if id_train_suppr > nombre_trains_manuel_interface:
            del st.session_state.roulement_manuel[id_train_suppr]

    onglet1, onglet2 = st.tabs(["Édition des roulements", "Vue d'ensemble"])

    with onglet1:
        for id_train_manuel in range(1, nombre_trains_manuel_interface + 1):
            if id_train_manuel not in st.session_state.roulement_manuel:
                st.session_state.roulement_manuel[id_train_manuel] = []

            with st.expander(f"Train {id_train_manuel}", expanded=False):
                st.caption(f"Construction du parcours du train {id_train_manuel}")

                if st.session_state.roulement_manuel[id_train_manuel]:
                    df_etapes_train_manuel = pd.DataFrame(st.session_state.roulement_manuel[id_train_manuel])
                    st.write("Parcours actuel:")
                    st.dataframe(df_etapes_train_manuel[["depart", "heure_depart", "arrivee", "heure_arrivee", "temps_trajet"]])

                nombre_etapes_train_manuel = len(st.session_state.roulement_manuel[id_train_manuel]) + 1
                derniere_gare_train_manuel, derniere_heure_train_manuel = None, None

                if st.session_state.roulement_manuel[id_train_manuel]:
                    derniere_etape_train_manuel = st.session_state.roulement_manuel[id_train_manuel][-1]
                    derniere_gare_train_manuel, derniere_heure_train_manuel = derniere_etape_train_manuel["arrivee"], derniere_etape_train_manuel["heure_arrivee"]
                    st.info(f"Dernière position connue: {derniere_gare_train_manuel} à {derniere_heure_train_manuel}")

                st.subheader(f"Étape {nombre_etapes_train_manuel} du train {id_train_manuel}")
                colonne1_manuel, colonne2_manuel = st.columns(2)

                with colonne1_manuel:
                    index_depart_defaut_manuel = 0
                    if derniere_gare_train_manuel and derniere_gare_train_manuel in dataframe_gares["gare"].tolist():
                        index_depart_defaut_manuel = dataframe_gares["gare"].tolist().index(derniere_gare_train_manuel)

                    gare_depart_etape_manuel_saisie = st.selectbox(f"Gare de départ Étape {nombre_etapes_train_manuel} T{id_train_manuel}", options=dataframe_gares["gare"],
                                             index=index_depart_defaut_manuel,
                                             key=f"train{id_train_manuel}_etape{nombre_etapes_train_manuel}_dep_gare")

                    index_arrivee_defaut_manuel = (index_depart_defaut_manuel + 1) % len(dataframe_gares["gare"]) if len(dataframe_gares["gare"]) > 0 else 0
                    gare_arrivee_etape_manuel_saisie = st.selectbox(f"Gare d'arrivée Étape {nombre_etapes_train_manuel} T{id_train_manuel}", options=dataframe_gares["gare"].tolist(),
                                              index=index_arrivee_defaut_manuel,
                                              key=f"train{id_train_manuel}_etape{nombre_etapes_train_manuel}_arr_gare")
                with colonne2_manuel:
                    heure_depart_defaut_etape_manuel = heure_debut_service
                    if derniere_heure_train_manuel and derniere_gare_train_manuel == gare_depart_etape_manuel_saisie:
                        temps_retournement_min_manuel = obtenir_temps_retournement_defaut(derniere_gare_train_manuel, st.session_state.missions)
                        try:
                            heure_depart_defaut_etape_manuel = (datetime.strptime(derniere_heure_train_manuel, "%H:%M") + timedelta(minutes=temps_retournement_min_manuel)).time()
                        except ValueError: pass
                    elif derniere_heure_train_manuel:
                        try: heure_depart_defaut_etape_manuel = (datetime.strptime(derniere_heure_train_manuel, "%H:%M") + timedelta(minutes=1)).time()
                        except ValueError: pass

                    methode_saisie_heure_manuel = st.radio("Méthode de saisie de l'heure", ["Sélecteur", "Manuel (HH:MM)"], horizontal=True, key=f"train{id_train_manuel}_etape{nombre_etapes_train_manuel}_input_method")
                    heure_depart_saisie_manuel = heure_depart_defaut_etape_manuel
                    if methode_saisie_heure_manuel == "Sélecteur":
                        heure_depart_saisie_manuel = st.time_input(f"Heure de départ Étape {nombre_etapes_train_manuel} T{id_train_manuel}", value=heure_depart_defaut_etape_manuel, key=f"train{id_train_manuel}_etape{nombre_etapes_train_manuel}_dep_time")
                    else:
                        try: heure_depart_saisie_manuel = datetime.strptime(st.text_input(f"Heure départ (HH:MM) É{nombre_etapes_train_manuel} T{id_train_manuel}", value=heure_depart_defaut_etape_manuel.strftime("%H:%M"), key=f"train{id_train_manuel}_etape{nombre_etapes_train_manuel}_dep_time_manual"), "%H:%M").time()
                        except ValueError: st.error("Format HH:MM invalide."); heure_depart_saisie_manuel = heure_depart_defaut_etape_manuel

                    temps_trajet_defaut_etape_manuel = obtenir_temps_trajet_defaut_etape_manuelle(gare_depart_etape_manuel_saisie, gare_arrivee_etape_manuel_saisie, st.session_state.missions, dataframe_gares)
                    temps_trajet_saisi_manuel = st.number_input(f"Temps de trajet (min) Étape {nombre_etapes_train_manuel} T{id_train_manuel}", min_value=1, max_value=720, value=temps_trajet_defaut_etape_manuel, key=f"train{id_train_manuel}_etape{nombre_etapes_train_manuel}_tps_trajet")

                datetime_depart_estime_manuel = datetime.combine(datetime.today(), heure_depart_saisie_manuel)
                datetime_arrivee_estime_manuel = datetime_depart_estime_manuel + timedelta(minutes=temps_trajet_saisi_manuel)
                st.info(f"Heure d'arrivée estimée pour {gare_depart_etape_manuel_saisie} → {gare_arrivee_etape_manuel_saisie}: {datetime_arrivee_estime_manuel.strftime('%H:%M')}")

                if st.button(f"Ajouter cette étape au train {id_train_manuel}", key=f"train{id_train_manuel}_etape{nombre_etapes_train_manuel}_add_button"):
                    mission_correspondante_pour_etape = trouver_mission_pour_od(gare_depart_etape_manuel_saisie, gare_arrivee_etape_manuel_saisie, st.session_state.missions)

                    if mission_correspondante_pour_etape and isinstance(mission_correspondante_pour_etape.get("passing_points"), list) and mission_correspondante_pour_etape.get("passing_points"):
                        st.info(f"Décomposition de l'étape {gare_depart_etape_manuel_saisie}-{gare_arrivee_etape_manuel_saisie} selon la mission et ses points de passage.")
                        horaire_mission_detaille = []
                        horaire_mission_detaille.append({"gare": mission_correspondante_pour_etape["origine"], "time_offset_min": 0, "dist": dataframe_gares.loc[dataframe_gares["gare"] == mission_correspondante_pour_etape["origine"], "distance"].iloc[0]})
                        for pp in mission_correspondante_pour_etape["passing_points"]:
                            if isinstance(pp, dict) and "gare" in pp and "temps_depuis_origine" in pp :
                                horaire_mission_detaille.append({"gare": pp["gare"], "time_offset_min": pp["temps_depuis_origine"], "dist": dataframe_gares.loc[dataframe_gares["gare"] == pp["gare"], "distance"].iloc[0]})
                        horaire_mission_detaille.append({"gare": mission_correspondante_pour_etape["terminus"], "time_offset_min": mission_correspondante_pour_etape["temps_trajet"], "dist": dataframe_gares.loc[dataframe_gares["gare"] == mission_correspondante_pour_etape["terminus"], "distance"].iloc[0]})

                        horaire_mission_detaille.sort(key=lambda x: x["time_offset_min"])

                        horaire_mission_detaille_unique = []
                        dernier_point_unique = None
                        for point_horaire in horaire_mission_detaille:
                            if not dernier_point_unique or not (point_horaire["gare"] == dernier_point_unique["gare"] and point_horaire["time_offset_min"] == dernier_point_unique["time_offset_min"]):
                                horaire_mission_detaille_unique.append(point_horaire)
                            dernier_point_unique = point_horaire
                        horaire_mission_detaille = horaire_mission_detaille_unique

                        datetime_depart_base_etape = datetime.combine(datetime.today(), heure_depart_saisie_manuel)
                        nombre_segments_ajoutes = 0
                        for indice_segment_mission in range(len(horaire_mission_detaille) - 1):
                            point_depart_segment = horaire_mission_detaille[indice_segment_mission]
                            point_arrivee_segment = horaire_mission_detaille[indice_segment_mission+1]

                            if point_depart_segment["gare"] == point_arrivee_segment["gare"] and point_depart_segment["time_offset_min"] == point_arrivee_segment["time_offset_min"]: continue

                            datetime_depart_segment = datetime_depart_base_etape + timedelta(minutes=point_depart_segment["time_offset_min"])
                            datetime_arrivee_segment = datetime_depart_base_etape + timedelta(minutes=point_arrivee_segment["time_offset_min"])
                            duree_segment_mission = (datetime_arrivee_segment - datetime_depart_segment).total_seconds() / 60

                            if duree_segment_mission <= 0 and point_depart_segment["gare"] != point_arrivee_segment["gare"]:
                                st.warning(f"Segment {point_depart_segment['gare']}->{point_arrivee_segment['gare']} a une durée nulle ou négative ({duree_segment_mission}min). Ignoré.")
                                continue

                            dict_sous_etape = {
                                "depart": point_depart_segment["gare"], "dist_depart": point_depart_segment["dist"],
                                "heure_depart": datetime_depart_segment.strftime("%H:%M"),
                                "arrivee": point_arrivee_segment["gare"], "dist_arrivee": point_arrivee_segment["dist"],
                                "heure_arrivee": datetime_arrivee_segment.strftime("%H:%M"),
                                "temps_trajet": int(round(duree_segment_mission))
                            }
                            st.session_state.roulement_manuel[id_train_manuel].append(dict_sous_etape)
                            nombre_segments_ajoutes +=1
                        if nombre_segments_ajoutes > 0:
                             st.success(f"{nombre_segments_ajoutes} segment(s) de la mission décomposée ajoutés au train {id_train_manuel}.")
                        else:
                             st.warning("Aucun segment valide n'a pu être généré à partir de la mission pour cette étape.")
                    else:
                        distance_depart_etape = dataframe_gares.loc[dataframe_gares["gare"] == gare_depart_etape_manuel_saisie, "distance"].iloc[0]
                        distance_arrivee_etape = dataframe_gares.loc[dataframe_gares["gare"] == gare_arrivee_etape_manuel_saisie, "distance"].iloc[0]
                        datetime_depart_etape_obj = datetime.combine(datetime.today(), heure_depart_saisie_manuel)
                        datetime_arrivee_etape_obj = datetime_depart_etape_obj + timedelta(minutes=temps_trajet_saisi_manuel)

                        dict_etape_simple = {
                            "depart": gare_depart_etape_manuel_saisie, "dist_depart": distance_depart_etape,
                            "heure_depart": datetime_depart_etape_obj.strftime("%H:%M"),
                            "arrivee": gare_arrivee_etape_manuel_saisie, "dist_arrivee": distance_arrivee_etape,
                            "heure_arrivee": datetime_arrivee_etape_obj.strftime("%H:%M"),
                            "temps_trajet": temps_trajet_saisi_manuel
                        }
                        st.session_state.roulement_manuel[id_train_manuel].append(dict_etape_simple)
                        st.success(f"Étape directe {gare_depart_etape_manuel_saisie}-{gare_arrivee_etape_manuel_saisie} ajoutée au train {id_train_manuel}.")
                    st.rerun()

                if st.session_state.roulement_manuel[id_train_manuel] and st.button(f"Supprimer la dernière étape du train {id_train_manuel}", key=f"train{id_train_manuel}_remove_last"):
                    st.session_state.roulement_manuel[id_train_manuel].pop()
                    st.success(f"Dernière étape supprimée du train {id_train_manuel}")
                    st.rerun()

    with onglet2:
        st.subheader("Vue d'ensemble des roulements")
        apercu_toutes_etapes_manuelles = []
        for id_train_apercu, etapes_train_apercu in st.session_state.roulement_manuel.items():
            if not etapes_train_apercu: continue
            for indice_etape_apercu, etape_apercu in enumerate(etapes_train_apercu):
                apercu_toutes_etapes_manuelles.append({
                    "Train": id_train_apercu, "Étape": indice_etape_apercu + 1,
                    "Départ": etape_apercu["depart"], "Heure départ": etape_apercu["heure_depart"],
                    "Arrivée": etape_apercu["arrivee"], "Heure arrivée": etape_apercu["heure_arrivee"],
                    "Temps trajet (min)": etape_apercu["temps_trajet"]
                })
        if apercu_toutes_etapes_manuelles:
            df_apercu_manuel = pd.DataFrame(apercu_toutes_etapes_manuelles)
            st.dataframe(df_apercu_manuel, use_container_width=True)
            if st.download_button("Télécharger la configuration (CSV)", df_apercu_manuel.to_csv(index=False, sep=";", encoding='utf-8-sig').encode('utf-8-sig'), file_name="config_roulements.csv", mime="text/csv"):
                st.success("Configuration téléchargée.")
        else:
            st.warning("Aucun roulement manuel défini.")

if mode_generation == "Manuel" and any(st.session_state.roulement_manuel.values()):
    st.header("5. Vérification de cohérence des fréquences (Mode Manuel)")
    sections_analyse_frequence = {}
    for indice_mission_coherence, mission_coherence in enumerate(st.session_state.missions):
        origine_mission_coh = mission_coherence["origine"]
        terminus_mission_coh = mission_coherence["terminus"]
        frequence_mission_coh = mission_coherence["frequence"]
        cle_affichage_section_coh = f"{origine_mission_coh} → {terminus_mission_coh}"

        if cle_affichage_section_coh not in sections_analyse_frequence:
            sections_analyse_frequence[cle_affichage_section_coh] = {
                "freq_objectif": frequence_mission_coh,
                "trains_par_heure": {},
                "origine_mission": origine_mission_coh,
                "terminus_mission": terminus_mission_coh,
                "passing_points_mission": mission_coherence.get("passing_points", []),
                "temps_trajet_mission": mission_coherence["temps_trajet"]
            }
        else:
            sections_analyse_frequence[cle_affichage_section_coh]["freq_objectif"] = max(sections_analyse_frequence[cle_affichage_section_coh]["freq_objectif"], frequence_mission_coh)

    comptes_segments_coh = {}
    for id_train_coh, etapes_train_coh in st.session_state.roulement_manuel.items():
        for etape_coh in etapes_train_coh:
            gare_depart_etape_coh = etape_coh["depart"]
            gare_arrivee_etape_coh = etape_coh["arrivee"]
            cle_segment_coh = f"{gare_depart_etape_coh} → {gare_arrivee_etape_coh}"

            if cle_segment_coh not in comptes_segments_coh:
                comptes_segments_coh[cle_segment_coh] = {"trains_par_heure": {}}

            heure_depart_etape_coh = datetime.strptime(etape_coh["heure_depart"], "%H:%M").hour
            comptes_actuels_segment_coh = comptes_segments_coh[cle_segment_coh]["trains_par_heure"]
            comptes_actuels_segment_coh[heure_depart_etape_coh] = comptes_actuels_segment_coh.get(heure_depart_etape_coh, 0) + 1

    for cle_mission_coh, details_mission_coh in sections_analyse_frequence.items():
        st.subheader(f"Analyse pour Mission: {cle_mission_coh}")
        frequence_objectif_coh = details_mission_coh["freq_objectif"]

        chemin_mission_coh = [(details_mission_coh["origine_mission"], 0)]
        points_passage_actuels_coh = details_mission_coh.get("passing_points_mission", [])
        if isinstance(points_passage_actuels_coh, list):
            for point_passage_coh in points_passage_actuels_coh:
                 if isinstance(point_passage_coh, dict) and "gare" in point_passage_coh and "temps_depuis_origine" in point_passage_coh:
                    chemin_mission_coh.append((point_passage_coh["gare"], point_passage_coh["temps_depuis_origine"]))
        chemin_mission_coh.append((details_mission_coh["terminus_mission"], details_mission_coh["temps_trajet_mission"]))
        chemin_mission_coh.sort(key=lambda x: x[1])

        chemin_mission_unique_coh = []
        dernier_element_chemin_unique_coh = None
        for element_chemin_coh in chemin_mission_coh:
            if not dernier_element_chemin_unique_coh or not (element_chemin_coh[0] == dernier_element_chemin_unique_coh[0] and element_chemin_coh[1] == dernier_element_chemin_unique_coh[1]):
                chemin_mission_unique_coh.append(element_chemin_coh)
            dernier_element_chemin_unique_coh = element_chemin_coh
        chemin_mission_coh = chemin_mission_unique_coh

        total_heures_respectees_coh = 0
        total_heures_segments_analysees_coh = 0
        tous_segments_pleinement_respectes_coh = True

        for indice_segment_chemin_coh in range(len(chemin_mission_coh) - 1):
            origine_segment_coh, _ = chemin_mission_coh[indice_segment_chemin_coh]
            terminus_segment_coh, _ = chemin_mission_coh[indice_segment_chemin_coh+1]
            if origine_segment_coh == terminus_segment_coh: continue

            cle_segment_actuel_a_verifier_coh = f"{origine_segment_coh} → {terminus_segment_coh}"
            trains_reels_sur_segment_par_heure_coh = comptes_segments_coh.get(cle_segment_actuel_a_verifier_coh, {}).get("trains_par_heure", {})

            st.markdown(f"**Segment de la mission: {cle_segment_actuel_a_verifier_coh}**")
            donnees_analyse_segment_coh = []
            segment_respecte_pour_cette_mission_coh = True

            liste_heures_service_coh = list(range(int(heure_debut_service.hour), int(heure_fin_service.hour) + 1))

            if not liste_heures_service_coh:
                st.info(f"Plage de service nulle pour le segment {cle_segment_actuel_a_verifier_coh}, analyse non applicable.")
                tous_segments_pleinement_respectes_coh = False
                continue

            if frequence_objectif_coh >= 1.0:
                for heure_analyse_coh in liste_heures_service_coh:
                    nombre_trains_reels_coh = trains_reels_sur_segment_par_heure_coh.get(heure_analyse_coh, 0)
                    statut_heure_coh = "✓" if nombre_trains_reels_coh >= frequence_objectif_coh else "❌"
                    if statut_heure_coh == "❌":
                        segment_respecte_pour_cette_mission_coh = False
                        tous_segments_pleinement_respectes_coh = False
                    donnees_analyse_segment_coh.append({
                        "Heure": f"{heure_analyse_coh:02d}:00", "Trains sur segment": nombre_trains_reels_coh,
                        "Objectif Mission": f"{frequence_objectif_coh:.1f}", "Statut": statut_heure_coh
                    })
                    if statut_heure_coh == "✓":
                        total_heures_respectees_coh += 1
                    total_heures_segments_analysees_coh += 1
            else:
                longueur_creneau_heures_coh = 0
                if frequence_objectif_coh > 0:
                    longueur_creneau_heures_coh = int(round(1.0 / frequence_objectif_coh))
                else:
                    longueur_creneau_heures_coh = len(liste_heures_service_coh) + 1

                statut_par_heure_coh = {h: "Pending" for h in liste_heures_service_coh}

                for heure_debut_creneau_coh in liste_heures_service_coh:
                    if statut_par_heure_coh[heure_debut_creneau_coh] != "Pending":
                        continue

                    train_trouve_dans_creneau_coh = False
                    if longueur_creneau_heures_coh > 0 and longueur_creneau_heures_coh <= len(liste_heures_service_coh) +1:
                        for decalage_heure_dans_creneau_coh in range(longueur_creneau_heures_coh):
                            heure_a_verifier_dans_creneau_coh = heure_debut_creneau_coh + decalage_heure_dans_creneau_coh
                            if heure_a_verifier_dans_creneau_coh > heure_fin_service.hour:
                                break
                            if trains_reels_sur_segment_par_heure_coh.get(heure_a_verifier_dans_creneau_coh, 0) > 0:
                                train_trouve_dans_creneau_coh = True
                                break

                    for indice_couverture_creneau_coh in range(longueur_creneau_heures_coh):
                        heure_a_marquer_dans_creneau_coh = heure_debut_creneau_coh + indice_couverture_creneau_coh
                        if heure_a_marquer_dans_creneau_coh <= heure_fin_service.hour:
                            if statut_par_heure_coh[heure_a_marquer_dans_creneau_coh] == "Pending":
                                statut_par_heure_coh[heure_a_marquer_dans_creneau_coh] = "✓" if train_trouve_dans_creneau_coh else "❌"
                        else:
                            break

                for heure_analyse_coh in liste_heures_service_coh:
                    nombre_trains_reels_affichage_coh = trains_reels_sur_segment_par_heure_coh.get(heure_analyse_coh, 0)
                    statut_heure_coh = statut_par_heure_coh[heure_analyse_coh]
                    if statut_heure_coh == "❌":
                        segment_respecte_pour_cette_mission_coh = False
                        tous_segments_pleinement_respectes_coh = False

                    texte_objectif_mission_affichage_coh = f"1 train / {longueur_creneau_heures_coh}h" if longueur_creneau_heures_coh > 0 and longueur_creneau_heures_coh <= len(liste_heures_service_coh) else f"Moy. {frequence_objectif_coh:.2f}/h"
                    donnees_analyse_segment_coh.append({
                        "Heure": f"{heure_analyse_coh:02d}:00", "Trains sur segment": nombre_trains_reels_affichage_coh,
                        "Objectif Mission": texte_objectif_mission_affichage_coh, "Statut": statut_heure_coh
                    })
                    if statut_heure_coh == "✓":
                        total_heures_respectees_coh += 1
                    total_heures_segments_analysees_coh += 1

            if donnees_analyse_segment_coh:
                df_analyse_segment_coh = pd.DataFrame(donnees_analyse_segment_coh)
                st.dataframe(df_analyse_segment_coh, use_container_width=True)
                if not segment_respecte_pour_cette_mission_coh:
                     st.warning(f"L'objectif n'est pas respecté pour toutes les heures ou plages sur le segment {cle_segment_actuel_a_verifier_coh}.")

        if total_heures_segments_analysees_coh > 0:
            pourcentage_moyen_conformite_mission_coh = (total_heures_respectees_coh / total_heures_segments_analysees_coh) * 100
            if tous_segments_pleinement_respectes_coh:
                st.success(f"✅ Objectif de fréquence respecté à 100% pour TOUS les segments et TOUTES les heures de cette mission.")
            elif pourcentage_moyen_conformite_mission_coh >=75 :
                 st.warning(f"⚠️ Objectif de fréquence globalement respecté à {pourcentage_moyen_conformite_mission_coh:.1f}% pour l'ensemble des segments de cette mission.")
            else:
                 st.error(f"❌ Objectif de fréquence globalement respecté seulement à {pourcentage_moyen_conformite_mission_coh:.1f}% pour l'ensemble des segments de cette mission.")
        elif len(chemin_mission_coh) > 1 :
             st.info("Analyse de conformité non applicable (pas d'heures de service à analyser pour les segments de cette mission).")
        else:
            st.info("Analyse de conformité non applicable (la mission n'a pas de segments définis).")

st.header("6. Paramètres d'affichage graphique")
datetime_debut_service_obj = datetime.combine(datetime.min, heure_debut_service)
datetime_fin_service_obj = datetime.combine(datetime.min, heure_fin_service)
if datetime_fin_service_obj <= datetime_debut_service_obj:
    datetime_fin_service_obj += timedelta(days=1)
duree_service_secondes = (datetime_fin_service_obj - datetime_debut_service_obj).total_seconds()
duree_service_heures = duree_service_secondes / 3600

fenetre_graphique_heures = st.number_input("Durée de la fenêtre graphique (h)", min_value=1, max_value=max(1,int(duree_service_heures)) or 24, value=min(5, max(1,int(duree_service_heures)) or 5) )

valeur_max_slider_fenetre = max(0, int(duree_service_heures - fenetre_graphique_heures))
decalage_heure_debut_fenetre = 0
if valeur_max_slider_fenetre > 0 :
    decalage_heure_debut_fenetre = st.slider("Début de la fenêtre horaire (h depuis début service)", 0, valeur_max_slider_fenetre, 0)
elif duree_service_heures > 0 :
    st.caption(f"La durée de service ({duree_service_heures:.1f}h) est inférieure ou égale à la fenêtre graphique ({fenetre_graphique_heures}h). Affichage complet.")
else:
    st.caption("Durée de service nulle ou invalide. Vérifiez les heures de début et de fin.")

lancer_calcul_graphique = st.button("Lancer le calcul et afficher le graphique")

if lancer_calcul_graphique:
    df_gares_triees = dataframe_gares.sort_values("distance").reset_index(drop=True)
    gare_vers_position_y = {ligne["gare"]: ligne["distance"] for _, ligne in df_gares_triees.iterrows()}
    chronologie_trajets = {}

    if mode_generation == "Rotation optimisée":
        decalage_a_utiliser_opt = decalage_par_gare_missions if 'decalage_par_gare_missions' in locals() else 0

        def generer_tous_trajets_optimises(liste_missions_utilisateur, heure_debut_service_optim_time, heure_fin_service_optim_time, df_gares_local_optim, decalage_par_gare_saisi=0):
            chronologie_complete_optim = {}
            id_train_actuel_optim = 1

            datetime_debut_service_optim = datetime.combine(datetime.today(), heure_debut_service_optim_time)
            datetime_fin_service_optim = datetime.combine(datetime.today(), heure_fin_service_optim_time)

            if datetime_fin_service_optim.time() <= datetime_debut_service_optim.time() and datetime_fin_service_optim <= datetime_debut_service_optim :
                datetime_fin_service_optim += timedelta(days=1)

            trains_disponibles_optim = [] # Structure: {"id_train": ..., "disponible_a_partir_de": ..., "en_gare_actuelle": ...}
            evenements = [] # Structure: {"type": "...", "heure": ..., "details": ...}

            def preparer_horaire_mission_interne(config_mission_interne):
                # ... (Identique à la version précédente)
                horaire_brut_mission = [{"gare": config_mission_interne["origine"], "time_offset_min": 0}]
                points_passage_mission_interne = config_mission_interne.get("passing_points", [])
                if isinstance(points_passage_mission_interne, list):
                    for point_passage_interne in points_passage_mission_interne:
                        if isinstance(point_passage_interne, dict) and "gare" in point_passage_interne and "temps_depuis_origine" in point_passage_interne:
                            horaire_brut_mission.append({"gare": point_passage_interne["gare"], "time_offset_min": point_passage_interne["temps_depuis_origine"]})
                horaire_brut_mission.append({"gare": config_mission_interne["terminus"], "time_offset_min": config_mission_interne["temps_trajet"]})
                horaire_brut_mission.sort(key=lambda x: x["time_offset_min"])

                horaire_unique_mission = []
                dernier_point_horaire = None
                for point_horaire in horaire_brut_mission:
                    if not dernier_point_horaire or not (point_horaire["gare"] == dernier_point_horaire["gare"] and point_horaire["time_offset_min"] == dernier_point_horaire["time_offset_min"]):
                        horaire_unique_mission.append(point_horaire)
                    dernier_point_horaire = point_horaire
                return horaire_unique_mission

            # 1. Générer les événements de départ "aller" initiaux
            decalages_depart_par_mission_origine = {} # Pour gérer le décalage entre séries de missions 'aller'
            for indice_mission_aller, mission_originale_aller in enumerate(liste_missions_utilisateur):
                minutes_reference_parsees = []
                minutes_reference_texte_optim = mission_originale_aller.get("reference_minutes", "0")
                try:
                    minutes_brutes_texte = [m.strip() for m in minutes_reference_texte_optim.split(',') if m.strip()]
                    for minute_ref_texte in minutes_brutes_texte:
                        if minute_ref_texte.isdigit():
                            minute_ref_entier = int(minute_ref_texte)
                            if 0 <= minute_ref_entier <= 59:
                                if minute_ref_entier not in minutes_reference_parsees:
                                    minutes_reference_parsees.append(minute_ref_entier)
                            else:
                                st.warning(f"Minute de référence {minute_ref_entier} hors plage (0-59) pour mission Aller {indice_mission_aller+1} ({mission_originale_aller.get('origine','?')}), ignorée.")
                    if not minutes_reference_parsees and minutes_brutes_texte:
                         st.warning(f"Aucune minute de référence valide (mission Aller {indice_mission_aller+1}, {mission_originale_aller.get('origine','?')}) pour '{minutes_reference_texte_optim}'. Utilisation de :00 par défaut.")
                         minutes_reference_parsees = [0]
                    elif not minutes_reference_parsees and not minutes_brutes_texte:
                         minutes_reference_parsees = [0]
                    minutes_reference_parsees.sort()
                except Exception as e:
                    st.warning(f"Erreur lors du parsing des minutes de référence '{minutes_reference_texte_optim}' (mission Aller {indice_mission_aller+1}, {mission_originale_aller.get('origine','?')}): {e}. Utilisation de :00 par défaut.")
                    minutes_reference_parsees = [0]

                if mission_originale_aller["frequence"] <= 0:
                    continue
                intervalle_mission_optim = timedelta(hours=1 / mission_originale_aller["frequence"])
                decalage_actuel_gare_origine_minutes = decalages_depart_par_mission_origine.get(mission_originale_aller["origine"], 0)

                for minute_reference_actuelle in minutes_reference_parsees:
                    debut_effectif_creneau_mission = datetime_debut_service_optim + timedelta(minutes=decalage_actuel_gare_origine_minutes)
                    premier_depart_tentative = debut_effectif_creneau_mission.replace(minute=minute_reference_actuelle, second=0, microsecond=0)

                    heure_premier_depart_reelle = premier_depart_tentative
                    if premier_depart_tentative.time() < debut_effectif_creneau_mission.time() :
                        heure_premier_depart_reelle = (debut_effectif_creneau_mission + timedelta(hours=1)).replace(minute=minute_reference_actuelle, second=0, microsecond=0)

                    if heure_premier_depart_reelle < datetime_debut_service_optim:
                        debut_absolu_temporaire_base_ref = datetime_debut_service_optim.replace(minute=minute_reference_actuelle, second=0, microsecond=0)
                        if debut_absolu_temporaire_base_ref < datetime_debut_service_optim:
                            heure_premier_depart_reelle = (datetime_debut_service_optim + timedelta(hours=1)).replace(minute=minute_reference_actuelle, second=0, microsecond=0)
                        else:
                            heure_premier_depart_reelle = debut_absolu_temporaire_base_ref

                    curseur_temps_mission_optim = heure_premier_depart_reelle
                    while curseur_temps_mission_optim < datetime_fin_service_optim and \
                          (curseur_temps_mission_optim + timedelta(minutes=mission_originale_aller["temps_trajet"])) <= datetime_fin_service_optim :
                        if curseur_temps_mission_optim >= datetime_debut_service_optim:
                            evenements.append({
                                "type": "depart_aller_planifie",
                                "heure": curseur_temps_mission_optim,
                                "details": {
                                    "mission_config": mission_originale_aller,
                                    "origine": mission_originale_aller["origine"],
                                    "terminus": mission_originale_aller["terminus"]
                                }
                            })
                        curseur_temps_mission_optim += intervalle_mission_optim

                if decalage_par_gare_saisi > 0 and minutes_reference_parsees :
                     # Applique le décalage pour la *prochaine* mission 'aller' partant de la même gare
                    decalages_depart_par_mission_origine[mission_originale_aller["origine"]] = decalage_actuel_gare_origine_minutes + decalage_par_gare_saisi

            # 2. Boucle principale de traitement des événements
            while evenements:
                evenements.sort(key=lambda x: x["heure"]) # Toujours traiter l'événement le plus proche
                evenement_actuel = evenements.pop(0)
                heure_evenement = evenement_actuel["heure"]

                if heure_evenement > datetime_fin_service_optim: # Ne pas traiter les événements hors service
                    continue

                if evenement_actuel["type"] == "depart_aller_planifie":
                    details_aller = evenement_actuel["details"]
                    config_mission_aller = details_aller["mission_config"]
                    origine_aller = details_aller["origine"]
                    terminus_aller = details_aller["terminus"]

                    train_choisi_pour_aller = None
                    trains_disponibles_optim.sort(key=lambda t: t["disponible_a_partir_de"])
                    for indice_train_dispo, etat_train in enumerate(trains_disponibles_optim):
                        if etat_train["en_gare_actuelle"] == origine_aller and etat_train["disponible_a_partir_de"] <= heure_evenement:
                            train_choisi_pour_aller = trains_disponibles_optim.pop(indice_train_dispo)
                            break

                    id_train_assigne_aller = -1
                    if train_choisi_pour_aller:
                        id_train_assigne_aller = train_choisi_pour_aller["id_train"]
                    else:
                        id_train_assigne_aller = id_train_actuel_optim
                        chronologie_complete_optim[id_train_assigne_aller] = []
                        id_train_actuel_optim +=1

                    # Générer les segments pour ce trajet aller
                    horaire_aller_prepare = preparer_horaire_mission_interne(config_mission_aller)
                    datetime_arrivee_finale_aller = heure_evenement # Init
                    segments_aller_actuel = []

                    if len(horaire_aller_prepare) < 2: continue # Mission mal formée

                    for i_seg in range(len(horaire_aller_prepare) - 1):
                        p_dep = horaire_aller_prepare[i_seg]
                        p_arr = horaire_aller_prepare[i_seg+1]
                        seg_dt_dep_abs = heure_evenement + timedelta(minutes=p_dep["time_offset_min"])
                        seg_dt_arr_abs = heure_evenement + timedelta(minutes=p_arr["time_offset_min"])

                        if seg_dt_arr_abs > datetime_fin_service_optim: break # Ne pas dépasser la fin de service

                        if seg_dt_arr_abs <= seg_dt_dep_abs and p_dep["gare"] != p_arr["gare"]: continue

                        segments_aller_actuel.append({"start": seg_dt_dep_abs, "end": seg_dt_arr_abs, "origine": p_dep["gare"], "terminus": p_arr["gare"]})
                        datetime_arrivee_finale_aller = seg_dt_arr_abs

                    if segments_aller_actuel: # Si des segments valides ont été créés
                        chronologie_complete_optim[id_train_assigne_aller].extend(segments_aller_actuel)

                        # Créer un événement pour la disponibilité du train pour un retour
                        if origine_aller != terminus_aller: # Seulement si ce n'est pas une boucle A->A
                            temps_retournement_aller = timedelta(minutes=config_mission_aller["temps_retournement"])
                            heure_dispo_pour_retour = datetime_arrivee_finale_aller + temps_retournement_aller
                            if heure_dispo_pour_retour < datetime_fin_service_optim : # Si le train peut être prêt avant la fin de service
                                evenements.append({
                                    "type": "train_disponible_pour_retour",
                                    "heure": heure_dispo_pour_retour,
                                    "details": {
                                        "id_train_concerne": id_train_assigne_aller,
                                        "en_gare": terminus_aller, # Gare où il est arrivé
                                        "mission_originale_pour_construire_retour": config_mission_aller
                                    }
                                })
                        else: # Si c'était une boucle, le rendre dispo pour un autre aller
                             temps_retournement_boucle = timedelta(minutes=config_mission_aller["temps_retournement"])
                             heure_dispo_apres_boucle = datetime_arrivee_finale_aller + temps_retournement_boucle
                             trains_disponibles_optim.append({
                                 "id_train": id_train_assigne_aller,
                                 "disponible_a_partir_de": heure_dispo_apres_boucle,
                                 "en_gare_actuelle": terminus_aller # qui est aussi l'origine
                             })


                elif evenement_actuel["type"] == "train_disponible_pour_retour":
                    details_dispo_retour = evenement_actuel["details"]
                    id_train_pour_retour = details_dispo_retour["id_train_concerne"]
                    gare_depart_retour = details_dispo_retour["en_gare"] # Terminus de l'aller
                    mission_originale = details_dispo_retour["mission_originale_pour_construire_retour"]

                    # Configurer la mission retour
                    config_mission_retour = {
                        "origine": gare_depart_retour, # Terminus de l'aller devient origine du retour
                        "terminus": mission_originale["origine"], # Origine de l'aller devient terminus du retour
                        "frequence": mission_originale["frequence"], # Non utilisé pour cadencer le retour, mais conservé
                        "temps_trajet": mission_originale["temps_trajet"],
                        "temps_retournement": mission_originale["temps_retournement"], # Sera utilisé à la fin du retour
                        "reference_minutes": mission_originale["reference_minutes"], # Non utilisé pour cadencer le retour
                        "passing_points": []
                    }
                    temps_total_trajet_ref = mission_originale["temps_trajet"]
                    pp_inverses_retour = []
                    pp_actuels_aller_ref = mission_originale.get("passing_points", [])
                    if isinstance(pp_actuels_aller_ref, list):
                        for pp_aller_ref in reversed(pp_actuels_aller_ref):
                            if isinstance(pp_aller_ref, dict) and "gare" in pp_aller_ref and "temps_depuis_origine" in pp_aller_ref:
                                pp_inverses_retour.append({
                                    "gare": pp_aller_ref["gare"],
                                    "temps_depuis_origine": temps_total_trajet_ref - pp_aller_ref["temps_depuis_origine"]
                                })
                    config_mission_retour["passing_points"] = sorted(pp_inverses_retour, key=lambda x: x["temps_depuis_origine"])

                    # Générer les segments pour ce trajet retour
                    horaire_retour_prepare = preparer_horaire_mission_interne(config_mission_retour)
                    datetime_arrivee_finale_retour = heure_evenement # Heure de départ du retour
                    segments_retour_actuel = []

                    if len(horaire_retour_prepare) < 2: continue # Mission retour mal formée

                    for i_seg_r in range(len(horaire_retour_prepare) - 1):
                        p_dep_r = horaire_retour_prepare[i_seg_r]
                        p_arr_r = horaire_retour_prepare[i_seg_r+1]
                        seg_dt_dep_abs_r = heure_evenement + timedelta(minutes=p_dep_r["time_offset_min"]) # heure_evenement est l'heure de départ du retour
                        seg_dt_arr_abs_r = heure_evenement + timedelta(minutes=p_arr_r["time_offset_min"])

                        if seg_dt_arr_abs_r > datetime_fin_service_optim: break

                        if seg_dt_arr_abs_r <= seg_dt_dep_abs_r and p_dep_r["gare"] != p_arr_r["gare"]: continue

                        segments_retour_actuel.append({"start": seg_dt_dep_abs_r, "end": seg_dt_arr_abs_r, "origine": p_dep_r["gare"], "terminus": p_arr_r["gare"]})
                        datetime_arrivee_finale_retour = seg_dt_arr_abs_r

                    if segments_retour_actuel:
                        if id_train_pour_retour not in chronologie_complete_optim: # Devrait déjà exister
                            chronologie_complete_optim[id_train_pour_retour] = []
                        chronologie_complete_optim[id_train_pour_retour].extend(segments_retour_actuel)

                        # Rendre le train disponible à son point d'origine initial pour un prochain aller
                        temps_retournement_fin_retour = timedelta(minutes=config_mission_retour["temps_retournement"])
                        heure_dispo_apres_retour = datetime_arrivee_finale_retour + temps_retournement_fin_retour
                        if heure_dispo_apres_retour < datetime_fin_service_optim:
                            trains_disponibles_optim.append({
                                "id_train": id_train_pour_retour,
                                "disponible_a_partir_de": heure_dispo_apres_retour,
                                "en_gare_actuelle": config_mission_retour["terminus"] # Origine initiale de la mission aller
                            })

            return chronologie_complete_optim

        chronologie_trajets = generer_tous_trajets_optimises(st.session_state.missions, heure_debut_service, heure_fin_service, dataframe_gares, decalage_a_utiliser_opt)

    else: # Mode Manuel
        for id_train_manuel_graph, etapes_train_manuel_graph in st.session_state.roulement_manuel.items():
            if not etapes_train_manuel_graph: continue
            chronologie_trajets[id_train_manuel_graph] = []
            for etape_graph in etapes_train_manuel_graph:
                try:
                    heure_debut_obj_graph = datetime.strptime(etape_graph["heure_depart"], "%H:%M").time()
                    heure_fin_obj_graph = datetime.strptime(etape_graph["heure_arrivee"], "%H:%M").time()

                    datetime_debut_service_initial_graph = datetime.combine(datetime.today(), heure_debut_service)

                    datetime_debut_segment_actuel_graph = datetime.combine(datetime_debut_service_initial_graph.date(), heure_debut_obj_graph)
                    datetime_fin_segment_actuel_graph = datetime.combine(datetime_debut_service_initial_graph.date(), heure_fin_obj_graph)

                    if chronologie_trajets[id_train_manuel_graph]:
                        datetime_fin_dernier_segment_graph = chronologie_trajets[id_train_manuel_graph][-1]["end"]
                        if datetime_debut_segment_actuel_graph < datetime_fin_dernier_segment_graph and datetime_debut_segment_actuel_graph.time() < datetime_fin_dernier_segment_graph.time():
                             datetime_debut_segment_actuel_graph += timedelta(days=1)
                             datetime_fin_segment_actuel_graph += timedelta(days=1)


                    if datetime_fin_segment_actuel_graph < datetime_debut_segment_actuel_graph :
                         datetime_fin_segment_actuel_graph += timedelta(days=1)

                    chronologie_trajets[id_train_manuel_graph].append({
                        "start": datetime_debut_segment_actuel_graph,
                        "end": datetime_fin_segment_actuel_graph,
                        "origine": etape_graph["depart"],
                        "terminus": etape_graph["arrivee"]
                    })
                except ValueError as e:
                    st.error(f"Erreur de format d'heure pour le train {id_train_manuel_graph}, étape {etape_graph['depart']}->{etape_graph['arrivee']}: {e}. Étape ignorée pour le graphique.")

    st.header("Graphique horaire")
    if not chronologie_trajets or all(not trajets for trajets in chronologie_trajets.values()):
        st.warning("Aucun train à afficher. Vérifiez la configuration des missions ou les roulements manuels.")
    else:
        figure_graph, axes_graph = plt.subplots(figsize=(17, 6))
        try:
            palette_couleurs_graph = plt.colormaps.get_cmap('tab20')
            nombre_couleurs_palette = palette_couleurs_graph.N
        except AttributeError:
            palette_couleurs_graph = plt.cm.get_cmap('tab20')
            nombre_couleurs_palette = len(palette_couleurs_graph.colors)

        ids_trains_tries_graph = sorted(chronologie_trajets.keys())

        for indice_train_graph, id_train_graph in enumerate(ids_trains_tries_graph):
            trajets_train_graph = chronologie_trajets[id_train_graph]
            if not trajets_train_graph: continue

            couleur_train_graph = palette_couleurs_graph(indice_train_graph % nombre_couleurs_palette if nombre_couleurs_palette > 0 else 0)

            trajets_train_graph.sort(key=lambda t: t["start"])

            for indice_trajet_graph, trajet_graph in enumerate(trajets_train_graph):
                if trajet_graph["origine"] not in gare_vers_position_y or trajet_graph["terminus"] not in gare_vers_position_y:
                    st.error(f"Gare non reconnue dans le trajet: {trajet_graph['origine']} ou {trajet_graph['terminus']}. Segment ignoré pour le train {id_train_graph}.")
                    continue

                coordonnees_x_trajet = [trajet_graph["start"], trajet_graph["end"]]
                coordonnees_y_trajet = [gare_vers_position_y[trajet_graph["origine"]], gare_vers_position_y[trajet_graph["terminus"]]]

                axes_graph.plot(coordonnees_x_trajet, coordonnees_y_trajet, marker='o', markersize=4, label=f"Train {id_train_graph}" if indice_trajet_graph == 0 else "", color=couleur_train_graph, linewidth=1.5)

                if indice_trajet_graph < len(trajets_train_graph) - 1:
                    prochain_trajet_graph = trajets_train_graph[indice_trajet_graph+1]
                    if trajet_graph["terminus"] == prochain_trajet_graph["origine"] and prochain_trajet_graph["start"] > trajet_graph["end"]:
                        position_y_gare_attente = gare_vers_position_y[trajet_graph["terminus"]]
                        axes_graph.plot([trajet_graph["end"], prochain_trajet_graph["start"]], [position_y_gare_attente, position_y_gare_attente], linestyle='--', color=couleur_train_graph, alpha=0.7, linewidth=1.0)

        axes_graph.set_yticks(df_gares_triees["distance"])
        axes_graph.set_yticklabels([f"{ligne['gare']} ({ligne['distance']:.0f} km)" for _, ligne in df_gares_triees.iterrows()], fontsize=8)
        axes_graph.set_ylabel("Gares et Distance (km)")

        datetime_debut_affichage_initial_graph = datetime.combine(datetime.today(), heure_debut_service)
        limite_debut_x_graph = datetime_debut_affichage_initial_graph + timedelta(hours=decalage_heure_debut_fenetre)
        limite_fin_x_graph = limite_debut_x_graph + timedelta(hours=fenetre_graphique_heures)
        axes_graph.set_xlim(limite_debut_x_graph, limite_fin_x_graph)

        if fenetre_graphique_heures <= 2:
            axes_graph.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
            axes_graph.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0,60,5)))
        elif fenetre_graphique_heures <= 6:
            axes_graph.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            axes_graph.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0,30]))
        elif fenetre_graphique_heures <= 12:
            axes_graph.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            axes_graph.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        else:
            axes_graph.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(fenetre_graphique_heures/12))))
            axes_graph.xaxis.set_minor_locator(mdates.HourLocator(interval=max(1, int(fenetre_graphique_heures/24))))

        axes_graph.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.xticks(rotation=45, ha="right")

        axes_graph.grid(True, which="both", axis="both", linestyle=":", color="grey", alpha=0.5)
        axes_graph.set_xlabel("Heure")
        axes_graph.set_title(f"Graphique horaire ferroviaire ({len(chronologie_trajets)} trains/services) - Fenêtre de {fenetre_graphique_heures}h")

        elements_legende_graph, labels_legende_graph = axes_graph.get_legend_handles_labels()
        if elements_legende_graph:
            legende_par_label_graph = dict(zip(labels_legende_graph[:20], elements_legende_graph[:20]))
            axes_graph.legend(legende_par_label_graph.values(), legende_par_label_graph.keys(), loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        st.pyplot(figure_graph)

        lignes_export_excel = []
        for id_train_export, trajets_train_export in chronologie_trajets.items():
            for trajet_export in trajets_train_export:
                 lignes_export_excel.append({
                    "Train": id_train_export,
                    "Début": trajet_export["start"].strftime("%Y-%m-%d %H:%M"),
                    "Fin": trajet_export["end"].strftime("%Y-%m-%d %H:%M"),
                    "Origine": trajet_export["origine"],
                    "Terminus": trajet_export["terminus"]
                })
        if lignes_export_excel:
            df_export_excel = pd.DataFrame(lignes_export_excel)
            tampon_excel = BytesIO()
            df_export_excel.to_excel(tampon_excel, index=False, sheet_name="Roulements")
            st.download_button("Télécharger roulements (Excel)", tampon_excel.getvalue(), file_name="roulements_horaires.xlsx", mime="application/vnd.ms-excel")

            tampon_pdf = BytesIO()
            try:
                figure_graph.savefig(tampon_pdf, format="pdf", bbox_inches='tight')
                st.download_button("Télécharger graphique (PDF)", tampon_pdf.getvalue(), file_name="graphique_horaire.pdf", mime="application/pdf")
            except Exception as erreur_generation_pdf:
                st.error(f"Erreur lors de la génération du PDF: {erreur_generation_pdf}")

        st.caption("Chaque série de segments de même couleur représente un service de train. Marqueurs: arrêts. Les lignes horizontales en pointillés indiquent un temps d'arrêt/retournement en gare.")

