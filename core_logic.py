# -*- coding: utf-8 -*-
"""
core_logic.py

Ce module contient le "cerveau" de l'application : les fonctions de calcul complexes
qui ne dépendent pas de l'interface utilisateur.

"""
from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO, StringIO
import heapq
import bisect
from functools import lru_cache
from collections import defaultdict
import json


def _find_end_of_single_track_block(df_gares, start_idx, direction):
    """
    Trouve l'index de la prochaine gare qui permet un croisement (VE, D, ou Terminus)
    dans la direction donnée.
    Retourne l'index de la gare de fin de bloc.
    """
    n = len(df_gares)
    curr = start_idx + direction
    while 0 <= curr < n:
        gare_name = df_gares.iloc[curr]['gare']
        infra = _get_infra_at_gare(df_gares, gare_name)
        # Une gare VE ou D marque la fin du cantonnement de voie unique "strict"
        # (D = début double voie, VE = évitement).
        if infra in ['VE', 'D']:
            return curr
        curr += direction
    return start_idx # Ne devrait pas arriver si terminus bien défini


def _get_infra_at_gare(df_gares, gare_name):
    """Helper pour obtenir la valeur 'infra' d'une gare."""
    try:
        # Assurer que df_gares a bien une colonne 'gare' et 'infra'
        if df_gares is None or 'gare' not in df_gares.columns or 'infra' not in df_gares.columns:
            return 'F'

        # Utiliser .loc pour éviter SettingWithCopyWarning potentiel si df_gares est une slice
        infra_series = df_gares.loc[df_gares['gare'] == gare_name, 'infra']
        if infra_series.empty:
            return 'F' # Gare non trouvée

        infra = infra_series.iloc[0]
        return str(infra).strip().upper() if pd.notna(infra) else 'F' # Défaut F si NaN
    except (IndexError, KeyError) as e:
        return 'F' # Défaut F en cas d'erreur


def _can_cross_at(df_gares, gare_name, is_terminus=False):
    """
    Retourne True si la gare permet un croisement (Voie d'Évitement).
    Un point 'D' ne permet pas de croisement *sur place*.
    """
    infra_type = _get_infra_at_gare(df_gares, gare_name)

    if infra_type == 'VE':
        return True
    elif infra_type in ['D', 'F']:
        return False

    for col in ['can_cross', 'allow_cross']:
        if col in df_gares.columns:
            try:
                # Utiliser .loc pour être sûr
                val_series = df_gares.loc[df_gares['gare'] == gare_name, col]
                if val_series.empty: continue # Gare non trouvée pour cette colonne
                val = val_series.iloc[0]
                if pd.isna(val): continue
                if isinstance(val, str):
                    return val.strip().upper() not in ('F', 'FALSE', '0', '')
                return bool(val)
            except (IndexError, KeyError):
                pass # Essayer colonne suivante

    # Par défaut : les terminus permettent le croisement/stationnement
    return is_terminus


def generer_tous_trajets_optimises(missions, heure_debut, heure_fin, df_gares):
    """
    Génère la chronologie complète des trajets en mode "Rotation optimisée".
    VERSION CORRIGÉE : Gestion des conflits stricte (pas de passage forcé).
    Logique avertissement points 'D' corrigée.
    Planification retour corrigée.
    """
    # Validation initiale df_gares
    if df_gares is None or df_gares.empty or 'gare' not in df_gares.columns or 'distance' not in df_gares.columns:
         # Utiliser st.error si disponible (exécuté dans Streamlit), sinon print
         error_msg = "DataFrame des gares invalide ou vide."
         if st: st.error(error_msg)
         else: print(f"Erreur: {error_msg}")
         return {}, {"infra_violations": [], "other": [error_msg]}

    # S'assurer que 'infra' existe, sinon la créer avec 'F' par défaut
    if 'infra' not in df_gares.columns:
         df_gares = df_gares.copy() # Eviter modif inplace de l'original
         df_gares['infra'] = 'F'


    # --- Initialisation ---
    infra_violation_warnings = []
    other_warnings = []
    chronologie_reelle = {}
    id_train_counter = 1
    event_counter = 0

    dt_debut_service = datetime.combine(datetime.today(), heure_debut)
    dt_fin_service = datetime.combine(datetime.today(), heure_fin)
    if dt_fin_service <= dt_debut_service:
        dt_fin_service += timedelta(days=1)

    trains = {} # Stocke l'état {id: {"loc": gare, "dispo_a": datetime}}
    occupation_cantons = defaultdict(list) # {(gare_a, gare_b): [(start_dt, end_dt, train_id), ...]}
    evenements = [] # Heapq : (datetime, counter, type_event, details)

    # --- Préparation des demandes de départ ---
    gare_set = set(df_gares['gare']) # Pour vérification rapide
    for i, mission in enumerate(missions):
        mission_id = f"M{i+1}" # Pour logs
        # Validation Mission
        if not isinstance(mission, dict):
             other_warnings.append(f"Mission {mission_id} invalide (n'est pas un dictionnaire). Ignorée.")
             continue
        origine = mission.get("origine")
        terminus = mission.get("terminus")
        frequence = mission.get("frequence")

        # Vérifier gares
        if not origine or origine not in gare_set:
            other_warnings.append(f"Origine '{origine}' de {mission_id} introuvable. Mission ignorée.")
            continue
        if not terminus or terminus not in gare_set:
            other_warnings.append(f"Terminus '{terminus}' de {mission_id} introuvable. Mission ignorée.")
            continue
        if origine == terminus:
             other_warnings.append(f"Origine et Terminus identiques pour {mission_id}. Mission ignorée.")
             continue

        # Vérifier fréquence
        if not isinstance(frequence, (int, float)) or frequence <= 0:
            other_warnings.append(f"Fréquence '{frequence}' invalide pour {mission_id}. Mission ignorée.")
            continue

        try: # Calcul intervalle robuste
             intervalle = timedelta(hours=1 / frequence)
             if intervalle.total_seconds() <= 0: raise ValueError("Intervalle nul ou négatif")
        except (ZeroDivisionError, ValueError):
             other_warnings.append(f"Fréquence '{frequence}' invalide (intervalle <= 0) pour {mission_id}. Mission ignorée.")
             continue


        # Vérifier minutes de référence
        try:
            minutes_ref_str = mission.get("reference_minutes", "0")
            minutes_ref = sorted(list(set([
                int(m.strip()) for m in minutes_ref_str.split(',')
                if m.strip().isdigit()
            ])))
            if not minutes_ref: minutes_ref = [0] # Défaut à 0 si vide ou invalide après filtrage
        except Exception as e:
            other_warnings.append(f"Format minutes réf. '{minutes_ref_str}' invalide pour {mission_id}: {e}. Utilisation de [0].")
            minutes_ref = [0]


        # Génération des événements de demande
        for minute_ref in minutes_ref:
            offset_hours = minute_ref // 60
            offset_minutes = minute_ref % 60

            # Calcul du premier départ théorique
            curseur_temps = dt_debut_service.replace(
                minute=offset_minutes, second=0, microsecond=0
            ) + timedelta(hours=offset_hours)

            # Ajuster au premier départ DANS ou APRES dt_debut_service
            while curseur_temps < dt_debut_service:
                 # Avancer par intervalle pour respecter le cadencement dès le début
                 curseur_temps += intervalle


            # Générer les départs jusqu'à la fin de service
            while curseur_temps < dt_fin_service:
                event_counter += 1
                heapq.heappush(evenements, (
                    curseur_temps, event_counter, "demande_depart_aller",
                    {"mission": mission, "mission_id": mission_id} # Ajouter ID pour logs
                ))
                curseur_temps += intervalle # Avancer pour le prochain

    # Sérialiser une seule fois pour le cache (si df_gares est valide)
    try:
         df_gares_json = df_gares.to_json() if df_gares is not None else None
    except Exception as e_json:
         other_warnings.append(f"Erreur sérialisation df_gares: {e_json}. Le cache horaire sera désactivé.")
         df_gares_json = None


    # --- Boucle principale de simulation ---
    while evenements:
        heure, _, type_event, details = heapq.heappop(evenements)

        # Ignorer événements après fin de service
        if heure >= dt_fin_service:
            continue

        # --- Gestion Demande de Départ ---
        if type_event == "demande_depart_aller":
            mission_cfg = details["mission"]
            mission_id = details["mission_id"]
            origine = mission_cfg["origine"]

            # Trouver le train dispo le plus tôt à l'origine
            train_assigne_id = None
            earliest_dispo = datetime.max
            for id_t, t in trains.items():
                # Vérifier si le train existe bien et a les clés nécessaires
                if isinstance(t, dict) and t.get("loc") == origine and isinstance(t.get("dispo_a"), datetime) and t["dispo_a"] <= heure:
                     if t["dispo_a"] < earliest_dispo:
                          earliest_dispo = t["dispo_a"]
                          train_assigne_id = id_t

            # Si aucun train trouvé, en créer un nouveau
            if train_assigne_id is None:
                train_assigne_id = id_train_counter
                trains[train_assigne_id] = {"id": train_assigne_id, "loc": origine, "dispo_a": heure}
                chronologie_reelle[train_assigne_id] = []
                id_train_counter += 1
            else:
                # Marquer le train comme occupé à partir de maintenant (ou de sa dispo si > heure)
                trains[train_assigne_id]["dispo_a"] = max(heure, earliest_dispo)


            # Programmer la tentative de mouvement à l'heure réelle de disponibilité/demande
            heure_programmation_mvt = max(heure, trains[train_assigne_id]["dispo_a"])
            event_counter += 1
            heapq.heappush(evenements, (heure_programmation_mvt, event_counter, "tentative_mouvement", {
                "id_train": train_assigne_id, "mission": mission_cfg, "mission_id": mission_id,
                "trajet_spec": "aller", "index_etape": 0, "retry_count": 0
            }))

        # --- Gestion Tentative de Mouvement ---
        elif type_event == "tentative_mouvement":
            # Récupération détails
            id_train = details["id_train"]
            mission_cfg = details["mission"]
            mission_id = details["mission_id"]
            trajet_spec = details["trajet_spec"]
            index_etape = details["index_etape"]

            # Obtenir l'horaire détaillé (potentiellement depuis cache)
            horaire = None # Init
            try:
                mission_key = json.dumps(mission_cfg, sort_keys=True)
                if df_gares_json:
                     horaire = construire_horaire_mission_cached(mission_key, trajet_spec, df_gares_json)
                else: # Pas de cache
                     horaire = construire_horaire_mission(mission_cfg, trajet_spec, df_gares)
            except Exception as e_horaire:
                 other_warnings.append(f"Erreur construction horaire {mission_id} ({trajet_spec}) Train {id_train}: {e_horaire}")


            # Si horaire invalide ou fin atteinte, abandonner ce mouvement
            if not horaire or not isinstance(horaire, list) or index_etape >= len(horaire) - 1:
                # print(f"Debug: Horaire invalide ou fin atteinte pour T{id_train}, M{mission_id}, {trajet_spec}, etape {index_etape}") # Debug
                continue # Le train reste où il est, dispo à "dispo_a"

            # Points de départ et arrivée de l'étape
            try:
                pt_depart = horaire[index_etape]
                pt_arrivee = horaire[index_etape + 1]
                gare_dep = pt_depart.get("gare")
                gare_arr = pt_arrivee.get("gare")
                # Validation des points
                if not isinstance(pt_depart, dict) or not isinstance(pt_arrivee, dict) or \
                   not gare_dep or not gare_arr:
                    raise ValueError("Format point horaire invalide")
            except (IndexError, ValueError) as e_pt:
                 other_warnings.append(f"Point horaire invalide {mission_id} étape {index_etape} Train {id_train}: {e_pt}. Mouvement ignoré.")
                 continue


            # Calcul durée (assurer non négative)
            duree_min_trajet = max(0, pt_arrivee.get("time_offset_min", 0) - pt_depart.get("time_offset_min", 0))

            # Heure de départ réelle = max(heure programmée, dispo train)
            dispo_train = trains.get(id_train, {}).get("dispo_a", heure) # Heure actuelle si train inconnu
            # Assurer que dispo_train est un datetime
            if not isinstance(dispo_train, datetime): dispo_train = heure

            heure_depart_reelle = max(heure, dispo_train)

            # Vérifier si départ possible avant fin service
            if heure_depart_reelle >= dt_fin_service:
                 continue

            # --- Vérification Conflits ---
            conflit = None
            # Pas de conflit si pas de mouvement ou si gares identiques (arrêt sur place)
            if duree_min_trajet > 0 and gare_dep != gare_arr:
                 try:
                      conflit = verifier_conflit(
                          heure_depart_reelle, duree_min_trajet,
                          gare_dep, gare_arr,
                          id_train, occupation_cantons, df_gares, debug=False #a modifier pour désactiver / activer debug
                      )
                 except Exception as e_conflit:
                      other_warnings.append(f"Erreur vérification conflit {gare_dep}->{gare_arr} Train {id_train}: {e_conflit}")
                      conflit = {"type": "erreur_interne", "fin_conflit": heure_depart_reelle + timedelta(minutes=1)} # Bloquer par sécurité


            # --- Gestion Résultat Conflit ---
            if conflit is None: # Pas de conflit -> Mouvement
                heure_arrivee_reelle = heure_depart_reelle + timedelta(minutes=duree_min_trajet)

                # Vérifier si arrivée avant fin service
                if heure_arrivee_reelle > dt_fin_service:
                     continue # Trajet non complété

                # Enregistrer trajet si mouvement
                if duree_min_trajet > 0 and gare_dep != gare_arr:
                    chronologie_reelle.setdefault(id_train, []).append({
                        "start": heure_depart_reelle, "end": heure_arrivee_reelle,
                        "origine": gare_dep, "terminus": gare_arr
                    })
                    # Occuper canton
                    occupation_tuple = (heure_depart_reelle, heure_arrivee_reelle, id_train)
                    # Clé directionnelle pour l'occupation
                    bisect.insort(occupation_cantons[(gare_dep, gare_arr)], occupation_tuple)

                # Mettre à jour état train (même si durée nulle, la loc change)
                trains[id_train]["loc"] = gare_arr
                trains[id_train]["dispo_a"] = heure_arrivee_reelle

                # Programmer suite
                if index_etape + 1 < len(horaire) - 1: # Encore des étapes
                    event_counter += 1
                    heapq.heappush(evenements, (
                        heure_arrivee_reelle, event_counter, "tentative_mouvement", {
                            "id_train": id_train, "mission": mission_cfg, "mission_id": mission_id,
                            "trajet_spec": trajet_spec, "index_etape": index_etape + 1, "retry_count": 0
                        }
                    ))
                else: # Fin de mission
                    event_counter += 1
                    heapq.heappush(evenements, (
                        heure_arrivee_reelle, event_counter, "fin_mission", {
                            "id_train": id_train, "mission": mission_cfg, "mission_id": mission_id,
                            "trajet_spec": trajet_spec, "gare_finale": gare_arr
                        }
                    ))

            else: # Conflit -> Attente
                retry_count = details.get("retry_count", 0)
                # Utiliser heure départ réelle + 1min comme fallback si fin_conflit invalide ou None
                fin_possible = conflit.get("fin_conflit")
                if not isinstance(fin_possible, datetime):
                     fin_possible = heure_depart_reelle + timedelta(minutes=1)
                # Assurer que fin_possible est au moins légèrement après l'heure actuelle pour éviter boucle
                fin_possible = max(fin_possible, heure_depart_reelle + timedelta(seconds=1))


                train_en_conflit = conflit.get('avec_train', 'inconnu')
                conflit_type = conflit.get("type")

                # Générer avertissement seulement la première fois et si pertinent
                if retry_count == 0:
                    gare_depart_infra = _get_infra_at_gare(df_gares, gare_dep)
                    # Avertissement seulement si retenu en 'F'
                    if gare_depart_infra == 'F' and conflit_type not in ["erreur_infra", "erreur_interne"]:
                        msg = (
                            f"Train {id_train} ({gare_dep} → {gare_arr}) : "
                            f"RETENU à {gare_dep} (point 'F') pour attendre "
                            f"Train {train_en_conflit} jusqu'à env. {fin_possible.strftime('%H:%M:%S')}."
                        )
                        infra_violation_warnings.append(msg)
                    # Avertissement pour nez-à-nez (même si en VE/D)
                    elif conflit_type == "face_a_face":
                         msg = (
                            f"Train {id_train} ({gare_dep} → {gare_arr}) : "
                            f"Conflit 'nez-à-nez' (section VU) avec Train {train_en_conflit}. "
                            f"Attente à {gare_dep} (point '{gare_depart_infra}') jusqu'à env. {fin_possible.strftime('%H:%M:%S')}."
                        )
                         infra_violation_warnings.append(msg)

                # Reprogrammer la tentative après la fin du conflit
                new_details = details.copy()
                new_details["retry_count"] = retry_count + 1
                event_counter += 1
                heapq.heappush(evenements, (fin_possible, event_counter, "tentative_mouvement", new_details))


        # --- Gestion Fin de Mission ---
        elif type_event == "fin_mission":
            id_train = details["id_train"]
            mission_cfg = details["mission"]
            mission_id = details["mission_id"]
            gare_finale = details["gare_finale"]

            # Assurer mise à jour localisation (important!)
            if id_train in trains:
                trains[id_train]["loc"] = gare_finale
                heure_arrivee_mission = heure # L'heure de l'event est l'heure d'arrivée
            else: continue # Train inconnu? Ne devrait pas arriver.


            if details["trajet_spec"] == "aller": # Fin aller -> Programmer retour
                temps_retournement = mission_cfg.get("temps_retournement_B", 10)
                # Assurer temps retournement non négatif
                if not isinstance(temps_retournement, (int, float)) or temps_retournement < 0: temps_retournement = 0

                heure_dispo_pour_retour = heure_arrivee_mission + timedelta(minutes=temps_retournement)
                trains[id_train]["dispo_a"] = heure_dispo_pour_retour


                horaire_retour_calc = None
                temps_trajet_retour_calc = -1 # Init invalide
                try:
                    mission_key = json.dumps(mission_cfg, sort_keys=True)
                    if df_gares_json:
                        horaire_retour_calc = construire_horaire_mission_cached(mission_key, "retour", df_gares_json)
                    else:
                        horaire_retour_calc = construire_horaire_mission(mission_cfg, "retour", df_gares)

                    if isinstance(horaire_retour_calc, list) and len(horaire_retour_calc) > 1:
                        temps_trajet_retour_calc = horaire_retour_calc[-1].get("time_offset_min", -1) # Récupérer durée
                        if not isinstance(temps_trajet_retour_calc, (int, float)) or temps_trajet_retour_calc < 0:
                             temps_trajet_retour_calc = -1 # Marquer comme invalide

                except Exception as e_estim_h:
                     other_warnings.append(f"Erreur calcul horaire retour pour estimation {mission_id} T{id_train}: {e_estim_h}")


                # Vérifier si durée valide calculée
                if temps_trajet_retour_calc >= 0:
                      heure_fin_retour_estimee = heure_dispo_pour_retour + timedelta(minutes=temps_trajet_retour_calc)

                      # Programmer retour si fin avant fin de service
                      if heure_fin_retour_estimee <= dt_fin_service:
                          event_counter += 1
                          heapq.heappush(evenements, (
                              heure_dispo_pour_retour, event_counter, "tentative_mouvement", {
                                  "id_train": id_train, "mission": mission_cfg, "mission_id": mission_id,
                                  "trajet_spec": "retour", "index_etape": 0, "retry_count": 0
                              }
                          ))


            else: # Fin retour -> Train dispo pour nouvel aller
                temps_retournement = mission_cfg.get("temps_retournement_A", 10)
                if not isinstance(temps_retournement, (int, float)) or temps_retournement < 0: temps_retournement = 0
                heure_dispo_finale = heure_arrivee_mission + timedelta(minutes=temps_retournement)
                trains[id_train]["dispo_a"] = heure_dispo_finale
                # Reste à gare_finale (origine A)

    # --- Nettoyage Final ---
    trains_a_supprimer = [tid for tid, trajets in chronologie_reelle.items() if not trajets]
    for tid in trains_a_supprimer:
        del chronologie_reelle[tid]

    # Retourner résultats
    return chronologie_reelle, {
        "infra_violations": infra_violation_warnings,
        "other": other_warnings
    }

# ... (construire_horaire_mission_cached) ...
@lru_cache(maxsize=256)
def construire_horaire_mission_cached(mission_key, trajet_spec, df_gares_json):
    """Version mise en cache pour accélérer les appels répétés."""
    # Gérer cas où df_gares_json est None
    if df_gares_json is None:
         try:
              mission_cfg = json.loads(mission_key)
              return None # Plus sûr de retourner None
         except:
              return None # Erreur parsing clé ou accès df_gares


    try:
        # Utiliser StringIO pour que read_json fonctionne avec une chaîne
        df_gares_local = pd.read_json(StringIO(df_gares_json))
        mission_cfg = json.loads(mission_key)
        # Appeler la fonction non cachée avec les données désérialisées
        return construire_horaire_mission(mission_cfg, trajet_spec, df_gares_local)
    except Exception as e:
         # Log l'erreur pour aider au diagnostic
         print(f"Erreur critique dans construire_horaire_mission_cached: {e}")
         import traceback
         print(traceback.format_exc())
         return None # Retourner None en cas d'erreur


# ... (construire_horaire_mission) ...
def construire_horaire_mission(mission_config, trajet_spec, df_gares):
    """
    Construit un horaire détaillé incluant TOUTES les gares physiques.
    Les time_offset_min sont les temps d'ARRIVÉE.
    Correction interpolation retour + NameError user_times.
    """
    if df_gares is None or df_gares.empty or 'gare' not in df_gares.columns or 'distance' not in df_gares.columns:
         # print("Error: df_gares invalide dans construire_horaire_mission") # Debug
         return [] # Protection

    # --- Définition horaire de base (Aller / Retour) ---
    origine_base, terminus_base = None, None
    temps_trajet_base = 0
    passing_points_base = []

    if trajet_spec == 'aller':
        origine_base = mission_config.get("origine")
        terminus_base = mission_config.get("terminus")
        temps_trajet_base = mission_config.get("temps_trajet", 0)
        passing_points_base = [
            pp for pp in mission_config.get("passing_points", [])
            if isinstance(pp, dict) and "gare" in pp and isinstance(pp.get("time_offset_min"), (int, float)) and pp["time_offset_min"] >= 0
        ]
    else: # Retour
        origine_base = mission_config.get("terminus")
        terminus_base = mission_config.get("origine")
        temps_trajet_base_config = mission_config.get("temps_trajet", 0) # Base pour inversion

        if mission_config.get("trajet_asymetrique"):
            temps_trajet_base = mission_config.get("temps_trajet_retour", temps_trajet_base_config)
            passing_points_base = [
                pp for pp in mission_config.get("passing_points_retour", [])
                 if isinstance(pp, dict) and "gare" in pp and isinstance(pp.get("time_offset_min"), (int, float)) and pp["time_offset_min"] >= 0
            ]
            passing_points_base.sort(key=lambda x: x["time_offset_min"])
        else: # Retour symétrique (inversion)
            temps_trajet_base = temps_trajet_base_config
            pp_aller = [
                pp for pp in mission_config.get("passing_points", [])
                 if isinstance(pp, dict) and "gare" in pp and isinstance(pp.get("time_offset_min"), (int, float)) and pp["time_offset_min"] >= 0
            ]
            if isinstance(temps_trajet_base, (int, float)) and temps_trajet_base > 0:
                passing_points_base = []
                for pp in pp_aller:
                     inverted_time = temps_trajet_base - pp["time_offset_min"]
                     if inverted_time >= 0:
                          passing_points_base.append({"gare": pp["gare"], "time_offset_min": inverted_time})
                passing_points_base.sort(key=lambda x: x["time_offset_min"])
            else: passing_points_base = []

    # Validation initiale
    if not origine_base or not terminus_base or \
       not isinstance(temps_trajet_base, (int, float)) or temps_trajet_base <= 0:
         return []

    # Construire horaire de base structuré
    base_horaire_struct = [{"gare": origine_base, "time_offset_min": 0}]
    base_horaire_struct.extend(passing_points_base)
    base_horaire_struct.append({"gare": terminus_base, "time_offset_min": temps_trajet_base})

    # Dédoublonnage et tri
    seen = set()
    deduplicated_horaire = []
    for point in sorted(base_horaire_struct, key=lambda x: x.get('time_offset_min', 0)):
        if isinstance(point, dict) and 'gare' in point and 'time_offset_min' in point:
             identifier = (point['gare'], point['time_offset_min'])
             if identifier not in seen:
                 # Vérifier temps >= 0 et croissant
                 if point['time_offset_min'] >= (deduplicated_horaire[-1]['time_offset_min'] if deduplicated_horaire else -1):
                      deduplicated_horaire.append(point)
                      seen.add(identifier)
    base_horaire = deduplicated_horaire
    if len(base_horaire) < 2 or base_horaire[0]['gare'] == base_horaire[-1]['gare']:
         return []

    # --- Interpolation sur gares physiques ---
    final_horaire = []
    try:
        gares_sorted = df_gares.sort_values('distance').reset_index(drop=True)
        gares_physiques_set = set(gares_sorted['gare'])
        if origine_base not in gares_physiques_set or terminus_base not in gares_physiques_set:
             return []

        gare_to_dist = pd.Series(gares_sorted.distance.values, index=gares_sorted.gare).to_dict()
        gare_to_index = {gare: idx for idx, gare in gares_sorted['gare'].items()}

        idx_origine = gare_to_index[origine_base]
        idx_terminus = gare_to_index[terminus_base]
        direction = 1 if idx_terminus > idx_origine else -1

        user_times = {p['gare']: p['time_offset_min'] for p in base_horaire}


        # Itérer sur les segments définis par l'utilisateur (points de base)
        for i in range(len(base_horaire) - 1):
            start_point = base_horaire[i]
            end_point = base_horaire[i+1]
            gare_start, gare_end = start_point['gare'], end_point['gare']
            t1, t2 = start_point['time_offset_min'], end_point['time_offset_min']

            # Vérifier que les gares existent et récupérer distances/indices
            if gare_start not in gare_to_index or gare_end not in gare_to_index: continue
            idx_start_seg = gare_to_index[gare_start]
            idx_end_seg = gare_to_index[gare_end]
            d1, d2 = gare_to_dist[gare_start], gare_to_dist[gare_end]

            # Validation segment utilisateur
            if t2 < t1: continue # Temps décroissant

            seg_dist_total = abs(d2 - d1)
            seg_time_total = t2 - t1

            # Indices de début et fin dans gares_sorted pour ce segment
            phys_idx_start = min(idx_start_seg, idx_end_seg)
            phys_idx_end = max(idx_start_seg, idx_end_seg)

            # Sélectionner les gares physiques DANS l'ordre physique (distance)
            segment_gares_physiques_df = gares_sorted.iloc[phys_idx_start : phys_idx_end + 1]

            # Trier dans le sens du trajet (aller ou retour)
            # Utiliser sort_index est correct ici car gares_sorted est déjà trié par distance (index physique)
            segment_gares_physiques_df = segment_gares_physiques_df.sort_index(ascending=(direction == 1))


            # Interpolation pour chaque gare physique dans ce segment
            for _, station in segment_gares_physiques_df.iterrows():
                gare_actuelle = station['gare']
                dist_actuelle = gare_to_dist.get(gare_actuelle)
                if dist_actuelle is None: continue

                temps_calcule = -1 # Init
                # Utiliser temps utilisateur si défini et cohérent
                if gare_actuelle in user_times and t1 <= user_times[gare_actuelle] <= t2:
                    temps_calcule = user_times[gare_actuelle]
                # Conditions d'interpolation
                elif seg_dist_total < 1e-6: temps_calcule = t1 # Dist nulle
                elif seg_time_total <= 0: temps_calcule = t1 # Temps nul/négatif
                else: # Interpolation
                    dist_from_start = abs(dist_actuelle - d1)
                    temps_calcule = t1 + seg_time_total * (dist_from_start / seg_dist_total)

                if temps_calcule >= 0:
                     final_horaire.append({'gare': gare_actuelle, 'time_offset_min': round(temps_calcule)})

    except Exception as e:
         print(f"Erreur critique pendant interpolation horaire {origine_base}->{terminus_base} ({trajet_spec}): {e}")
         import traceback
         print(traceback.format_exc())
         return []

    # --- Nettoyage Final ---
    if not final_horaire:
        return base_horaire # Retour base si interpolation vide

    seen_final = set()
    final_deduplicated_horaire = []
    # Pré-calculer index physique pour tri stable
    final_horaire_with_index = []
    for point in final_horaire:
         # Vérifier si la gare existe encore dans le mapping (sécurité)
         if point['gare'] in gare_to_index:
             point['physical_index'] = gare_to_index[point['gare']]
             final_horaire_with_index.append(point)


    # Tri multi-critères: temps, puis index physique
    for point in sorted(final_horaire_with_index, key=lambda x: (x.get('time_offset_min', 0), x.get('physical_index', 0))):
         identifier = (point['gare'], point['time_offset_min'])
         if identifier not in seen_final:
             # Vérifier bornes globales
             if base_horaire[0]['time_offset_min'] <= point['time_offset_min'] <= base_horaire[-1]['time_offset_min']:
                  final_point = {'gare': point['gare'], 'time_offset_min': point['time_offset_min']}
                  final_deduplicated_horaire.append(final_point)
                  seen_final.add(identifier)

    # Si après tout ça, le résultat est invalide, retourner l'horaire de base
    if len(final_deduplicated_horaire) < 2:
         # Vérifier si l'horaire de base lui-même est valide
         if len(base_horaire) >= 2: return base_horaire
         else: return [] # Retourner vide si base aussi invalide

    return final_deduplicated_horaire


def verifier_conflit(h_depart, duree, gare_dep, gare_arr, id_train_courant,
                     occupation_cantons, df_gares, debug=False):
    """
    Vérifie les conflits avec une logique de "Bloc de Voie Unique".
    Si on s'engage sur une VU depuis une VE/D, on vérifie que TOUT le bloc jusqu'à la prochaine VE/D est libre.
    """
    h_arrivee = h_depart + timedelta(minutes=duree)
    if gare_dep == gare_arr or duree <= 0: return None

    gares_sorted = df_gares.sort_values('distance').reset_index(drop=True)
    gare_to_index = {g: i for i, g in enumerate(gares_sorted['gare'])}

    idx_dep = gare_to_index.get(gare_dep)
    idx_arr = gare_to_index.get(gare_arr)

    if idx_dep is None or idx_arr is None: return None # Sécurité

    direction = 1 if idx_arr > idx_dep else -1

    # 1. Identifier si on entre dans un bloc critique
    infra_dep = _get_infra_at_gare(df_gares, gare_dep)

    # Si on part d'un point de croisement (VE) ou d'une transition (D) vers une voie unique (F),
    # on doit scanner jusqu'au prochain point de croisement.
    # Note: Si infra_dep est 'F', on est déjà "au milieu" (retardé précédemment),
    # on vérifie juste le segment immédiat pour ne pas s'empiler.

    check_full_block = (infra_dep in ['VE', 'D', 'Terminus'])
    # (Note: Terminus traité comme VE implicitement dans _can_cross_at, ici simplifié)

    target_scan_idx = idx_arr

    if check_full_block:
        # Regarder si le segment immédiat est une VOIE UNIQUE
        # On regarde l'infra de la gare SUIVANTE. Si c'est F, c'est du VU.
        # Si c'est D (début double), ce n'est pas du VU critique.
        next_gare_idx = idx_dep + direction
        if 0 <= next_gare_idx < len(gares_sorted):
            next_infra = _get_infra_at_gare(df_gares, gares_sorted.iloc[next_gare_idx]['gare'])
            # Si la gare suivante est F, on s'engage en VU.
            # Si c'est VE, le bloc ne fait qu'un segment, c'est OK.
            if next_infra == 'F':
                # On doit trouver la fin du bloc
                end_block_idx = _find_end_of_single_track_block(df_gares, idx_dep, direction)
                # On scanne jusqu'à cette gare là, au lieu de juste gare_arr
                target_scan_idx = end_block_idx

    # Définition de la plage d'index à vérifier
    range_start = idx_dep
    range_end = target_scan_idx
    step = direction

    # Boucle de vérification sur tous les segments du bloc identifié
    curr = range_start
    while curr != range_end:
        idx_a = curr
        idx_b = curr + step
        gare_a = gares_sorted.iloc[idx_a]['gare']
        gare_b = gares_sorted.iloc[idx_b]['gare']

        # Vérif occupation sur ce segment (gare_a, gare_b) ou (gare_b, gare_a)
        # On cherche des trains OPPOSÉS ou MÊME SENS qui occuperaient le segment
        # pendant notre traversée estimée du BLOC entier.

        # Estimation grossière du temps de passage à ce segment précis
        # (On prend large : [h_depart, h_arrivee_du_bloc])
        # Pour être rigoureux, on devrait interpoler, mais prendre la fenêtre large est plus sûr pour la sécurité.
        t_check_start = h_depart
        t_check_end = h_arrivee + timedelta(minutes=5) # Marge de sécurité

        seg_key = (gare_a, gare_b) if gare_a < gare_b else (gare_b, gare_a)
        occupations = occupation_cantons.get(seg_key, [])

        # Mais attention, occupation_cantons stocke (start, end, id).
        # Pour un segment de voie unique, toute occupation dans la fenêtre est un danger
        # SI c'est un train adverse (nez à nez) OU un train devant (rattrapage).

        for (occ_start, occ_end, occ_id) in occupations:
            if occ_id == id_train_courant: continue

            # Chevauchement temporel ?
            if max(t_check_start, occ_start) < min(t_check_end, occ_end):
                # Conflit !
                # Si on scannait tout le bloc et qu'on trouve un train loin devant,
                # on doit attendre ICI (gare_dep) que le bloc se libère.

                return {
                    "fin_conflit": occ_end + timedelta(minutes=1),
                    "avec_train": occ_id,
                    "type": "bloc_vu_occupe",
                    "points": [gare_a, gare_b]
                }

        curr += step

    return None


# ... (generer_tous_trajets_sans_conflits) ...
def generer_tous_trajets_sans_conflits(missions, heure_debut, heure_fin, df_gares):
    chronologie_optimale = {}
    id_train_counter = 1
    warnings = []
    dt_debut_service = datetime.combine(datetime.today(), heure_debut)
    dt_fin_service = datetime.combine(datetime.today(), heure_fin)
    if dt_fin_service <= dt_debut_service: dt_fin_service += timedelta(days=1)
    trains_disponibles = []
    evenements = [] # Utiliser heapq pour la priorité temporelle

    # Helper pour ajouter event avec priorité
    evt_counter = 0
    def add_event(time, type, details):
        nonlocal evt_counter
        evt_counter += 1
        heapq.heappush(evenements, (time, evt_counter, type, details))

    # Préparer les départs initiaux
    gare_set = set(df_gares['gare']) if df_gares is not None else set()
    for i, mission in enumerate(missions):
         # Validation minimale de la mission
         if not isinstance(mission, dict) or \
            mission.get("origine") not in gare_set or \
            mission.get("terminus") not in gare_set or \
            not isinstance(mission.get("frequence"), (int, float)) or \
            mission.get("frequence", 0) <= 0:
              # warnings.append(f"Mission {i+1} invalide ou incomplète. Ignorée.")
              continue # Ignorer mission invalide

         frequence = mission["frequence"]
         intervalle = timedelta(hours=1 / frequence)

         # Minutes de référence (avec fallback)
         try:
             minutes_ref_str = mission.get("reference_minutes", "0")
             minutes_ref = sorted(list(set([int(m.strip()) for m in minutes_ref_str.split(',') if m.strip().isdigit()])))
             if not minutes_ref: minutes_ref = [0]
         except: minutes_ref = [0]

         for minute in minutes_ref:
             # Calcul premier départ
             curseur_temps = dt_debut_service.replace(minute=(minute % 60), hour=(dt_debut_service.hour + minute // 60), second=0, microsecond=0)

             # Ajuster au début de service en avançant par intervalle
             while curseur_temps < dt_debut_service:
                  curseur_temps += intervalle

             # Ajouter les départs dans la fenêtre
             while curseur_temps < dt_fin_service:
                 temps_traj_estime = mission.get("temps_trajet", 60) # Défaut 60min
                 if isinstance(temps_traj_estime, (int, float)) and temps_traj_estime >= 0:
                      heure_arrivee_estimee = curseur_temps + timedelta(minutes=temps_traj_estime)
                      if heure_arrivee_estimee <= dt_fin_service:
                           add_event(curseur_temps, "depart_aller", {"mission": mission})
                 curseur_temps += intervalle


    # Boucle de simulation sans conflit
    while evenements:
        heure_event, _, type_event, details = heapq.heappop(evenements)

        if heure_event >= dt_fin_service: continue

        if type_event == "depart_aller":
            mission_cfg = details["mission"]
            train_assigne = None
            earliest_dispo_time = datetime.max
            best_train_idx = -1
            # Trouver train dispo le plus tôt
            for i_train, train in enumerate(trains_disponibles):
                if train["gare"] == mission_cfg["origine"] and train["disponible_a"] <= heure_event:
                    if train["disponible_a"] < earliest_dispo_time:
                         earliest_dispo_time = train["disponible_a"]
                         best_train_idx = i_train

            if best_train_idx != -1:
                 train_assigne = trains_disponibles.pop(best_train_idx)
                 # Départ réel = max(heure demandée, dispo réelle)
                 heure_depart_reel = max(heure_event, earliest_dispo_time)
            else: # Nouveau train
                train_assigne = {"id": id_train_counter, "gare": mission_cfg["origine"]}
                chronologie_optimale[train_assigne["id"]] = []
                id_train_counter += 1
                heure_depart_reel = heure_event # Dispo immédiatement


            # Générer trajets aller
            horaire_aller = construire_horaire_mission(mission_cfg, "aller", df_gares)
            heure_arrivee_finale = heure_depart_reel
            trajet_aller_possible = True
            if not horaire_aller: trajet_aller_possible = False # Impossible si horaire vide

            for i in range(len(horaire_aller) - 1):
                p_dep, p_arr = horaire_aller[i], horaire_aller[i+1]
                duree_segment = max(0, p_arr.get("time_offset_min", 0) - p_dep.get("time_offset_min", 0))

                t_dep_theorique = heure_depart_reel + timedelta(minutes=p_dep.get("time_offset_min", 0))
                t_dep = max(t_dep_theorique, heure_arrivee_finale) # Ne peut partir avant arrivée précédente
                t_arr = t_dep + timedelta(minutes=duree_segment)

                if t_arr > dt_fin_service:
                     trajet_aller_possible = False; break

                if duree_segment > 0:
                     chronologie_optimale.setdefault(train_assigne["id"], []).append({
                          "start": t_dep, "end": t_arr,
                          "origine": p_dep.get("gare"), "terminus": p_arr.get("gare")
                      })
                heure_arrivee_finale = t_arr

            # Programmer retour si possible
            if trajet_aller_possible:
                temps_retournement_B = mission_cfg.get("temps_retournement_B", 10)
                heure_dispo_retour = heure_arrivee_finale + timedelta(minutes=temps_retournement_B)

                if heure_dispo_retour < dt_fin_service:
                    horaire_retour_test = construire_horaire_mission(mission_cfg, "retour", df_gares)
                    if horaire_retour_test:
                         temps_retour_estime = horaire_retour_test[-1].get("time_offset_min", 60)
                         if heure_dispo_retour + timedelta(minutes=temps_retour_estime) <= dt_fin_service:
                              add_event(heure_dispo_retour, "disponible_pour_retour", {
                                   "train": train_assigne, "gare": mission_cfg.get("terminus"), "mission_aller": mission_cfg
                               })


        elif type_event == "disponible_pour_retour":
            train = details["train"]
            mission_aller = details["mission_aller"]
            heure_depart_retour = heure_event # L'heure de l'event est la dispo

            # Générer trajets retour
            horaire_retour = construire_horaire_mission(mission_aller, "retour", df_gares)
            heure_arrivee_finale_retour = heure_depart_retour
            trajet_retour_possible = True
            if not horaire_retour: trajet_retour_possible = False

            for i in range(len(horaire_retour) - 1):
                 p_dep, p_arr = horaire_retour[i], horaire_retour[i+1]
                 duree_segment = max(0, p_arr.get("time_offset_min", 0) - p_dep.get("time_offset_min", 0))

                 t_dep_theorique = heure_depart_retour + timedelta(minutes=p_dep.get("time_offset_min", 0))
                 t_dep = max(t_dep_theorique, heure_arrivee_finale_retour)
                 t_arr = t_dep + timedelta(minutes=duree_segment)

                 if t_arr > dt_fin_service:
                      trajet_retour_possible = False; break

                 if duree_segment > 0:
                     chronologie_optimale.setdefault(train["id"], []).append({
                          "start": t_dep, "end": t_arr,
                          "origine": p_dep.get("gare"), "terminus": p_arr.get("gare")
                      })
                 heure_arrivee_finale_retour = t_arr

            # Mettre train dispo si retour complété
            if trajet_retour_possible:
                 temps_retournement_A = mission_aller.get("temps_retournement_A", 10)
                 heure_dispo_finale = heure_arrivee_finale_retour + timedelta(minutes=temps_retournement_A)
                 if heure_dispo_finale < dt_fin_service:
                     trains_disponibles.append({
                          "id": train["id"], "gare": mission_aller.get("origine"), "disponible_a": heure_dispo_finale
                      })
                     # Trier la liste des dispos peut être utile
                     trains_disponibles.sort(key=lambda t: t["disponible_a"])

    # Nettoyer chronologie vide
    trains_a_suppr = [tid for tid, traj in chronologie_optimale.items() if not traj]
    for tid in trains_a_suppr: del chronologie_optimale[tid]

    return chronologie_optimale, warnings

# ... (preparer_roulement_manuel) ...
def preparer_roulement_manuel(roulement_manuel):
    chronologie_trajets = {}
    for id_train, etapes_train in roulement_manuel.items():
        if not etapes_train: continue
        chronologie_trajets[id_train] = []
        last_arrival_time = None
        for etape in etapes_train:
            try:
                # Vérifier que les heures sont bien des strings avant conversion
                heure_dep_str = str(etape.get("heure_depart", "00:00"))
                heure_arr_str = str(etape.get("heure_arrivee", "00:00"))

                dt_debut = datetime.combine(datetime.today(), datetime.strptime(heure_dep_str, "%H:%M").time())
                dt_fin = datetime.combine(datetime.today(), datetime.strptime(heure_arr_str, "%H:%M").time())
                # Gérer passage minuit pour départ et arrivée
                if last_arrival_time and dt_debut < last_arrival_time: dt_debut += timedelta(days=1)
                if dt_fin < dt_debut: dt_fin += timedelta(days=1)
                chronologie_trajets[id_train].append({"start": dt_debut, "end": dt_fin, "origine": etape.get("depart"), "terminus": etape.get("arrivee")})
                last_arrival_time = dt_fin
            except (ValueError, KeyError, TypeError) as e: # Ajout TypeError
                print(f"Skipping invalid step for train {id_train}: {etape}. Error: {e}")
                continue
    return chronologie_trajets


# ... (importer_roulements_fichier) ...
def importer_roulements_fichier(uploaded_file, dataframe_gares):
    try:
        df_import = None
        if uploaded_file.name.endswith('.csv'):
            try:
                 # Essayer différents encodages si besoin
                 try:
                     df_import = pd.read_csv(uploaded_file, sep=';', dtype=str, encoding='utf-8')
                 except UnicodeDecodeError:
                      uploaded_file.seek(0) # Rembobiner le fichier
                      df_import = pd.read_csv(uploaded_file, sep=';', dtype=str, encoding='latin1')

            except pd.errors.ParserError as e:
                 return None, f"Erreur de parsing CSV: {e}. Vérifiez le séparateur (;) et l'encodage (UTF-8 ou Latin-1)."
            except Exception as e:
                 return None, f"Erreur lecture CSV: {e}"

        elif uploaded_file.name.endswith(('.xlsx', '.xls')): # Accepter .xls aussi
            try:
                 df_import = pd.read_excel(uploaded_file, dtype=str, engine='openpyxl' if uploaded_file.name.endswith('.xlsx') else None)
            except Exception as e:
                 return None, f"Erreur lecture Excel: {e}"
        else: # Format non supporté
             return None, "Format de fichier non supporté. Utilisez CSV (séparateur ';', encodage UTF-8 ou Latin-1) ou XLSX/XLS."

        if df_import is None:
             return None, "Impossible de lire le fichier."


        # Standardiser les noms de colonnes (insensible à la casse, accents, espaces)
        df_import.columns = (df_import.columns.str.lower()
                             .str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8') # Enlever accents
                             .str.replace(r'[^a-z0-9_]', '', regex=True) # Garder lettres, chiffres, underscore
                             .str.replace(r'\s+', '_', regex=True)) # Remplacer espaces multiples par underscore unique

        # Mappages possibles pour les colonnes requises
        col_maps = {
            "train_id": ["train", "trainid", "idtrain", "numero_train", "train_no"],
            "origine": ["depart", "origine", "depuis", "gare_depart"],
            "heure_depart": ["heure_depart", "heuredepart", "debut", "departle", "h_dep", "hdep"],
            "terminus": ["arrivee", "terminus", "term", "destination", "vers", "gare_arrivee"],
            "heure_arrivee": ["heure_arrivee", "heurearrivee", "fin", "arriveele", "h_arr", "harr"],
            "temps_trajet": ["temps_trajet", "tempstrajet", "duree", "dureemin", "tps_trajet"]
        }

        # Renommer les colonnes trouvées
        rename_dict = {}
        found_cols = {}
        for target_col, possible_names in col_maps.items():
            for name in possible_names:
                if name in df_import.columns:
                    # Gérer cas où plusieurs colonnes sources matchent la même cible (prendre la première)
                    if target_col not in found_cols:
                         rename_dict[name] = target_col
                         found_cols[target_col] = True
                    break # Prendre le premier match pour cette cible

        df_import = df_import.rename(columns=rename_dict)

        required_cols = list(col_maps.keys())
        missing = [col for col in required_cols if col not in found_cols]
        if missing:
            # Essayer de déduire temps_trajet si heures dispo
            if "temps_trajet" in missing and "heure_depart" in found_cols and "heure_arrivee" in found_cols:
                 try:
                      # Fonction pour parser heure robuste (HH:MM ou HH:MM:SS)
                      def parse_time_robust(time_str):
                           time_str = str(time_str).strip()
                           for fmt in ("%H:%M:%S", "%H:%M"):
                                try: return pd.to_datetime(time_str, format=fmt, errors='raise').time()
                                except ValueError: pass
                           return pd.NaT # Retourner NaT si échec

                      start_times = df_import['heure_depart'].apply(parse_time_robust)
                      end_times = df_import['heure_arrivee'].apply(parse_time_robust)

                      # Créer des datetimes arbitraires pour calculer la différence
                      start_dt = start_times.apply(lambda t: datetime.combine(datetime.today(), t) if pd.notna(t) else pd.NaT)
                      end_dt = end_times.apply(lambda t: datetime.combine(datetime.today(), t) if pd.notna(t) else pd.NaT)


                      valid_times = start_dt.notna() & end_dt.notna()
                      delta = end_dt[valid_times] - start_dt[valid_times]
                      # Gérer passage minuit
                      delta = delta + pd.Timedelta(days=1) * (delta < pd.Timedelta(0))
                      # Calculer temps en minutes, gérer NaT
                      df_import.loc[valid_times, 'temps_trajet'] = (delta.dt.total_seconds() / 60).fillna(0).astype(int)
                      # Marquer comme trouvé si au moins une valeur a été calculée
                      if valid_times.any():
                           found_cols["temps_trajet"] = True
                           if "temps_trajet" in missing: missing.remove("temps_trajet") # Recalculé

                 except Exception as e_calc:
                      print(f"Avertissement: N'a pas pu calculer temps_trajet depuis heures: {e_calc}")

            # Si toujours manquant après tentative de calcul
            if missing:
                 return None, f"Fichier invalide. Colonnes manquantes ou non reconnues: {', '.join(missing)}"


        new_roulement = {}
        gares_valides = set(dataframe_gares['gare']) if dataframe_gares is not None else set()

        # Nettoyage et validation ligne par ligne
        processed_rows = 0
        skipped_rows = 0
        for index, row in df_import.iterrows():
            try:
                # Nettoyer les données (espaces, etc.)
                train_id_str = str(row.get('train_id', '')).strip()
                origine = str(row.get('origine', '')).strip()
                terminus = str(row.get('terminus', '')).strip()
                heure_depart_str = str(row.get('heure_depart', '')).strip()
                heure_arrivee_str = str(row.get('heure_arrivee', '')).strip()
                temps_trajet_str = str(row.get('temps_trajet', '')).strip()

                # Validation Train ID
                if not train_id_str: skipped_rows += 1; continue # Ignorer ligne si pas de Train ID
                train_id = int(float(train_id_str)) # Tenter conversion robuste

                # Validation Gares
                if not origine or origine not in gares_valides:
                    print(f"Ligne {index+2} ignorée: Gare départ '{origine}' invalide ou inconnue.")
                    skipped_rows += 1; continue
                if not terminus or terminus not in gares_valides:
                    print(f"Ligne {index+2} ignorée: Gare arrivée '{terminus}' invalide ou inconnue.")
                    skipped_rows += 1; continue

                # Validation et formatage Heures (format HH:MM)
                # Helper pour parser et formater
                def format_heure(heure_in):
                    heure_in_str = str(heure_in).strip()
                    for fmt in ("%H:%M:%S", "%H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"): # Ajouter formats courants
                         try: return pd.to_datetime(heure_in_str, format=fmt, errors='raise').strftime('%H:%M')
                         except ValueError: pass
                    # Essayer parser générique pandas si formats échouent
                    try: return pd.to_datetime(heure_in_str, errors='raise').strftime('%H:%M')
                    except: raise ValueError(f"Format heure '{heure_in_str}' non reconnu")

                try:
                    heure_depart_str_fmt = format_heure(heure_depart_str)
                    heure_arrivee_str_fmt = format_heure(heure_arrivee_str)
                except ValueError as e_heure:
                     print(f"Ligne {index+2} ignorée: {e_heure}")
                     skipped_rows += 1; continue

                # Validation Temps Trajet
                try:
                    if not temps_trajet_str or temps_trajet_str.lower() == 'nan': # Recalculer si vide ou NaN
                        start_dt_val = pd.to_datetime(heure_depart_str_fmt, format='%H:%M')
                        end_dt_val = pd.to_datetime(heure_arrivee_str_fmt, format='%H:%M')
                        delta_val = end_dt_val - start_dt_val
                        if delta_val < pd.Timedelta(0): delta_val += pd.Timedelta(days=1)
                        temps_trajet = int(delta_val.total_seconds() / 60)
                    else:
                        temps_trajet = int(float(temps_trajet_str)) # Tenter conversion robuste
                    if temps_trajet < 0: raise ValueError("Temps trajet négatif")
                except ValueError:
                     print(f"Ligne {index+2} ignorée: Temps trajet '{temps_trajet_str}' invalide.")
                     skipped_rows += 1; continue


                etape = {
                    "depart": origine,
                    "heure_depart": heure_depart_str_fmt,
                    "arrivee": terminus,
                    "heure_arrivee": heure_arrivee_str_fmt,
                    "temps_trajet": temps_trajet
                }
                new_roulement.setdefault(train_id, []).append(etape)
                processed_rows += 1

            except Exception as e_row: # Capturer autres erreurs inattendues par ligne
                print(f"Ligne {index+2} ignorée (erreur inattendue): {row.to_dict()}. Erreur : {e_row}")
                skipped_rows += 1
                continue

        if skipped_rows > 0:
             print(f"Avertissement: {skipped_rows} lignes ignorées lors de l'importation (voir logs pour détails).")
        if processed_rows == 0:
             return None, f"Aucune ligne valide n'a pu être importée. Vérifiez le format du fichier et les noms des colonnes/gares."

        # Trier les étapes pour chaque train par heure de départ
        for train_id in new_roulement:
            new_roulement[train_id].sort(key=lambda x: datetime.strptime(x['heure_depart'], "%H:%M").time())

        return new_roulement, None

    except Exception as e:
        import traceback
        print(f"Erreur critique lors import fichier: {traceback.format_exc()}")
        return None, f"Erreur critique lors de la lecture du fichier : {e}"

# ... (analyser_frequences_manuelles) ...
def analyser_frequences_manuelles(roulement_manuel, missions, heure_debut_service, heure_fin_service):
    comptes_par_heure = {}
    for etapes in roulement_manuel.values():
        for etape in etapes:
            # Ignorer les étapes avec temps de trajet nul ou négatif pour l'analyse
            if etape.get("temps_trajet", 0) <= 0: continue
            cle_segment = f"{etape['depart']} → {etape['arrivee']}"
            try:
                heure = datetime.strptime(etape['heure_depart'], "%H:%M").hour
                comptes_par_heure.setdefault(cle_segment, {}).setdefault(heure, 0)
                comptes_par_heure[cle_segment][heure] += 1
            except (ValueError, TypeError):
                 print(f"Étape ignorée dans l'analyse de fréquence (heure invalide): {etape}")
                 continue # Ignorer étape si heure invalide

    resultats_analyse = {}
    # Calculer la liste des heures en gérant le passage de minuit
    heures_service = []
    current_hour_dt = datetime.combine(datetime.today(), heure_debut_service)
    end_dt = datetime.combine(datetime.today(), heure_fin_service)
    if end_dt <= current_hour_dt: end_dt += timedelta(days=1) # Gérer passage minuit pour fin

    while current_hour_dt < end_dt:
         heures_service.append(current_hour_dt.hour)
         current_hour_dt += timedelta(hours=1)


    for mission in missions:
        # Ignorer missions sans fréquence définie ou invalide
        freq = mission.get('frequence')
        if freq is None or not isinstance(freq, (int, float)) or freq <= 0: continue

        cle_mission = f"{mission['origine']} → {mission['terminus']}"
        objectif = freq # Utiliser la fréquence directement
        donnees_analyse = []
        heures_respectees = 0
        trains_reels_segment = comptes_par_heure.get(cle_mission, {})

        for heure in heures_service:
            trains_reels = trains_reels_segment.get(heure, 0)
            statut = "✓" if trains_reels >= objectif else "❌"
            if statut == "✓": heures_respectees += 1
            donnees_analyse.append({"Heure": f"{heure:02d}:00", "Trains sur segment": trains_reels, "Objectif": f"≥ {objectif:.1f}", "Statut": statut}) # Afficher objectif avec décimale

        if donnees_analyse: # Seulement si on a pu analyser des heures
             df_analyse = pd.DataFrame(donnees_analyse)
             conformite = (heures_respectees / len(heures_service)) * 100 if heures_service else 100
             resultats_analyse[cle_mission] = {"df": df_analyse, "conformite": conformite}
        # Ne pas ajouter la mission à l'analyse si aucune heure n'a pu être traitée


    return resultats_analyse


# ... (generer_exports) ...
def generer_exports(chronologie_trajets, figure):
     # --- Fonction inchangée ---
    lignes_export = []
    # Trier par train puis par heure de début
    for id_t in sorted(chronologie_trajets.keys()):
         trajets_tries = sorted(chronologie_trajets[id_t], key=lambda t: t['start'])
         for t in trajets_tries:
              # Formater les heures pour Excel/CSV
              start_str = t["start"].strftime('%Y-%m-%d %H:%M:%S')
              end_str = t["end"].strftime('%Y-%m-%d %H:%M:%S')
              lignes_export.append({
                   "Train": id_t,
                   "Début": start_str, # Utiliser format complet pour éviter ambiguïté jour
                   "Fin": end_str,
                   "Origine": t["origine"],
                   "Terminus": t["terminus"]
               })

    df_export = pd.DataFrame(lignes_export)

    # Export Excel
    excel_buffer = BytesIO()
    try:
         with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
              df_export.to_excel(writer, index=False, sheet_name="Roulements")
              # Auto-ajustement largeur colonnes (optionnel mais recommandé)
              workbook = writer.book
              worksheet = writer.sheets['Roulements']
              for i, col in enumerate(df_export.columns):
                   # Calculer largeur max basé sur données et titre colonne
                   column_len = df_export[col].astype(str).map(len)
                   max_len = max(column_len.max() if not column_len.empty else 0, len(col)) + 2 # Ajouter marge
                   worksheet.set_column(i, i, max_len)
    except Exception as e_excel:
         print(f"Erreur écriture Excel: {e_excel}")
         # Retourner buffer vide ou lever une erreur?
         excel_buffer = BytesIO() # Retourner vide en cas d'erreur
    excel_buffer.seek(0)


    # Export PDF (Graphique)
    pdf_buffer = BytesIO()
    if figure is not None: # Vérifier que la figure existe
        try:
            figure.savefig(pdf_buffer, format="pdf", bbox_inches='tight', dpi=300) # Augmenter DPI pour qualité
        except Exception as e_pdf:
             print(f"Erreur lors de la sauvegarde du graphique en PDF: {e_pdf}")
             # Créer un PDF vide ou avec message d'erreur? Pour l'instant, buffer vide.
             pass # Le buffer sera vide si erreur
    pdf_buffer.seek(0)

    return excel_buffer, pdf_buffer


# ... (reset_caches) ...
def reset_caches():
    """Permet de réinitialiser les caches pour éviter les incohérences lors d’un rechargement d’infra."""
    construire_horaire_mission_cached.cache_clear()


