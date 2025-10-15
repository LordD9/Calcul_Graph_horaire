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

def _can_cross_at(df_gares, gare_name, is_terminus=False):
    """
    Retourne True si la gare permet un croisement.
    Si une colonne 'can_cross' ou 'allow_cross' est présente, on lit sa valeur ('T'/'F', bool, 1/0).
    Sinon : si is_terminus -> True, sinon False.
    """
    for col in ['can_cross', 'allow_cross']:
        if col in df_gares.columns:
            try:
                val = df_gares.loc[df_gares['gare'] == gare_name, col].iloc[0]
                if isinstance(val, str):
                    return val.strip().upper() not in ('F', 'FALSE', '0', '')
                return bool(val)
            except Exception:
                return is_terminus
    return is_terminus

def generer_tous_trajets_optimises(missions, heure_debut, heure_fin, df_gares):
    """
    Génère la chronologie complète des trajets en mode "Rotation optimisée"
    en tenant compte des contraintes d'infrastructure pour éviter les conflits.
    Retourne un dictionnaire d'avertissements structuré.
    """
    if 'infra' not in df_gares.columns or df_gares['infra'].isnull().all():
        # Retourne le même format de données même pour le cas simple
        chronologie_sans_conflits, warnings_sans_conflits = generer_tous_trajets_sans_conflits(missions, heure_debut, heure_fin, df_gares)
        return chronologie_sans_conflits, {"infra_violations": [], "other": warnings_sans_conflits}

    infra_violation_warnings = []
    other_warnings = []
    chronologie_reelle = {}
    id_train_counter = 1
    event_counter = 0

    dt_debut_service = datetime.combine(datetime.today(), heure_debut)
    dt_fin_service = datetime.combine(datetime.today(), heure_fin)
    if dt_fin_service <= dt_debut_service:
        dt_fin_service += timedelta(days=1)

    trains = {}
    occupation_cantons = defaultdict(list)
    evenements = []

    # préparation des demandes départs (minute de référence & fréquence)
    for i, mission in enumerate(missions):
        try:
            minutes_ref = sorted(list(set([int(m.strip()) for m in mission.get("reference_minutes", "0").split(',') if m.strip().isdigit()])))
            if not minutes_ref:
                minutes_ref = [0]
        except Exception:
            other_warnings.append(f"Format des minutes de référence invalide pour M{i+1}.")
            minutes_ref = [0]

        if mission.get("frequence", 0) <= 0:
            continue

        intervalle = timedelta(hours=1 / mission["frequence"])
        for minute_ref in minutes_ref:
            offset_hours = minute_ref // 60
            offset_minutes = minute_ref % 60

            curseur_temps = dt_debut_service.replace(minute=offset_minutes, second=0, microsecond=0)
            curseur_temps += timedelta(hours=offset_hours)

            while curseur_temps < dt_debut_service:
                curseur_temps += timedelta(hours=1)

            while curseur_temps < dt_fin_service:
                event_counter += 1
                heapq.heappush(evenements, (curseur_temps, event_counter, "demande_depart_aller", {"mission": mission}))
                curseur_temps += intervalle

    df_gares_json = df_gares.to_json()

    while evenements:
        heure, _, type_event, details = heapq.heappop(evenements)
        if heure > dt_fin_service:
            continue

        if type_event == "demande_depart_aller":
            mission_cfg = details["mission"]
            origine = mission_cfg["origine"]

            train_assigne_id = None
            for id_t, t in trains.items():
                if t["loc"] == origine and t["dispo_a"] <= heure:
                    train_assigne_id = id_t
                    break

            if not train_assigne_id:
                train_assigne_id = id_train_counter
                trains[train_assigne_id] = {"id": train_assigne_id, "loc": origine, "dispo_a": heure}
                chronologie_reelle[train_assigne_id] = []
                id_train_counter += 1

            trains[train_assigne_id]["dispo_a"] = heure + timedelta(seconds=1)

            event_counter += 1
            heapq.heappush(evenements, (heure, event_counter, "tentative_mouvement", {
                "id_train": train_assigne_id, "mission": mission_cfg, "trajet_spec": "aller",
                "index_etape": 0, "retry_count": 0
            }))

        elif type_event == "tentative_mouvement":
            id_train = details["id_train"]
            mission_cfg = details["mission"]
            trajet_spec = details["trajet_spec"]
            index_etape = details["index_etape"]
            mission_key = json.dumps(mission_cfg, sort_keys=True)
            horaire = construire_horaire_mission_cached(mission_key, trajet_spec, df_gares_json)

            if not horaire or index_etape >= len(horaire) - 1:
                continue

            pt_depart, pt_arrivee = horaire[index_etape], horaire[index_etape + 1]
            duree_min_trajet = pt_arrivee["time_offset_min"] - pt_depart["time_offset_min"]
            if duree_min_trajet < 0:
                continue

            heure_depart_reelle = max(heure, trains.get(id_train, {}).get("dispo_a", heure))
            conflit = verifier_conflit(heure_depart_reelle, duree_min_trajet, pt_depart["gare"], pt_arrivee["gare"], id_train, occupation_cantons, df_gares)

            if conflit is None:
                heure_arrivee_reelle = heure_depart_reelle + timedelta(minutes=duree_min_trajet)
                if duree_min_trajet > 0:
                    chronologie_reelle.setdefault(id_train, []).append({
                        "start": heure_depart_reelle, "end": heure_arrivee_reelle,
                        "origine": pt_depart["gare"], "terminus": pt_arrivee["gare"]
                    })
                    occupation_tuple = (heure_depart_reelle, heure_arrivee_reelle, id_train)
                    bisect.insort(occupation_cantons[(pt_depart["gare"], pt_arrivee["gare"])], occupation_tuple)

                trains[id_train]["loc"] = pt_arrivee["gare"]
                trains[id_train]["dispo_a"] = heure_arrivee_reelle

                if index_etape + 1 < len(horaire) - 1:
                    event_counter += 1
                    heapq.heappush(evenements, (heure_arrivee_reelle, event_counter, "tentative_mouvement", {
                        "id_train": id_train, "mission": mission_cfg, "trajet_spec": trajet_spec,
                        "index_etape": index_etape + 1, "retry_count": 0
                    }))
                else:
                    event_counter += 1
                    heapq.heappush(evenements, (heure_arrivee_reelle, event_counter, "fin_mission", {
                        "id_train": id_train, "mission": mission_cfg, "trajet_spec": trajet_spec
                    }))
            else:
                retry_count = details.get("retry_count", 0)
                fin_possible = conflit.get("fin_conflit", heure + timedelta(minutes=1))

                if (fin_possible - heure_depart_reelle) > timedelta(minutes=15) or retry_count > 5:
                    train_en_conflit = conflit.get('avec_train', 'inconnu')
                    msg = (f"Train {id_train} ({pt_depart['gare']} → {pt_arrivee['gare']}) vers "
                           f"{heure_depart_reelle.strftime('%H:%M')} : Passage forcé malgré un conflit "
                           f"avec le Train {train_en_conflit}.")
                    infra_violation_warnings.append(msg)

                    heure_arrivee_reelle = heure_depart_reelle + timedelta(minutes=duree_min_trajet)
                    if duree_min_trajet > 0:
                        chronologie_reelle.setdefault(id_train, []).append({
                            "start": heure_depart_reelle, "end": heure_arrivee_reelle,
                            "origine": pt_depart["gare"], "terminus": pt_arrivee["gare"], "infra_violation": True
                        })
                        occupation_tuple = (heure_depart_reelle, heure_arrivee_reelle, id_train)
                        bisect.insort(occupation_cantons[(pt_depart["gare"], pt_arrivee["gare"])], occupation_tuple)

                    trains[id_train]["loc"] = pt_arrivee["gare"]
                    trains[id_train]["dispo_a"] = heure_arrivee_reelle

                    if index_etape + 1 < len(horaire) - 1:
                        event_counter += 1
                        heapq.heappush(evenements, (heure_arrivee_reelle, event_counter, "tentative_mouvement", {
                            "id_train": id_train, "mission": mission_cfg, "trajet_spec": trajet_spec,
                            "index_etape": index_etape + 1, "retry_count": 0
                        }))
                    else:
                        event_counter += 1
                        heapq.heappush(evenements, (heure_arrivee_reelle, event_counter, "fin_mission", {
                            "id_train": id_train, "mission": mission_cfg, "trajet_spec": trajet_spec
                        }))
                else:
                    new_details = details.copy()
                    new_details["retry_count"] = retry_count + 1
                    event_counter += 1
                    heapq.heappush(evenements, (fin_possible, event_counter, "tentative_mouvement", new_details))

        elif type_event == "fin_mission":
            id_train = details["id_train"]
            mission_cfg = details["mission"]

            if details["trajet_spec"] == "aller":
                temps_retournement = mission_cfg.get("temps_retournement_B", 10)
                heure_dispo_pour_retour = heure + timedelta(minutes=temps_retournement)
                trains[id_train]["dispo_a"] = heure_dispo_pour_retour

                if heure_dispo_pour_retour < dt_fin_service:
                    event_counter += 1
                    heapq.heappush(evenements, (heure_dispo_pour_retour, event_counter, "tentative_mouvement", {
                        "id_train": id_train, "mission": mission_cfg, "trajet_spec": "retour",
                        "index_etape": 0, "retry_count": 0
                    }))
                else:
                    other_warnings.append(f"Train {id_train} : retour programmé après la fin du service ({heure_dispo_pour_retour.strftime('%H:%M')}).")
            else:
                temps_retournement = mission_cfg.get("temps_retournement_A", 10)
                heure_dispo_finale = heure + timedelta(minutes=temps_retournement)
                trains[id_train]["dispo_a"] = heure_dispo_finale

    return chronologie_reelle, {"infra_violations": infra_violation_warnings, "other": other_warnings}


@lru_cache(maxsize=256)
def construire_horaire_mission_cached(mission_key, trajet_spec, df_gares_json):
    """Version mise en cache pour accélérer les appels répétés."""
    df_gares_local = pd.read_json(StringIO(df_gares_json))
    mission_cfg = json.loads(mission_key)
    return construire_horaire_mission(mission_cfg, trajet_spec, df_gares_local)

def construire_horaire_mission(mission_config, trajet_spec, df_gares):
    """
    Construit un horaire détaillé incluant TOUTES les gares physiques sur le parcours,
    en interpolant les temps de passage.
    """
    if trajet_spec == 'aller':
        base_horaire = [{"gare": mission_config["origine"], "time_offset_min": 0}]
        base_horaire.extend(mission_config.get("passing_points", []))
        base_horaire.append({"gare": mission_config["terminus"], "time_offset_min": mission_config["temps_trajet"]})
    else: # retour
        origine_retour = mission_config["terminus"]
        terminus_retour = mission_config["origine"]
        if mission_config.get("trajet_asymetrique"):
            temps_trajet_retour = mission_config.get("temps_trajet_retour", mission_config["temps_trajet"])
            pp_retour = mission_config.get("passing_points_retour", [])
        else:
            temps_trajet_retour = mission_config["temps_trajet"]
            pp_retour = sorted([{"gare": pp["gare"], "time_offset_min": temps_trajet_retour - pp["time_offset_min"]} for pp in mission_config.get("passing_points", [])], key=lambda x: x["time_offset_min"])
        base_horaire = [{"gare": origine_retour, "time_offset_min": 0}]
        base_horaire.extend(pp_retour)
        base_horaire.append({"gare": terminus_retour, "time_offset_min": temps_trajet_retour})

    base_horaire = sorted([dict(t) for t in {tuple(d.items()) for d in base_horaire}], key=lambda x: x['time_offset_min'])
    if len(base_horaire) < 2: return []

    try:
        gares_sorted = df_gares.sort_values('distance').reset_index(drop=True)
        idx_origine = gares_sorted[gares_sorted['gare'] == base_horaire[0]['gare']].index[0]
        idx_terminus = gares_sorted[gares_sorted['gare'] == base_horaire[-1]['gare']].index[0]
        direction = 1 if idx_terminus > idx_origine else -1
        path_df = gares_sorted.iloc[idx_origine : idx_terminus + direction : direction]
    except (IndexError, KeyError):
        return []

    final_horaire = []
    user_times = {p['gare']: p['time_offset_min'] for p in base_horaire}

    for i in range(len(base_horaire) - 1):
        start_point = base_horaire[i]
        end_point = base_horaire[i+1]

        t1, t2 = start_point['time_offset_min'], end_point['time_offset_min']
        d1 = df_gares.loc[df_gares['gare'] == start_point['gare'], 'distance'].iloc[0]
        d2 = df_gares.loc[df_gares['gare'] == end_point['gare'], 'distance'].iloc[0]

        seg_dist_total = abs(d2 - d1)
        seg_time_total = t2 - t1

        seg_path_gares = path_df[(path_df['distance'] >= min(d1, d2)) & (path_df['distance'] <= max(d1, d2))]

        for _, station in seg_path_gares.iterrows():
            if station['gare'] in user_times:
                temps_calcule = user_times[station['gare']]
            elif seg_dist_total == 0:
                temps_calcule = t1
            else:
                dist_from_start = abs(station['distance'] - d1)
                temps_calcule = t1 + seg_time_total * (dist_from_start / seg_dist_total)

            final_horaire.append({'gare': station['gare'], 'time_offset_min': round(temps_calcule)})

    if not final_horaire:
        # Fallback pour éviter le vide en cas d'erreur d'interpolation
        return base_horaire

    return sorted([dict(t) for t in {tuple(d.items()) for d in final_horaire}], key=lambda x: x['time_offset_min'])

def verifier_conflit(h_depart, duree, gare_dep, gare_arr, id_train_courant, occupation_cantons, df_gares):
    """
    Vérifie les conflits d'infrastructure en utilisant une recherche optimisée.
    """
    h_arrivee = h_depart + timedelta(minutes=duree)
    try:
        gares_sorted = df_gares.sort_values('distance').reset_index(drop=True)
        idx_dep = gares_sorted[gares_sorted['gare'] == gare_dep].index[0]
        idx_arr = gares_sorted[gares_sorted['gare'] == gare_arr].index[0]
    except IndexError:
        return None # Gare non trouvée, pas de vérification possible

    start_idx, end_idx = min(idx_dep, idx_arr), max(idx_dep, idx_arr)
    portion = gares_sorted.iloc[start_idx:end_idx+1]

    # Pre-calculate resolution possibilities for the physical segment
    is_double_segment = any(str(x).strip().upper() in ('D', 'DOUBLE', '2') for x in portion.get('infra', []))
    if is_double_segment:
        return None  # No conflict possible on a double track segment

    crossing_points = [
        st['gare'] for _, st in portion.iterrows()
        if _can_cross_at(df_gares, st['gare'], is_terminus=(st['gare'] in [gare_dep, gare_arr]))
    ]

    # Check for temporal overlaps efficiently
    for seg in [(gare_dep, gare_arr), (gare_arr, gare_dep)]:
        occupations = occupation_cantons.get(seg, [])
        if not occupations:
            continue

        # Find the first occupation that could overlap
        idx = bisect.bisect_left(occupations, (h_depart - timedelta(hours=1),)) # Search a bit before

        # Check from this index onwards
        for i in range(max(0, idx - 1), len(occupations)): # Check one before for safety
            h_deb_occup, h_fin_occup, id_train_occup = occupations[i]

            if h_deb_occup > h_arrivee:
                break # No more overlaps possible

            if h_fin_occup > h_depart and id_train_occup != id_train_courant: # Overlap detected
                fin_conflit = h_fin_occup + timedelta(seconds=1)
                conflit_details = {"fin_conflit": fin_conflit, "avec_train": id_train_occup}
                if crossing_points:
                    conflit_details["type"] = "croisement_possible"
                    conflit_details["points"] = crossing_points
                else:
                    conflit_details["type"] = "face_a_face"
                    conflit_details["points"] = []
                return conflit_details

    return None # No conflicts found


def generer_tous_trajets_sans_conflits(missions, heure_debut, heure_fin, df_gares):
    chronologie_optimale = {}
    id_train_counter = 1
    warnings = []
    dt_debut_service = datetime.combine(datetime.today(), heure_debut)
    dt_fin_service = datetime.combine(datetime.today(), heure_fin)
    if dt_fin_service <= dt_debut_service: dt_fin_service += timedelta(days=1)
    trains_disponibles = []
    evenements = []

    def preparer_horaire_mission_simple(mission_config):
        horaire = [{"gare": mission_config["origine"], "time_offset_min": 0}]
        horaire.extend(mission_config.get("passing_points", []))
        horaire.append({"gare": mission_config["terminus"], "time_offset_min": mission_config["temps_trajet"]})
        horaire_unique = [dict(t) for t in {tuple(d.items()) for d in horaire}]
        return sorted(horaire_unique, key=lambda x: x["time_offset_min"])

    for i, mission in enumerate(missions):
        try:
            minutes_ref = sorted(list(set([int(m.strip()) for m in mission.get("reference_minutes", "0").split(',') if m.strip().isdigit()])))
            if not minutes_ref: minutes_ref = [0]
        except:
            warnings.append(f"Format des minutes de référence invalide pour M{i+1}.")
            minutes_ref = [0]
        if mission["frequence"] <= 0: continue
        intervalle = timedelta(hours=1 / mission["frequence"])
        for minute in minutes_ref:
            curseur_temps = dt_debut_service.replace(minute=minute, second=0, microsecond=0)
            if curseur_temps < dt_debut_service: curseur_temps += timedelta(hours=1)
            while curseur_temps < dt_fin_service:
                if (curseur_temps + timedelta(minutes=mission["temps_trajet"])) <= dt_fin_service:
                    evenements.append({"type": "depart_aller", "heure": curseur_temps, "details": {"mission": mission}})
                curseur_temps += intervalle

    evenements.sort(key=lambda x: x["heure"])
    while evenements:
        event = evenements.pop(0)
        if event["type"] == "depart_aller":
            mission_cfg = event["details"]["mission"]
            train_assigne = None
            for i_train, train in enumerate(sorted(trains_disponibles, key=lambda t: t["disponible_a"])):
                if train["gare"] == mission_cfg["origine"] and train["disponible_a"] <= event["heure"]:
                    train_assigne = trains_disponibles.pop(i_train)
                    break
            if not train_assigne:
                train_assigne = {"id": id_train_counter, "gare": mission_cfg["origine"]}
                chronologie_optimale[train_assigne["id"]] = []
                id_train_counter += 1
            horaire = preparer_horaire_mission_simple(mission_cfg)
            heure_arrivee_finale = event["heure"]
            for i in range(len(horaire) - 1):
                p_dep, p_arr = horaire[i], horaire[i+1]
                t_dep, t_arr = event["heure"] + timedelta(minutes=p_dep["time_offset_min"]), event["heure"] + timedelta(minutes=p_arr["time_offset_min"])
                chronologie_optimale[train_assigne["id"]].append({"start": t_dep, "end": t_arr, "origine": p_dep["gare"], "terminus": p_arr["gare"]})
                heure_arrivee_finale = t_arr
            temps_retournement = mission_cfg.get("temps_retournement_B", 10)
            heure_dispo_retour = heure_arrivee_finale + timedelta(minutes=temps_retournement)
            if heure_dispo_retour < dt_fin_service:
                evenements.append({"type": "disponible_pour_retour", "heure": heure_dispo_retour, "details": {"train": train_assigne, "gare": mission_cfg["terminus"], "mission_aller": mission_cfg}})
                evenements.sort(key=lambda x: x["heure"])
        elif event["type"] == "disponible_pour_retour":
            train, mission_aller = event["details"]["train"], event["details"]["mission_aller"]
            horaire_retour = construire_horaire_mission(mission_aller, "retour", df_gares)
            heure_arrivee_finale = event["heure"]
            for i in range(len(horaire_retour) - 1):
                 p_dep, p_arr = horaire_retour[i], horaire_retour[i+1]
                 t_dep, t_arr = event["heure"] + timedelta(minutes=p_dep["time_offset_min"]), event["heure"] + timedelta(minutes=p_arr["time_offset_min"])
                 if t_arr > dt_fin_service: break
                 chronologie_optimale[train["id"]].append({"start": t_dep, "end": t_arr, "origine": p_dep["gare"], "terminus": p_arr["gare"]})
                 heure_arrivee_finale = t_arr
            temps_retournement = mission_aller.get("temps_retournement_A", 10)
            heure_dispo_finale = heure_arrivee_finale + timedelta(minutes=temps_retournement)
            if heure_dispo_finale < dt_fin_service:
                trains_disponibles.append({"id": train["id"], "gare": mission_aller["origine"], "disponible_a": heure_dispo_finale})
    return chronologie_optimale, warnings


def preparer_roulement_manuel(roulement_manuel):
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
    try:
        if uploaded_file.name.endswith('.csv'):
            df_import = pd.read_csv(uploaded_file, sep=';')
            df_import = df_import.rename(columns={
                "Départ": "origine", "Arrivée": "terminus",
                "Heure départ": "heure_depart", "Heure arrivée": "heure_arrivee",
                "Temps trajet (min)": "temps_trajet", "Train": "train_id"
            })
        else:
            df_import = pd.read_excel(uploaded_file)
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
                heure_depart_str = str(row['heure_depart'])
                heure_arrivee_str = str(row['heure_arrivee'])
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
                print(f"Ligne ignorée lors de l'import : {row}. Erreur : {e}")
                continue

        for train_id in new_roulement:
            new_roulement[train_id].sort(key=lambda x: datetime.strptime(x['heure_depart'], "%H:%M").time())

        return new_roulement, None

    except Exception as e:
        return None, f"Erreur critique lors de la lecture du fichier : {e}"


def analyser_frequences_manuelles(roulement_manuel, missions, heure_debut_service, heure_fin_service):
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
    lignes_export = [{"Train": id_t, "Début": t["start"], "Fin": t["end"], "Origine": t["origine"], "Terminus": t["terminus"]} for id_t, trajets in chronologie_trajets.items() for t in trajets]
    df_export = pd.DataFrame(lignes_export)
    excel_buffer = BytesIO()
    df_export.to_excel(excel_buffer, index=False, sheet_name="Roulements")
    excel_buffer.seek(0)
    pdf_buffer = BytesIO()
    figure.savefig(pdf_buffer, format="pdf", bbox_inches='tight')
    pdf_buffer.seek(0)
    return excel_buffer, pdf_buffer

def reset_caches():
    """Permet de réinitialiser les caches pour éviter les incohérences lors d’un rechargement d’infra."""
    construire_horaire_mission_cached.cache_clear()

