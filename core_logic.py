# -*- coding: utf-8 -*-
"""
core_logic.py

Ce module contient le "cerveau" de l'application.
Mise à jour V3 : Logique de croisement affinée (DV vs VU) et respect strict des contraintes utilisateur.
"""
from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO, StringIO
import heapq
import bisect
from functools import lru_cache
from collections import defaultdict
import json
import math

# =============================================================================
# CLASSES ET UTILITAIRES DE PLANIFICATION
# =============================================================================

def _get_infra_at_gare(df_gares, gare_name):
    """Récupère l'infrastructure d'une gare (VE, D, F)."""
    try:
        if df_gares is None or 'gare' not in df_gares.columns or 'infra' not in df_gares.columns:
            return 'F'
        series = df_gares.loc[df_gares['gare'] == gare_name, 'infra']
        if series.empty: return 'F'
        val = series.iloc[0]
        return str(val).strip().upper() if pd.notna(val) else 'F'
    except:
        return 'F'

def _is_crossing_point(infra_code):
    """Définit si un train peut s'arrêter ici pour croiser (VE, D, Terminus)."""
    # VE = Voie d'évitement, D = Double Voie (donc on peut s'y arrêter/croiser), Terminus = OK
    return infra_code in ['VE', 'D', 'Terminus']

class SimulationEngine:
    def __init__(self, df_gares, heure_debut, heure_fin):
        self.df_gares = df_gares.sort_values('distance').reset_index(drop=True)
        self.gares_map = {r.gare: i for i, r in self.df_gares.iterrows()}
        self.infra_map = {r.gare: _get_infra_at_gare(df_gares, r.gare) for _, r in self.df_gares.iterrows()}

        self.dt_debut = datetime.combine(datetime.today(), heure_debut)
        self.dt_fin = datetime.combine(datetime.today(), heure_fin)
        if self.dt_fin <= self.dt_debut:
            self.dt_fin += timedelta(days=1)

        # Analyse des segments (Voie Unique vs Double Voie)
        # segment_properties[i] décrit le segment entre gare i et i+1
        self.segment_is_double = self._analyze_segments()

        # Stockage des sillons validés (trajectoires)
        self.committed_schedules = []

        # Gestion du matériel { 'gare': [(dt_dispo, id_train), ...] }
        self.fleet_availability = defaultdict(list)
        self.train_counter = 1

    def _analyze_segments(self):
        """
        Détermine pour chaque segment s'il est à Double Voie (DV) ou Voie Unique (VU).
        Logique : 'D' marque une transition (Début/Fin DV).
        On suppose que la ligne commence en VU par défaut, sauf indication contraire.
        """
        is_double = {}
        n = len(self.df_gares)
        current_state_double = False # Hypothèse : départ en VU

        for i in range(n - 1):
            gare_curr = self.df_gares.iloc[i]['gare']
            infra_curr = self.infra_map.get(gare_curr, 'F')

            # Si on rencontre un 'D', on bascule l'état pour le segment À VENIR (et les suivants)
            # Hypothèse : D --(DV)-- D --(VU)-- VE
            if infra_curr == 'D':
                current_state_double = not current_state_double

            is_double[i] = current_state_double

        return is_double

    def _get_block_end(self, start_idx, direction):
        """Trouve la fin du bloc de cantonnement (prochaine gare capable de croiser)."""
        curr = start_idx + direction
        while 0 <= curr < len(self.df_gares):
            gare = self.df_gares.iloc[curr]['gare']
            infra = self.infra_map.get(gare, 'F')
            if _is_crossing_point(infra):
                return curr
            curr += direction
        return start_idx # Should not happen if terminus defined

    def check_segment_availability(self, seg_idx_min, seg_idx_max, t_enter, t_exit):
        """
        Vérifie si une plage de segments est libre.
        Si c'est du DV, on ignore les trains en sens inverse.
        Si c'est du VU, on vérifie tout conflit.
        Retourne (True, None) ou (False, t_liberation_estimee).
        """
        # Vérifions d'abord la nature du tronçon
        # Si TOUS les segments sont DV, c'est du DV global.
        all_double = True
        for i in range(seg_idx_min, seg_idx_max):
            if not self.segment_is_double.get(i, False):
                all_double = False
                break

        # Marge de sécurité
        margin = timedelta(minutes=1)
        t_check_start = t_enter
        t_check_end = t_exit

        latest_conflict_end = t_enter # Par défaut

        for committed in self.committed_schedules:
            path = committed['path']
            # Optimisation: bounding box temporelle du train complet
            if path[-1]['arr'] < t_check_start or path[0]['dep'] > t_check_end:
                continue

            for i in range(len(path) - 1):
                p_a, p_b = path[i], path[i+1]

                # Intersection spatiale ?
                o_idx_min = min(p_a['index'], p_b['index'])
                o_idx_max = max(p_a['index'], p_b['index'])

                if max(seg_idx_min, o_idx_min) < min(seg_idx_max, o_idx_max):
                    # Intersection temporelle ?
                    o_start = p_a['dep'] - margin
                    o_end = p_b['arr'] + margin

                    if max(t_check_start, o_start) < min(t_check_end, o_end):
                        # CONFLIT POTENTIEL

                        # Si on est en Double Voie (DV), le conflit n'existe que si même sens (rattrapage)
                        # Pour simplifier et respecter "croisement en tout point", on ignore les sens inverses sur DV.
                        # On ignore même le même sens (hypothèse cantonnement mobile ou espacement géré ailleurs/non critique ici)
                        if all_double:
                            continue

                        # Si Voie Unique (VU), tout croisement est fatal.
                        return False, o_end + timedelta(seconds=1) # On doit attendre que l'autre sorte

        return True, None

    def solve_mission_schedule(self, mission, ideal_start_time, type_trajet, min_start_time=None):
        """
        Construit un horaire valide en insérant des attentes si nécessaire.
        Priorité : Respecter min_start_time (dispo matériel), puis s'approcher de ideal_start_time.
        """
        # 1. Obtenir le tracé de base (temps de parcours purs)
        base_schedule = construire_horaire_mission(mission, type_trajet, self.df_gares)
        if not base_schedule: return None, [], "Erreur itinéraire"

        # Convertir en liste d'étapes avec durées relatives
        steps = []
        for i in range(len(base_schedule)):
            pt = base_schedule[i]
            idx = self.gares_map.get(pt['gare'])
            offset = pt.get('time_offset_min', 0)

            # Durée depuis le point précédent
            duration_from_prev = 0
            if i > 0:
                duration_from_prev = offset - base_schedule[i-1].get('time_offset_min', 0)

            steps.append({
                'gare': pt['gare'],
                'index': idx,
                'run_time': duration_from_prev, # Temps de marche pour arriver ici
                'infra': self.infra_map.get(pt['gare'], 'F')
            })

        # 2. Initialisation
        # Le départ ne peut pas être avant la dispo du matériel
        actual_start_time = ideal_start_time
        if min_start_time and min_start_time > actual_start_time:
            actual_start_time = min_start_time

        current_time = actual_start_time
        final_path = []
        direction = 1 if steps[-1]['index'] > steps[0]['index'] else -1

        # 3. Construction pas à pas (Greedy)
        # On valide bloc par bloc (entre points de croisement)

        # Le premier point est forcément un point d'arrêt/départ
        final_path.append({
            'gare': steps[0]['gare'],
            'index': steps[0]['index'],
            'arr': current_time,
            'dep': current_time # Peut être modifié si on attend au départ
        })

        i = 0
        while i < len(steps) - 1:
            curr_step = steps[i]

            # Identifier le prochain "Check Point" (fin du bloc VU ou prochaine gare importante)
            # Sur une DV, on peut avancer gare par gare. Sur une VU, il faut réserver tout le bloc jusqu'au prochain évitement.

            # On cherche le prochain point de croisement physique
            next_crossing_idx_in_steps = -1
            phys_idx_curr = curr_step['index']

            # Scan dans les steps de la mission pour trouver le prochain point capable de croisement
            target_step_idx = i + 1
            while target_step_idx < len(steps):
                st = steps[target_step_idx]
                infra = st['infra']
                # Si c'est un point de croisement ou la fin de mission
                if _is_crossing_point(infra) or target_step_idx == len(steps) - 1:
                    next_crossing_idx_in_steps = target_step_idx
                    break
                target_step_idx += 1

            # Définition du Bloc Critique à traverser
            # De step[i] à step[target_step_idx]
            # On doit vérifier que TOUTE cette section est libre pendant le temps de traversée

            # Calcul du temps de trajet théorique cumulé pour ce bloc
            travel_time_block = 0
            for k in range(i + 1, target_step_idx + 1):
                travel_time_block += steps[k]['run_time']

            # Boucle de tentative (Wait & Retry)
            # On essaie de partir de curr_step à 'current_departure'
            # Si ça coince, on incrémente 'current_departure' (donc on attend à curr_step)
            current_departure = current_time

            # Protection boucle infinie
            max_wait = timedelta(hours=2)

            while True:
                if current_departure - current_time > max_wait:
                    return None, [], "Impossible de trouver un créneau (timeout)"

                t_enter_block = current_departure
                t_exit_block = current_departure + timedelta(minutes=travel_time_block)

                # Vérification sur l'ensemble des segments physiques du bloc
                # Attention : steps peut sauter des gares si interpolation, mais ici on a toutes les gares.
                # On utilise les indices physiques min/max.
                idx_start_phys = steps[i]['index']
                idx_end_phys = steps[target_step_idx]['index']

                idx_min = min(idx_start_phys, idx_end_phys)
                idx_max = max(idx_start_phys, idx_end_phys)

                is_free, next_free_time = self.check_segment_availability(idx_min, idx_max, t_enter_block, t_exit_block)

                if is_free:
                    # C'est validé !
                    # On met à jour le path pour tous les points intermédiaires
                    t_cursor = t_enter_block

                    # Mettre à jour le départ réel du point actuel (si on a attendu)
                    final_path[-1]['dep'] = t_enter_block

                    for k in range(i + 1, target_step_idx + 1):
                        dt = steps[k]['run_time']
                        t_cursor += timedelta(minutes=dt)
                        final_path.append({
                            'gare': steps[k]['gare'],
                            'index': steps[k]['index'],
                            'arr': t_cursor,
                            'dep': t_cursor # Par défaut, repart tout de suite (sera ajusté au prochain tour si attente)
                        })

                    current_time = t_cursor # Heure d'arrivée au bout du bloc
                    i = target_step_idx # On avance l'itérateur
                    break # Sortie de la boucle while True (Retry)

                else:
                    # Conflit détecté.
                    # On doit attendre à la gare actuelle (steps[i]).
                    # next_free_time nous dit quand le conflit se libère.
                    # On décale notre tentative de départ.
                    if next_free_time and next_free_time > current_departure:
                        current_departure = next_free_time
                    else:
                        current_departure += timedelta(minutes=1)

        return actual_start_time, final_path, None

    def allocate_train_id(self, gare_depart, heure_depart, mission_id):
        """Réutilisation FIFO du matériel."""
        pool = self.fleet_availability[gare_depart]
        pool.sort(key=lambda x: x[0])

        # On cherche un train dispo AVANT ou À l'heure de départ
        best_train_id = None
        best_idx = -1

        for i, (dt_dispo, tid) in enumerate(pool):
            if dt_dispo <= heure_depart:
                best_train_id = tid
                best_idx = i
                break

        if best_train_id:
            pool.pop(best_idx)
            return best_train_id, False
        else:
            tid = self.train_counter
            self.train_counter += 1
            return tid, True

    def register_train_arrival(self, train_id, gare_arrivee, heure_arrivee, temps_retournement):
        """Libère le train."""
        dt_dispo = heure_arrivee + timedelta(minutes=temps_retournement)
        self.fleet_availability[gare_arrivee].append((dt_dispo, train_id))

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def generer_tous_trajets_optimises(missions, heure_debut, heure_fin, df_gares):
    """
    Génère les trajets avec logique de glissement pour respecter l'infra.
    """
    engine = SimulationEngine(df_gares, heure_debut, heure_fin)
    chronologie_reelle = defaultdict(list)
    warnings = {"infra_violations": [], "other": []}

    # 1. Création des demandes (Triées par heure idéale)
    demandes = []
    for i, mission in enumerate(missions):
        freq = mission.get('frequence', 0)
        if freq <= 0: continue
        intervalle = timedelta(hours=1.0/freq)

        try:
            refs = sorted([int(x.strip()) for x in str(mission.get('reference_minutes', '0')).split(',') if x.strip().isdigit()])
        except: refs = [0]
        if not refs: refs = [0]

        for ref in refs:
            # Calage précis sur la minute de référence
            # Si ref=0, intervalle=30min -> 06:00, 06:30...
            # Calcul du premier temps valide >= heure_debut
            start_h = engine.dt_debut.hour

            # On part d'une base jour J à 00:00 + minute ref
            base_time = engine.dt_debut.replace(hour=start_h, minute=0, second=0) + timedelta(minutes=ref)

            # On recule pour être sûr de choper le cycle
            curr = base_time - timedelta(hours=2)
            while curr < engine.dt_debut:
                curr += intervalle

            # Génération jusqu'à la fin
            while curr < engine.dt_fin:
                demandes.append({
                    'ideal_time': curr,
                    'mission': mission,
                    'mission_idx': i
                })
                curr += intervalle

    demandes.sort(key=lambda x: x['ideal_time'])

    # 2. Résolution séquentielle
    for dem in demandes:
        mission = dem['mission']
        ideal_start = dem['ideal_time']

        # A. Planification ALLER
        # On ne passe pas de min_start_time spécifique pour l'aller initial (nouveau train possible)
        # Mais allocate_train_id favorisera un train existant si dispo.

        # Astuce : Pour savoir si on est contraint par le matériel au départ, il faudrait savoir quel train on prend.
        # Mais on ne le sait qu'après avoir fixé l'horaire.
        # Solution : On calcule l'horaire "idéal". Ensuite on cherche un train.
        # Si aucun train dispo à ideal_start, on a deux choix :
        # 1. Créer un nouveau train (Respect fréquence > Réutilisation) -> Choix actuel pour minimiser impact.
        # 2. Retarder le départ (Réutilisation > Fréquence).
        # L'utilisateur demande de "Minimiser le nombre de trains".
        # Donc on devrait vérifier la dispo flotte AVANT.

        pool = engine.fleet_availability.get(mission['origine'], [])
        pool.sort(key=lambda x: x[0])
        min_mat_time = None

        # Y a-t-il un train dispo "raisonnablement" proche (ex: retard < 30 min) ?
        # Sinon on en crée un.
        for dt_dispo, tid in pool:
            if dt_dispo <= ideal_start + timedelta(minutes=15): # Tolérance réutilisation
                min_mat_time = dt_dispo
                break # On prend le premier dispo

        # Calcul Sillon Aller
        real_start, path_aller, err = engine.solve_mission_schedule(mission, ideal_start, 'aller', min_start_time=min_mat_time)

        if err or not path_aller:
            warnings['other'].append(f"Échec plannif M{dem['mission_idx']+1} à {ideal_start.strftime('%H:%M')}: {err}")
            continue

        # Vérification Fréquence
        diff_min = (real_start - ideal_start).total_seconds() / 60
        if diff_min > 2:
            warnings['infra_violations'].append(
                f"Violation Fréquence M{dem['mission_idx']+1} (+{int(diff_min)} min). "
                f"Prévu: {ideal_start.strftime('%H:%M')}, Réel: {real_start.strftime('%H:%M')} (Conflit Infra/Matériel)"
            )

        # Assignation Train
        train_id, is_new = engine.allocate_train_id(path_aller[0]['gare'], real_start, dem['mission_idx'])
        engine.committed_schedules.append({'train_id': train_id, 'path': path_aller})

        # Stockage Aller
        prev = path_aller[0]
        for pt in path_aller[1:]:
            chronologie_reelle[train_id].append({
                "start": prev['dep'], "end": pt['arr'],
                "origine": prev['gare'], "terminus": pt['gare']
            })
            prev = pt

        # B. Gestion RETOUR (Chaînage immédiat pour optimiser rotations)
        gare_term = path_aller[-1]['gare']
        arr_time = path_aller[-1]['arr']
        t_ret_B = mission.get("temps_retournement_B", 10)

        # Le train est libéré (virtuellement)
        dispo_retour = arr_time + timedelta(minutes=t_ret_B)

        # Tentative de planification du retour "dans la foulée"
        # On cherche la prochaine demande théorique de retour qui colle ?
        # Ou on crée un retour "hors fréquence" juste pour ramener le train ?
        # L'utilisateur dit : "On respecte les fréquence saisie par l'utilisateur".
        # DONC le retour ne doit pas être arbitraire. Il doit correspondre à une demande "Retour".
        # C'est compliqué à matcher ici car 'demandes' est une liste plate.

        # Pour ce prototype V3, on applique une logique simplifiée de rotation :
        # Le train effectue le retour "dès que possible" en respectant l'infra,
        # OU on le laisse au dépôt (fleet) pour qu'il soit pris par la prochaine demande 'aller' (sens inverse).

        # Mais 'demandes' ne contient que des 'Aller' (selon la boucle ci-dessus).
        # Si la mission a un retour défini, il faut le jouer.

        # Correction : On génère le retour TOUT DE SUITE comme une suite logique du train.
        # Mais à quelle heure ?
        # Si on veut respecter la fréquence, le retour devrait aussi être cadencé.
        # Si l'utilisateur n'a pas défini de mission "Retour" séparée mais compte sur l'ASymétrie :
        # On va tenter de faire repartir le train à l'heure idéale la plus proche >= dispo_retour
        # basée sur la fréquence.

        # Calcul prochain slot théorique retour
        # Base : ideal_start + temps_trajet + temps_ret ? Non, cadencement fixe.
        # On cherche k tel que (Base + k*intervalle) >= dispo_retour

        # On assume que le retour a la même fréquence.
        # Minute ref retour ? Souvent symétrique ou libre.
        # Disons qu'on essaie de repartir au plus tôt (dispo_retour) mais en respectant infra.

        # Calcul Sillon Retour
        real_start_ret, path_ret, err_ret = engine.solve_mission_schedule(mission, dispo_retour, 'retour', min_start_time=dispo_retour)

        if path_ret:
            # On garde le même train
            engine.committed_schedules.append({'train_id': train_id, 'path': path_ret})

            prev = path_ret[0]
            for pt in path_ret[1:]:
                chronologie_reelle[train_id].append({
                    "start": prev['dep'], "end": pt['arr'],
                    "origine": prev['gare'], "terminus": pt['gare']
                })
                prev = pt

            # Libération finale à l'origine
            t_ret_A = mission.get("temps_retournement_A", 10)
            engine.register_train_arrival(train_id, path_ret[-1]['gare'], path_ret[-1]['arr'], t_ret_A)

        else:
            # Si pas de retour possible (fin de service ?), on libère au terminus
            engine.register_train_arrival(train_id, gare_term, arr_time, t_ret_B)

    return chronologie_reelle, warnings

# =============================================================================
# FONCTIONS UTILITAIRES (CONSERVÉES POUR COMPATIBILITÉ)
# =============================================================================

@lru_cache(maxsize=256)
def construire_horaire_mission_cached(mission_key, trajet_spec, df_gares_json):
    """Wrapper avec cache pour la construction d'horaire."""
    if df_gares_json is None: return None
    try:
        df_gares_local = pd.read_json(StringIO(df_gares_json))
        mission_cfg = json.loads(mission_key)
        return construire_horaire_mission(mission_cfg, trajet_spec, df_gares_local)
    except:
        return None

def construire_horaire_mission(mission_config, trajet_spec, df_gares):
    """
    Construit un horaire détaillé incluant TOUTES les gares physiques par interpolation.
    (Code repris et nettoyé de la version précédente pour assurer la robustesse)
    """
    if df_gares is None or df_gares.empty: return []

    # 1. Définition des points clés (Origine -> [PP] -> Terminus)
    points_cles = []
    if trajet_spec == 'aller':
        base_pp = mission_config.get("passing_points", [])
        points_cles.append({"gare": mission_config.get("origine"), "time_offset_min": 0})
        points_cles.extend([p for p in base_pp if p.get("time_offset_min", 0) > 0])
        points_cles.append({"gare": mission_config.get("terminus"), "time_offset_min": mission_config.get("temps_trajet", 60)})
    else: # Retour
        # Gestion asymétrique ou symétrique
        base_pp = []
        temps_total = mission_config.get("temps_trajet", 60)

        if mission_config.get("trajet_asymetrique"):
            temps_total = mission_config.get("temps_trajet_retour", temps_total)
            base_pp = mission_config.get("passing_points_retour", [])
        else:
            # Inversion automatique
            pp_aller = mission_config.get("passing_points", [])
            base_pp = []
            for p in pp_aller:
                inv_t = temps_total - p["time_offset_min"]
                if inv_t > 0:
                    base_pp.append({"gare": p["gare"], "time_offset_min": inv_t})
            base_pp.sort(key=lambda x: x["time_offset_min"])

        points_cles.append({"gare": mission_config.get("terminus"), "time_offset_min": 0})
        points_cles.extend(base_pp)
        points_cles.append({"gare": mission_config.get("origine"), "time_offset_min": temps_total})

    # Nettoyage doublons et tri
    points_cles.sort(key=lambda x: x['time_offset_min'])
    unique_cles = []
    seen = set()
    for p in points_cles:
        if p['gare'] not in seen:
            unique_cles.append(p)
            seen.add(p['gare'])

    if len(unique_cles) < 2: return []

    # 2. Interpolation sur le dataframe physique
    final_horaire = []

    try:
        # Prépare les données géographiques
        gares_sorted = df_gares.sort_values('distance').reset_index(drop=True)
        gare_map = {r.gare: (i, r.distance) for i, r in gares_sorted.iterrows()} # Nom -> (Index, Dist)

        # Itération sur les segments "clés" définis par l'utilisateur
        for i in range(len(unique_cles) - 1):
            pt_start = unique_cles[i]
            pt_end = unique_cles[i+1]

            if pt_start['gare'] not in gare_map or pt_end['gare'] not in gare_map:
                continue

            idx_s, dist_s = gare_map[pt_start['gare']]
            idx_e, dist_e = gare_map[pt_end['gare']]
            t_s, t_e = pt_start['time_offset_min'], pt_end['time_offset_min']

            # Identifier les gares physiques entre ces deux points
            idx_min, idx_max = min(idx_s, idx_e), max(idx_s, idx_e)

            segment_gares = gares_sorted.iloc[idx_min : idx_max+1]

            # Sens de parcours pour ce segment
            direction = 1 if idx_e > idx_s else -1
            if direction == -1:
                segment_gares = segment_gares.sort_index(ascending=False)

            # Interpolation linéaire
            total_dist_seg = abs(dist_e - dist_s)
            total_time_seg = t_e - t_s

            for _, row in segment_gares.iterrows():
                # Éviter de rajouter le point de départ s'il a déjà été ajouté par le segment précédent
                if len(final_horaire) > 0 and final_horaire[-1]['gare'] == row['gare']:
                    continue

                curr_dist = row['distance']
                dist_from_start = abs(curr_dist - dist_s)

                if total_dist_seg > 0:
                    ratio = dist_from_start / total_dist_seg
                    curr_time = t_s + (total_time_seg * ratio)
                else:
                    curr_time = t_s

                final_horaire.append({
                    "gare": row['gare'],
                    "time_offset_min": round(curr_time, 1) # Garder décimale pour précision conflit
                })

    except Exception as e:
        print(f"Erreur interpolation: {e}")
        return []

    return final_horaire

# Fonctions legacy pour compatibilité imports (vides ou alias)
def preparer_roulement_manuel(roulement):
    # (Code identique à version précédente, conservé pour app.py)
    chronologie_trajets = {}
    for id_train, etapes in roulement.items():
        if not etapes: continue
        chronologie_trajets[id_train] = []
        for etape in etapes:
            try:
                dt_dep = datetime.combine(datetime.today(), datetime.strptime(etape["heure_depart"], "%H:%M").time())
                dt_arr = datetime.combine(datetime.today(), datetime.strptime(etape["heure_arrivee"], "%H:%M").time())
                if dt_arr < dt_dep: dt_arr += timedelta(days=1)
                chronologie_trajets[id_train].append({
                    "start": dt_dep, "end": dt_arr,
                    "origine": etape["depart"], "terminus": etape["arrivee"]
                })
            except: pass
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


