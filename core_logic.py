# -*- coding: utf-8 -*-
"""
core_logic.py
=============

Cœur du moteur de simulation ferroviaire.

Ce module contient la logique fondamentale pour :
1.  **La simulation des circulations** : Vérification des conflits de circulation sur voie unique (`SimulationEngine`).
2.  **La génération d'horaires** : Calcul des heures de passage, gestion des croisements.
3.  **L'interface avec l'optimisation** : Point d'entrée pour les algorithmes d'optimisation (`generer_tous_trajets_optimises`).
4.  **L'analyse de performance** : Calcul de l'indice d'homogénéité (Gini), import/export de données.

Classes principales :
- `SimulationEngine` : Gère l'état de la simulation, la vérification des segments libres et l'allocation des rames.

Fonctions clés :
- `generer_tous_trajets_optimises` : Orchestrateur principal pour la génération d'horaires.
- `evaluer_configuration` : Calcule le score d'une solution proposée.
"""

from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO
from collections import defaultdict
import itertools

from empreinte import memoiser
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

POIDS_EXCES_PARCOURS = 60.0


def _get_infra_at_gare(df_gares, gare_name):
    """
    Récupère le code d'infrastructure pour une gare donnée.

    Args:
        df_gares (pd.DataFrame): DataFrame contenant les infos des gares (colonnes 'gare', 'infra').
        gare_name (str): Nom de la gare recherchée.

    Returns:
        str: Code infra ('F', 'VE', 'D', 'Terminus', etc.). Retourne 'F' par défaut.
    """
    try:
        if df_gares is None or 'gare' not in df_gares.columns or 'infra' not in df_gares.columns:
            return 'F'
        series = df_gares.loc[df_gares['gare'] == gare_name, 'infra']
        if series.empty: 
            return 'F'
        val = series.iloc[0]
        return str(val).strip().upper() if pd.notna(val) else 'F'
    except:
        return 'F'

def _is_crossing_point(infra_code):
    """
    Détermine si un train peut s'arrêter ou croiser à un point donné.
    
    Un point de croisement est une gare disposant d'une infrastructure adéquate :
    - 'VE' (Voie d'Évitement)
    - 'D' (Double voie / Début double voie)
    - 'Terminus'

    Args:
        infra_code (str): Le code infrastructure de la gare.

    Returns:
        bool: True si le croisement est possible, False sinon.
    """
    return infra_code in ['VE', 'D', 'Terminus']

def _prochain_creneau(earliest, offset_minute, intervalle, dt_debut):
    """Premier instant ≥ ``earliest`` sur la grille de cadencement.

    La grille = {ancre + k·intervalle}, ancrée à ``dt_debut`` calé sur
    ``offset_minute``. Ancrer à dt_debut (plutôt qu'un simple modulo 60) garde
    les retours en phase avec les allers même pour des intervalles non diviseurs
    de 60 (ex. fréquence 1,5/h → pas de 40 min). Garantit des départs réguliers :
    tous les retours d'une mission tombent aux mêmes minutes d'heure en heure.

    Args:
        earliest (datetime): instant au plus tôt (dispo du train après retournement).
        offset_minute (int): minute de référence du départ retour (0-59).
        intervalle (timedelta): pas de cadencement de la mission (60/fréquence).
        dt_debut (datetime): début de service, ancre de la grille.

    Returns:
        datetime: le créneau retenu (≥ earliest).
    """
    offset_minute = int(offset_minute) % 60
    creneau = dt_debut.replace(minute=offset_minute, second=0, microsecond=0)
    while creneau < earliest:
        creneau += intervalle
    while creneau - intervalle >= earliest:
        creneau -= intervalle
    return creneau

def enumerer_rencontres(mission, df_gares, heure_debut, heure_fin,
                        reference_minute_str, frequence_par_heure,
                        turnaround_buffer=0, retour_offset=None):
    """
    Retourne la liste déterministe des rencontres aller×retour idéalisées.
    Chaque entrée: {'aller_idx', 'retour_idx', 'natural_ve', 'candidate_ve'}

    Si ``retour_offset`` est fourni, les départs retour sont calés sur la grille
    de cadencement (cohérent avec le moteur), au lieu de partir au plus tôt.
    """
    from datetime import datetime, timedelta, time as dt_time
    import math

    if frequence_par_heure <= 0:
        return []

    horaire_aller = construire_horaire_mission(mission, 'aller', df_gares)
    horaire_retour = construire_horaire_mission(mission, 'retour', df_gares)
    if not horaire_aller or len(horaire_aller) < 2 or not horaire_retour:
        return []

    duree_aller = horaire_aller[-1].get('time_offset_min', 0)
    duree_retour = horaire_retour[-1].get('time_offset_min', 0)
    if duree_aller <= 0:
        return []

    t_ret_min = mission.get('temps_retournement_B', 10) + turnaround_buffer
    intervalle_min = 60.0 / frequence_par_heure

    def to_min(t):
        if isinstance(t, dt_time):
            return t.hour * 60 + t.minute
        return 0
    debut_min = to_min(heure_debut)
    fin_min = to_min(heure_fin)
    if fin_min <= debut_min:
        fin_min += 24 * 60

    try:
        minutes_ref = sorted(set(
            int(m.strip()) for m in reference_minute_str.split(',')
            if m.strip().isdigit()
        ))
        if not minutes_ref:
            minutes_ref = [0]
    except Exception:
        minutes_ref = [0]

    ve_gares = []
    for pt in horaire_aller[1:-1]:
        if _is_crossing_point(_get_infra_at_gare(df_gares, pt['gare'])):
            ve_gares.append({'gare': pt['gare'], 'time_offset_min': pt['time_offset_min']})

    if not ve_gares:
        return []

    allers = []
    for mref in minutes_ref:
        t = mref - debut_min
        while t < 0:
            t += intervalle_min
        while t < (fin_min - debut_min):
            allers.append(t)
            t += intervalle_min
    allers.sort()

    retours = []
    if retour_offset is not None:
        anchor_t = (int(retour_offset) % 60) - (debut_min % 60)
        for t_a in allers:
            t_r0 = t_a + duree_aller + t_ret_min
            k = math.ceil((t_r0 - anchor_t) / intervalle_min - 1e-9)
            t_r = anchor_t + k * intervalle_min
            if t_r < (fin_min - debut_min):
                retours.append(t_r)
    else:
        for t_a in allers:
            t_r = t_a + duree_aller + t_ret_min
            if t_r < (fin_min - debut_min):
                retours.append(t_r)
    retours.sort()

    rencontres = []
    for aller_idx, t_a in enumerate(allers):
        t_a_arrive = t_a + duree_aller
        for retour_idx, t_r in enumerate(retours):
            t_r_arrive = t_r + duree_retour
            if t_a >= t_r_arrive or t_r >= t_a_arrive:
                continue
            t_meet = (t_a + t_r + duree_aller) / 2.0
            offset_from_origin = t_meet - t_a
            if offset_from_origin <= 0 or offset_from_origin >= duree_aller:
                continue
            best_ve = min(ve_gares, key=lambda v: abs(v['time_offset_min'] - offset_from_origin))
            best_idx = ve_gares.index(best_ve)
            candidates = [ve_gares[best_idx]['gare']]
            if best_idx > 0:
                candidates.append(ve_gares[best_idx - 1]['gare'])
            if best_idx < len(ve_gares) - 1:
                candidates.append(ve_gares[best_idx + 1]['gare'])
            rencontres.append({
                'aller_idx': aller_idx,
                'retour_idx': retour_idx,
                'natural_ve': best_ve['gare'],
                'candidate_ve': candidates,
            })

    return rencontres


def calculer_indice_homogeneite(horaires):
    """
    Calcule l'homogénéité du cadencement via un coefficient de Gini inversé.

    L'indice varie de 0.0 (très irrégulier) à 1.0 (cadencement parfaitement régulier).

    Args:
        horaires (list): Liste d'objets datetime ou timestamps représentant les passages.

    Returns:
        float: Score d'homogénéité entre 0.0 et 1.0.
    """
    if len(horaires) < 2:
        return 1.0

    horaires_tries = sorted(horaires)
    intervalles = [(horaires_tries[i+1] - horaires_tries[i]).total_seconds() / 60.0 
                   for i in range(len(horaires_tries) - 1) if (horaires_tries[i+1] - horaires_tries[i]).total_seconds() / 60.0 > 0.1]

    if not intervalles or sum(intervalles) == 0:
        return 0.0

    n = len(intervalles)
    if n == 0:
        return 0.0
    
    intervalles.sort()
    somme_ponderee = sum((i + 1) * val for i, val in enumerate(intervalles))
    somme_totale = sum(intervalles)

    if somme_totale == 0:
        return 0.0
    
    gini = (2.0 * somme_ponderee) / (n * somme_totale) - (n + 1.0) / n
    return max(0.0, min(1.0, 1.0 - gini))


class SimulationEngine:
    """
    Moteur de simulation gérant l'état des circulations et la détection des conflits.

    Cette classe maintient l'état global de la grille horaire en cours de construction :
    - Segments de voie occupés.
    - Disponibilité des rames (flotte).
    - Respect des contraintes de voie unique (sécurité).
    """
    
    def __init__(self, df_gares, heure_debut, heure_fin):
        """
        Initialise le moteur de simulation.

        Args:
            df_gares (pd.DataFrame): Données des gares (séquencées par distance).
            heure_debut (time): Heure de début de service.
            heure_fin (time): Heure de fin de service.
        """
        self.df_gares = df_gares.sort_values('distance').reset_index(drop=True)
        self.gares_ordonnees = gares_ordonnees(self.df_gares)
        self.gares_map = {r.gare: i for i, r in self.df_gares.iterrows()}
        self.infra_map = {r.gare: _get_infra_at_gare(df_gares, r.gare) for _, r in self.df_gares.iterrows()}

        self.dt_debut = datetime.combine(datetime.today(), heure_debut)
        self.dt_fin = datetime.combine(datetime.today(), heure_fin)
        if self.dt_fin <= self.dt_debut:
            self.dt_fin += timedelta(days=1)

        self.segment_is_double = self._analyze_segments()
        self.reset()

    def reset(self):
        """Réinitialise l'état complet (horaires validés, flotte) pour une nouvelle simulation."""
        self.committed_schedules = []
        self.fleet_availability = defaultdict(list)
        self.train_counter = 1
        self.last_crossing_extensions = []

    def _analyze_segments(self):
        """
        Analyse l'infrastructure pour identifier les segments à double voie.
        
        Returns:
            dict: Mapping {index_segment: bool (True si double voie)}.
        """
        is_double = {}
        n = len(self.df_gares)
        current_state_double = False
        for i in range(n - 1):
            gare_curr = self.df_gares.iloc[i]['gare']
            infra_curr = self.infra_map.get(gare_curr, 'F')
            if infra_curr == 'D':
                current_state_double = not current_state_double
            is_double[i] = current_state_double
        return is_double

    def check_segment_availability(self, seg_idx_min, seg_idx_max, t_enter, t_exit):
        """
        Vérifie si un ensemble de segments (tronçon) est libre sur une plage horaire donnée.

        Args:
            seg_idx_min (int): Index de début du tronçon.
            seg_idx_max (int): Index de fin du tronçon.
            t_enter (datetime): Heure d'entrée sur le tronçon.
            t_exit (datetime): Heure de sortie du tronçon.

        Returns:
            tuple: (bool is_free, datetime next_available_time_if_occupied)
        """
        margin = timedelta(minutes=1)

        all_double = all(self.segment_is_double.get(i, False) for i in range(seg_idx_min, seg_idx_max))

        for committed in self.committed_schedules:
            path = committed['path']
            if not path or path[-1]['arr'] < t_enter or path[0]['dep'] > t_exit:
                continue

            for i in range(len(path) - 1):
                p_a, p_b = path[i], path[i+1]
                o_idx_min = min(p_a['index'], p_b['index'])
                o_idx_max = max(p_a['index'], p_b['index'])

                if max(seg_idx_min, o_idx_min) < min(seg_idx_max, o_idx_max):
                    if all_double:
                        continue
                    o_start = p_a['dep']
                    o_end = p_b['arr']
                    if not (t_exit <= o_start or t_enter >= o_end):
                        return False, o_end + margin

        return True, None

    def solve_mission_schedule(self, mission, ideal_start_time, direction, crossing_strategy=None):
        """
        Construit un sillon valide (horaire) pour une mission, en gérant les conflits de circulation.

        Cette méthode tente de tracer le train point par point (bloc par bloc entre deux évitements).
        Si un conflit est détecté sur une section à voie unique, elle retarde le départ
        depuis le dernier point d'arrêt valide jusqu'à libération de la voie.

        Args:
            mission (dict): Configuration de la mission (gares, temps de parcours).
            ideal_start_time (datetime): Heure de départ souhaitée.
            direction (str): 'aller' ou 'retour'.
            crossing_strategy (CrossingStrategy, optional): Stratégie d'optimisation des croisements.

        Returns:
            tuple: (datetime real_departure, list path_steps, str error_message)
        """
        base_schedule = horaire_mission(mission, direction, self.gares_ordonnees)
        if not base_schedule:
            return None, [], "Erreur itinéraire"
    
        steps = []
        for i, pt in enumerate(base_schedule):
            idx = self.gares_map.get(pt['gare'])
            if i > 0:
                duration = max(0, pt['time_offset_min'] - base_schedule[i-1]['depart_offset_min'])
            else:
                duration = 0
            
            steps.append({
                'gare': pt['gare'],
                'index': idx,
                'run_time': duration,
                'duree_arret': pt.get('duree_arret_min', 0),
                'infra': self.infra_map.get(pt['gare'], 'F')
            })
    
        current_time = ideal_start_time
        final_path = [{
            'gare': steps[0]['gare'],
            'index': steps[0]['index'],
            'arr': current_time,
            'dep': current_time + timedelta(minutes=steps[0]['duree_arret'])
        }]
        current_time += timedelta(minutes=steps[0]['duree_arret'])
        
        self.last_crossing_extensions = []
    
        i = 0
        max_attempts = 500
        
        while i < len(steps) - 1:
            target_idx = i + 1
            while target_idx < len(steps):
                if _is_crossing_point(steps[target_idx]['infra']) or target_idx == len(steps) - 1:
                    break
                target_idx += 1
    
            travel_time_block = sum(steps[k]['run_time'] + steps[k]['duree_arret'] 
                                for k in range(i + 1, target_idx))
            travel_time_block += steps[target_idx]['run_time']
    
            current_departure = current_time
            max_delay = timedelta(minutes=crossing_strategy.max_acceptable_delay if crossing_strategy else 240)
            
            planned_stop_extension = 0
            if crossing_strategy and steps[target_idx]['gare'] in crossing_strategy.stop_durations:
                planned_stop_extension = crossing_strategy.stop_durations[steps[target_idx]['gare']]
            
            attempt = 0
            
            while attempt < max_attempts:
                idx_min = min(steps[i]['index'], steps[target_idx]['index'])
                idx_max = max(steps[i]['index'], steps[target_idx]['index'])
    
                t_enter = current_departure
                t_exit = current_departure + timedelta(minutes=travel_time_block + planned_stop_extension)
    
                is_free, next_t = self.check_segment_availability(idx_min, idx_max, t_enter, t_exit)
    
                if is_free:
                    final_path[-1]['dep'] = current_departure
                    t_cursor = current_departure
    
                    for k in range(i + 1, target_idx + 1):
                        st_k = steps[k]
                        t_cursor += timedelta(minutes=st_k['run_time'])
                        
                        base_stop = st_k['duree_arret']
                        extra_stop = 0
                        
                        if crossing_strategy and st_k['gare'] in crossing_strategy.stop_durations:
                            extra_stop = crossing_strategy.stop_durations[st_k['gare']]
                            if extra_stop > 0:
                                self.last_crossing_extensions.append({
                                    'gare': st_k['gare'],
                                    'extension_minutes': extra_stop,
                                    'reason': 'strategic_crossing'
                                })
                        
                        final_path.append({
                            'gare': st_k['gare'],
                            'index': st_k['index'],
                            'arr': t_cursor,
                            'dep': t_cursor + timedelta(minutes=base_stop + extra_stop)
                        })
                        t_cursor += timedelta(minutes=base_stop + extra_stop)
    
                    current_time = t_cursor
                    i = target_idx
                    break
                    
                else:
                    current_departure = next_t
                    
                    if current_departure - ideal_start_time > max_delay:
                        return None, [], "Impasse infra (délai max dépassé)"
                    
                    attempt += 1
            
            if attempt >= max_attempts:
                return None, [], "Trop de tentatives de résolution"
    
        return final_path[0]['dep'], final_path, None

    def allocate_train_id(self, gare, target_time, type_materiel, mission_id, 
                         can_inject=True, allow_cross_mission_sharing=True):
        """Alloue une rame à un départ."""
        pool = self.fleet_availability[gare]
        pool.sort()

        for i, (dispo_t, tid, mat_type, orig_mission) in enumerate(pool):
            if mat_type != type_materiel:
                continue
            if dispo_t > target_time + timedelta(minutes=2):
                continue
            
            same_mission = (orig_mission == mission_id)
            if same_mission or allow_cross_mission_sharing:
                return pool.pop(i)[1]

        if can_inject:
            tid = self.train_counter
            self.train_counter += 1
            return tid

        return None

    def register_arrival(self, tid, gare, arr_time, turnaround_min, type_materiel, mission_id):
        """Libère une rame dans une gare après son temps de retournement."""
        self.fleet_availability[gare].append(
            (arr_time + timedelta(minutes=turnaround_min), tid, type_materiel, mission_id)
        )


def evaluer_configuration(engine, requests, allow_cross_mission_sharing=True, crossing_strategies=None):
    """Évalue une configuration complète avec support des stratégies de croisement."""
    if crossing_strategies is None:
        crossing_strategies = {}
        
    engine.reset()
    total_delay_min = 0
    total_crossing_extensions = 0 
    
    trajets_resultat = defaultdict(list)
    failures = []
    mission_station_times = defaultdict(lambda: defaultdict(list))
    
    sorted_reqs = sorted(requests, key=lambda x: x['ideal_dep'])
    
    for req in sorted_reqs:
        mission_id = f"M{req.get('m_idx', 0)}"
        type_materiel = req['mission'].get('type_materiel', 'electrique')
        inject_allowed = True
        
        if req['type'] == 'retour':
            inject_allowed = req['mission'].get('inject_from_terminus_2', False)
        
        crossing_strategy = crossing_strategies.get(mission_id, None)
        
        real_dep, path, err = engine.solve_mission_schedule(
            req['mission'], req['ideal_dep'], req['type'], crossing_strategy
        )

        if err or not path:
            failures.append({
                "time": req['ideal_dep'],
                "mission": f"M{req.get('m_idx', 0)+1} ({req['type']})",
                "reason": err or "Échec construction horaire",
                "is_infra_violation": ("infra" in (err or "").lower())
            })
            continue
        
        tid = engine.allocate_train_id(
            path[0]['gare'], 
            real_dep, 
            type_materiel,
            mission_id,
            can_inject=inject_allowed,
            allow_cross_mission_sharing=allow_cross_mission_sharing
        )

        if tid is None:
            failures.append({
                "time": req['ideal_dep'],
                "mission": f"M{req.get('m_idx', 0)+1} ({req['type']})",
                "reason": "Pas de rame disponible",
                "is_infra_violation": False
            })
            continue

        tid = int(tid)
        engine.committed_schedules.append({'train_id': tid, 'path': path})
        
        delay = (real_dep - req['ideal_dep']).total_seconds() / 60
        total_delay_min += delay

        m_key = f"M{req.get('m_idx', 0)}_{req['type']}"
        for step in path:
            is_terminus = (step == path[-1])
            time_to_record = step['arr'] if is_terminus else step['dep']
            mission_station_times[m_key][step['gare']].append(time_to_record)

        mission_label = f"{req['mission']['origine']} → {req['mission']['terminus']}"
        if req['type'] == 'retour':
             mission_label = f"{req['mission']['terminus']} → {req['mission']['origine']}"

        for k in range(len(path)-1):
            p_curr, p_next = path[k], path[k+1]
            trajets_resultat[tid].append({
                "start": p_curr['dep'],
                "end": p_next['arr'],
                "origine": p_curr['gare'],
                "terminus": p_next['gare'],
                "mission": mission_label,
                "is_mission_start": (k == 0)
            })
            if p_next['dep'] > p_next['arr']:
                trajets_resultat[tid].append({
                    "start": p_next['arr'],
                    "end": p_next['dep'],
                    "origine": p_next['gare'],
                    "terminus": p_next['gare'],
                    "mission": mission_label,
                    "is_mission_start": False
                })

        t_ret = req['mission'].get(
            'temps_retournement_B' if req['type'] == 'aller' else 'temps_retournement_A', 
            10
        )
        engine.register_arrival(tid, path[-1]['gare'], path[-1]['dep'], t_ret, type_materiel, mission_id)
        
        if hasattr(engine, 'last_crossing_extensions'):
            for ext in engine.last_crossing_extensions:
                total_crossing_extensions += ext.get('extension_minutes', 0)

    global_homogeneity_score = 0
    total_stations_checked = 0
    homogeneite_par_mission = {}

    for m_key, stations_data in mission_station_times.items():
        scores_gares = []
        for gare, horaires in stations_data.items():
            if len(horaires) > 1:
                g_score = calculer_indice_homogeneite(horaires)
                scores_gares.append(g_score)
                global_homogeneity_score += g_score
                total_stations_checked += 1

        avg_score = sum(scores_gares) / len(scores_gares) if scores_gares else 1.0
        try:
            midx, sens = m_key.split('_')
            ui_key = f"Mission {int(midx[1:])+1} ({sens.capitalize()})"
        except:
            ui_key = m_key
        homogeneite_par_mission[ui_key] = avg_score

    avg_homogeneity = global_homogeneity_score / total_stations_checked if total_stations_checked > 0 else 1.0

    penalty_trains = engine.train_counter * 2000
    penalty_failures = len(failures) * 5000
    penalty_delay = total_delay_min * 10
    bonus_homogeneity = avg_homogeneity * 3000

    score = penalty_trains + penalty_failures + penalty_delay - bonus_homogeneity
    _max_arret_legacy = 5
    if total_crossing_extensions <= _max_arret_legacy:
        score += total_crossing_extensions * 15
    else:
        excess = total_crossing_extensions - _max_arret_legacy
        score += _max_arret_legacy * 15 + (excess ** 2) * 100

    return score, dict(trajets_resultat), failures, homogeneite_par_mission, total_delay_min, engine.train_counter


def _calculer_duree_mission_max(missions, df_gares):
    """
    Calcule la durée maximale d'une mission aller.
    
    Args:
        missions (list): Liste des missions
        df_gares (pd.DataFrame): DataFrame des gares
        
    Returns:
        int: Durée maximale en minutes
    """
    duree_max = 0
    for mission in missions:
        if mission.get('frequence', 0) <= 0:
            continue
        horaire = construire_horaire_mission(mission, 'aller', df_gares)
        if horaire and len(horaire) > 0:
            duree = horaire[-1].get('time_offset_min', 0)
            duree_max = max(duree_max, duree)
    return duree_max

def executer_simulation_evenementielle(
    missions, df_gares, heure_debut, heure_fin,
    allow_sharing=True,
    turnaround_buffers=None,
    crossing_strategies=None,
    adjusted_reference_minutes=None,
    crossing_pair_assignments=None,
    retour_reference_offsets=None,
):
    """
    Moteur événementiel unifié pour la simulation ferroviaire.

    Seuls les ALLERS sont programmés, les RETOURS sont générés dynamiquement
    quand une rame devient disponible.

    Args:
        missions (list): Liste des configurations de missions.
        df_gares (pd.DataFrame): Données d'infrastructure.
        heure_debut (time): Début de service.
        heure_fin (time): Fin de service.
        allow_sharing (bool): Partage inter-missions autorisé.
        turnaround_buffers (dict): {mission_id: int} minutes supplémentaires ajoutées
            au temps de retournement minimum (utile pour améliorer les croisements).
        crossing_strategies (dict): {mission_id: CrossingStrategy} stratégies de croisement
            aux points VE (arrêts prolongés pour laisser passer).
        adjusted_reference_minutes (dict): {m_idx: str} minutes de référence surchargées
            (ex: {"0": "15", "1": "30"}). Si None, utilise mission['reference_minutes'].
        crossing_pair_assignments (dict): {mission_id: [gares VE préférées]} pour
            orienter le point de croisement (cf. crossing_pairs côté optimiseur).
        retour_reference_offsets (dict): {mission_id: int|None} minute de référence
            du départ retour (cadencement). Si fournie pour une mission, le retour
            part au prochain créneau de la grille (≥ dispo après retournement),
            garantissant des départs réguliers. None/absent = comportement ASAP.

    Returns:
        tuple: (chronologie, warnings, stats_homogeneite)
    """
    import heapq

    if turnaround_buffers is None:
        turnaround_buffers = {}
    if crossing_strategies is None:
        crossing_strategies = {}
    if adjusted_reference_minutes is None:
        adjusted_reference_minutes = {}
    if retour_reference_offsets is None:
        retour_reference_offsets = {}

    def _buffer_directionnel(mission_id, side):
        """Buffer de retournement pour un côté ('A' origine / 'B' terminus).

        Accepte un int (compat : appliqué aux deux côtés) ou un dict {'A','B'}.
        """
        raw = turnaround_buffers.get(mission_id, 0)
        if isinstance(raw, dict):
            return raw.get(side, 0)
        return raw

    def _preferred_ves(mission_id, sens):
        """VE préférées pour une mission et un sens ('aller'/'retour').

        Accepte une liste simple (compat : appliquée aux deux sens) ou un dict
        directionnel {'aller': [...], 'retour': [...]}.
        """
        raw = crossing_pair_assignments.get(mission_id) if crossing_pair_assignments else None
        if raw is None:
            return []
        if isinstance(raw, dict):
            return raw.get(sens, []) or []
        return raw

    engine = SimulationEngine(df_gares, heure_debut, heure_fin)
    gares = engine.gares_ordonnees
    infra_map = engine.infra_map

    infra_violation_warnings = []
    other_warnings = []
    chronologie_reelle = {}
    id_train_counter = 1
    event_counter = 0

    trains = {}
    evenements = []

    for m_idx, mission in enumerate(missions):
        if mission.get('frequence', 0) <= 0:
            continue

        mission_id = f"M{m_idx+1}"
        frequence = mission['frequence']
        intervalle = timedelta(hours=1.0 / frequence)

        ref_str = adjusted_reference_minutes.get(str(m_idx), mission.get("reference_minutes", "0"))
        try:
            minutes_ref = sorted(list(set([
                int(m.strip()) for m in ref_str.split(',')
                if m.strip().isdigit()
            ])))
            if not minutes_ref:
                minutes_ref = [0]
        except Exception:
            minutes_ref = [0]

        heure_debut_mission = engine.dt_debut

        if mission.get('inject_from_terminus_2', False):
            horaire_aller = horaire_mission(mission, "aller", gares)
            if horaire_aller:
                temps_trajet_aller = horaire_aller[-1].get("time_offset_min", 0)
                t_ret_b = mission.get("temps_retournement_B", 10)
                buf = _buffer_directionnel(mission_id, "B")
                temps_avant_service = temps_trajet_aller + t_ret_b + buf + 60
                heure_debut_mission = engine.dt_debut - timedelta(minutes=temps_avant_service)

        for minute_ref in minutes_ref:
            offset_hours = minute_ref // 60
            offset_minutes = minute_ref % 60

            curseur_temps = heure_debut_mission.replace(
                minute=offset_minutes, second=0, microsecond=0
            ) + timedelta(hours=offset_hours)

            while curseur_temps < heure_debut_mission:
                curseur_temps += intervalle

            while curseur_temps < engine.dt_fin:
                event_counter += 1
                is_fictif = curseur_temps < engine.dt_debut
                heapq.heappush(evenements, (
                    curseur_temps, event_counter, "demande_depart_aller",
                    {"mission": mission, "mission_id": mission_id, "m_idx": m_idx,
                     "is_aller_fictif": is_fictif}
                ))
                curseur_temps += intervalle

    while evenements:
        heure, _, type_event, details = heapq.heappop(evenements)

        if heure >= engine.dt_fin:
            continue

        if type_event == "demande_depart_aller":
            mission_cfg = details["mission"]
            mission_id = details["mission_id"]
            origine = mission_cfg["origine"]
            is_aller_fictif = details.get("is_aller_fictif", False)
            type_materiel = mission_cfg.get("type_materiel", "diesel")

            train_assigne_id = None
            earliest_dispo = datetime.max

            for id_t, t in trains.items():
                if t.get("loc") == origine and t.get("dispo_a", datetime.max) <= heure:
                    if not allow_sharing and t.get("mission_id") != mission_id:
                        continue
                    if t.get("type_materiel") != type_materiel:
                        continue
                    if t["dispo_a"] < earliest_dispo:
                        earliest_dispo = t["dispo_a"]
                        train_assigne_id = id_t

            if train_assigne_id is None:
                train_assigne_id = id_train_counter
                trains[train_assigne_id] = {
                    "id": train_assigne_id,
                    "loc": origine,
                    "dispo_a": heure,
                    "mission_id": mission_id,
                    "type_materiel": type_materiel,
                }
                chronologie_reelle[train_assigne_id] = []
                id_train_counter += 1
            else:
                trains[train_assigne_id]["dispo_a"] = max(heure, earliest_dispo)

            heure_programmation = max(heure, trains[train_assigne_id]["dispo_a"])
            event_counter += 1
            heapq.heappush(evenements, (heure_programmation, event_counter, "tentative_mouvement", {
                "id_train": train_assigne_id,
                "mission": mission_cfg,
                "mission_id": mission_id,
                "trajet_spec": "aller",
                "index_etape": 0,
                "retry_count": 0,
                "is_trajet_fictif": is_aller_fictif,
            }))

        elif type_event == "tentative_mouvement":
            id_train = details["id_train"]
            mission_cfg = details["mission"]
            mission_id = details["mission_id"]
            trajet_spec = details["trajet_spec"]
            index_etape = details["index_etape"]
            is_trajet_fictif = details.get("is_trajet_fictif", False)
            use_preferred_ve = details.get("use_preferred_ve", True)

            cs = crossing_strategies.get(mission_id) if crossing_strategies else None

            horaire = horaire_mission(mission_cfg, trajet_spec, gares)

            if not horaire or index_etape >= len(horaire) - 1:
                continue

            bloc_gares = [horaire[index_etape]]
            next_crossing_idx = index_etape + 1

            while next_crossing_idx < len(horaire):
                gare_name = horaire[next_crossing_idx]["gare"]
                infra = infra_map.get(gare_name, "F")
                bloc_gares.append(horaire[next_crossing_idx])

                if _is_crossing_point(infra) or next_crossing_idx == len(horaire) - 1:
                    break
                next_crossing_idx += 1

            pt_depart_bloc = bloc_gares[0]
            pt_arrivee_bloc = bloc_gares[-1]

            duree_bloc_min = max(0, pt_arrivee_bloc["time_offset_min"] - pt_depart_bloc["depart_offset_min"])

            duree_arret_commercial = pt_arrivee_bloc.get("duree_arret_min", 0)
            duree_arret_final = duree_arret_commercial

            if cs and pt_arrivee_bloc["gare"] in cs.stop_durations:
                duree_arret_final = max(duree_arret_final, cs.stop_durations[pt_arrivee_bloc["gare"]])

            gare_dep_bloc = pt_depart_bloc.get("gare")
            gare_arr_bloc = pt_arrivee_bloc.get("gare")

            preferred_extension_applied = False
            if use_preferred_ve and crossing_pair_assignments:
                preferred_ves = _preferred_ves(mission_id, trajet_spec)
                if preferred_ves and gare_arr_bloc not in preferred_ves:
                    look_idx = next_crossing_idx + 1
                    extended = list(bloc_gares)
                    found_preferred = False
                    while look_idx < len(horaire):
                        g = horaire[look_idx]
                        g_name = g["gare"]
                        g_infra = infra_map.get(g_name, "F")
                        extended.append(g)
                        if _is_crossing_point(g_infra):
                            if g_name in preferred_ves:
                                found_preferred = True
                                look_idx += 1
                            break
                        look_idx += 1
                    if found_preferred:
                        new_arrivee = extended[-1]
                        new_duree = max(0, new_arrivee["time_offset_min"] - pt_depart_bloc["depart_offset_min"])
                        preferred_extension_applied = True
                        bloc_gares = extended
                        next_crossing_idx = look_idx - 1
                        pt_arrivee_bloc = new_arrivee
                        gare_arr_bloc = new_arrivee["gare"]
                        duree_bloc_min = new_duree
                        duree_arret_commercial = new_arrivee.get("duree_arret_min", 0)
                        duree_arret_final = duree_arret_commercial
                        if cs and gare_arr_bloc in cs.stop_durations:
                            duree_arret_final = max(duree_arret_final,
                                                    cs.stop_durations[gare_arr_bloc])

            dispo_train = trains.get(id_train, {}).get("dispo_a", heure)
            heure_depart_reelle = max(heure, dispo_train)

            if heure_depart_reelle >= engine.dt_fin:
                continue

            conflit = False
            fin_conflit = None

            if duree_bloc_min > 0 and gare_dep_bloc != gare_arr_bloc:
                idx_dep = engine.gares_map.get(gare_dep_bloc)
                idx_arr = engine.gares_map.get(gare_arr_bloc)

                if idx_dep is not None and idx_arr is not None:
                    idx_min = min(idx_dep, idx_arr)
                    idx_max = max(idx_dep, idx_arr)

                    t_enter = heure_depart_reelle
                    t_exit = heure_depart_reelle + timedelta(minutes=duree_bloc_min + duree_arret_final)

                    is_free, next_t = engine.check_segment_availability(idx_min, idx_max, t_enter, t_exit)

                    if not is_free:
                        conflit = True
                        fin_conflit = next_t

            if not conflit:
                heure_arrivee_finale = heure_depart_reelle + timedelta(minutes=duree_bloc_min)


                mission_label = f"{mission_cfg['origine']} → {mission_cfg['terminus']}"
                if trajet_spec == "retour":
                    mission_label = f"{mission_cfg['terminus']} → {mission_cfg['origine']}"

                if not is_trajet_fictif:
                    conflict_wait = (heure_depart_reelle - dispo_train).total_seconds() / 60.0
                    if index_etape > 0 and conflict_wait > 0.5:
                        wait_entry = {
                            "start": dispo_train,
                            "end": heure_depart_reelle,
                            "origine": gare_dep_bloc,
                            "terminus": gare_dep_bloc,
                            "mission": mission_label,
                            "is_mission_start": False,
                            "crossing_extension_min": conflict_wait,
                            "subi": True,
                        }
                        chronologie_reelle.setdefault(id_train, []).append(wait_entry)

                    first_seg = True
                    for i in range(len(bloc_gares) - 1):
                        pt_curr = bloc_gares[i]
                        pt_next = bloc_gares[i + 1]

                        duree_segment = max(0, pt_next["time_offset_min"] - pt_curr["depart_offset_min"])

                        base_offset = pt_depart_bloc["depart_offset_min"]
                        offset_arr_curr = max(0, pt_curr["time_offset_min"] - base_offset) if i > 0 else 0
                        offset_arr_next = max(0, pt_next["time_offset_min"] - base_offset)
                        arret_inter = pt_curr["duree_arret_min"] if i > 0 else 0
                        offset_dep_curr = offset_arr_curr + arret_inter

                        h_arr_curr = heure_depart_reelle + timedelta(minutes=offset_arr_curr)
                        h_dep_curr = heure_depart_reelle + timedelta(minutes=offset_dep_curr)
                        h_arr_next = heure_depart_reelle + timedelta(minutes=offset_arr_next)

                        if i > 0 and arret_inter > 0:
                            arret_inter_entry = {
                                "start": h_arr_curr,
                                "end": h_dep_curr,
                                "origine": pt_curr["gare"],
                                "terminus": pt_curr["gare"],
                                "mission": mission_label,
                                "is_mission_start": False,
                            }
                            chronologie_reelle.setdefault(id_train, []).append(arret_inter_entry)

                        if duree_segment > 0 or (pt_curr["gare"] != pt_next["gare"]):
                            seg_entry = {
                                "start": h_dep_curr,
                                "end": h_arr_next,
                                "origine": pt_curr["gare"],
                                "terminus": pt_next["gare"],
                                "mission": mission_label,
                                "is_mission_start": (first_seg and index_etape == 0),
                            }
                            if first_seg and details.get("preferred_ve_failed"):
                                seg_entry["crossing_assignment_violated"] = True
                            chronologie_reelle.setdefault(id_train, []).append(seg_entry)
                            first_seg = False

                    if duree_arret_final > 0:
                        arr_gare = gare_arr_bloc
                        arr_time = heure_arrivee_finale
                        dep_time = heure_arrivee_finale + timedelta(minutes=duree_arret_final)
                        crossing_ext = max(0, duree_arret_final - duree_arret_commercial)
                        arret_entry = {
                            "start": arr_time,
                            "end": dep_time,
                            "origine": arr_gare,
                            "terminus": arr_gare,
                            "mission": mission_label,
                            "is_mission_start": False,
                            "crossing_extension_min": crossing_ext,
                        }
                        chronologie_reelle.setdefault(id_train, []).append(arret_entry)

                idx_dep = engine.gares_map[gare_dep_bloc]
                idx_arr = engine.gares_map[gare_arr_bloc]

                path_bloc = [
                    {'gare': gare_dep_bloc, 'index': idx_dep, 'arr': heure_depart_reelle, 'dep': heure_depart_reelle},
                    {'gare': gare_arr_bloc, 'index': idx_arr, 'arr': heure_arrivee_finale,
                     'dep': heure_arrivee_finale + timedelta(minutes=duree_arret_final)},
                ]
                engine.committed_schedules.append({'train_id': id_train, 'path': path_bloc})

                trains[id_train]["loc"] = gare_arr_bloc
                trains[id_train]["dispo_a"] = heure_arrivee_finale + timedelta(minutes=duree_arret_final)

                if next_crossing_idx < len(horaire) - 1:
                    event_counter += 1
                    heapq.heappush(evenements, (
                        trains[id_train]["dispo_a"], event_counter, "tentative_mouvement",
                        {
                            "id_train": id_train, "mission": mission_cfg, "mission_id": mission_id,
                            "trajet_spec": trajet_spec, "index_etape": next_crossing_idx,
                            "retry_count": 0, "is_trajet_fictif": is_trajet_fictif,
                            "use_preferred_ve": True,
                        }
                    ))
                else:
                    event_counter += 1
                    heapq.heappush(evenements, (
                        trains[id_train]["dispo_a"], event_counter, "fin_mission",
                        {
                            "id_train": id_train, "mission": mission_cfg, "mission_id": mission_id,
                            "trajet_spec": trajet_spec, "gare_finale": gare_arr_bloc,
                            "is_trajet_fictif": is_trajet_fictif,
                        }
                    ))
            else:
                if use_preferred_ve:
                    new_details = details.copy()
                    new_details["use_preferred_ve"] = False
                    new_details["retry_count"] = 0
                    if preferred_extension_applied:
                        new_details["preferred_ve_failed"] = True
                    event_counter += 1
                    heapq.heappush(evenements, (
                        heure_depart_reelle, event_counter, "tentative_mouvement", new_details
                    ))
                else:
                    retry_count = details.get("retry_count", 0)
                    if retry_count < 500:
                        new_details = details.copy()
                        new_details["retry_count"] = retry_count + 1
                        event_counter += 1
                        heapq.heappush(evenements, (
                            fin_conflit, event_counter, "tentative_mouvement", new_details
                        ))
                    else:
                        gares_sans_ve = []
                        for gare_info in bloc_gares[1:-1]:
                            gare_inter = gare_info["gare"]
                            infra_inter = infra_map.get(gare_inter, "F")
                            if not _is_crossing_point(infra_inter):
                                gares_sans_ve.append(f"{gare_inter} ({infra_inter})")

                        reason_detail = f"Impossible de trouver un créneau libre après 500 tentatives pour le bloc {gare_dep_bloc} → {gare_arr_bloc}"
                        if gares_sans_ve:
                            reason_detail += f". Gares sans voie d'évitement dans le bloc : {', '.join(gares_sans_ve)}"

                        infra_violation_warnings.append({
                            "time": heure_depart_reelle,
                            "gare": f"{gare_dep_bloc} → {gare_arr_bloc}",
                            "mission": mission_id,
                            "reason": reason_detail,
                            "is_infra_violation": True,
                        })

        elif type_event == "fin_mission":
            id_train = details["id_train"]
            mission_cfg = details["mission"]
            mission_id = details["mission_id"]
            gare_finale = details["gare_finale"]
            is_trajet_fictif = details.get("is_trajet_fictif", False)

            if id_train not in trains:
                continue

            trains[id_train]["loc"] = gare_finale
            heure_arrivee_mission = heure

            if details["trajet_spec"] == "aller":
                buf_b = _buffer_directionnel(mission_id, "B")
                t_ret_min = mission_cfg.get("temps_retournement_B", 10)
                heure_dispo_retour = heure_arrivee_mission + timedelta(minutes=t_ret_min + buf_b)

                offset_retour = retour_reference_offsets.get(mission_id)
                heure_depart_retour = heure_dispo_retour
                if offset_retour is not None:
                    freq = mission_cfg.get("frequence", 0)
                    if freq > 0:
                        intervalle_ret = timedelta(hours=1.0 / freq)
                        heure_depart_retour = _prochain_creneau(
                            heure_dispo_retour, offset_retour, intervalle_ret, engine.dt_debut
                        )

                trains[id_train]["dispo_a"] = heure_depart_retour

                horaire_retour = horaire_mission(mission_cfg, "retour", gares)
                if horaire_retour and len(horaire_retour) > 1:
                    if heure_depart_retour < engine.dt_fin:
                        is_retour_fictif = heure_depart_retour < engine.dt_debut
                        event_counter += 1
                        heapq.heappush(evenements, (heure_depart_retour, event_counter, "tentative_mouvement", {
                            "id_train": id_train, "mission": mission_cfg, "mission_id": mission_id,
                            "trajet_spec": "retour", "index_etape": 0, "retry_count": 0,
                            "is_trajet_fictif": is_retour_fictif,
                        }))
            else:
                buf_a = _buffer_directionnel(mission_id, "A")
                t_ret_min = mission_cfg.get("temps_retournement_A", 10)
                trains[id_train]["dispo_a"] = heure_arrivee_mission + timedelta(minutes=t_ret_min + buf_a)

    trains_a_supprimer = [tid for tid, trajets in chronologie_reelle.items() if not trajets]
    for tid in trains_a_supprimer:
        del chronologie_reelle[tid]

    warnings = {"infra_violations": infra_violation_warnings, "other": other_warnings}
    stats_homogeneite = _calculer_stats_homogeneite(chronologie_reelle)

    return chronologie_reelle, warnings, stats_homogeneite


def _calculer_exces_parcours(chronologie, durees_theoriques):
    """Somme (minutes) du temps de parcours excédentaire vs la durée théorique.

    Découpe chaque rame en courses sur les frontières ``is_mission_start`` et
    compare la durée réelle (du départ de la course à l'arrivée du dernier
    segment de circulation) à la durée théorique de la mission/sens. L'excès
    capture aussi bien les arrêts stratégiques aux VE que les attentes subies en
    ligne, puisque tous deux repoussent les segments suivants.

    Args:
        chronologie (dict): {train_id: [segments]}.
        durees_theoriques (dict): {label_mission: duree_min}, label au format
            "Origine → Terminus" (sens inclus dans la flèche).

    Returns:
        float: total des minutes excédentaires (toujours ≥ 0 par course).
    """
    if not chronologie or not durees_theoriques:
        return 0.0

    total = 0.0
    for trajets in chronologie.values():
        if not trajets:
            continue
        trajets_tries = sorted(trajets, key=lambda x: x['start'])

        courses = []
        current = None
        for t in trajets_tries:
            if t.get('is_mission_start'):
                if current is not None:
                    courses.append(current)
                current = [t]
            elif current is not None:
                current.append(t)
        if current is not None:
            courses.append(current)

        for segs in courses:
            travel = [s for s in segs if s['origine'] != s['terminus']]
            if not travel:
                continue
            label = travel[0]['mission']
            theo = durees_theoriques.get(label)
            if not theo or theo <= 0:
                continue
            span = (travel[-1]['end'] - travel[0]['start']).total_seconds() / 60.0
            exces = span - theo
            if exces > 0:
                total += exces

    return total


def _score_chronologie_bruit(chronologie, warnings, max_arret_ligne_min=5,
                             durees_theoriques=None):
    """Score bas niveau sans appel Streamlit — utilisé par l'optimisation interne.

    Le terme de temps de parcours (``durees_theoriques``) crée le gradient
    « trajets plus courts » : chaque minute d'attente (subie OU stratégique) est
    comptée une seule fois, au même tarif, via l'excès de parcours. La pénalité
    quadratique d'arrêt n'intervient plus qu'au-delà du plafond utilisateur
    (``max_arret_ligne_min``) pour faire respecter cette borne, sur les deux
    types d'attente indifféremment.
    """
    nb_rames = len(chronologie) if chronologie else 0
    nb_violations = len(warnings.get("infra_violations", []))
    nb_fails = len(warnings.get("other", []))

    stats = _calculer_stats_homogeneite(chronologie)
    avg_gini = 0
    count = 0
    for v in stats.values():
        if isinstance(v, (int, float)):
            avg_gini += v
            count += 1
    avg_gini = avg_gini / count if count > 0 else 1.0

    penalty_overflow = 0.0
    violated_count = 0
    if chronologie:
        for steps in chronologie.values():
            for step in steps:
                if step.get('crossing_assignment_violated'):
                    violated_count += 1
                ext = step.get('crossing_extension_min', 0)
                if ext > max_arret_ligne_min:
                    over = ext - max_arret_ligne_min
                    penalty_overflow += (over ** 2) * 800.0
    penalty_overflow += violated_count * 300

    penalty_parcours = _calculer_exces_parcours(chronologie, durees_theoriques) * POIDS_EXCES_PARCOURS

    return (nb_rames * 2000 + nb_violations * 50000 + nb_fails * 3000
            - avg_gini * 3000 + penalty_overflow + penalty_parcours)


def generer_tous_trajets_optimises(missions, df_gares, heure_debut, heure_fin,
                                   allow_sharing=True, optimization_config=None,
                                   progress_callback=None, search_strategy='simple',
                                   crossing_strategies=None):
    """
    Orchestrateur principal — délègue toujours au moteur événementiel.
    
    Modes de recherche :
    - simple : exécution directe sans exploration
    - smart_progressive / fast / smart : délégation vers optimisation_logic
    - exhaustif / genetic : délégation vers optimisation_logic

    Args:
        missions, df_gares, heure_debut, heure_fin : paramètres infrastructure/service.
        allow_sharing (bool): Partage inter-missions.
        optimization_config (OptimizationConfig, optional): Si présent, délègue.
        progress_callback (callable, optional): Barre de progression UI.
        search_strategy (str): 'simple', 'smart_progressive', 'fast', 'smart', 'exhaustif', 'genetic'.
        crossing_strategies (dict): Stratégies de croisement pré-calculées.

    Returns:
        tuple: (chronologie, warnings, stats_homogeneite)
    """

    if optimization_config is not None:
        from optimisation_logic import optimiser_graphique_horaire
        chronologie, warnings, stats = optimiser_graphique_horaire(
            missions, df_gares, heure_debut, heure_fin,
            optimization_config, allow_sharing=allow_sharing,
            progress_callback=progress_callback
        )
        stats_homogeneite = _calculer_stats_homogeneite(chronologie)
        return chronologie, warnings, stats_homogeneite

    if search_strategy == 'simple':
        chronologie, warnings, stats_homogeneite = executer_simulation_evenementielle(
            missions, df_gares, heure_debut, heure_fin,
            allow_sharing=allow_sharing,
            turnaround_buffers={},
            crossing_strategies=crossing_strategies,
        )
        if progress_callback:
            progress_callback(1, 1, 0, len(chronologie), 0)
        return chronologie, warnings, stats_homogeneite

    from optimisation_logic import optimiser_graphique_horaire, OptimizationConfig
    config = OptimizationConfig(mode=search_strategy)
    chronologie, warnings, _ = optimiser_graphique_horaire(
        missions, df_gares, heure_debut, heure_fin,
        config, allow_sharing=allow_sharing,
        progress_callback=progress_callback
    )
    stats_homogeneite = _calculer_stats_homogeneite(chronologie)
    return chronologie, warnings, stats_homogeneite


def gares_ordonnees(df_gares):
    """(gare, distance) triées : tout ce que l'horaire théorique lit de l'infrastructure."""
    gs = df_gares.sort_values('distance')
    return tuple(zip(gs['gare'].tolist(), gs['distance'].astype(float).tolist()))


def construire_horaire_mission(mission, direction, df_gares):
    return horaire_mission(mission, direction, gares_ordonnees(df_gares))


@memoiser(maxsize=256)
def horaire_mission(mission, direction, gares):
    if direction not in ['aller', 'retour']:
        return []
    
    o = mission.get('origine')
    t = mission.get('terminus')
    
    if not o or not t:
        return []
    
    t_trajet = mission.get('temps_trajet', 60)
    pass_pts = mission.get('passing_points', [])
    
    if direction == 'retour':
        o, t = t, o
        trajet_asym = mission.get('trajet_asymetrique', False)
        if trajet_asym:
            t_trajet = mission.get('temps_trajet_retour', t_trajet)
            pass_pts = mission.get('passing_points_retour', [])
        else:
            pass_pts = [{"gare": p["gare"],
                        "time_offset_min": (
                            t_trajet - p["time_offset_min"]
                            - (p.get("duree_arret_min", 0) if p.get("arret_commercial", False) else 0)
                        ),
                        "arret_commercial": p.get("arret_commercial", False),
                        "duree_arret_min": p.get("duree_arret_min", 0)}
                       for p in reversed(pass_pts)]
    
    pts = [{"gare": o, "time_offset_min": 0, "duree_arret_min": 0}]
    
    for p in pass_pts:
        duree_arret = p.get("duree_arret_min", 0) if p.get("arret_commercial", False) else 0
        pts.append({
            "gare": p["gare"],
            "time_offset_min": p["time_offset_min"],
            "duree_arret_min": duree_arret
        })
    
    for m in mission.get('missions_intermediaires', []):
        dur = m.get('temps_trajet', 0)
        if dur > 0:
            pts.append({"gare": m.get("terminus"), "time_offset_min": dur, "duree_arret_min": 0})
            pts.append({"gare": m.get("origine"), "time_offset_min": dur, "duree_arret_min": 0})

    pts.append({"gare": t, "time_offset_min": t_trajet, "duree_arret_min": 0})

    pts.sort(key=lambda x: x['time_offset_min'])
    
    unique = []
    seen = set()
    for p in pts:
        if p['gare'] not in seen:
            unique.append(p)
            seen.add(p['gare'])

    res = []
    gmap = {gare: (i, distance) for i, (gare, distance) in enumerate(gares)}

    for i in range(len(unique) - 1):
        s, e = unique[i], unique[i+1]
        if s['gare'] not in gmap or e['gare'] not in gmap: 
            continue
        i_s, d_s = gmap[s['gare']]
        i_e, d_e = gmap[e['gare']]
        seg = gares[min(i_s, i_e) : max(i_s, i_e)+1]
        if i_e < i_s:
            seg = seg[::-1]

        for gare, distance in seg:
            if res and res[-1]['gare'] == gare:
                continue
            d_arret = 0
            for op in unique:
                if op['gare'] == gare:
                    d_arret = op.get('duree_arret_min', 0)
                    break
            dist_p = abs(distance - d_s)
            tot_d = abs(d_e - d_s)
            ratio = dist_p / tot_d if tot_d > 0 else 0
            t = s['time_offset_min'] + ((e['time_offset_min'] - s['time_offset_min']) * ratio)
            res.append({"gare": gare, "time_offset_min": round(t, 1), "duree_arret_min": d_arret,
                        "depart_offset_min": round(t, 1) + d_arret})
    return res

def preparer_roulement_manuel(roulement):
    """Prépare les roulements manuels pour la simulation."""
    res = {}
    for tid, etapes in roulement.items():
        res[tid] = []
        for e in etapes:
            try:
                d = datetime.combine(datetime.today(), datetime.strptime(e["heure_depart"], "%H:%M").time())
                a = datetime.combine(datetime.today(), datetime.strptime(e["heure_arrivee"], "%H:%M").time())
                if a < d: 
                    a += timedelta(days=1)
                res[tid].append({"start": d, "end": a, "origine": e["depart"], "terminus": e["arrivee"]})
            except: 
                pass
    return res

def importer_roulements_fichier(uploaded_file, dataframe_gares):
    """Importe les roulements depuis un fichier Excel."""
    try:
        df = pd.read_excel(uploaded_file)
        required_cols = ['Train', 'Début', 'Fin', 'Origine', 'Terminus']
        if not all(col in df.columns for col in required_cols):
            return None, f"Colonnes manquantes. Attendu: {required_cols}"
        
        chronologie = {}
        for train_id, group in df.groupby('Train'):
            trajets = []
            for _, row in group.iterrows():
                debut = pd.to_datetime(row['Début'])
                fin = pd.to_datetime(row['Fin'])
                trajets.append({
                    'depart': row['Origine'], 
                    'heure_depart': debut.strftime("%H:%M"),
                    'arrivee': row['Terminus'], 
                    'heure_arrivee': fin.strftime("%H:%M"),
                    'temps_trajet': int((fin - debut).total_seconds() / 60)
                })
            chronologie[train_id] = trajets
        return chronologie, None
    except Exception as e:
        return None, str(e)

def analyser_frequences_manuelles(roulement_manuel, missions, heure_debut_service, heure_fin_service):
    """Analyse les fréquences des roulements manuels."""
    comptes = {}
    for etapes in roulement_manuel.values():
        for e in etapes:
            cle = f"{e['depart']} → {e['arrivee']}"
            try:
                h = datetime.strptime(e['heure_depart'], "%H:%M").hour
                comptes.setdefault(cle, {}).setdefault(h, 0)
                comptes[cle][h] += 1
            except: 
                continue

    resultats = {}
    heures = []
    curr = datetime.combine(datetime.today(), heure_debut_service)
    end = datetime.combine(datetime.today(), heure_fin_service)
    if end <= curr: 
        end += timedelta(days=1)
    while curr < end:
        heures.append(curr.hour)
        curr += timedelta(hours=1)

    for m in missions:
        if m.get('frequence', 0) <= 0: 
            continue
        cle = f"{m['origine']} → {m['terminus']}"
        donnees = []
        respectees = 0
        for h in heures:
            reel = comptes.get(cle, {}).get(h, 0)
            statut = "✓" if reel >= m['frequence'] else "❌"
            if statut == "✓": 
                respectees += 1
            donnees.append({
                "Heure": f"{h:02d}:00", 
                "Trains": reel, 
                "Objectif": f"≥ {m['frequence']}", 
                "Statut": statut
            })
        if donnees:
            resultats[cle] = {
                "df": pd.DataFrame(donnees), 
                "conformite": (respectees / len(heures)) * 100
            }
    return resultats

_PDF_DPI = 300

def _add_logo_to_figure(fig, logo_img):
    """Insère le logo en coin inférieur gauche via figimage (intégration directe dans le PDF)."""
    import numpy as np
    logo_h, logo_w = logo_img.shape[:2]
    target_w = max(1, int(fig.get_figwidth() * _PDF_DPI * 0.12))
    target_h = max(1, int(target_w * logo_h / logo_w))
    ri = np.clip((np.arange(target_h) * logo_h / target_h + 0.5).astype(int), 0, logo_h - 1)
    ci = np.clip((np.arange(target_w) * logo_w / target_w + 0.5).astype(int), 0, logo_w - 1)
    logo_resampled = logo_img[np.ix_(ri, ci)]
    return fig.figimage(logo_resampled, xo=10, yo=10, zorder=10)


def generer_exports(chronologie, figure, figures_batterie=None, logo_path=None):
    """Génère les fichiers d'export Excel et PDF (multi-pages si graphes batterie)."""
    rows = []
    for tid in sorted(chronologie.keys()):
        for t in sorted(chronologie[tid], key=lambda x: x['start']):
            rows.append({
                "Train": tid,
                "Début": t["start"].strftime('%Y-%m-%d %H:%M:%S'),
                "Fin": t["end"].strftime('%Y-%m-%d %H:%M:%S'),
                "Origine": t["origine"],
                "Terminus": t["terminus"]
            })
    df = pd.DataFrame(rows)

    bx = BytesIO()
    with pd.ExcelWriter(bx, engine='xlsxwriter') as wr:
        df.to_excel(wr, index=False, sheet_name="Tableau de Marche")
    bx.seek(0)

    all_figures = [f for f in ([figure] + list(figures_batterie or [])) if f is not None]

    logo_img = None
    if logo_path:
        try:
            logo_img = mpimg.imread(logo_path)
        except Exception:
            logo_img = None

    bp = BytesIO()
    if all_figures:
        with PdfPages(bp) as pdf:
            for fig in all_figures:
                logo_patch = None
                if logo_img is not None:
                    logo_patch = _add_logo_to_figure(fig, logo_img)
                pdf.savefig(fig, bbox_inches='tight', dpi=_PDF_DPI)
                if logo_patch is not None and logo_patch in fig.images:
                    fig.images.remove(logo_patch)
    bp.seek(0)

    return bx, bp

def reset_caches():
    horaire_mission.vider()
    from optimisation_logic import reset_eval_cache
    reset_eval_cache()

def _calculer_stats_homogeneite(chronologie):
    """Calcule les statistiques d'homogénéité PAR MISSION ET PAR SENS (aller/retour séparés)."""
    stats = {}
    missions_horaires = defaultdict(list)
    
    for train_id, trajets in chronologie.items():
        if not trajets: 
            continue
        trajets_tries = sorted(trajets, key=lambda x: x['start'])
        
        if any('mission' in t for t in trajets_tries):
            for t in trajets_tries:
                if t.get('is_mission_start', False):
                    missions_horaires[t['mission']].append(t['start'])
            continue
        
        current_start = trajets_tries[0]
        current_end = trajets_tries[0]
        
        for i in range(1, len(trajets_tries)):
            seg = trajets_tries[i]
            if (seg['origine'] == current_end['terminus']) and \
               ((seg['start'] - current_end['end']).total_seconds() / 60.0 < 20):
                current_end = seg
            else:
                mission_key = f"{current_start['origine']} → {current_end['terminus']}"
                missions_horaires[mission_key].append(current_start['start'])
                current_start = seg
                current_end = seg
        
        mission_key = f"{current_start['origine']} → {current_end['terminus']}"
        missions_horaires[mission_key].append(current_start['start'])

    for mission_key, horaires in missions_horaires.items():
        if len(horaires) < 2:
            stats[mission_key] = 1.0
            continue
        stats[mission_key] = calculer_indice_homogeneite(horaires)
    
    return stats
