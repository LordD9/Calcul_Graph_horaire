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
from functools import lru_cache
import json

# =============================================================================
# 1. UTILITAIRES
# =============================================================================

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
    # Correction : le Gini doit être entre 0 et 1, on ne retourne que si positif
    # Un cadencement parfait a un Gini de 0 (tous les intervalles identiques)
    # Un cadencement imparfait a un Gini proche de 1
    # On inverse pour avoir 1 = parfait, 0 = imparfait
    return max(0.0, min(1.0, 1.0 - gini))

# =============================================================================
# 2. MOTEUR DE SIMULATION
# =============================================================================

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

        # Optimisation : vérifier si tous les segments sont en double voie
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
        base_schedule = construire_horaire_mission(mission, direction, self.df_gares)
        if not base_schedule:
            return None, [], "Erreur itinéraire"
    
        steps = []
        for i, pt in enumerate(base_schedule):
            idx = self.gares_map.get(pt['gare'])
            if i > 0:
                # Calcul du temps écoulé total entre les deux arrivées (selon saisie utilisateur)
                delta_total = pt['time_offset_min'] - base_schedule[i-1]['time_offset_min']
                
                # On soustrait le temps d'arrêt de la gare précédente pour obtenir le temps de roulage pur
                prev_arret = base_schedule[i-1].get('duree_arret_min', 0)
                
                # Le temps de marche ne peut pas être négatif (max 0)
                duration = max(0, delta_total - prev_arret)
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
        max_attempts = 500  # Augmenté de 50 à 500 pour plus de persistence
        
        while i < len(steps) - 1:
            # Trouver le prochain point de croisement
            target_idx = i + 1
            while target_idx < len(steps):
                if _is_crossing_point(steps[target_idx]['infra']) or target_idx == len(steps) - 1:
                    break
                target_idx += 1
    
            # Calculer temps de trajet total pour ce bloc
            travel_time_block = sum(steps[k]['run_time'] + steps[k]['duree_arret'] 
                                for k in range(i + 1, target_idx))
            travel_time_block += steps[target_idx]['run_time']
    
            current_departure = current_time
            max_delay = timedelta(minutes=crossing_strategy.max_acceptable_delay if crossing_strategy else 240)  # 4h par défaut
            
            # Extension d'arrêt planifiée pour croisement
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
    
                    # Ajouter tous les points du bloc
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

# =============================================================================
# 3. FONCTION D'ÉVALUATION
# =============================================================================

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
        
        # Accumulation des extensions d'arrêt
        if hasattr(engine, 'last_crossing_extensions'):
            for ext in engine.last_crossing_extensions:
                total_crossing_extensions += ext.get('extension_minutes', 0)

    # Calcul homogénéité
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
    score += total_crossing_extensions * 15

    return score, dict(trajets_resultat), failures, homogeneite_par_mission, total_delay_min, engine.train_counter

# =============================================================================
# 4. OPTIMISATION GLOBALE
# =============================================================================

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

def generer_tous_trajets_optimises(missions, df_gares, heure_debut, heure_fin, 
                                   allow_sharing=True, optimization_config=None, 
                                   progress_callback=None, search_strategy='smart', 
                                   crossing_strategies=None):
    """
    Fonction principale de génération de la grille horaire (Orchestrateur).

    Cette fonction pilote la création de l'ensemble des trajets pour toutes les missions demandées.
    Elle peut fonctionner en plusieurs modes :
    1.  **Délégation** : Si `optimization_config` est fourni, elle délègue à `optimisation_logic`.
    2.  **Simulation directe** : Sinon, elle exécute une simulation séquentielle (avec ou sans stratégies simples).

    Args:
        missions (list): Liste des configurations de missions.
        df_gares (pd.DataFrame): Données d'infrastructure.
        heure_debut (time): Début de service.
        heure_fin (time): Fin de service.
        allow_sharing (bool): Si True, permet le chaînage de missions différentes (inter-opérabilité).
        optimization_config (OptimizationConfig, optional): Configuration pour l'algo génétique/avancé.
        progress_callback (callable, optional): Fonction de rappel pour la barre de progression UI.
        search_strategy (str): Stratégie de recherche ('simple', 'smart', 'exhaustif', etc.).
        crossing_strategies (dict, optional): Stratégies spécifiques de croisement pré-calculées.

    Returns:
        tuple: (dict chronologie, dict warnings, dict stats)
            - chronologie : {train_id: [liste_trajets]}
            - warnings : {infra_violations: [], other: []}
            - stats : Statistiques de performance (homogénéité, etc.)
    """
    
    # Délégation vers optimisation_logic si config présente
    if optimization_config is not None:
        from optimisation_logic import optimiser_graphique_horaire
        chronologie, warnings, stats = optimiser_graphique_horaire(
            missions, df_gares, heure_debut, heure_fin,
            optimization_config, allow_sharing=allow_sharing, 
            progress_callback=progress_callback
        )
        stats_homogeneite = _calculer_stats_homogeneite(chronologie)
        return chronologie, warnings, stats_homogeneite
    
    engine = SimulationEngine(df_gares, heure_debut, heure_fin)
    
    # Génération des Allers
    aller_requests = []
    for m_idx, m in enumerate(missions):
        freq = m.get('frequence', 1)
        if freq <= 0: 
            continue
        
        intervalle = timedelta(hours=1.0/freq)
        refs = [int(x.strip()) for x in str(m.get('reference_minutes', '0')).split(',') 
                if x.strip().isdigit()] or [0]
        
        for r in refs:
            curr = engine.dt_debut.replace(minute=0, second=0, microsecond=0) + \
                   timedelta(minutes=r) - timedelta(hours=1)
            while curr < engine.dt_debut: 
                curr += intervalle
            while curr < engine.dt_fin:
                aller_requests.append({'ideal_dep': curr, 'mission': m, 'type': 'aller', 'm_idx': m_idx})
                curr += intervalle
    
    # Génération des Retours avec optimisation
    missions_avec_retour = [m for m in missions if m.get('frequence', 0) > 0]
    
    if search_strategy == 'simple':
        # MODE SIMPLE : Logique événementielle de l'ancienne version
        # Seuls les ALLERS sont programmés initialement, les RETOURS sont créés dynamiquement
        from collections import defaultdict
        import heapq
        
        infra_violation_warnings = []
        other_warnings = []
        chronologie_reelle = {}
        id_train_counter = 1
        event_counter = 0
        
        trains = {}  # {id: {"loc": gare, "dispo_a": datetime}}
        occupation_cantons = defaultdict(list)
        evenements = []  # heapq
        
        # NOUVEAU : Calculer la durée max des missions aller pour savoir combien de temps avant de commencer
        duree_mission_max_min = _calculer_duree_mission_max(missions, df_gares)
        
        # Générer les demandes de départ ALLER (y compris les allers fictifs avant le début du service)
        for m_idx, mission in enumerate(missions):
            if mission.get('frequence', 0) <= 0:
                continue
            
            mission_id = f"M{m_idx+1}"
            frequence = mission['frequence']
            intervalle = timedelta(hours=1.0 / frequence)
            
            # Minutes de référence
            minutes_ref_str = mission.get("reference_minutes", "0")
            try:
                minutes_ref = sorted(list(set([
                    int(m.strip()) for m in minutes_ref_str.split(',')
                    if m.strip().isdigit()
                ])))
                if not minutes_ref:
                    minutes_ref = [0]
            except:
                minutes_ref = [0]
            
            # NOUVEAU : Déterminer l'heure de début pour cette mission
            # Si injection depuis terminus 2 autorisée, commencer plus tôt pour générer des allers fictifs
            heure_debut_mission = engine.dt_debut
            
            if mission.get('inject_from_terminus_2', False):
                # Calculer combien de temps avant le début du service on doit commencer
                # pour avoir des trains disponibles au terminus 2 au début du service
                horaire_aller = construire_horaire_mission(mission, "aller", df_gares)
                if horaire_aller:
                    temps_trajet_aller = horaire_aller[-1].get("time_offset_min", 0)
                    temps_retournement_B = mission.get("temps_retournement_B", 10)
                    
                    # On commence (temps_aller + temps_retournement + 1h) avant le début du service
                    # Le +1h assure qu'on a au moins un cycle complet avant le début
                    temps_avant_service = temps_trajet_aller + temps_retournement_B + 60
                    heure_debut_mission = engine.dt_debut - timedelta(minutes=temps_avant_service)
            
            # Générer les événements de demande ALLER (y compris les allers fictifs si besoin)
            for minute_ref in minutes_ref:
                offset_hours = minute_ref // 60
                offset_minutes = minute_ref % 60
                
                curseur_temps = heure_debut_mission.replace(
                    minute=offset_minutes, second=0, microsecond=0
                ) + timedelta(hours=offset_hours)
                
                # Ajuster au premier départ dans ou après heure_debut_mission
                while curseur_temps < heure_debut_mission:
                    curseur_temps += intervalle
                
                # Générer les départs jusqu'à la fin de service
                while curseur_temps < engine.dt_fin:
                    event_counter += 1
                    is_fictif = curseur_temps < engine.dt_debut  # Marqueur pour les allers avant le début du service
                    heapq.heappush(evenements, (
                        curseur_temps, event_counter, "demande_depart_aller",
                        {"mission": mission, "mission_id": mission_id, "is_aller_fictif": is_fictif}
                    ))
                    curseur_temps += intervalle
        
        # Boucle événementielle principale
        while evenements:
            heure, _, type_event, details = heapq.heappop(evenements)
            
            if heure >= engine.dt_fin:
                continue
            
            # Gestion demande de départ ALLER
            if type_event == "demande_depart_aller":
                mission_cfg = details["mission"]
                mission_id = details["mission_id"]
                origine = mission_cfg["origine"]
                is_aller_fictif = details.get("is_aller_fictif", False)
                
                # Trouver train disponible à l'origine
                train_assigne_id = None
                earliest_dispo = datetime.max
                
                for id_t, t in trains.items():
                    if t.get("loc") == origine and t.get("dispo_a", datetime.max) <= heure:
                        if t["dispo_a"] < earliest_dispo:
                            earliest_dispo = t["dispo_a"]
                            train_assigne_id = id_t
                
                # Si pas de train, en créer un
                if train_assigne_id is None:
                    train_assigne_id = id_train_counter
                    trains[train_assigne_id] = {
                        "id": train_assigne_id, 
                        "loc": origine, 
                        "dispo_a": heure
                    }
                    chronologie_reelle[train_assigne_id] = []
                    id_train_counter += 1
                else:
                    trains[train_assigne_id]["dispo_a"] = max(heure, earliest_dispo)
                
                # Programmer la tentative de mouvement
                heure_programmation = max(heure, trains[train_assigne_id]["dispo_a"])
                event_counter += 1
                heapq.heappush(evenements, (heure_programmation, event_counter, "tentative_mouvement", {
                    "id_train": train_assigne_id,
                    "mission": mission_cfg,
                    "mission_id": mission_id,
                    "trajet_spec": "aller",
                    "index_etape": 0,
                    "retry_count": 0,
                    "is_trajet_fictif": is_aller_fictif  # Propagation du flag
                }))
            
            # Gestion tentative de mouvement (LOGIQUE BLOC PHYSIQUE)
            elif type_event == "tentative_mouvement":
                id_train = details["id_train"]
                mission_cfg = details["mission"]
                mission_id = details["mission_id"]
                trajet_spec = details["trajet_spec"]
                index_etape = details["index_etape"]
                is_trajet_fictif = details.get("is_trajet_fictif", False)
                
                # Construire l'horaire complet de la mission
                horaire = construire_horaire_mission(mission_cfg, trajet_spec, df_gares)
                
                if not horaire or index_etape >= len(horaire) - 1:
                    continue
                
                # NOUVEAU : Identifier le bloc complet (jusqu'au prochain point de croisement)
                bloc_gares = [horaire[index_etape]]
                next_crossing_idx = index_etape + 1
                
                # Chercher le prochain point de croisement valide
                while next_crossing_idx < len(horaire):
                    gare_name = horaire[next_crossing_idx]["gare"]
                    infra = _get_infra_at_gare(df_gares, gare_name)
                    bloc_gares.append(horaire[next_crossing_idx])
                    
                    # Arrêter au prochain point de croisement ou à la fin
                    if _is_crossing_point(infra) or next_crossing_idx == len(horaire) - 1:
                        break
                    next_crossing_idx += 1
                
                # Calculer le temps total du bloc
                pt_depart_bloc = bloc_gares[0]
                pt_arrivee_bloc = bloc_gares[-1]
                
                duree_bloc_min = max(0, 
                    pt_arrivee_bloc.get("time_offset_min", 0) - 
                    pt_depart_bloc.get("time_offset_min", 0)
                )
                
                # Ajouter le temps d'arrêt à destination si c'est un point de croisement
                duree_arret_final = pt_arrivee_bloc.get("duree_arret_min", 0)
                
                gare_dep_bloc = pt_depart_bloc.get("gare")
                gare_arr_bloc = pt_arrivee_bloc.get("gare")
                
                # Déterminer l'heure de départ effective
                dispo_train = trains.get(id_train, {}).get("dispo_a", heure)
                heure_depart_reelle = max(heure, dispo_train)
                
                if heure_depart_reelle >= engine.dt_fin:
                    continue
                
                # VÉRIFICATION CRUCIALE : Occupation du BLOC COMPLET
                conflit = False
                fin_conflit = None
                
                if duree_bloc_min > 0 and gare_dep_bloc != gare_arr_bloc:
                    idx_dep = engine.gares_map.get(gare_dep_bloc)
                    idx_arr = engine.gares_map.get(gare_arr_bloc)
                    
                    if idx_dep is not None and idx_arr is not None:
                        idx_min = min(idx_dep, idx_arr)
                        idx_max = max(idx_dep, idx_arr)
                        
                        # Temps d'entrée et sortie du bloc
                        t_enter = heure_depart_reelle
                        # IMPORTANT : Inclure le temps d'arrêt à destination dans l'occupation
                        t_exit = heure_depart_reelle + timedelta(minutes=duree_bloc_min + duree_arret_final)
                        
                        # Vérifier disponibilité du bloc complet
                        is_free, next_t = engine.check_segment_availability(idx_min, idx_max, t_enter, t_exit)
                        
                        if not is_free:
                            conflit = True
                            fin_conflit = next_t
                            # Note: L'avertissement sera généré seulement si échec de résolution (après 500 tentatives)
                
                if not conflit:
                    # Mouvement du bloc complet réussi
                    heure_arrivee_finale = heure_depart_reelle + timedelta(minutes=duree_bloc_min)
                    
                    if heure_arrivee_finale > engine.dt_fin:
                        continue
                    
                    # NOUVEAU : Ne tracer que si le trajet n'est pas fictif
                    # Un trajet est fictif si c'est un aller avant le début du service
                    if not is_trajet_fictif:
                        # Enregistrer TOUS les segments du bloc dans chronologie
                        for i in range(len(bloc_gares) - 1):
                            pt_curr = bloc_gares[i]
                            pt_next = bloc_gares[i + 1]
                            
                            delta_total = pt_next.get("time_offset_min", 0) - pt_curr.get("time_offset_min", 0)
                            prev_arret = pt_curr.get("duree_arret_min", 0)
                            duree_segment = max(0, delta_total - prev_arret) 
                            
                            if duree_segment > 0:
                                offset_dep = pt_curr.get("time_offset_min", 0) - pt_depart_bloc.get("time_offset_min", 0)
                                offset_arr = pt_next.get("time_offset_min", 0) - pt_depart_bloc.get("time_offset_min", 0)
                                
                                h_dep_segment = heure_depart_reelle + timedelta(minutes=offset_dep)
                                h_arr_segment = heure_depart_reelle + timedelta(minutes=offset_arr)
                                
                                chronologie_reelle.setdefault(id_train, []).append({
                                    "start": h_dep_segment,
                                    "end": h_arr_segment,
                                    "origine": pt_curr["gare"],
                                    "terminus": pt_next["gare"]
                                })
                    
                    # Enregistrer l'occupation du BLOC COMPLET dans engine
                    idx_dep = engine.gares_map[gare_dep_bloc]
                    idx_arr = engine.gares_map[gare_arr_bloc]
                    
                    path_bloc = [{
                        'gare': gare_dep_bloc, 
                        'index': idx_dep,
                        'arr': heure_depart_reelle, 
                        'dep': heure_depart_reelle
                    }, {
                        'gare': gare_arr_bloc, 
                        'index': idx_arr,
                        'arr': heure_arrivee_finale, 
                        'dep': heure_arrivee_finale + timedelta(minutes=duree_arret_final)
                    }]
                    engine.committed_schedules.append({'train_id': id_train, 'path': path_bloc})
                    
                    # Mettre à jour l'état du train
                    trains[id_train]["loc"] = gare_arr_bloc
                    # IMPORTANT : Le train est dispo APRÈS le temps d'arrêt
                    trains[id_train]["dispo_a"] = heure_arrivee_finale + timedelta(minutes=duree_arret_final)
                    
                    # Programmer la suite de la mission si nécessaire
                    if next_crossing_idx < len(horaire) - 1:
                        event_counter += 1
                        heapq.heappush(evenements, (
                            trains[id_train]["dispo_a"], 
                            event_counter, 
                            "tentative_mouvement", 
                            {
                                "id_train": id_train,
                                "mission": mission_cfg,
                                "mission_id": mission_id,
                                "trajet_spec": trajet_spec,
                                "index_etape": next_crossing_idx,
                                "retry_count": 0,
                                "is_trajet_fictif": is_trajet_fictif
                            }
                        ))
                    else:
                        # Fin de la mission
                        event_counter += 1
                        heapq.heappush(evenements, (
                            trains[id_train]["dispo_a"], 
                            event_counter, 
                            "fin_mission", 
                            {
                                "id_train": id_train,
                                "mission": mission_cfg,
                                "mission_id": mission_id,
                                "trajet_spec": trajet_spec,
                                "gare_finale": gare_arr_bloc,
                                "is_trajet_fictif": is_trajet_fictif
                            }
                        ))
                else:
                    # Conflit détecté -> Reprogrammer
                    retry_count = details.get("retry_count", 0)
                    if retry_count < 500:  # Limite de retry
                        new_details = details.copy()
                        new_details["retry_count"] = retry_count + 1
                        event_counter += 1
                        heapq.heappush(evenements, (
                            fin_conflit, 
                            event_counter, 
                            "tentative_mouvement", 
                            new_details
                        ))
                    else:
                        # Échec après trop de tentatives - VRAIE VIOLATION
                        # Identifier les gares problématiques dans le bloc
                        gares_sans_ve = []
                        for gare_info in bloc_gares[1:-1]:  # Gares intermédiaires
                            gare_inter = gare_info["gare"]
                            infra_inter = _get_infra_at_gare(df_gares, gare_inter)
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
                            "is_infra_violation": True
                        })
            
            # Gestion fin de mission
            elif type_event == "fin_mission":
                id_train = details["id_train"]
                mission_cfg = details["mission"]
                mission_id = details["mission_id"]
                gare_finale = details["gare_finale"]
                is_trajet_fictif = details.get("is_trajet_fictif", False)
                
                if id_train in trains:
                    trains[id_train]["loc"] = gare_finale
                    heure_arrivee_mission = heure
                else:
                    continue
                
                if details["trajet_spec"] == "aller":
                    # Fin aller -> Programmer retour
                    temps_retournement = mission_cfg.get("temps_retournement_B", 10)
                    heure_dispo_pour_retour = heure_arrivee_mission + timedelta(minutes=temps_retournement)
                    trains[id_train]["dispo_a"] = heure_dispo_pour_retour
                    
                    # Calculer durée retour pour vérifier si faisable
                    horaire_retour = construire_horaire_mission(mission_cfg, "retour", df_gares)
                    if horaire_retour and len(horaire_retour) > 1:
                        temps_trajet_retour = horaire_retour[-1].get("time_offset_min", 0)
                        heure_fin_retour_estimee = heure_dispo_pour_retour + timedelta(minutes=temps_trajet_retour)
                        
                        if heure_fin_retour_estimee <= engine.dt_fin:
                            # NOUVEAU : Le retour n'est plus fictif si on est après le début du service
                            is_retour_fictif = heure_dispo_pour_retour < engine.dt_debut
                            
                            event_counter += 1
                            heapq.heappush(evenements, (heure_dispo_pour_retour, event_counter, "tentative_mouvement", {
                                "id_train": id_train,
                                "mission": mission_cfg,
                                "mission_id": mission_id,
                                "trajet_spec": "retour",
                                "index_etape": 0,
                                "retry_count": 0,
                                "is_trajet_fictif": is_retour_fictif
                            }))
                else:
                    # Fin retour -> Train dispo pour nouvel aller
                    temps_retournement = mission_cfg.get("temps_retournement_A", 10)
                    heure_dispo_finale = heure_arrivee_mission + timedelta(minutes=temps_retournement)
                    trains[id_train]["dispo_a"] = heure_dispo_finale
        
        # Nettoyage - supprimer les trains sans trajets
        trains_a_supprimer = [tid for tid, trajets in chronologie_reelle.items() if not trajets]
        for tid in trains_a_supprimer:
            del chronologie_reelle[tid]
        
        warnings = {
            "infra_violations": infra_violation_warnings,
            "other": other_warnings
        }
        stats_homogeneite = _calculer_stats_homogeneite(chronologie_reelle)
        
        if progress_callback:
            progress_callback(1, 1, 0, len(chronologie_reelle), 0)
        
        return chronologie_reelle, warnings, stats_homogeneite
    
    if search_strategy == 'smart_progressive':
        # Stratégie SMART PROGRESSIVE : recherche par affinement progressif
        # Cette fonction doit être définie dans optimisation_logic.py
        try:
            from optimisation_logic import _optimisation_smart_progressive
            return _optimisation_smart_progressive(
                engine, missions, missions_avec_retour, aller_requests, 
                allow_sharing, crossing_strategies, progress_callback
            )
        except ImportError:
            # Si la fonction n'existe pas, utiliser la stratégie smart normale
            search_strategy = 'smart'
    
    # Stratégies classiques
    if search_strategy == 'smart':
        cadencements_a_tester = list(range(0, 60, 5))  # Test tous les 5 minutes
    elif search_strategy == 'fast':
        cadencements_a_tester = list(range(0, 60, 10))  # Test tous les 10 minutes
    else:  # exhaustive
        cadencements_a_tester = list(range(60))
    
    all_combinations = list(itertools.product(cadencements_a_tester, repeat=len(missions_avec_retour)))
    
    # Limiter le nombre de combinaisons pour éviter une explosion
    if len(all_combinations) > 10000 and search_strategy != 'exhaustive':
        all_combinations = list(itertools.product(list(range(0, 60, 10)), repeat=len(missions_avec_retour)))
    
    best_score = float('inf')
    best_chronologie = {}
    best_warnings = {'infra_violations': [], 'other': []}
    best_rames = 0
    best_combo = None
    checkpoint = max(1, len(all_combinations) // 20)
    
    for combo_idx, cadencements_combo in enumerate(all_combinations):
        retour_requests = []
        for mission_idx, (m_idx, m) in enumerate([(i, m) for i, m in enumerate(missions) if m.get('frequence', 0) > 0]):
            target_arr_min = cadencements_combo[mission_idx]
            t_trajet_ret = m.get('temps_trajet_retour', m.get('temps_trajet', 60))
            dep_min = (target_arr_min - t_trajet_ret) % 60
            intervalle = timedelta(hours=1.0/m['frequence'])
            
            curr = engine.dt_debut.replace(minute=0, second=0, microsecond=0) + \
                   timedelta(minutes=dep_min) - timedelta(hours=1)
            while curr < engine.dt_debut: 
                curr += intervalle
            while curr < engine.dt_fin:
                retour_requests.append({'ideal_dep': curr, 'mission': m, 'type': 'retour', 'm_idx': m_idx})
                curr += intervalle
        
        score, chrono, fails, stats, delay, nb_rames = evaluer_configuration(
            engine, aller_requests + retour_requests,
            allow_cross_mission_sharing=allow_sharing,
            crossing_strategies=crossing_strategies
        )
        
        if score < best_score:
            best_score = score
            best_chronologie = chrono
            best_rames = nb_rames
            best_combo = cadencements_combo
            best_warnings = {'infra_violations': [], 'other': []}
            for fail in fails:
                msg = f"{fail['mission']} à {fail['time'].strftime('%H:%M')}: {fail['reason']}"
                if fail.get('is_infra_violation', False):
                    best_warnings['infra_violations'].append(msg)
                else:
                    best_warnings['other'].append(msg)
        
        if (combo_idx + 1) % checkpoint == 0 and progress_callback:
            progress_callback(combo_idx + 1, len(all_combinations), 0, best_rames, 0)
    
    if progress_callback:
        progress_callback(len(all_combinations), len(all_combinations), 0, best_rames, 0)
    
    stats_homogeneite = _calculer_stats_homogeneite(best_chronologie)
    return best_chronologie, best_warnings, stats_homogeneite

# Note: _optimisation_smart_progressive est défini dans optimisation_logic.py si nécessaire

# =============================================================================
# 5. CONSTRUCTION DES HORAIRES DE MISSION
# =============================================================================

@lru_cache(maxsize=128)
def construire_horaire_mission(mission_tuple, direction, df_gares_tuple):
    """Version cachée de la construction d'horaire (pour performances)."""
    mission = json.loads(mission_tuple)
    df_gares = pd.DataFrame(json.loads(df_gares_tuple))
    return _construire_horaire_mission_impl(mission, direction, df_gares)

def construire_horaire_mission_cached(mission, direction, df_gares):
    """Wrapper pour cacher les appels répétés."""
    mission_json = json.dumps(mission, sort_keys=True)
    df_json = df_gares.to_json(orient='records')
    return construire_horaire_mission((mission_json, direction, df_json))

def _construire_horaire_mission_impl(mission, direction, df_gares):
    """Implémentation réelle de la construction d'horaire."""
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
                        "time_offset_min": t_trajet - p["time_offset_min"],
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
    
    # Supprimer les doublons
    unique = []
    seen = set()
    for p in pts:
        if p['gare'] not in seen:
            unique.append(p)
            seen.add(p['gare'])

    res = []
    gs = df_gares.sort_values('distance').reset_index(drop=True)
    gmap = {r.gare: (i, r.distance) for i, r in gs.iterrows()}

    for i in range(len(unique) - 1):
        s, e = unique[i], unique[i+1]
        if s['gare'] not in gmap or e['gare'] not in gmap: 
            continue
        i_s, d_s = gmap[s['gare']]
        i_e, d_e = gmap[e['gare']]
        seg = gs.iloc[min(i_s, i_e) : max(i_s, i_e)+1]
        if i_e < i_s: 
            seg = seg.sort_index(ascending=False)

        for _, row in seg.iterrows():
            if res and res[-1]['gare'] == row['gare']: 
                continue
            d_arret = 0
            for op in unique:
                if op['gare'] == row['gare']:
                    d_arret = op.get('duree_arret_min', 0)
                    break
            dist_p = abs(row['distance'] - d_s)
            tot_d = abs(d_e - d_s)
            ratio = dist_p / tot_d if tot_d > 0 else 0
            t = s['time_offset_min'] + ((e['time_offset_min'] - s['time_offset_min']) * ratio)
            res.append({"gare": row['gare'], "time_offset_min": round(t, 1), "duree_arret_min": d_arret})
    return res

# Wrapper pour compatibilité
def construire_horaire_mission(mission, direction, df_gares):
    """Construction d'horaire de mission (sans cache)."""
    return _construire_horaire_mission_impl(mission, direction, df_gares)

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

def generer_exports(chronologie, figure):
    """Génère les fichiers d'export Excel et PDF."""
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
    
    bp = BytesIO()
    if figure: 
        figure.savefig(bp, format="pdf", bbox_inches='tight')
    bp.seek(0)
    
    return bx, bp

def reset_caches():
    """Réinitialise les caches."""
    construire_horaire_mission_cached.cache_clear()

def _calculer_stats_homogeneite(chronologie):
    """Calcule les statistiques d'homogénéité PAR MISSION ET PAR SENS (aller/retour séparés)."""
    stats = {}
    missions_horaires = defaultdict(list)
    
    for train_id, trajets in chronologie.items():
        if not trajets: 
            continue
        trajets_tries = sorted(trajets, key=lambda x: x['start'])
        
        # Si les trajets ont déjà l'info mission (mode optimisé)
        if any('mission' in t for t in trajets_tries):
            for t in trajets_tries:
                if t.get('is_mission_start', False):
                    # La clé mission contient déjà le sens (ex: "A → B" ou "B → A")
                    missions_horaires[t['mission']].append(t['start'])
            continue
        
        # Mode manuel : reconstruire les missions EN DISTINGUANT LE SENS
        current_start = trajets_tries[0]
        current_end = trajets_tries[0]
        
        for i in range(1, len(trajets_tries)):
            seg = trajets_tries[i]
            # Vérifier si c'est la continuation de la mission actuelle
            # (même origine/terminus ET temps de connexion < 20min)
            if (seg['origine'] == current_end['terminus']) and \
               ((seg['start'] - current_end['end']).total_seconds() / 60.0 < 20):
                current_end = seg
            else:
                # Fin de mission - enregistrer avec le SENS EXPLICITE
                # Format: "Gare A → Gare B" (le sens est dans la flèche →)
                mission_key = f"{current_start['origine']} → {current_end['terminus']}"
                missions_horaires[mission_key].append(current_start['start'])
                current_start = seg
                current_end = seg
        
        # Ne pas oublier la dernière mission
        mission_key = f"{current_start['origine']} → {current_end['terminus']}"
        missions_horaires[mission_key].append(current_start['start'])

    # Calculer le coefficient de Gini pour chaque mission/sens
    # Chaque clé "A → B" aura son propre coefficient, distinct de "B → A"
    for mission_key, horaires in missions_horaires.items():
        if len(horaires) < 2:
            stats[mission_key] = 1.0
            continue
        stats[mission_key] = calculer_indice_homogeneite(horaires)
    
    return stats
