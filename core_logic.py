# -*- coding: utf-8 -*-
"""
core_logic.py
=============

Version : 5.0 (Nettoyée et optimisée)
- Code mort supprimé
- Cache optimisé
- Fonctions non utilisées retirées
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
    """Récupère le type d'infrastructure pour une gare donnée."""
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
    """Détermine si un train peut s'arrêter ou croiser à un point donné."""
    return infra_code in ['VE', 'D', 'Terminus']

def calculer_indice_homogeneite(horaires):
    """Calcule l'homogénéité du cadencement via coefficient de Gini inversé."""
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
    """Classe gérant l'état de la simulation avec optimisations."""
    
    def __init__(self, df_gares, heure_debut, heure_fin):
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
        """Réinitialise l'état pour une nouvelle tentative de simulation."""
        self.committed_schedules = []
        self.fleet_availability = defaultdict(list)
        self.train_counter = 1
        self.last_crossing_extensions = []

    def _analyze_segments(self):
        """Détermine pour chaque inter-gare si c'est VU ou DV."""
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
        """Vérifie si un tronçon est libre sur une plage horaire."""
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
        """Construit un sillon valide avec gestion stratégique des croisements."""
        base_schedule = construire_horaire_mission(mission, direction, self.df_gares)
        if not base_schedule:
            return None, [], "Erreur itinéraire"
    
        steps = []
        for i, pt in enumerate(base_schedule):
            idx = self.gares_map.get(pt['gare'])
            duration = pt['time_offset_min'] - base_schedule[i-1]['time_offset_min'] if i > 0 else 0
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
        max_attempts = 50
        
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
            max_delay = timedelta(minutes=crossing_strategy.max_acceptable_delay if crossing_strategy else 60)
            
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
        mission_id = f"Mission_{req.get('m_idx', 0)}"
        strategy = crossing_strategies.get(mission_id)
        
        real_dep, path, err = engine.solve_mission_schedule(
            req['mission'], 
            req['ideal_dep'], 
            req['type'],
            crossing_strategy=strategy
        )

        if not path:
            failures.append({
                "time": req['ideal_dep'],
                "mission": f"M{req.get('m_idx', 0)+1} ({req['type']})",
                "reason": err or "Impossible de planifier",
                "is_infra_violation": False
            })
            continue

        type_materiel = req['mission'].get('type_materiel', 'diesel')
        inject_allowed = (req['type'] == 'aller') or \
                        (req['type'] == 'retour' and req['mission'].get('inject_from_terminus_2', False))
        
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

def generer_tous_trajets_optimises(missions, df_gares, heure_debut, heure_fin, 
                                   allow_sharing=True, optimization_config=None, 
                                   progress_callback=None, search_strategy='smart', 
                                   crossing_strategies=None):
    """Optimisation globale avec support des stratégies de croisement et recherche progressive."""
    
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
    
    if search_strategy == 'smart_progressive':
        # Stratégie SMART PROGRESSIVE : recherche par affinement progressif
        return _optimisation_smart_progressive(
            engine, missions, missions_avec_retour, aller_requests, 
            allow_sharing, crossing_strategies, progress_callback
        )
    
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
        
        if progress_callback and ((combo_idx + 1) % checkpoint == 0 or combo_idx == len(all_combinations) - 1):
            progress_callback(combo_idx + 1, len(all_combinations), best_score, best_rames, 0)
    
    final_stats = _calculer_stats_homogeneite(best_chronologie)
    return best_chronologie, best_warnings, final_stats


def _optimisation_smart_progressive(engine, missions, missions_avec_retour, aller_requests,
                                     allow_sharing, crossing_strategies, progress_callback):
    """
    Recherche progressive par affinement :
    1. Recherche grossière (pas de 10 min)
    2. Affinement autour des meilleures zones (pas de 5 min)
    3. Recherche fine dans la zone optimale (pas de 1 min)
    """
    
    # Phase 1 : Recherche grossière (pas de 10 min)
    step_sizes = [10, 5, 2, 1]  # Pas progressifs
    best_overall_score = float('inf')
    best_overall_chronologie = {}
    best_overall_warnings = {'infra_violations': [], 'other': []}
    best_overall_combo = None
    
    # Pour chaque taille de pas
    search_center = None
    search_radius = 30  # Rayon initial de recherche autour du meilleur
    
    total_work_estimate = sum(((60 // step) ** len(missions_avec_retour)) for step in step_sizes)
    work_done = 0
    
    for step_idx, step_size in enumerate(step_sizes):
        # Définir l'espace de recherche
        if search_center is None:
            # Première phase : recherche complète
            search_space = [list(range(0, 60, step_size))] * len(missions_avec_retour)
        else:
            # Phases suivantes : recherche autour du meilleur
            search_space = []
            for center_val in search_center:
                min_val = max(0, center_val - search_radius)
                max_val = min(59, center_val + search_radius)
                values = list(range(min_val, max_val + 1, step_size))
                if not values:
                    values = [center_val]
                search_space.append(values)
        
        all_combinations = list(itertools.product(*search_space))
        
        best_score_phase = float('inf')
        best_combo_phase = None
        best_chronologie_phase = {}
        best_warnings_phase = {'infra_violations': [], 'other': []}
        
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
            
            if score < best_score_phase:
                best_score_phase = score
                best_combo_phase = cadencements_combo
                best_chronologie_phase = chrono
                best_warnings_phase = {'infra_violations': [], 'other': []}
                for fail in fails:
                    msg = f"{fail['mission']} à {fail['time'].strftime('%H:%M')}: {fail['reason']}"
                    if fail.get('is_infra_violation', False): 
                        best_warnings_phase['infra_violations'].append(msg)
                    else: 
                        best_warnings_phase['other'].append(msg)
            
            work_done += 1
            if progress_callback and work_done % max(1, total_work_estimate // 20) == 0:
                progress_callback(work_done, total_work_estimate, best_overall_score, 
                                len(best_overall_chronologie), 0)
        
        # Mettre à jour le meilleur global si nécessaire
        if best_score_phase < best_overall_score:
            best_overall_score = best_score_phase
            best_overall_chronologie = best_chronologie_phase
            best_overall_warnings = best_warnings_phase
            best_overall_combo = best_combo_phase
        
        # Préparer la prochaine phase
        if best_combo_phase is not None and step_idx < len(step_sizes) - 1:
            search_center = best_combo_phase
            # Réduire le rayon à chaque étape
            search_radius = step_sizes[step_idx + 1] * 3
    
    # Callback final
    if progress_callback:
        progress_callback(total_work_estimate, total_work_estimate, 
                        best_overall_score, len(best_overall_chronologie), 0)
    
    final_stats = _calculer_stats_homogeneite(best_overall_chronologie)
    return best_overall_chronologie, best_overall_warnings, final_stats

# =============================================================================
# 5. FONCTIONS UTILITAIRES ET EXPORT
# =============================================================================

@lru_cache(maxsize=256)
def construire_horaire_mission_cached(mission_key, trajet_spec, df_gares_json):
    """Version cachée de construire_horaire_mission."""
    if df_gares_json is None: 
        return None
    try:
        df_gares_local = pd.read_json(BytesIO(df_gares_json.encode('utf-8')))
        mission_cfg = json.loads(mission_key)
        return construire_horaire_mission(mission_cfg, trajet_spec, df_gares_local)
    except:
        return None

def construire_horaire_mission(m, direction, df_gares):
    """Construit l'horaire complet d'une mission."""
    if df_gares is None or df_gares.empty: 
        return []
    
    pts = []
    if direction == 'aller':
        pts.append({"gare": m.get("origine"), "time_offset_min": 0, "duree_arret_min": 0})
        pts.extend(m.get("passing_points", []))
        pts.append({"gare": m.get("terminus"), "time_offset_min": m.get("temps_trajet", 60), "duree_arret_min": 0})
    else:
        dur = m.get("temps_trajet_retour", m.get("temps_trajet", 60))
        if m.get("trajet_asymetrique"):
            base_pp = m.get("passing_points_retour", [])
        else:
            base_pp = [{"gare": p["gare"], "time_offset_min": dur - p["time_offset_min"], 
                       "duree_arret_min": p.get("duree_arret_min", 0)} 
                      for p in m.get("passing_points", [])]
        pts.append({"gare": m.get("terminus"), "time_offset_min": 0, "duree_arret_min": 0})
        pts.extend(base_pp)
        pts.append({"gare": m.get("origine"), "time_offset_min": dur, "duree_arret_min": 0})

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
    """Calcule les statistiques d'homogénéité par mission."""
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
        
        # Heuristique pour mode manuel
        current_start = trajets_tries[0]
        current_end = trajets_tries[0]
        for i in range(1, len(trajets_tries)):
            seg = trajets_tries[i]
            if (seg['origine'] == current_end['terminus']) and \
               ((seg['start'] - current_end['end']).total_seconds() / 60.0 < 20):
                current_end = seg
            else:
                missions_horaires[f"{current_start['origine']} → {current_end['terminus']}"].append(current_start['start'])
                current_start = seg
                current_end = seg
        missions_horaires[f"{current_start['origine']} → {current_end['terminus']}"].append(current_start['start'])

    for mission_key, horaires in missions_horaires.items():
        if len(horaires) < 2:
            stats[mission_key] = 1.0
            continue
        stats[mission_key] = calculer_indice_homogeneite(horaires)
    
    return stats
