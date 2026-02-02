# -*- coding: utf-8 -*-
"""
core_logic.py
=============

Version : 4.2 (Corrections des violations d'infrastructure et scoring amélioré)

CORRECTIONS MAJEURES:
1. Violations d'infrastructure = UNIQUEMENT croisements sur voie unique
   - Deux trains qui se croisent ailleurs que sur VE, D ou Terminus
   - Ce n'est PAS une violation si un trajet ne peut pas être planifié
   
2. Scoring amélioré:
   - Pénalise les trajets non planifiés (échecs de planification)
   - Privilégie les solutions avec moins de rames
   - Maintient la pénalité sur les retards
   
3. Performance:
   - Suppression des prints inutiles pendant les calculs
   - Messages uniquement pour l'interface web
"""

from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO
from collections import defaultdict
import itertools
from functools import lru_cache
import json

from optimisation_logic import (
    OptimizationConfig,
    CrossingOptimization,
    optimiser_graphique_horaire)

# =============================================================================
# 1. UTILITAIRES
# =============================================================================

def _get_infra_at_gare(df_gares, gare_name):
    """Récupère le type d'infrastructure pour une gare donnée."""
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
    """Détermine si un train peut s'arrêter ou croiser à un point donné."""
    return infra_code in ['VE', 'D', 'Terminus']

def calculer_indice_homogeneite(horaires):
    """Calcule l'homogénéité du cadencement."""
    if len(horaires) < 2:
        return 1.0

    horaires_tries = sorted(horaires)
    intervalles = []
    for i in range(len(horaires_tries) - 1):
        diff = (horaires_tries[i+1] - horaires_tries[i]).total_seconds() / 60.0
        if diff > 0.1:
            intervalles.append(diff)

    if not intervalles or sum(intervalles) == 0:
        return 0.0

    n = len(intervalles)
    intervalles.sort()
    somme_ponderee = sum((i + 1) * val for i, val in enumerate(intervalles))
    somme_totale = sum(intervalles)

    gini = (2.0 * somme_ponderee) / (n * somme_totale) - (n + 1.0) / n
    return max(0.0, 1.0 - gini)

# =============================================================================
# 2. MOTEUR DE SIMULATION
# =============================================================================

class SimulationEngine:
    """Classe gérant l'état de la simulation."""
    
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

        all_double = True
        for i in range(seg_idx_min, seg_idx_max):
            if not self.segment_is_double.get(i, False):
                all_double = False
                break

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

    def solve_mission_schedule(self, mission, ideal_start_time, direction):
        """
        Construit un sillon valide pour une mission donnée.
        Accepte des retards pour trouver une solution.
        """
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

        i = 0
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
            max_delay = timedelta(hours=1)
            
            while True:
                idx_min = min(steps[i]['index'], steps[target_idx]['index'])
                idx_max = max(steps[i]['index'], steps[target_idx]['index'])

                t_enter = current_departure
                t_exit = current_departure + timedelta(minutes=travel_time_block)

                is_free, next_t = self.check_segment_availability(idx_min, idx_max, t_enter, t_exit)

                if is_free:
                    final_path[-1]['dep'] = current_departure
                    t_cursor = current_departure

                    for k in range(i + 1, target_idx + 1):
                        t_cursor += timedelta(minutes=steps[k]['run_time'])
                        st_k = steps[k]
                        final_path.append({
                            'gare': st_k['gare'],
                            'index': st_k['index'],
                            'arr': t_cursor,
                            'dep': t_cursor + timedelta(minutes=st_k['duree_arret'])
                        })
                        t_cursor += timedelta(minutes=st_k['duree_arret'])

                    current_time = t_cursor
                    i = target_idx
                    break
                else:
                    current_departure = next_t
                    if current_departure - ideal_start_time > max_delay:
                        return None, [], "Impasse infra (délai > 1 heure)"

        return final_path[0]['dep'], final_path, None

    def allocate_train_id(self, gare, target_time, type_materiel, mission_id, 
                         can_inject=True, allow_cross_mission_sharing=True):
        """
        Alloue une rame à un départ.
        
        COMPORTEMENT:
        - Partage TOUJOURS autorisé au sein d'une même mission
        - Partage entre missions différentes contrôlé par allow_cross_mission_sharing
        
        Args:
            gare: Gare de départ
            target_time: Heure de départ souhaitée
            type_materiel: Type de matériel requis
            mission_id: ID de la mission (ex: "Mission_0")
            can_inject: Si True, peut créer une nouvelle rame
            allow_cross_mission_sharing: Si True, permet le partage entre missions
            
        Returns:
            int or None: ID de la rame allouée
        """
        pool = self.fleet_availability[gare]
        pool.sort()

        # Recherche d'une rame compatible
        for i, (dispo_t, tid, mat_type, orig_mission) in enumerate(pool):
            # Vérification 1: Même type de matériel
            if mat_type != type_materiel:
                continue
            
            # Vérification 2: Disponible à temps
            if dispo_t > target_time + timedelta(minutes=2):
                continue
            
            # Vérification 3: Règles de partage
            same_mission = (orig_mission == mission_id)
            
            if same_mission:
                # ✅ TOUJOURS autorisé au sein de la même mission
                return pool.pop(i)[1]
            elif allow_cross_mission_sharing:
                # ✅ Autorisé entre missions si allow_cross_mission_sharing=True
                return pool.pop(i)[1]
            # else: ❌ Pas autorisé entre missions différentes

        # Pas de rame compatible trouvée : injection si autorisée
        if can_inject:
            tid = self.train_counter
            self.train_counter += 1
            return tid

        return None

    def register_arrival(self, tid, gare, arr_time, turnaround_min, type_materiel, mission_id):
        """
        Libère une rame dans une gare après son temps de retournement.
        
        Enregistre également l'ID de la mission d'origine pour les règles de partage.
        
        Args:
            tid: ID de la rame
            gare: Gare d'arrivée
            arr_time: Heure d'arrivée
            turnaround_min: Temps de retournement en minutes
            type_materiel: Type de matériel
            mission_id: ID de la mission (pour les règles de partage)
        """
        self.fleet_availability[gare].append(
            (arr_time + timedelta(minutes=turnaround_min), tid, type_materiel, mission_id)
        )

# =============================================================================
# 3. FONCTION D'ÉVALUATION (VERSION AMÉLIORÉE)
# =============================================================================

def evaluer_configuration(engine, requests, allow_cross_mission_sharing=True):
    """
    Évalue une configuration complète avec score multi-critères amélioré.
    
    CHANGEMENTS IMPORTANTS:
    1. Les violations d'infrastructure sont UNIQUEMENT les croisements sur voie unique
    2. Les échecs de planification (trajets non planifiés) sont pénalisés mais ne sont PAS des violations
    3. Le nombre de rames est plus fortement pénalisé pour privilégier les solutions économiques
    
    Score = (Rames × 2000) + (Échecs × 5000) + (Retard × 10) - (Homogénéité × 3000)
    
    Returns:
        tuple: (score, chronologie, failures, stats_homogeneite, total_delay, nb_rames)
    """
    engine.reset()
    total_delay_min = 0
    trajets_resultat = defaultdict(list)
    failures = []
    mission_station_times = defaultdict(lambda: defaultdict(list))
    
    # Compteurs pour violations RÉELLES d'infrastructure
    infra_violations = []

    sorted_reqs = sorted(requests, key=lambda x: x['ideal_dep'])

    for req in sorted_reqs:
        real_dep, path, err = engine.solve_mission_schedule(
            req['mission'], 
            req['ideal_dep'], 
            req['type']
        )

        if not path:
            # IMPORTANT: Ce n'est PAS une violation d'infrastructure
            # C'est un échec de planification (pénalisé dans le score)
            failures.append({
                "time": req['ideal_dep'],
                "mission": f"M{req.get('m_idx', 0)+1} ({req['type']})",
                "reason": err or "Impossible de planifier",
                "is_infra_violation": False  # Nouvelle propriété
            })
            continue

        type_materiel = req['mission'].get('type_materiel', 'diesel')
        mission_id = f"Mission_{req.get('m_idx', 0)}"
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
            # Pas de violation d'infrastructure, juste manque de matériel
            failures.append({
                "time": req['ideal_dep'],
                "mission": f"M{req.get('m_idx', 0)+1} ({req['type']})",
                "reason": "Pas de rame disponible",
                "is_infra_violation": False
            })
            continue

        tid = int(tid)
        engine.committed_schedules.append({'train_id': tid, 'path': path})
        
        # Calcul du retard (pénalise les départs retardés)
        delay = (real_dep - req['ideal_dep']).total_seconds() / 60
        total_delay_min += delay

        m_key = f"M{req.get('m_idx', 0)}_{req['type']}"
        for step in path:
            is_terminus = (step == path[-1])
            time_to_record = step['arr'] if is_terminus else step['dep']
            mission_station_times[m_key][step['gare']].append(time_to_record)

        # Déterminer le label de la mission pour les stats
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
                    # Les arrêts ne sont pas des débuts de mission
                    "mission": mission_label,
                    "is_mission_start": False
                })

        t_ret = req['mission'].get(
            'temps_retournement_B' if req['type'] == 'aller' else 'temps_retournement_A', 
            10
        )
        engine.register_arrival(tid, path[-1]['gare'], path[-1]['dep'], t_ret, type_materiel, mission_id)

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

    # NOUVEAU SCORING: Pénalités renforcées
    penalty_trains = engine.train_counter * 2000  # Augmenté de 1000 → 2000
    penalty_failures = len(failures) * 5000  # Augmenté de 1M → 5000 (plus réaliste)
    penalty_delay = total_delay_min * 10
    bonus_homogeneity = avg_homogeneity * 3000  # Réduit de 5000 → 3000

    score = penalty_trains + penalty_failures + penalty_delay - bonus_homogeneity

    return score, dict(trajets_resultat), failures, homogeneite_par_mission, total_delay_min, engine.train_counter

# =============================================================================
# 4. OPTIMISATION GLOBALE (VERSION SANS PRINT EXCESSIFS)
# =============================================================================

def generer_tous_trajets_optimises(missions, df_gares, heure_debut, heure_fin, 
                                   allow_sharing=True, optimization_config=None, 
                                   progress_callback=None, search_strategy='smart'):
    """
    Optimisation globale avec corrections des règles et affichage minimal.
    
    CHANGEMENTS:
    - Suppression des prints pendant les calculs
    - Messages uniquement via progress_callback pour l'interface
    - Violations redéfinies correctement
    
    Args:
        missions: Liste des missions
        heure_debut, heure_fin: Période de service
        df_gares: Infrastructure
        allow_sharing: Contrôle le partage ENTRE missions (partage intra-mission toujours autorisé)
        optimization_config: OptimizationConfig ou None (mode standard)
        progress_callback: Fonction de callback pour la progression
        search_strategy: 'smart' ou 'exhaustive'
    
    Returns:
        tuple: (chronologie, warnings, stats_homogeneite)
    """
    # SI une configuration d'optimisation avancée est fournie
    if optimization_config is not None:
        from optimisation_logic import optimiser_graphique_horaire
        
        chronologie, warnings, stats = optimiser_graphique_horaire(
            missions,
            df_gares,
            heure_debut,
            heure_fin,
            optimization_config,
            allow_sharing=allow_sharing,
            progress_callback=progress_callback
        )
        
        # Calculer les statistiques d'homogénéité
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
                aller_requests.append({
                    'ideal_dep': curr,
                    'mission': m, 
                    'type': 'aller', 
                    'm_idx': m_idx
                })
                curr += intervalle
    
    # Génération des cadencements possibles pour les retours
    missions_avec_retour = [m for m in missions if m.get('frequence', 0) > 0]
    
    if search_strategy == 'smart':
        cadencements_a_tester = list(range(0, 60, 5))
    else:
        cadencements_a_tester = list(range(60))
    
    # Génération de toutes les combinaisons
    all_combinations = list(itertools.product(cadencements_a_tester, 
                                             repeat=len(missions_avec_retour)))
    
    total_combinations = len(all_combinations)
    
    if total_combinations > 10000 and search_strategy == 'exhaustive' and total_combinations > 50000:
        search_strategy = 'smart'
        cadencements_a_tester = list(range(0, 60, 5))
        all_combinations = list(itertools.product(cadencements_a_tester, 
                                                 repeat=len(missions_avec_retour)))
        total_combinations = len(all_combinations)
    
    # Test de toutes les combinaisons
    best_score = float('inf')
    best_chronologie = {}
    best_stats = {}
    best_warnings = {'infra_violations': [], 'other': []}
    best_combination = None
    best_delay = 0
    best_rames = 0
    
    checkpoint = max(1, total_combinations // 20)
    
    for combo_idx, cadencements_combo in enumerate(all_combinations):
        # Construction des requêtes retour
        retour_requests = []
        
        for mission_idx, (m_idx, m) in enumerate([(i, m) for i, m in enumerate(missions) 
                                                   if m.get('frequence', 0) > 0]):
            target_arr_min = cadencements_combo[mission_idx]
            t_trajet_ret = m.get('temps_trajet_retour', m.get('temps_trajet', 60))
            dep_min = (target_arr_min - t_trajet_ret) % 60
            intervalle = timedelta(hours=1.0/m['frequence'])
            
            curr = engine.dt_debut.replace(minute=0, second=0, microsecond=0) + \
                   timedelta(minutes=dep_min) - timedelta(hours=1)

            while curr < engine.dt_debut:
                curr += intervalle

            while curr < engine.dt_fin:
                retour_requests.append({
                    'ideal_dep': curr,
                    'mission': m,
                    'type': 'retour',
                    'm_idx': m_idx
                })
                curr += intervalle
        
        # Évaluation
        all_requests = aller_requests + retour_requests
        score, chrono, fails, stats, delay, nb_rames = evaluer_configuration(
            engine, 
            all_requests,
            allow_cross_mission_sharing=allow_sharing
        )
        
        # Mise à jour si meilleur
        if score < best_score:
            best_score = score
            best_chronologie = chrono
            best_stats = stats
            best_combination = cadencements_combo
            best_delay = delay
            best_rames = nb_rames
            
            # Séparer les vraies violations des échecs de planification
            best_warnings = {'infra_violations': [], 'other': []}
            for fail in fails:
                msg = f"{fail['mission']} à {fail['time'].strftime('%H:%M')}: {fail['reason']}"
                if fail.get('is_infra_violation', False):
                    best_warnings['infra_violations'].append(msg)
                else:
                    best_warnings['other'].append(msg)
        
        # Progression (uniquement via callback)
        if progress_callback and ((combo_idx + 1) % checkpoint == 0 or combo_idx == total_combinations - 1):
            progress = (combo_idx + 1) / total_combinations
            progress_callback(combo_idx + 1, total_combinations, best_score, best_rames, best_delay)
    
    # Recalculer les stats avec le format standardisé (A -> B)
    final_stats = _calculer_stats_homogeneite(best_chronologie)
    return best_chronologie, best_warnings, final_stats

# =============================================================================
# 5. FONCTIONS UTILITAIRES
# =============================================================================

@lru_cache(maxsize=256)
def construire_horaire_mission_cached(mission_key, trajet_spec, df_gares_json):
    """
    Wrapper avec cache pour la construction d'horaire.
    """
    if df_gares_json is None: return None
    try:
        df_gares_local = pd.read_json(BytesIO(df_gares_json.encode('utf-8')))
        mission_cfg = json.loads(mission_key)
        return construire_horaire_mission(mission_cfg, trajet_spec, df_gares_local)
    except Exception as e:
        return None

def construire_horaire_mission(m, direction, df_gares):
    """
    Interpole les horaires de passage à toutes les gares intermédiaires.
    """
    if df_gares is None or df_gares.empty: return []
    pts = []

    # Construction des points pivots selon le sens
    if direction == 'aller':
        pts.append({"gare": m.get("origine"), "time_offset_min": 0, "duree_arret_min": 0})
        pts.extend(m.get("passing_points", []))
        pts.append({"gare": m.get("terminus"), "time_offset_min": m.get("temps_trajet", 60), "duree_arret_min": 0})
    else:
        dur = m.get("temps_trajet_retour", m.get("temps_trajet", 60))
        if m.get("trajet_asymetrique"):
            base_pp = m.get("passing_points_retour", [])
        else:
            base_pp = [{"gare": p["gare"], "time_offset_min": dur - p["time_offset_min"], "duree_arret_min": p.get("duree_arret_min", 0)} for p in m.get("passing_points", [])]
        pts.append({"gare": m.get("terminus"), "time_offset_min": 0, "duree_arret_min": 0})
        pts.extend(base_pp)
        pts.append({"gare": m.get("origine"), "time_offset_min": dur, "duree_arret_min": 0})

    pts.sort(key=lambda x: x['time_offset_min'])

    # Interpolation linéaire
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
        if s['gare'] not in gmap or e['gare'] not in gmap: continue
        i_s, d_s = gmap[s['gare']]
        i_e, d_e = gmap[e['gare']]
        seg = gs.iloc[min(i_s, i_e) : max(i_s, i_e)+1]
        if i_e < i_s: seg = seg.sort_index(ascending=False)

        for _, row in seg.iterrows():
            if res and res[-1]['gare'] == row['gare']: continue

            # Récupération durée arrêt
            d_arret = 0
            for op in unique:
                if op['gare'] == row['gare']:
                    d_arret = op.get('duree_arret_min', 0)
                    break

            # Calcul temps passage
            dist_p = abs(row['distance'] - d_s)
            tot_d = abs(d_e - d_s)
            ratio = dist_p / tot_d if tot_d > 0 else 0
            t = s['time_offset_min'] + ((e['time_offset_min'] - s['time_offset_min']) * ratio)

            res.append({"gare": row['gare'], "time_offset_min": round(t, 1), "duree_arret_min": d_arret})
    
    return res

def preparer_roulement_manuel(roulement):
    """Convertit le dictionnaire de roulement manuel en format standard."""
    res = {}
    for tid, etapes in roulement.items():
        res[tid] = []
        for e in etapes:
            try:
                d = datetime.combine(datetime.today(), datetime.strptime(e["heure_depart"], "%H:%M").time())
                a = datetime.combine(datetime.today(), datetime.strptime(e["heure_arrivee"], "%H:%M").time())
                if a < d:
                    a += timedelta(days=1)
                res[tid].append({
                    "start": d,
                    "end": a,
                    "origine": e["depart"],
                    "terminus": e["arrivee"]
                })
            except:
                pass
    return res

def importer_roulements_fichier(uploaded_file, dataframe_gares):
    """
    Import un roulement depuis un fichier Excel exporté.
    
    Format attendu:
    - Train: numéro du train
    - Début: datetime de début
    - Fin: datetime de fin  
    - Origine: gare d'origine
    - Terminus: gare de terminus
    
    Returns:
        tuple: (chronologie, error_message)
    """
    try:
        # Lire le fichier Excel
        df = pd.read_excel(uploaded_file)
        
        # Vérifier les colonnes
        required_cols = ['Train', 'Début', 'Fin', 'Origine', 'Terminus']
        if not all(col in df.columns for col in required_cols):
            return None, f"Colonnes manquantes. Attendu: {required_cols}"
        
        # Convertir en chronologie
        chronologie = {}
        
        for train_id, group in df.groupby('Train'):
            trajets = []
            
            for _, row in group.iterrows():
                debut = pd.to_datetime(row['Début'])
                fin = pd.to_datetime(row['Fin'])
                duree = int((fin - debut).total_seconds() / 60)
                
                trajet = {
                    'depart': row['Origine'],
                    'heure_depart': debut.strftime("%H:%M"),
                    'arrivee': row['Terminus'],
                    'heure_arrivee': fin.strftime("%H:%M"),
                    'temps_trajet': duree
                }
                trajets.append(trajet)
            
            chronologie[train_id] = trajets
        
        return chronologie, None
        
    except Exception as e:
        return None, str(e)

def analyser_frequences_manuelles(roulement_manuel, missions, heure_debut_service, heure_fin_service):
    """Analyse le respect des fréquences horaires."""
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
    """Génère les fichiers d'export."""
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
        worksheet = wr.sheets['Tableau de Marche']
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)
    bx.seek(0)

    bp = BytesIO()
    if figure:
        figure.savefig(bp, format="pdf", bbox_inches='tight')
    bp.seek(0)

    return bx, bp

def reset_caches():
    """Vide les caches."""
    construire_horaire_mission_cached.cache_clear()

def _calculer_stats_homogeneite(chronologie):
    """
    Calcule les statistiques d'homogénéité par mission (origine -> terminus).
    Gère les modes Automatique (tags) et Manuel (heuristique de coupure).
    """
    from collections import defaultdict
    
    stats = {}
    missions_horaires = defaultdict(list)
    
    for train_id, trajets in chronologie.items():
        if not trajets:
            continue
            
        trajets_tries = sorted(trajets, key=lambda x: x['start'])
        
        # 1. Mode Automatique / Optimisé (si les tags existent)
        if any('mission' in t for t in trajets_tries):
            for t in trajets_tries:
                if t.get('is_mission_start', False):
                    missions_horaires[t['mission']].append(t['start'])
            continue

        # 2. Mode Manuel / Heuristique (Reconstruction des missions)
        current_start = trajets_tries[0]
        current_end = trajets_tries[0]
        
        for i in range(1, len(trajets_tries)):
            seg = trajets_tries[i]
            
            # Critères de continuité : même gare de jonction et délai court (< 20 min)
            gap_minutes = (seg['start'] - current_end['end']).total_seconds() / 60.0
            is_connected = (seg['origine'] == current_end['terminus'])
            
            if is_connected and gap_minutes < 20:
                current_end = seg # On prolonge la mission
            else:
                # Rupture : Fin de la mission précédente
                key = f"{current_start['origine']} → {current_end['terminus']}"
                missions_horaires[key].append(current_start['start'])
                
                # Nouvelle mission
                current_start = seg
                current_end = seg
        
        # Enregistrer la dernière mission
        key = f"{current_start['origine']} → {current_end['terminus']}"
        missions_horaires[key].append(current_start['start'])
    
    # Calcul du Gini
    for mission_key, horaires in missions_horaires.items():
        if len(horaires) < 2:
            stats[mission_key] = 1.0
            continue
        
        horaires_tries = sorted(horaires)
        intervalles = []
        
        for i in range(len(horaires_tries) - 1):
            diff = (horaires_tries[i+1] - horaires_tries[i]).total_seconds() / 60.0
            if diff > 0.1: 
                intervalles.append(diff)
        
        if not intervalles or sum(intervalles) == 0:
            stats[mission_key] = 0.0
            continue
        
        n = len(intervalles)
        intervalles.sort()
        somme_ponderee = sum((i + 1) * val for i, val in enumerate(intervalles))
        somme_totale = sum(intervalles)
        
        gini = (2.0 * somme_ponderee) / (n * somme_totale) - (n + 1.0) / n
        stats[mission_key] = max(0.0, 1.0 - gini)
    
    return stats
