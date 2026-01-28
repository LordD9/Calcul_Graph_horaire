# -*- coding: utf-8 -*-
"""
core_logic.py
=============

Version : 4.1 (Corrections Partage Intra-Mission + Flexibilité Allers)

CORRECTIONS MAJEURES:
1. allow_sharing redéfini:
   - True: Partage autorisé entre missions ET au sein d'une mission
   - False: Partage UNIQUEMENT au sein de la même mission (pas entre missions)
   
2. Flexibilité des Allers:
   - Les horaires de départ peuvent être retardés pour éviter des conflits
   - L'objectif reste de respecter l'horaire théorique, mais des compromis sont possibles
   - Les retards sont pénalisés dans le score (×10 par minute)
"""

from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO
from collections import defaultdict
import itertools
from functools import lru_cache

# =============================================================================
# 1. UTILITAIRES (inchangés)
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
        NOUVEAU: Accepte des retards pour trouver une solution.
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
        
        NOUVEAU COMPORTEMENT:
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
        
        NOUVEAU: Enregistre également l'ID de la mission d'origine pour les règles de partage.
        
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
# 3. FONCTION D'ÉVALUATION
# =============================================================================

def evaluer_configuration(engine, requests, allow_cross_mission_sharing=True):
    """
    Évalue une configuration complète avec score multi-critères.
    
    NOUVEAU PARAMÈTRE:
    - allow_cross_mission_sharing: Si False, les rames ne peuvent être partagées
      QU'AU SEIN de leur mission d'origine
    
    Score = (Rames × 1000) + (Échecs × 1M) + (Retard × 10) - (Homogénéité × 5000)
    """
    engine.reset()
    total_delay_min = 0
    trajets_resultat = defaultdict(list)
    failures = []
    mission_station_times = defaultdict(lambda: defaultdict(list))

    sorted_reqs = sorted(requests, key=lambda x: x['ideal_dep'])

    for req in sorted_reqs:
        real_dep, path, err = engine.solve_mission_schedule(
            req['mission'], 
            req['ideal_dep'], 
            req['type']
        )

        if not path:
            failures.append({
                "time": req['ideal_dep'],
                "mission": f"M{req.get('m_idx', 0)+1} ({req['type']})",
                "reason": err or "Infrastructure saturée"
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
            failures.append({
                "time": req['ideal_dep'],
                "mission": f"M{req.get('m_idx', 0)+1} ({req['type']})",
                "reason": "Pas de rame disponible"
            })
            continue

        tid = int(tid)
        engine.committed_schedules.append({'train_id': tid, 'path': path})
        
        # Calcul du retard (IMPORTANT: pénalise les départs retardés)
        delay = (real_dep - req['ideal_dep']).total_seconds() / 60
        total_delay_min += delay

        m_key = f"M{req.get('m_idx', 0)}_{req['type']}"
        for step in path:
            is_terminus = (step == path[-1])
            time_to_record = step['arr'] if is_terminus else step['dep']
            mission_station_times[m_key][step['gare']].append(time_to_record)

        for k in range(len(path)-1):
            p_curr, p_next = path[k], path[k+1]
            trajets_resultat[tid].append({
                "start": p_curr['dep'],
                "end": p_next['arr'],
                "origine": p_curr['gare'],
                "terminus": p_next['gare']
            })
            if p_next['dep'] > p_next['arr']:
                trajets_resultat[tid].append({
                    "start": p_next['arr'],
                    "end": p_next['dep'],
                    "origine": p_next['gare'],
                    "terminus": p_next['gare']
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

    # Score multi-critères
    penalty_trains = engine.train_counter * 1000
    penalty_failures = len(failures) * 1000000
    penalty_delay = total_delay_min * 10  # Pénalise les retards
    bonus_homogeneity = avg_homogeneity * 5000

    score = penalty_trains + penalty_failures + penalty_delay - bonus_homogeneity

    return score, dict(trajets_resultat), failures, homogeneite_par_mission, total_delay_min, engine.train_counter

# =============================================================================
# 4. OPTIMISATION GLOBALE
# =============================================================================

def generer_tous_trajets_optimises(missions, heure_debut, heure_fin, df_gares, 
                                   allow_sharing=True, search_strategy='smart'):
    """
    Optimisation vraiment globale avec correction des règles de partage.
    
    NOUVEAU COMPORTEMENT DE allow_sharing:
    - True: Partage autorisé ENTRE missions et AU SEIN des missions
    - False: Partage UNIQUEMENT AU SEIN d'une même mission (pas entre missions)
    
    FLEXIBILITÉ DES ALLERS:
    - Les horaires de départ peuvent être retardés pour éviter des conflits
    - Les retards sont pénalisés dans le score (×10 par minute)
    - L'optimisation trouvera le meilleur compromis retard vs nombre de rames
    
    Args:
        missions: Liste des missions
        heure_debut, heure_fin: Période de service
        df_gares: Infrastructure
        allow_sharing: Contrôle le partage ENTRE missions (partage intra-mission toujours autorisé)
        search_strategy: 'smart' (rapide) ou 'exhaustive' (optimal)
    
    Returns:
        tuple: (chronologie, warnings, stats_homogeneite)
    """
    engine = SimulationEngine(df_gares, heure_debut, heure_fin)
    
    print("=" * 80)
    print("OPTIMISATION GLOBALE v4.1")
    print("=" * 80)
    print(f"Missions: {len(missions)}")
    print(f"Stratégie: {search_strategy}")
    print(f"Partage entre missions: {allow_sharing}")
    print(f"Partage intra-mission: TOUJOURS activé")
    print(f"Flexibilité allers: Retards autorisés (pénalisés ×10/min)")
    print("-" * 80)
    
    # 1. Génération des Allers (horaires de départ SOUHAITÉS mais pas fixes)
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
                    'ideal_dep': curr,  # Horaire SOUHAITÉ (peut être retardé)
                    'mission': m, 
                    'type': 'aller', 
                    'm_idx': m_idx
                })
                curr += intervalle

    print(f"Allers générés: {len(aller_requests)}")
    
    # 2. Génération des cadencements possibles pour les retours
    missions_avec_retour = [m for m in missions if m.get('frequence', 0) > 0]
    
    if search_strategy == 'smart':
        cadencements_a_tester = list(range(0, 60, 5))
    else:
        cadencements_a_tester = list(range(60))
    
    print(f"Cadencements par mission: {len(cadencements_a_tester)}")
    
    # 3. Génération de toutes les combinaisons
    all_combinations = list(itertools.product(cadencements_a_tester, 
                                             repeat=len(missions_avec_retour)))
    
    total_combinations = len(all_combinations)
    print(f"Combinaisons totales: {total_combinations:,}")
    
    if total_combinations > 10000:
        print("⚠️  Nombre élevé de combinaisons, calcul peut prendre du temps...")
        if search_strategy == 'exhaustive' and total_combinations > 50000:
            print("⚠️  Passage automatique en mode 'smart'")
            search_strategy = 'smart'
            cadencements_a_tester = list(range(0, 60, 5))
            all_combinations = list(itertools.product(cadencements_a_tester, 
                                                     repeat=len(missions_avec_retour)))
            total_combinations = len(all_combinations)
            print(f"    Nouvelles combinaisons: {total_combinations:,}")
    
    print("-" * 80)
    print("Recherche de l'optimum global...")
    
    # 4. Test de toutes les combinaisons
    best_score = float('inf')
    best_chronologie = {}
    best_stats = {}
    best_warnings = {'infra_violations': [], 'other': []}
    best_combination = None
    best_delay = 0
    best_rames = 0
    
    checkpoint = max(1, total_combinations // 20)
    
    for combo_idx, cadencements_combo in enumerate(all_combinations):
        # Construction des requêtes retour pour cette combinaison
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
        
        # Évaluation de cette combinaison
        all_requests = aller_requests + retour_requests
        score, chrono, fails, stats, delay, nb_rames = evaluer_configuration(
            engine, 
            all_requests,
            allow_cross_mission_sharing=allow_sharing  # NOUVEAU: Paramètre corrigé
        )
        
        # Mise à jour si meilleur
        if score < best_score:
            best_score = score
            best_chronologie = chrono
            best_stats = stats
            best_combination = cadencements_combo
            best_delay = delay
            best_rames = nb_rames
            
            best_warnings = {'infra_violations': [], 'other': []}
            for fail in fails:
                msg = f"{fail['mission']} à {fail['time'].strftime('%H:%M')}: {fail['reason']}"
                if "Infrastructure" in fail['reason'] or "Impasse" in fail['reason']:
                    best_warnings['infra_violations'].append(msg)
                else:
                    best_warnings['other'].append(msg)
        
        # Progression
        if (combo_idx + 1) % checkpoint == 0 or combo_idx == total_combinations - 1:
            progress = (combo_idx + 1) / total_combinations * 100
            print(f"  {progress:.1f}% ({combo_idx+1:,}/{total_combinations:,}) "
                  f"- Meilleur: score={best_score:.0f}, rames={best_rames}, retard={best_delay:.1f}min")
    
    print("-" * 80)
    print("RÉSULTATS OPTIMUM GLOBAL:")
    print(f"  Score: {best_score:.0f}")
    print(f"  Cadencements: {best_combination}")
    print(f"  Trains: {sum(len(t) for t in best_chronologie.values())}")
    print(f"  Rames: {best_rames}")
    print(f"  Retard total: {best_delay:.1f} min")
    print("=" * 80)
    
    return best_chronologie, best_warnings, best_stats

# =============================================================================
# 5. FONCTIONS UTILITAIRES (inchangées)
# =============================================================================

@lru_cache(maxsize=256)
def construire_horaire_mission_cached(mission_key, trajet_spec, df_gares_json):
    """
    Wrapper avec cache pour la construction d'horaire.
    Permet d'éviter de recalculer l'interpolation des gares à chaque itération de la simulation.
    """
    if df_gares_json is None: return None
    try:
        df_gares_local = pd.read_json(StringIO(df_gares_json))
        mission_cfg = json.loads(mission_key)
        return construire_horaire_mission(mission_cfg, trajet_spec, df_gares_local)
    except: return None

def construire_horaire_mission(m, direction, df_gares):
    """
    Interpole les horaires de passage à toutes les gares intermédiaires (physiques)
    entre l'origine et le terminus, en se basant sur les points de passage définis.
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
            # Inversion automatique des PP aller si symétrique
            base_pp = [{"gare": p["gare"], "time_offset_min": dur - p["time_offset_min"], "duree_arret_min": p.get("duree_arret_min", 0)} for p in m.get("passing_points", [])]
        pts.append({"gare": m.get("terminus"), "time_offset_min": 0, "duree_arret_min": 0})
        pts.extend(base_pp)
        pts.append({"gare": m.get("origine"), "time_offset_min": dur, "duree_arret_min": 0})

    pts.sort(key=lambda x: x['time_offset_min'])

    # Interpolation linéaire entre les points pivots
    unique = []
    seen = set()
    for p in pts:
        if p['gare'] not in seen: unique.append(p); seen.add(p['gare'])

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

            # Récupération durée arrêt si c'est un point pivot
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
    """Importe un fichier CSV ou Excel de roulements."""
    try:
        df_import = None
        if uploaded_file.name.endswith('.csv'):
            try:
                df_import = pd.read_csv(uploaded_file, sep=';', dtype=str, encoding='utf-8')
            except:
                uploaded_file.seek(0)
                df_import = pd.read_csv(uploaded_file, sep=';', dtype=str, encoding='latin1')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_import = pd.read_excel(uploaded_file, dtype=str)
        else:
            return None, "Format non supporté."

        df_import.columns = (
            df_import.columns.str.lower()
            .str.normalize('NFKD')
            .str.encode('ascii', errors='ignore')
            .str.decode('utf-8')
            .str.replace(r'[^a-z0-9_]', '', regex=True)
        )

        col_maps = {
            "train_id": ["train", "trainid", "idtrain", "numero_train"],
            "origine": ["depart", "origine", "gare_depart"],
            "heure_depart": ["heure_depart", "heuredepart", "h_dep"],
            "terminus": ["arrivee", "terminus", "gare_arrivee"],
            "heure_arrivee": ["heure_arrivee", "heurearrivee", "h_arr"],
            "temps_trajet": ["temps_trajet", "duree"]
        }

        rename_dict = {}
        for target, possibles in col_maps.items():
            for p in possibles:
                if p in df_import.columns:
                    rename_dict[p] = target
                    break
        df_import = df_import.rename(columns=rename_dict)

        new_roulement = {}
        for _, row in df_import.iterrows():
            tid = str(row.get('train_id', '')).strip()
            if not tid:
                continue
            try:
                h_dep = pd.to_datetime(row['heure_depart']).strftime('%H:%M')
                h_arr = pd.to_datetime(row['heure_arrivee']).strftime('%H:%M')
                new_roulement.setdefault(int(float(tid)), []).append({
                    "depart": row['origine'],
                    "heure_depart": h_dep,
                    "arrivee": row['terminus'],
                    "heure_arrivee": h_arr,
                    "temps_trajet": int(float(row.get('temps_trajet', 0)))
                })
            except:
                continue

        for tid in new_roulement:
            new_roulement[tid].sort(key=lambda x: x['heure_depart'])

        return new_roulement, None
    except Exception as e:
        return None, f"Erreur import : {e}"

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
    pass
