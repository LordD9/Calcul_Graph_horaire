# -*- coding: utf-8 -*-
"""
core_logic.py - V12 (Injection Flexible & Homogénéité Structurée)
Ce module gère le cadencement itératif avec une priorité absolue à l'infrastructure.
Intègre la possibilité d'injecter des rames au Terminus 2 et calcule l'homogénéité du cadencement.
"""
from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO, StringIO
from collections import defaultdict
import json
import math
import copy
from functools import lru_cache

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
    return infra_code in ['VE', 'D', 'Terminus']

def calculer_indice_homogeneite(horaires):
    """
    Calcule l'homogénéité du cadencement via un indice basé sur le coefficient de Gini.
    Un indice de 1 signifie un cadencement parfaitement régulier (intervalles identiques).
    """
    if len(horaires) < 2:
        return 1.0

    # Calcul des intervalles (headways) en minutes
    horaires_tries = sorted(horaires)
    intervalles = []
    for i in range(len(horaires_tries) - 1):
        diff = (horaires_tries[i+1] - horaires_tries[i]).total_seconds() / 60.0
        intervalles.append(diff)

    if not intervalles or sum(intervalles) == 0:
        return 0.0

    # Calcul du coefficient de Gini sur les intervalles
    n = len(intervalles)
    intervalles.sort()
    # Formule du coefficient de Gini
    somme_diff = sum((i + 1) * val for i, val in enumerate(intervalles))
    moyenne = sum(intervalles) / n
    gini = (2.0 * somme_diff) / (n * sum(intervalles)) - (n + 1.0) / n

    # L'homogénéité est l'inverse de l'inégalité (1 - Gini)
    return max(0.0, 1.0 - gini)

class SimulationEngine:
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
                o_idx_min, o_idx_max = min(p_a['index'], p_b['index']), max(p_a['index'], p_b['index'])

                if max(seg_idx_min, o_idx_min) < min(seg_idx_max, o_idx_max):
                    if max(t_enter, p_a['dep'] - margin) < min(t_exit, p_b['arr'] + margin):
                        if all_double: continue
                        return False, p_b['arr'] + margin + timedelta(seconds=1)
        return True, None

    def solve_mission_schedule(self, mission, ideal_start_time, direction):
        """
        Calcule un sillon valide.
        Si aucun créneau n'est trouvé sans violation d'infra, retourne None.
        """
        base_schedule = construire_horaire_mission(mission, direction, self.df_gares)
        if not base_schedule: return None, [], "Erreur itinéraire"

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

            travel_time_block = sum(steps[k]['run_time'] + steps[k]['duree_arret'] for k in range(i + 1, target_idx))
            travel_time_block += steps[target_idx]['run_time']

            current_departure = current_time
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
                    # Fenêtre de recherche limitée pour éviter les boucles infinies
                    if current_departure - ideal_start_time > timedelta(hours=4):
                        return None, [], "Conflit insoluble"

        return final_path[0]['dep'], final_path, None

    def allocate_train_id(self, gare, target_time, can_inject=True):
        pool = self.fleet_availability[gare]
        pool.sort()
        for i, (dispo_t, tid) in enumerate(pool):
            if dispo_t <= target_time + timedelta(minutes=2):
                return pool.pop(i)[1]

        if can_inject:
            tid = self.train_counter
            self.train_counter += 1
            return tid
        return None

    def register_arrival(self, tid, gare, arr_time, turnaround_min):
        self.fleet_availability[gare].append((arr_time + timedelta(minutes=turnaround_min), tid))

# =============================================================================
# LOGIQUE D'OPTIMISATION DE LA GRILLE
# =============================================================================

def evaluer_configuration(engine, requests):
    """
    Simule la journée et collecte les statistiques.
    """
    engine.reset()
    total_delay_min = 0
    trajets_resultat = defaultdict(list)
    failures = []
    real_departures_per_mission = defaultdict(list)

    sorted_reqs = sorted(requests, key=lambda x: x['ideal_dep'])

    for req in sorted_reqs:
        # 1. Calcul du sillon (Respect STRICT de l'infra)
        real_dep, path, err = engine.solve_mission_schedule(req['mission'], req['ideal_dep'], req['type'])

        if not path:
            failures.append({
                "time": req['ideal_dep'],
                "mission": f"M{req.get('m_idx', 0)+1} ({req['type']})",
                "reason": err or "Infrastructure saturée"
            })
            continue

        # 2. Allocation Matériel (Injection possible au Terminus 2 si coché)
        inject_allowed = (req['type'] == 'aller') or (req['type'] == 'retour' and req['mission'].get('inject_from_terminus_2', False))
        tid = engine.allocate_train_id(path[0]['gare'], real_dep, can_inject=inject_allowed)

        if tid is None:
            failures.append({
                "time": req['ideal_dep'],
                "mission": f"M{req.get('m_idx', 0)+1} ({req['type']})",
                "reason": "Pas de rame disponible au terminus"
            })
            continue

        # 3. Validation
        tid = int(tid)
        engine.committed_schedules.append({'train_id': tid, 'path': path})
        delay = (real_dep - req['ideal_dep']).total_seconds() / 60
        total_delay_min += delay

        # Stockage pour le calcul de l'homogénéité (par mission et type)
        m_key = f"Mission {req.get('m_idx', 0)+1} ({req['type']})"
        real_departures_per_mission[m_key].append(real_dep)

        for k in range(len(path)-1):
            p_curr, p_next = path[k], path[k+1]
            trajets_resultat[tid].append({"start": p_curr['dep'], "end": p_next['arr'], "origine": p_curr['gare'], "terminus": p_next['gare']})
            if p_next['dep'] > p_next['arr']:
                trajets_resultat[tid].append({"start": p_next['arr'], "end": p_next['dep'], "origine": p_next['gare'], "terminus": p_next['gare']})

        t_ret = req['mission'].get('temps_retournement_B' if req['type'] == 'aller' else 'temps_retournement_A', 10)
        engine.register_arrival(tid, path[-1]['gare'], path[-1]['dep'], t_ret)

    # Score combiné
    score = (engine.train_counter * 100000) + total_delay_min + (len(failures) * 1000000)

    # Calcul des indices d'homogénéité par mission
    homogeneite = {}
    for m_key, departures in real_departures_per_mission.items():
        homogeneite[m_key] = calculer_indice_homogeneite(departures)

    return score, dict(trajets_resultat), failures, homogeneite

def generer_tous_trajets_optimises(missions, heure_debut, heure_fin, df_gares):
    engine = SimulationEngine(df_gares, heure_debut, heure_fin)
    final_requests = []

    # Allers
    for m_idx, m in enumerate(missions):
        freq = m.get('frequence', 1)
        if freq <= 0: continue
        intervalle = timedelta(hours=1.0/freq)
        refs = [int(x.strip()) for x in str(m.get('reference_minutes', '0')).split(',') if x.strip().isdigit()] or [0]
        for r in refs:
            curr = engine.dt_debut.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=r)
            while curr < engine.dt_debut: curr += intervalle
            while curr < engine.dt_fin:
                final_requests.append({'ideal_dep': curr, 'mission': m, 'type': 'aller', 'm_idx': m_idx})
                curr += intervalle

    # Optimisation itérative du Retour
    for m_idx, m in enumerate(missions):
        if m.get('frequence', 0) <= 0: continue
        best_min_score = float('inf')
        best_return_requests = []
        t_trajet_ret = m.get('temps_trajet_retour', m.get('temps_trajet', 60))
        intervalle = timedelta(hours=1.0/m['frequence'])

        for target_arr_min in range(60):
            test_return_reqs = []
            dep_min = (target_arr_min - t_trajet_ret) % 60
            curr = engine.dt_debut.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=dep_min)
            while curr < engine.dt_debut: curr += intervalle
            while curr < engine.dt_fin:
                test_return_reqs.append({'ideal_dep': curr, 'mission': m, 'type': 'retour', 'm_idx': m_idx})
                curr += intervalle

            score, _, _, _ = evaluer_configuration(engine, final_requests + test_return_reqs)
            if score < best_min_score:
                best_min_score = score
                best_return_requests = test_return_reqs
        final_requests.extend(best_return_requests)

    # Simulation finale
    score, chronologie, failures, homogeneite = evaluer_configuration(engine, final_requests)

    warnings = {"infra_violations": [], "other": []}
    for f in failures:
        warnings["other"].append(f"Trajet annulé : {f['time'].strftime('%H:%M')} - {f['mission']} -> {f['reason']}")

    return chronologie, warnings, homogeneite

# =============================================================================
# UTILITAIRES ET COMPATIBILITÉ UI
# =============================================================================

@lru_cache(maxsize=256)
def construire_horaire_mission_cached(mission_key, trajet_spec, df_gares_json):
    if df_gares_json is None: return None
    try:
        df_gares_local = pd.read_json(StringIO(df_gares_json))
        mission_cfg = json.loads(mission_key)
        return construire_horaire_mission(mission_cfg, trajet_spec, df_gares_local)
    except: return None

def construire_horaire_mission(m, direction, df_gares):
    if df_gares is None or df_gares.empty: return []
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
            base_pp = [{"gare": p["gare"], "time_offset_min": dur - p["time_offset_min"], "duree_arret_min": p.get("duree_arret_min", 0)} for p in m.get("passing_points", [])]
        pts.append({"gare": m.get("terminus"), "time_offset_min": 0, "duree_arret_min": 0})
        pts.extend(base_pp)
        pts.append({"gare": m.get("origine"), "time_offset_min": dur, "duree_arret_min": 0})

    pts.sort(key=lambda x: x['time_offset_min'])
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
    res = {}
    for tid, etapes in roulement.items():
        res[tid] = []
        for e in etapes:
            try:
                d = datetime.combine(datetime.today(), datetime.strptime(e["heure_depart"], "%H:%M").time())
                a = datetime.combine(datetime.today(), datetime.strptime(e["heure_arrivee"], "%H:%M").time())
                if a < d: a += timedelta(days=1)
                res[tid].append({"start": d, "end": a, "origine": e["depart"], "terminus": e["arrivee"]})
            except: pass
    return res

def importer_roulements_fichier(uploaded_file, dataframe_gares):
    try:
        df_import = None
        if uploaded_file.name.endswith('.csv'):
            try: df_import = pd.read_csv(uploaded_file, sep=';', dtype=str, encoding='utf-8')
            except:
                uploaded_file.seek(0)
                df_import = pd.read_csv(uploaded_file, sep=';', dtype=str, encoding='latin1')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_import = pd.read_excel(uploaded_file, dtype=str)
        else: return None, "Format non supporté."

        df_import.columns = (df_import.columns.str.lower().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
                             .str.replace(r'[^a-z0-9_]', '', regex=True))

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
            if not tid: continue
            try:
                h_dep = pd.to_datetime(row['heure_depart']).strftime('%H:%M')
                h_arr = pd.to_datetime(row['heure_arrivee']).strftime('%H:%M')
                new_roulement.setdefault(int(float(tid)), []).append({
                    "depart": row['origine'], "heure_depart": h_dep, "arrivee": row['terminus'],
                    "heure_arrivee": h_arr, "temps_trajet": int(float(row.get('temps_trajet', 0)))
                })
            except: continue
        for tid in new_roulement: new_roulement[tid].sort(key=lambda x: x['heure_depart'])
        return new_roulement, None
    except Exception as e: return None, f"Erreur import : {e}"

def analyser_frequences_manuelles(roulement_manuel, missions, heure_debut_service, heure_fin_service):
    comptes = {}
    for etapes in roulement_manuel.values():
        for e in etapes:
            cle = f"{e['depart']} → {e['arrivee']}"
            try:
                h = datetime.strptime(e['heure_depart'], "%H:%M").hour
                comptes.setdefault(cle, {}).setdefault(h, 0)
                comptes[cle][h] += 1
            except: continue
    resultats = {}
    heures = []
    curr = datetime.combine(datetime.today(), heure_debut_service)
    end = datetime.combine(datetime.today(), heure_fin_service)
    if end <= curr: end += timedelta(days=1)
    while curr < end:
        heures.append(curr.hour)
        curr += timedelta(hours=1)
    for m in missions:
        if m.get('frequence', 0) <= 0: continue
        cle = f"{m['origine']} → {m['terminus']}"
        donnees = []
        respectees = 0
        for h in heures:
            reel = comptes.get(cle, {}).get(h, 0)
            statut = "✓" if reel >= m['frequence'] else "❌"
            if statut == "✓": respectees += 1
            donnees.append({"Heure": f"{h:02d}:00", "Trains": reel, "Objectif": f"≥ {m['frequence']}", "Statut": statut})
        if donnees: resultats[cle] = {"df": pd.DataFrame(donnees), "conformite": (respectees / len(heures)) * 100}
    return resultats

def generer_exports(chronologie, figure):
    rows = []
    for tid in sorted(chronologie.keys()):
        for t in sorted(chronologie[tid], key=lambda x: x['start']):
            rows.append({"Train": tid, "Début": t["start"].strftime('%Y-%m-%d %H:%M:%S'), "Fin": t["end"].strftime('%Y-%m-%d %H:%M:%S'), "Origine": t["origine"], "Terminus": t["terminus"]})
    df = pd.DataFrame(rows)
    bx = BytesIO()
    with pd.ExcelWriter(bx, engine='xlsxwriter') as wr: df.to_excel(wr, index=False)
    bx.seek(0)
    bp = BytesIO()
    if figure: figure.savefig(bp, format="pdf", bbox_inches='tight')
    bp.seek(0)
    return bx, bp

def reset_caches():
    construire_horaire_mission_cached.cache_clear()