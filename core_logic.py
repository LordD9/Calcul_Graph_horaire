# -*- coding: utf-8 -*-
"""
core_logic.py
=============

Ce module constitue le moteur de calcul (Backend) de l'application de simulation ferroviaire.
Il gère :
1. La représentation de l'infrastructure (Gares, Voie Unique/Double).
2. La planification des horaires (Sillons) en respectant les contraintes de croisement.
3. L'optimisation des rotations de matériel (Minimisation du nombre de rames).
4. Le calcul d'indicateurs de qualité (Homogénéité/Gini).
5. L'exportation des résultats (Excel, PDF).

Auteur : [Votre Nom / Organisation]
Version : 2.1 (Correction Démarrage Matinal)
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
# 1. UTILITAIRES D'INFRASTRUCTURE
# =============================================================================

def _get_infra_at_gare(df_gares, gare_name):
    """
    Récupère le type d'infrastructure pour une gare donnée.

    Args:
        df_gares (pd.DataFrame): DataFrame contenant les colonnes 'gare' et 'infra'.
        gare_name (str): Nom de la gare recherchée.

    Returns:
        str: Code infrastructure ('VE', 'D', 'F') ou 'F' par défaut.
    """
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
    """
    Détermine si un train peut s'arrêter ou croiser à un point donné.

    Args:
        infra_code (str): Le code infra de la gare ('VE', 'D', 'F', ...).

    Returns:
        bool: True si croisement possible, False sinon.
    """
    # VE : Voie d'Évitement (Gare sur voie unique permettant le croisement)
    # D : Section à Double Voie (Croisement possible partout, donc arrêt possible)
    # Terminus : Implicitement un point de capacité
    return infra_code in ['VE', 'D', 'Terminus']

def calculer_indice_homogeneite(horaires):
    """
    Calcule l'homogénéité du cadencement (régularité des intervalles)
    en utilisant une adaptation du coefficient de Gini.

    Le score retourné est 1 - Gini.
    - 1.00 : Cadencement parfait (intervalles strictement identiques).
    - 0.00 : Irrégularité totale.

    Args:
        horaires (list[datetime]): Liste triée des heures de passage.

    Returns:
        float: Indice d'homogénéité (0 à 1).
    """
    if len(horaires) < 2:
        return 1.0

    # 1. Calcul des intervalles (headways) en minutes
    horaires_tries = sorted(horaires)
    intervalles = []
    for i in range(len(horaires_tries) - 1):
        diff = (horaires_tries[i+1] - horaires_tries[i]).total_seconds() / 60.0
        # On ignore les intervalles quasi-nuls (erreurs de données ou trains simultanés)
        if diff > 0.1:
            intervalles.append(diff)

    if not intervalles or sum(intervalles) == 0:
        return 0.0

    # 2. Calcul du coefficient de Gini sur les intervalles
    # Formule pour échantillon discret : G = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n + 1) / n
    n = len(intervalles)
    intervalles.sort()
    somme_ponderee = sum((i + 1) * val for i, val in enumerate(intervalles))
    somme_totale = sum(intervalles)

    gini = (2.0 * somme_ponderee) / (n * somme_totale) - (n + 1.0) / n

    # L'homogénéité est l'inverse de l'inégalité
    return max(0.0, 1.0 - gini)

# =============================================================================
# 2. MOTEUR DE SIMULATION
# =============================================================================

class SimulationEngine:
    """
    Classe gérant l'état de la simulation (occupation des voies, position des trains).
    Elle expose les méthodes pour vérifier les conflits et allouer les sillons.
    """
    def __init__(self, df_gares, heure_debut, heure_fin):
        # Préparation des données géographiques
        self.df_gares = df_gares.sort_values('distance').reset_index(drop=True)
        self.gares_map = {r.gare: i for i, r in self.df_gares.iterrows()}
        self.infra_map = {r.gare: _get_infra_at_gare(df_gares, r.gare) for _, r in self.df_gares.iterrows()}

        # Bornes temporelles
        self.dt_debut = datetime.combine(datetime.today(), heure_debut)
        self.dt_fin = datetime.combine(datetime.today(), heure_fin)
        if self.dt_fin <= self.dt_debut:
            self.dt_fin += timedelta(days=1)

        # Analyse statique de la ligne (détection des zones de double voie)
        self.segment_is_double = self._analyze_segments()

        # État dynamique (sera réinitialisé à chaque itération d'optimisation)
        self.reset()

    def reset(self):
        """Réinitialise l'état pour une nouvelle tentative de simulation."""
        self.committed_schedules = [] # Liste des sillons validés
        self.fleet_availability = defaultdict(list) # Disponibilité des rames par gare
        self.train_counter = 1

    def _analyze_segments(self):
        """
        Détermine pour chaque inter-gare si c'est une Voie Unique (VU) ou Double Voie (DV).
        Logique : Une gare marquée 'D' indique une transition de régime.
        """
        is_double = {}
        n = len(self.df_gares)
        current_state_double = False # Hypothèse : la ligne commence en VU
        for i in range(n - 1):
            gare_curr = self.df_gares.iloc[i]['gare']
            infra_curr = self.infra_map.get(gare_curr, 'F')
            if infra_curr == 'D':
                current_state_double = not current_state_double
            is_double[i] = current_state_double
        return is_double

    def check_segment_availability(self, seg_idx_min, seg_idx_max, t_enter, t_exit):
        """
        Vérifie si un tronçon (défini par index min/max) est libre sur une plage horaire.

        Logique :
        - Sur Double Voie (DV) : On considère que la capacité est infinie (simplification macro).
        - Sur Voie Unique (VU) : Aucun croisement n'est toléré. Si un autre train y est, conflit.

        Returns:
            tuple: (is_free (bool), next_free_time (datetime or None))
        """
        margin = timedelta(minutes=1) # Marge de sécurité

        # Vérification si tout le tronçon est en DV
        all_double = True
        for i in range(seg_idx_min, seg_idx_max):
            if not self.segment_is_double.get(i, False):
                all_double = False
                break

        # Vérification des conflits avec les trains déjà planifiés
        for committed in self.committed_schedules:
            path = committed['path']
            # Optimisation : si le train est hors de la fenêtre temporelle globale, on passe
            if not path or path[-1]['arr'] < t_enter or path[0]['dep'] > t_exit:
                continue

            for i in range(len(path) - 1):
                p_a, p_b = path[i], path[i+1]
                # Indices géographiques du segment occupé par le train existant
                o_idx_min = min(p_a['index'], p_b['index'])
                o_idx_max = max(p_a['index'], p_b['index'])

                # Intersection spatiale ?
                if max(seg_idx_min, o_idx_min) < min(seg_idx_max, o_idx_max):
                    # Intersection temporelle ?
                    # Note : p_a['dep'] et p_b['arr'] couvrent tout le temps de trajet sur le segment
                    if max(t_enter, p_a['dep'] - margin) < min(t_exit, p_b['arr'] + margin):
                        if all_double: continue # En DV, on ignore le conflit (croisement possible)

                        # Conflit détecté sur VU -> On renvoie l'heure de libération
                        return False, p_b['arr'] + margin + timedelta(seconds=1)
        return True, None

    def solve_mission_schedule(self, mission, ideal_start_time, direction):
        """
        Construit un sillon (horaire) valide pour une mission donnée.
        Tente de partir à `ideal_start_time`, mais retarde le départ en cas de conflit infra.

        Args:
            mission (dict): Paramètres de la mission.
            ideal_start_time (datetime): Heure de départ souhaitée.
            direction (str): 'aller' ou 'retour'.

        Returns:
            tuple: (real_start_time, path, error_message)
        """
        # 1. Construction du profil théorique (sans conflits) via interpolation
        base_schedule = construire_horaire_mission(mission, direction, self.df_gares)
        if not base_schedule: return None, [], "Erreur itinéraire (Gares inconnues ?)"

        # 2. Conversion en étapes séquentielles
        steps = []
        for i, pt in enumerate(base_schedule):
            idx = self.gares_map.get(pt['gare'])
            # Calcul du temps de marche depuis la gare précédente
            duration = pt['time_offset_min'] - base_schedule[i-1]['time_offset_min'] if i > 0 else 0
            steps.append({
                'gare': pt['gare'],
                'index': idx,
                'run_time': duration,
                'duree_arret': pt.get('duree_arret_min', 0),
                'infra': self.infra_map.get(pt['gare'], 'F')
            })

        # 3. Algorithme Glouton (Greedy) : Avancer de point de croisement en point de croisement
        current_time = ideal_start_time

        # Initialisation du chemin final
        final_path = [{
            'gare': steps[0]['gare'],
            'index': steps[0]['index'],
            'arr': current_time,
            'dep': current_time + timedelta(minutes=steps[0]['duree_arret'])
        }]
        current_time += timedelta(minutes=steps[0]['duree_arret'])

        i = 0
        while i < len(steps) - 1:
            # Identifier le prochain point où un croisement est possible (VE, D, Terminus)
            target_idx = i + 1
            while target_idx < len(steps):
                if _is_crossing_point(steps[target_idx]['infra']) or target_idx == len(steps) - 1:
                    break
                target_idx += 1

            # Calcul du temps nécessaire pour traverser tout ce bloc (marche + arrêts intermédiaires)
            travel_time_block = sum(steps[k]['run_time'] + steps[k]['duree_arret'] for k in range(i + 1, target_idx))
            travel_time_block += steps[target_idx]['run_time']

            # Recherche d'un créneau libre pour ce bloc
            current_departure = current_time
            while True:
                idx_min = min(steps[i]['index'], steps[target_idx]['index'])
                idx_max = max(steps[i]['index'], steps[target_idx]['index'])

                t_enter = current_departure
                t_exit = current_departure + timedelta(minutes=travel_time_block)

                is_free, next_t = self.check_segment_availability(idx_min, idx_max, t_enter, t_exit)

                if is_free:
                    # Créneau trouvé : on enregistre les horaires de toutes les gares intermédiaires
                    final_path[-1]['dep'] = current_departure # Mise à jour départ réel du bloc
                    t_cursor = current_departure

                    for k in range(i + 1, target_idx + 1):
                        t_cursor += timedelta(minutes=steps[k]['run_time']) # Marche
                        st_k = steps[k]
                        final_path.append({
                            'gare': st_k['gare'],
                            'index': st_k['index'],
                            'arr': t_cursor,
                            'dep': t_cursor + timedelta(minutes=st_k['duree_arret'])
                        })
                        t_cursor += timedelta(minutes=st_k['duree_arret']) # Arrêt

                    current_time = t_cursor
                    i = target_idx # Avancer au prochain bloc
                    break
                else:
                    # Conflit : on décale le départ à la fin du conflit détecté
                    current_departure = next_t

                    # Sécurité : Si on décale de plus de 30min, on abandonne (conflit insoluble)
                    if current_departure - ideal_start_time > timedelta(hours=1):
                        return None, [], "Impasse infra (délai > 1 heure)"

        # Le premier élément de final_path contient l'heure de départ réelle
        return final_path[0]['dep'], final_path, None

    def allocate_train_id(self, gare, target_time, can_inject=True):
        """
        Alloue une rame à un départ.
        - Cherche d'abord une rame disponible à la gare à l'heure cible.
        - Si aucune dispo et `can_inject` est True, crée une nouvelle rame.
        """
        pool = self.fleet_availability[gare]
        pool.sort()

        # Marge de tolérance : on accepte une rame qui arrive jusqu'à 2 min après le départ théorique
        # (le départ réel a déjà été calculé compatible infra, mais on vérifie la dispo matériel ici)
        for i, (dispo_t, tid) in enumerate(pool):
            if dispo_t <= target_time + timedelta(minutes=2):
                return pool.pop(i)[1] # On prend la rame et on la retire du pool

        if can_inject:
            tid = self.train_counter
            self.train_counter += 1
            return tid

        return None # Pas de solution

    def register_arrival(self, tid, gare, arr_time, turnaround_min):
        """Libère une rame dans une gare après son temps de retournement."""
        self.fleet_availability[gare].append((arr_time + timedelta(minutes=turnaround_min), tid))

# =============================================================================
# 3. LOGIQUE D'OPTIMISATION (ALGORITHME GÉNÉTIQUE SIMPLIFIÉ)
# =============================================================================

def evaluer_configuration(engine, requests):
    """
    Exécute une simulation complète pour un ensemble de requêtes horaires.
    Calcule un score de performance global.

    Score (à minimiser) =
      (Nb Rames * 1000)
    + (Nb Échecs * 1M)
    + (Minutes de retard * 10)
    - (Homogénéité * 5000)

    Returns:
        tuple: (score, chronologie_dict, liste_echecs, stats_homogeneite)
    """
    engine.reset()
    total_delay_min = 0
    trajets_resultat = defaultdict(list)
    failures = []

    # Structure pour stocker tous les horaires par mission et par gare pour calcul Gini
    mission_station_times = defaultdict(lambda: defaultdict(list))

    # Tri chronologique impératif pour la simulation
    sorted_reqs = sorted(requests, key=lambda x: x['ideal_dep'])

    for req in sorted_reqs:
        # A. Calcul du sillon (Respect STRICT de l'infra)
        real_dep, path, err = engine.solve_mission_schedule(req['mission'], req['ideal_dep'], req['type'])

        if not path:
            failures.append({
                "time": req['ideal_dep'],
                "mission": f"M{req.get('m_idx', 0)+1} ({req['type']})",
                "reason": err or "Infrastructure saturée"
            })
            continue

        # B. Allocation Matériel
        # L'injection est autorisée si c'est un aller OU si l'option Terminus 2 est active
        inject_allowed = (req['type'] == 'aller') or (req['type'] == 'retour' and req['mission'].get('inject_from_terminus_2', False))
        tid = engine.allocate_train_id(path[0]['gare'], real_dep, can_inject=inject_allowed)

        if tid is None:
            failures.append({
                "time": req['ideal_dep'],
                "mission": f"M{req.get('m_idx', 0)+1} ({req['type']})",
                "reason": "Pas de rame disponible au terminus (Injection interdite)"
            })
            continue

        # C. Validation et Enregistrement
        tid = int(tid)
        engine.committed_schedules.append({'train_id': tid, 'path': path})
        delay = (real_dep - req['ideal_dep']).total_seconds() / 60
        total_delay_min += delay

        # Enregistrement des temps de passage pour l'homogénéité (Clé par SENS)
        m_key = f"M{req.get('m_idx', 0)}_{req['type']}"
        for step in path:
            is_terminus = (step == path[-1])
            # Pour le cadencement, on regarde l'arrivée au terminus, et le départ aux autres gares
            time_to_record = step['arr'] if is_terminus else step['dep']
            mission_station_times[m_key][step['gare']].append(time_to_record)

        # Conversion du path en segments pour l'export/affichage
        for k in range(len(path)-1):
            p_curr, p_next = path[k], path[k+1]
            # Segment de marche
            trajets_resultat[tid].append({
                "start": p_curr['dep'],
                "end": p_next['arr'],
                "origine": p_curr['gare'],
                "terminus": p_next['gare']
            })
            # Segment d'arrêt commercial (si existant)
            if p_next['dep'] > p_next['arr']:
                trajets_resultat[tid].append({
                    "start": p_next['arr'],
                    "end": p_next['dep'],
                    "origine": p_next['gare'],
                    "terminus": p_next['gare']
                })

        # Libération de la rame
        t_ret = req['mission'].get('temps_retournement_B' if req['type'] == 'aller' else 'temps_retournement_A', 10)
        engine.register_arrival(tid, path[-1]['gare'], path[-1]['dep'], t_ret)

    # --- CALCUL DES INDICATEURS ---

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

        # Formatage clé pour UI : "M0_aller" -> "Mission 1 (Aller)"
        try:
            midx, sens = m_key.split('_')
            ui_key = f"Mission {int(midx[1:])+1} ({sens.capitalize()})"
        except:
            ui_key = m_key

        homogeneite_par_mission[ui_key] = avg_score

    avg_homogeneity = global_homogeneity_score / total_stations_checked if total_stations_checked > 0 else 1.0

    # Fonction de coût
    penalty_trains = engine.train_counter * 1000
    penalty_failures = len(failures) * 1000000
    penalty_delay = total_delay_min*10
    bonus_homogeneity = avg_homogeneity * 5000

    score = penalty_trains + penalty_failures + penalty_delay - bonus_homogeneity

    return score, dict(trajets_resultat), failures, homogeneite_par_mission

def generer_tous_trajets_optimises(missions, heure_debut, heure_fin, df_gares):
    """
    Point d'entrée principal pour le calcul des horaires.
    Orchestre l'optimisation itérative.

    Stratégie :
    1. Fixer les horaires Aller selon la fréquence et la minute de référence (Contrainte dure).
    2. Pour chaque mission, tester les 60 minutes possibles de cadencement Retour.
    3. Choisir la minute Retour qui minimise le parc matériel et maximise l'homogénéité.
    """
    engine = SimulationEngine(df_gares, heure_debut, heure_fin)
    final_requests = []

    # 1. Génération des Allers (Fixes)
    for m_idx, m in enumerate(missions):
        freq = m.get('frequence', 1)
        if freq <= 0: continue
        intervalle = timedelta(hours=1.0/freq)
        refs = [int(x.strip()) for x in str(m.get('reference_minutes', '0')).split(',') if x.strip().isdigit()] or [0]
        for r in refs:
            # CORRECTION : On remonte d'une heure pour être sûr de couvrir le début de service
            curr = engine.dt_debut.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=r) - timedelta(hours=1)

            # Avance jusqu'à la fenêtre demandée
            while curr < engine.dt_debut: curr += intervalle

            while curr < engine.dt_fin:
                final_requests.append({'ideal_dep': curr, 'mission': m, 'type': 'aller', 'm_idx': m_idx})
                curr += intervalle

    # 2. Optimisation des Retours (Variable)
    for m_idx, m in enumerate(missions):
        if m.get('frequence', 0) <= 0: continue
        best_min_score = float('inf')
        best_return_requests = []
        t_trajet_ret = m.get('temps_trajet_retour', m.get('temps_trajet', 60))
        intervalle = timedelta(hours=1.0/m['frequence'])

        # On teste chaque minute de cadencement possible (0 à 59)
        for target_arr_min in range(60):
            test_return_reqs = []
            dep_min = (target_arr_min - t_trajet_ret) % 60

            # CORRECTION : Idem, on commence tôt pour les injections Terminus 2
            curr = engine.dt_debut.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=dep_min) - timedelta(hours=1)

            # Recalage pour s'assurer qu'on couvre bien toute la plage, notamment dès 06h00
            while curr < engine.dt_debut: curr += intervalle

            while curr < engine.dt_fin:
                test_return_reqs.append({'ideal_dep': curr, 'mission': m, 'type': 'retour', 'm_idx': m_idx})
                curr += intervalle

            # Évaluation rapide (sans détail homogeneite)
            score, _, _, _ = evaluer_configuration(engine, final_requests + test_return_reqs)

            if score < best_min_score:
                best_min_score = score
                best_return_requests = test_return_reqs

        # On valide la meilleure configuration pour cette mission
        final_requests.extend(best_return_requests)

    # 3. Simulation finale détaillée
    score, chronologie, failures, homogeneite = evaluer_configuration(engine, final_requests)

    warnings = {"infra_violations": [], "other": []}
    for f in failures:
        warnings["other"].append(f"Trajet annulé : {f['time'].strftime('%H:%M')} - {f['mission']} -> {f['reason']}")

    return chronologie, warnings, homogeneite

# =============================================================================
# 4. UTILITAIRES ET INTERFACE (CACHE & EXPORT)
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
    """Convertit le dictionnaire de roulement manuel en format standard chronologie."""
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
    """
    Importe un fichier CSV ou Excel de roulements.
    Gère le nettoyage des colonnes (accents, casse) pour une robustesse maximale.
    """
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

        # Normalisation des colonnes
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
    """Analyse le respect des fréquences horaires pour un roulement manuel donné."""
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
    """
    Génère les fichiers d'export pour l'utilisateur :
    1. Un fichier Excel contenant le tableau de marche (liste des trains, horaires).
    2. Un fichier PDF contenant le graphique espace-temps.

    Args:
        chronologie (dict): Dictionnaire {train_id: [liste_de_segments]}.
                            Chaque segment contient {start, end, origine, terminus}.
        figure (matplotlib.figure.Figure): L'objet figure du graphique généré.

    Returns:
        tuple: (buffer_excel, buffer_pdf)
               Deux objets BytesIO prêts à être téléchargés par Streamlit.
    """
    # 1. Préparation des données pour Excel
    rows = []
    for tid in sorted(chronologie.keys()):
        for t in sorted(chronologie[tid], key=lambda x: x['start']):
            rows.append({"Train": tid, "Début": t["start"].strftime('%Y-%m-%d %H:%M:%S'), "Fin": t["end"].strftime('%Y-%m-%d %H:%M:%S'), "Origine": t["origine"], "Terminus": t["terminus"]})
    df = pd.DataFrame(rows)

    # 2. Création du fichier Excel en mémoire
    bx = BytesIO()
    # Utilisation du moteur xlsxwriter pour une meilleure compatibilité
    with pd.ExcelWriter(bx, engine='xlsxwriter') as wr:
        df.to_excel(wr, index=False, sheet_name="Tableau de Marche")
        # Ajustement automatique des colonnes
        worksheet = wr.sheets['Tableau de Marche']
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)
    bx.seek(0)

    # 3. Création du fichier PDF en mémoire
    bp = BytesIO()
    if figure:
        # Sauvegarde vectorielle (PDF) avec bbox_inches='tight' pour ne rien couper
        figure.savefig(bp, format="pdf", bbox_inches='tight')
    bp.seek(0)

    return bx, bp

def reset_caches():
    """Vide les caches de fonctions (lru_cache) pour libérer la mémoire."""
    construire_horaire_mission_cached.cache_clear()