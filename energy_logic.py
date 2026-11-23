# -*- coding: utf-8 -*-
"""
energy_logic.py
===============

Module de calcul de la consommation énergétique des trains.

Ce module modélise la physique du mouvement ferroviaire pour estimer la consommation
d'énergie en fonction des horaires, de l'infrastructure et des caractéristiques du matériel roulant.

Méthodologie :
1.  **Profil de Vitesse** : Reconstitution d'un profil de vitesse réaliste (accélération, croisière, freinage) 
    pour respecter les temps de parcours de la grille horaire.
2.  **Résistance à l'avancement** : Utilisation de l'équation de Davis ($A + Bv + Cv^2$).
3.  **Bilan Énergétique** : Intégration de la puissance instantanée pour calculer l'énergie consommée (traction)
    et régénérée (freinage).
4.  **Batterie** : Simulation de l'état de charge (SoC) pour les trains à batterie ou hybrides.

Fonctions principales :
- `calculer_consommation_trajet` : Calcule la consommation pour un trajet complet.
- `get_physical_profile` : Génère les courbes de vitesse/position/énergie.
"""

import math
import numpy as np
import pandas as pd
from datetime import timedelta

JOULES_PER_KWH = 3_600_000
GRAVITY_MS2 = 9.81
DEFAULT_V_MAX_KPH = 160.0

def get_default_energy_params():
    """
    Retourne les paramètres énergétiques par défaut (Matériel TER standard).
    
    Returns:
        dict: Dictionnaire contenant :
            - Caractéristiques physiques (masse, coeff Davis).
            - Performances (accélération, décélération).
            - Paramètres électriques (rendement, batterie, auxiliaires).
    """
    return {
        "masse_tonne": 50,
        "capacite_batterie_kwh": 800,
        "facteur_charge_C": 1.0,
        "recuperation_pct": 75,
        "soc_min_pct": 20,
        "soc_max_pct": 95,
        "capacite_eol_pct": 70,
        "simuler_fin_de_vie": False,

        "davis_A_N_t": 15.0,
        "davis_B_N_t_kph": 0.1,
        "davis_C_N_t_kph2": 0.0041,

        "accel_ms2": 1.0,
        "decel_ms2": 0.6,

        "facteur_aux_kwh_h": 43.0,
        "taux_aux_terminus_pct": 100,

        "rendement_thermique_pct": 38,
        "rendement_electrique_pct": 88,

        "kwh_per_liter_diesel": 10.0,
    }

def _calculate_phases(distance_m, v_start_kph, v_end_kph, v_cruise_kph, accel_ms2, decel_ms2):
    """
    Calcule le profil physique le plus RAPIDE possible pour une Vcruise donnée.
    Retourne les distances (m), temps (s) initiaux et v_moy (kph) pour chaque phase.
    """
    v_start_ms = max(0, v_start_kph / 3.6)
    v_end_ms = max(0, v_end_kph / 3.6)
    v_cruise_kph = max(v_cruise_kph, v_start_kph, v_end_kph, 0.1)
    v_cruise_ms = v_cruise_kph / 3.6
    accel_ms2 = max(0.01, accel_ms2)
    decel_ms2 = max(0.01, decel_ms2)

    t_to_cruise = (v_cruise_ms - v_start_ms) / accel_ms2 if v_cruise_ms > v_start_ms else 0
    d_to_cruise = v_start_ms * t_to_cruise + 0.5 * accel_ms2 * t_to_cruise**2
    t_from_cruise = (v_cruise_ms - v_end_ms) / decel_ms2 if v_cruise_ms > v_end_ms else 0
    d_from_cruise = v_cruise_ms * t_from_cruise - 0.5 * decel_ms2 * t_from_cruise**2

    dist_accel_decel = d_to_cruise + d_from_cruise
    if dist_accel_decel >= distance_m and not math.isclose(dist_accel_decel, distance_m):
        try:
            numerator = (2 * distance_m * accel_ms2 * decel_ms2 +
                         v_start_ms**2 * decel_ms2 +
                         v_end_ms**2 * accel_ms2)
            denominator = accel_ms2 + decel_ms2
            if numerator < 0 or denominator <= 0: raise ValueError("Calcul v_peak invalide")
            v_peak_sq = numerator / denominator
            v_peak_ms = np.sqrt(v_peak_sq)
            v_peak_ms = max(v_peak_ms, v_start_ms, v_end_ms)

            t_accel = (v_peak_ms - v_start_ms) / accel_ms2 if v_peak_ms > v_start_ms else 0
            d_accel = v_start_ms * t_accel + 0.5 * accel_ms2 * t_accel**2
            v_avg_accel = (v_start_kph + v_peak_ms * 3.6) / 2

            t_decel = (v_peak_ms - v_end_ms) / decel_ms2 if v_peak_ms > v_end_ms else 0
            d_decel = max(0, distance_m - d_accel)
            v_avg_decel = (v_peak_ms * 3.6 + v_end_kph) / 2

            t_cruise, d_cruise, v_avg_cruise = 0, 0, 0
        except (ValueError, ZeroDivisionError, RuntimeWarning) as e:
             v_avg_kph_fallback = max(v_start_kph, v_end_kph, 1)
             t_total_fallback = (distance_m / (v_avg_kph_fallback / 3.6)) if v_avg_kph_fallback > 0 else 3600*24
             return {"accel": (distance_m, t_total_fallback, v_avg_kph_fallback),
                     "cruise": (0, 0, 0), "decel": (0, 0, 0),
                     "total_time_sec": t_total_fallback}
    else:
        t_accel = t_to_cruise
        d_accel = d_to_cruise
        v_avg_accel = (v_start_kph + v_cruise_kph) / 2
        t_decel = t_from_cruise
        d_decel = d_from_cruise
        v_avg_decel = (v_cruise_kph + v_end_kph) / 2
        d_cruise = max(0, distance_m - d_accel - d_decel)
        t_cruise = (d_cruise / v_cruise_ms) if v_cruise_ms > 0 and d_cruise > 0 else 0
        v_avg_cruise = v_cruise_kph

    total_time_sec = t_accel + t_cruise + t_decel
    dist_check = d_accel + d_cruise + d_decel
    if not math.isclose(dist_check, distance_m, rel_tol=1e-5):
         diff = distance_m - dist_check
         if t_cruise >= t_accel and t_cruise >= t_decel and t_cruise > 0:
              d_cruise += diff
              t_cruise = (d_cruise / v_cruise_ms) if v_cruise_ms > 0 and d_cruise > 0 else 0
         elif t_decel >= t_accel and t_decel > 0: d_decel += diff
         elif t_accel > 0 : d_accel += diff
         total_time_sec = t_accel + t_cruise + t_decel

    return {"accel": (d_accel, t_accel, v_avg_accel),
            "cruise": (d_cruise, t_cruise, v_avg_cruise),
            "decel": (d_decel, t_decel, v_avg_decel),
            "total_time_sec": total_time_sec}



def get_physical_profile(distance_m, v_start_kph, v_end_kph, v_cruise_kph, accel_ms2, decel_ms2):
    """
    Calcule le profil physique (distances, temps, v_moy) le plus RAPIDE possible
    pour atteindre la v_cruise_kph donnée.
    Ne prend plus temps_planifie_sec en argument.
    """
    if distance_m <= 0:
        return {"accel": (0, 0, 0), "cruise": (0, 0, 0), "decel": (0, 0, 0)}

    phases = _calculate_phases(distance_m, v_start_kph, v_end_kph, v_cruise_kph, accel_ms2, decel_ms2)

    return {k: v for k, v in phases.items() if k != 'total_time_sec'}


def find_implicit_v_cruise(distance_m, v_start_kph, v_end_kph,
                            accel_ms2, decel_ms2, t_target_sec,
                            v_max_kph=DEFAULT_V_MAX_KPH):
    """
    Recherche la vitesse de croisière v_cruise (kph) telle que la durée totale
    du profil physique (accélération + croisière + décélération) égale
    `t_target_sec`. Cohérence horaire/énergie : la résistance à l'avancement
    sera intégrée à une vitesse compatible avec le temps planifié.

    Bisection sur v_cruise (relation t_total monotone décroissante avec v).
    Si même à v_max_kph la durée reste > t_target_sec, retourne v_max_kph
    (le train ne peut physiquement pas tenir l'horaire).
    """
    if distance_m <= 0 or t_target_sec <= 0:
        return max(v_start_kph, v_end_kph, 10.0)

    phases_max = _calculate_phases(distance_m, v_start_kph, v_end_kph,
                                   v_max_kph, accel_ms2, decel_ms2)
    if phases_max["total_time_sec"] >= t_target_sec:
        return v_max_kph

    v_low = max(v_start_kph, v_end_kph, 0.1)
    phases_low = _calculate_phases(distance_m, v_start_kph, v_end_kph,
                                   v_low, accel_ms2, decel_ms2)
    if phases_low["total_time_sec"] <= t_target_sec:
        return v_low

    v_high = v_max_kph
    for _ in range(50):
        v_mid = 0.5 * (v_low + v_high)
        phases_mid = _calculate_phases(distance_m, v_start_kph, v_end_kph,
                                       v_mid, accel_ms2, decel_ms2)
        if phases_mid["total_time_sec"] > t_target_sec:
            v_low = v_mid
        else:
            v_high = v_mid
        if (v_high - v_low) < 0.05:
            break

    return 0.5 * (v_low + v_high)


def find_v_cruise_no_decel(distance_m, v_start_kph, accel_ms2, t_target_sec,
                            v_max_kph=DEFAULT_V_MAX_KPH):
    """
    Trouve la vitesse de croisière pour un segment qui se TERMINE en plein
    mouvement (transition vers un point de passage sans arrêt) : profil
    accélération + croisière, sans phase de décélération finale.

    Le train entre à v_start_kph, accélère jusqu'à v_cruise puis croise
    jusqu'au bout. v_end == v_cruise. Sortie : v_cruise_kph (clampée).

    Résolution : ``0.5*(v_s + v_c)*(v_c - v_s)/a + v_c*(t - (v_c - v_s)/a) = d``
    se réécrit ``v_c² - 2*v_c*(v_s + t*a) + (v_s² + 2*d*a) = 0``. On prend la
    plus petite racine (l'autre est non physique : v > v_max).
    """
    if distance_m <= 0 or t_target_sec <= 0:
        return max(v_start_kph, 0.1)
    v_start_ms = max(0.0, v_start_kph / 3.6)
    a = max(0.01, accel_ms2)
    b = v_start_ms + t_target_sec * a
    discriminant = b * b - (v_start_ms * v_start_ms + 2.0 * distance_m * a)
    if discriminant < 0:
        return v_max_kph
    v_c_ms = b - math.sqrt(discriminant)
    v_c_kph = v_c_ms * 3.6
    if v_c_kph < v_start_kph:
        return v_start_kph
    if v_c_kph > v_max_kph:
        return v_max_kph
    return v_c_kph



def _calculer_force_resistance_davis_N(v_kph, masse_t, params):
    """Calcule la force de résistance (en Newtons) via l'équation de Davis."""
    A = params.get("davis_A_N_t", 20.0)
    B = params.get("davis_B_N_t_kph", 0.5)
    C = params.get("davis_C_N_t_kph2", 0.005)
    force_par_tonne_N = A + (B * abs(v_kph)) + (C * v_kph**2)
    return force_par_tonne_N * masse_t

def _get_puissance_infra(electrification_str):
    """Parse l'infrastructure pour renvoyer la puissance disponible en kW."""
    e_str = str(electrification_str).strip().upper()
    if e_str == "C1500": return 4000
    if e_str == "C25": return 6000
    if e_str.startswith("R"):
        try:
            return int(e_str[1:])
        except ValueError:
            return 0
    return 0

def calculer_consommation_trajet(trajets_train, mission, df_gares, energy_params):
    """
    Calcule la consommation avec gestion fine de la batterie (SoC, facteurs limitants, recharge statique).
    Gère explicitement les temps d'arrêt entre segments (recharge au terminus).
    """
    if not trajets_train:
        return {"total_kwh": 0, "batterie_log": [], "erreurs": []}

    params = energy_params
    type_materiel = mission.get("type_materiel", "diesel")
    masse_t = params.get("masse_tonne", 100)
    masse_kg = masse_t * 1000

    total_conso_brute_kwh = 0
    total_recup_kwh = 0
    total_conso_thermique_kwh = 0
    total_conso_electrique_kwh = 0
    total_conso_aux_kwh = 0
    total_conso_aux_electrique_kwh = 0
    total_conso_aux_thermique_kwh = 0
    total_distance_km = 0
    erreurs = []

    is_batterie = type_materiel == "batterie"
    capacite_nominale_kwh = params.get("capacite_batterie_kwh", 600)
    simuler_eol = params.get("simuler_fin_de_vie", False)
    capacite_eol_pct = params.get("capacite_eol_pct", 80)
    capacite_max_kwh = capacite_nominale_kwh * (capacite_eol_pct / 100.0) if simuler_eol else capacite_nominale_kwh
    soc_min_pct = params.get("soc_min_pct", 20) / 100.0
    soc_max_pct = params.get("soc_max_pct", 95) / 100.0
    niveau_min_kwh = capacite_max_kwh * soc_min_pct
    niveau_max_kwh = capacite_max_kwh * soc_max_pct
    facteur_charge_C = params.get("facteur_charge_C", 4.0)
    puissance_max_batterie_kw = capacite_max_kwh * facteur_charge_C
    niveau_batterie_kwh = niveau_max_kwh

    log_batterie = []

    recup_pct = params.get("recuperation_pct", 65) / 100.0
    kwh_per_liter = params.get("kwh_per_liter_diesel", 10.0)
    facteur_aux_kwh_h = params.get("facteur_aux_kwh_h", 50.0)
    try:
        taux_aux_terminus = max(0.0, min(100.0, float(params.get("taux_aux_terminus_pct", 100)))) / 100.0
    except (TypeError, ValueError):
        taux_aux_terminus = 1.0
    accel_ms2 = params.get("accel_ms2", 0.5)
    decel_ms2 = params.get("decel_ms2", 0.8)

    try:
        gares_info = df_gares.set_index('gare').to_dict('index')
    except KeyError:
        return {"total_kwh": 0, "batterie_log": [], "erreurs": ["Format gares incorrect"]}

    _df_gares_sorted = df_gares.sort_values('distance').reset_index(drop=True)
    _gares_dist = _df_gares_sorted['distance'].astype(float).tolist()
    _gares_rampe = (
        _df_gares_sorted['rampe_section_a_venir'].astype(float).tolist()
        if 'rampe_section_a_venir' in _df_gares_sorted.columns
        else [0.0] * len(_df_gares_sorted)
    )

    def _compute_delta_h_m(dist_dep_km, dist_arr_km):
        """
        Différence d'altitude (m) entre la gare de départ et la gare d'arrivée
        du trajet, en sommant la contribution de chaque sous-section traversée.

        Convention : rampe_section_a_venir d'une gare = pente (‰) de la section
        APRÈS cette gare dans le sens des kilométrages croissants. Pour un trajet
        en sens retour (kilométrage décroissant), le signe est inversé.
        """
        if dist_dep_km == dist_arr_km:
            return 0.0
        sens_increasing = dist_arr_km > dist_dep_km
        d_low = min(dist_dep_km, dist_arr_km)
        d_high = max(dist_dep_km, dist_arr_km)

        delta_h_inc = 0.0
        n = len(_gares_dist)
        for j in range(n - 1):
            d1 = _gares_dist[j]
            d2 = _gares_dist[j + 1]
            seg_lo = max(d1, d_low)
            seg_hi = min(d2, d_high)
            if seg_hi <= seg_lo:
                continue
            rampe_j = _gares_rampe[j] if j < len(_gares_rampe) else 0.0
            # (km) × (‰) → m  (car m = km*1000 × ‰/1000)
            delta_h_inc += (seg_hi - seg_lo) * float(rampe_j)

        return delta_h_inc if sens_increasing else -delta_h_inc

    def _termini_from_trajet(trajet):
        """Gares de retournement (origine/terminus de mission) pour un segment."""
        label = (trajet or {}).get("mission") or ""
        if " → " in label:
            a, b = label.split(" → ", 1)
            termini = {a.strip(), b.strip()}
            termini.discard("")
            if termini:
                return termini
        termini = set()
        for k in ("origine", "terminus"):
            v = mission.get(k)
            if v:
                termini.add(v)
        return termini

    def _est_arret_terminus(nom_gare, trajet=None):
        """True si le train est immobilisé à un terminus (sans voyageurs)."""
        return bool(nom_gare) and nom_gare in _termini_from_trajet(trajet)

    def _puissance_aux_kw(au_terminus=False):
        if au_terminus:
            return facteur_aux_kwh_h * taux_aux_terminus
        return facteur_aux_kwh_h

    def _source_aux_electrique(electrification_str):
        is_cat = str(electrification_str).strip().upper() in ["C1500", "C25"]
        return type_materiel in ["electrique", "batterie"] or (type_materiel == "bimode" and is_cat)

    def _comptabiliser_aux(conso_aux, electrification_str, force_electrique=False):
        nonlocal total_conso_brute_kwh, total_conso_electrique_kwh, total_conso_thermique_kwh
        nonlocal total_conso_aux_kwh, total_conso_aux_electrique_kwh, total_conso_aux_thermique_kwh
        if conso_aux <= 0:
            return
        total_conso_brute_kwh += conso_aux
        total_conso_aux_kwh += conso_aux
        if force_electrique or _source_aux_electrique(electrification_str):
            total_conso_electrique_kwh += conso_aux
            total_conso_aux_electrique_kwh += conso_aux
        else:
            total_conso_thermique_kwh += conso_aux
            total_conso_aux_thermique_kwh += conso_aux

    def _ajouter_aux_arret(duree_h, electrification_str, au_terminus=False):
        if duree_h <= 0:
            return
        conso_aux = _puissance_aux_kw(au_terminus) * duree_h
        _comptabiliser_aux(conso_aux, electrification_str)

    def simuler_arret(nom_gare, heure_debut, heure_fin, niveau_actuel, contexte="Arrêt", au_terminus=False):
        duree_h = (heure_fin - heure_debut).total_seconds() / 3600.0
        if duree_h <= 0: return niveau_actuel

        info_gare = gares_info.get(nom_gare, {})
        electrification = info_gare.get("electrification", "F")
        puissance_infra_kw = _get_puissance_infra(electrification)

        p_aux_kw = _puissance_aux_kw(au_terminus)
        conso_aux = p_aux_kw * duree_h
        _comptabiliser_aux(conso_aux, electrification, force_electrique=True)

        nouveau_niveau = niveau_actuel
        log_msg = ""

        if puissance_infra_kw > 0:
            puissance_dispo_charge = max(0, puissance_infra_kw - p_aux_kw)

            limite_puissance = min(puissance_dispo_charge, puissance_max_batterie_kw)
            raison_limite_puissance = "Infra" if puissance_dispo_charge < puissance_max_batterie_kw else "Capacité Charge"

            energie_rechargee_potentielle = limite_puissance * duree_h

            espace_dispo = niveau_max_kwh - niveau_actuel

            if energie_rechargee_potentielle >= espace_dispo:
                energie_rechargee_reelle = espace_dispo
                raison_limite = "Batterie Pleine"
            else:
                energie_rechargee_reelle = energie_rechargee_potentielle
                raison_limite = raison_limite_puissance

            nouveau_niveau += energie_rechargee_reelle

            if energie_rechargee_reelle > 0.01:
                log_msg = f"{contexte} sous {electrification} | Conso. Aux: {conso_aux:.1f} kWh | ΔBatt: +{energie_rechargee_reelle:.1f} kWh [{raison_limite}]"
            else:
                log_msg = f"{contexte} sous {electrification} | Conso. Aux: {conso_aux:.1f} kWh | ΔBatt: 0 kWh (Batterie pleine)"

        else:
            nouveau_niveau -= conso_aux
            log_msg = f"{contexte} sans infra | Conso. Aux: {conso_aux:.1f} kWh | ΔBatt: -{conso_aux:.1f} kWh"

        nouveau_niveau = max(niveau_min_kwh, min(nouveau_niveau, niveau_max_kwh))

        pct = (nouveau_niveau / capacite_max_kwh) * 100
        p_charge_C = 0.0
        if duree_h > 0 and 'energie_rechargee_reelle' in locals() and energie_rechargee_reelle > 0:
            p_charge_C = (energie_rechargee_reelle / duree_h) / capacite_max_kwh
        log_batterie.append((heure_fin, nouveau_niveau, f"{pct:.1f}%", log_msg, p_charge_C, True, nom_gare))

        return nouveau_niveau


    ajouter_log_batt = lambda h, k, m, p=0.0, is_stat=False, gare="": log_batterie.append((h, k, f"{(k/capacite_max_kwh)*100:.1f}%", m, p, is_stat, gare))
    eol_label = f" [Fin de vie {capacite_eol_pct}%]" if simuler_eol else ""
    ajouter_log_batt(trajets_train[0]["start"], niveau_batterie_kwh, f"Départ Mission{eol_label}", 0.0, True, trajets_train[0].get("origine", ""))

    v_precedente_kph = 0

    for i, trajet in enumerate(trajets_train):
        gare_depart_nom = trajet["origine"]
        gare_arrivee_nom = trajet["terminus"]

        info_depart = gares_info.get(gare_depart_nom)
        info_arrivee = gares_info.get(gare_arrivee_nom, {})

        if not info_depart:
            erreurs.append(f"Gare {gare_depart_nom} inconnue")
            continue

        distance_km = abs(info_arrivee.get('distance', 0) - info_depart.get('distance', 0))
        distance_m = distance_km * 1000

        if distance_km == 0:
            au_terminus = _est_arret_terminus(gare_depart_nom, trajet)
            if is_batterie:
                niveau_batterie_kwh = simuler_arret(
                    gare_depart_nom, trajet["start"], trajet["end"], niveau_batterie_kwh,
                    contexte="Stationnement", au_terminus=au_terminus,
                )
            else:
                duree_stat_h = max(0, (trajet["end"] - trajet["start"]).total_seconds() / 3600.0)
                elec_stat = info_depart.get("electrification", "F")
                _ajouter_aux_arret(duree_stat_h, elec_stat, au_terminus=au_terminus)

        else:
            duree_planifiee_h = max(0, (trajet["end"] - trajet["start"]).total_seconds() / 3600.0)
            duree_planifiee_sec = duree_planifiee_h * 3600.0

            conso_aux_segment_kwh = facteur_aux_kwh_h * duree_planifiee_h
            total_conso_aux_kwh += conso_aux_segment_kwh
            total_distance_km += distance_km

            if info_arrivee.get('distance', 0) < info_depart.get('distance', 0):
                electrification_seg = info_arrivee.get("electrification", "F").upper()
            else:
                electrification_seg = info_depart.get("electrification", "F").upper()
            
            is_catenary_segment = electrification_seg in ["C1500", "C25"]
            puissance_infra_kw = _get_puissance_infra(electrification_seg)

            v_initiale_kph = v_precedente_kph
            v_finale_target = 0
            v_cruise = find_implicit_v_cruise(
                distance_m, v_initiale_kph, v_finale_target,
                accel_ms2, decel_ms2, duree_planifiee_sec
            )

            profil = get_physical_profile(distance_m, v_initiale_kph, v_finale_target, v_cruise, accel_ms2, decel_ms2)
            duree_physique_sec = sum(t for _, t, _ in profil.values())
            duree_physique_h = duree_physique_sec / 3600.0

            _, _, v_avg_decel = profil["decel"]
            v_finale_kph_segment = 0


            dist_dep_km = float(info_depart.get('distance', 0))
            dist_arr_km = float(info_arrivee.get('distance', 0))
            delta_h = _compute_delta_h_m(dist_dep_km, dist_arr_km)
            e_pente_J = masse_kg * GRAVITY_MS2 * delta_h

            v_start_ms = v_initiale_kph / 3.6
            v_end_ms = v_finale_kph_segment / 3.6
            e_cin_J = 0.5 * masse_kg * (max(0, v_end_ms**2) - max(0, v_start_ms**2))

            e_resist_J = 0
            for phase in profil.values():
                d, _, v_avg = phase
                if d > 0:
                    f_res = _calculer_force_resistance_davis_N(v_avg, masse_t, params)
                    e_resist_J += f_res * d

            e_total_necessaire_J = e_resist_J + e_pente_J + e_cin_J

            conso_traction_kwh = 0
            recup_possible_kwh = 0
            source_actuelle = "electrique" if type_materiel in ["electrique", "batterie"] or (type_materiel == "bimode" and is_catenary_segment) else "thermique"

            rdt = params.get("rendement_electrique_pct", 88) / 100.0 if source_actuelle == "electrique" else params.get("rendement_thermique_pct", 38) / 100.0

            if e_total_necessaire_J > 0:
                conso_traction_kwh = (e_total_necessaire_J / rdt) / JOULES_PER_KWH
            else:
                if type_materiel in ["electrique", "batterie", "bimode"]:
                    recup_possible_kwh = (abs(e_total_necessaire_J) * recup_pct) / JOULES_PER_KWH

            total_recup_kwh += recup_possible_kwh
            if source_actuelle == "electrique":
                total_conso_electrique_kwh += (conso_traction_kwh + conso_aux_segment_kwh)
                total_conso_aux_electrique_kwh += conso_aux_segment_kwh
            else:
                total_conso_thermique_kwh += (conso_traction_kwh + conso_aux_segment_kwh)
                total_conso_aux_thermique_kwh += conso_aux_segment_kwh
            total_conso_brute_kwh += (conso_traction_kwh + conso_aux_segment_kwh)

            if is_batterie:
                conso_totale_segment = conso_traction_kwh + conso_aux_segment_kwh
                if is_catenary_segment:
                    p_conso_train = (conso_totale_segment / duree_planifiee_h) if duree_planifiee_h > 0 else 0
                    p_dispo_charge = max(0, puissance_infra_kw - p_conso_train)

                    p_target = min(p_dispo_charge, puissance_max_batterie_kw)
                    e_charge_potential = p_target * duree_planifiee_h

                    space = niveau_max_kwh - niveau_batterie_kwh
                    gain_net = min(e_charge_potential + recup_possible_kwh, space)
                    niveau_batterie_kwh += gain_net

                    limite_txt = "Infra" if p_dispo_charge < puissance_max_batterie_kw else "Capacité Charge"
                    if math.isclose(niveau_batterie_kwh, niveau_max_kwh): limite_txt = "Batterie Pleine"

                    log_msg = f"Traction sous caténaire ({electrification_seg}) | {distance_km:.1f} km | Conso. train: {conso_totale_segment:.1f} kWh | ΔBatt: +{gain_net:.1f} kWh [{limite_txt}]"
                    p_charge_C = 0.0
                    if duree_planifiee_h > 0 and gain_net > 0:
                        p_charge_C = (gain_net / duree_planifiee_h) / capacite_max_kwh
                else:
                    niveau_batterie_kwh -= conso_totale_segment
                    niveau_batterie_kwh += recup_possible_kwh
                    delta_batt = conso_totale_segment - recup_possible_kwh
                    log_msg = f"Traction sur batterie | {distance_km:.1f} km | Conso. train: {conso_totale_segment:.1f} kWh, Récup: +{recup_possible_kwh:.1f} kWh | ΔBatt: -{delta_batt:.1f} kWh"
                    p_charge_C = 0.0

                niveau_batterie_kwh = max(niveau_min_kwh, min(niveau_batterie_kwh, niveau_max_kwh))
                ajouter_log_batt(trajet["end"], niveau_batterie_kwh, log_msg, p_charge_C, False, "")

                if niveau_batterie_kwh <= niveau_min_kwh + 0.1:
                    erreurs.append(f"Batterie au seuil minimum à {gare_arrivee_nom}!")

            v_precedente_kph = v_finale_kph_segment

        if i < len(trajets_train) - 1:
            next_trajet = trajets_train[i+1]
            gap_seconds = (next_trajet["start"] - trajet["end"]).total_seconds()

            if gap_seconds > 60:
                gare_attente = trajet["terminus"]
                au_terminus = _est_arret_terminus(gare_attente, trajet)
                if is_batterie:
                    niveau_batterie_kwh = simuler_arret(
                        gare_attente,
                        trajet["end"],
                        next_trajet["start"],
                        niveau_batterie_kwh,
                        contexte="Attente/Terminus" if au_terminus else "Attente",
                        au_terminus=au_terminus,
                    )
                else:
                    gap_h = gap_seconds / 3600.0
                    info_term = gares_info.get(gare_attente, {})
                    elec_term = info_term.get("electrification", "F")
                    _ajouter_aux_arret(gap_h, elec_term, au_terminus=au_terminus)

    total_litres_diesel = total_conso_thermique_kwh / kwh_per_liter if kwh_per_liter > 0 else 0
    total_aux_litres_diesel = total_conso_aux_thermique_kwh / kwh_per_liter if kwh_per_liter > 0 else 0

    return {
        "total_conso_brute_kwh": total_conso_brute_kwh,
        "total_recup_kwh": total_recup_kwh,
        "total_conso_nette_kwh": total_conso_brute_kwh - total_recup_kwh,
        "total_conso_electrique_kwh": total_conso_electrique_kwh,
        "total_conso_thermique_kwh": total_conso_thermique_kwh,
        "total_conso_aux_kwh": total_conso_aux_kwh,
        "total_conso_aux_electrique_kwh": total_conso_aux_electrique_kwh,
        "total_conso_aux_thermique_kwh": total_conso_aux_thermique_kwh,
        "total_litres_diesel": total_litres_diesel,
        "total_aux_litres_diesel": total_aux_litres_diesel,
        "total_distance_km": total_distance_km,
        "batterie_log": log_batterie,
        "erreurs": erreurs
    }


def associer_mission_au_train(trajets, missions):
    """Trouve la mission correspondant au premier trajet d'une rame."""
    if not trajets:
        return None
    premier = trajets[0]
    orig = premier.get("origine")
    for mission in missions:
        if mission.get("origine") == orig:
            return mission
    for mission in missions:
        if mission.get("terminus") == orig:
            return mission
    return None


def calculer_energie_flotte(chronologie, missions, df_gares, energy_params):
    """Calcule l'energie de toutes les rames sans retoucher la grille horaire.

    Returns:
        (resultats_par_train, missions_par_train, erreurs)
        resultats_par_train[id] = (dict_conso, type_materiel)
    """
    resultats = {}
    missions_par_train = {}
    erreurs = []
    if not chronologie:
        return resultats, missions_par_train, erreurs
    for id_train, trajets in chronologie.items():
        if not trajets:
            continue
        mission = associer_mission_au_train(trajets, missions)
        if not mission:
            orig = trajets[0].get("origine", "?")
            erreurs.append(
                f"Impossible de trouver la mission pour le Train {id_train} "
                f"(demarrant a {orig}). Calcul energetique ignore."
            )
            continue
        missions_par_train[id_train] = mission
        type_mat = mission.get("type_materiel", "diesel")
        params_mat = (energy_params or {}).get(type_mat)
        if not params_mat:
            erreurs.append(
                f"Train {id_train}: parametres materiel '{type_mat}' absents, valeurs par defaut."
            )
            params_mat = get_default_energy_params()
        resultat = calculer_consommation_trajet(trajets, mission, df_gares, params_mat)
        resultats[id_train] = (resultat, type_mat)
        for err in resultat.get("erreurs") or []:
            erreurs.append(f"Train {id_train}: {err}")
    return resultats, missions_par_train, erreurs