# -*- coding: utf-8 -*-
"""
energy_logic.py

Ce module contient la logique de calcul de la consommation d'énergie.
Il est piloté par les horaires (temps planifié) fournis par core_logic.
Il calcule le profil de vitesse le plus rapide possible correspondant
à la vitesse moyenne de l'horaire et estime la consommation
en utilisant un profil de vitesse trapézoïdal et l'équation de Davis.

"""

import pandas as pd
from datetime import timedelta
import numpy as np
import math # Importer math pour isclose

# Constantes physiques
JOULES_PER_KWH = 3_600_000
GRAVITY_MS2 = 9.81
DEFAULT_V_MAX_KPH = 160.0 # Vitesse max par défaut si non calculable

def get_default_energy_params():
    """
    Retourne une structure de paramètres par défaut pour l'énergie,
    basée sur les analyses et la nouvelle physique (Davis).
    """
    return {
        "masse_tonne": 100,
        "capacite_batterie_kwh": 600, # Basé sur l'analyse train actuel
        "facteur_charge_C": 4.0, # Facteur de charge max (XC)
        "recuperation_pct": 65, # Basé sur l'analyse

        # Coeff de Davis (Force en Newtons par tonne)
        "davis_A_N_t": 20.0,  # Résistance mécanique (N/t)
        "davis_B_N_t_kph": 0.5, # Résistance roulements (N/t/kph)
        "davis_C_N_t_kph2": 0.005, # Résistance aéro (N/t/kph^2)

        # Performance (caractéristiques du matériel)
        "accel_ms2": 0.5,
        "decel_ms2": 0.8,

        "facteur_aux_kwh_h": 50.0,

        # Rendements (basés sur l'analyse)
        "rendement_thermique_pct": 38,
        "rendement_electrique_pct": 88,

        "kwh_per_liter_diesel": 10.0,
    }

# =============================================================================
# FONCTION UNIFIÉE DE CALCUL DU PROFIL PHYSIQUE (Inchangée V18)
# =============================================================================
def _calculate_phases(distance_m, v_start_kph, v_end_kph, v_cruise_kph, accel_ms2, decel_ms2):
    """
    Calcule le profil physique le plus RAPIDE possible pour une Vcruise donnée.
    Retourne les distances (m), temps (s) initiaux et v_moy (kph) pour chaque phase.
    """
    # ... (code _calculate_phases V15 inchangé) ...
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
        # Profil triangulaire
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
        # Profil trapézoïdal
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

    # Appeler _calculate_phases avec la v_cruise cible
    phases = _calculate_phases(distance_m, v_start_kph, v_end_kph, v_cruise_kph, accel_ms2, decel_ms2)

    # Exclure 'total_time_sec' du dictionnaire retourné
    return {k: v for k, v in phases.items() if k != 'total_time_sec'}





# =============================================================================
# Logique de calcul de consommation
# =============================================================================

def _calculer_force_resistance_davis_N(v_kph, masse_t, params):
    """Calcule la force de résistance (en Newtons) via l'équation de Davis."""
    A = params.get("davis_A_N_t", 20.0)
    B = params.get("davis_B_N_t_kph", 0.5)
    C = params.get("davis_C_N_t_kph2", 0.005)
    force_par_tonne_N = A + (B * abs(v_kph)) + (C * v_kph**2)
    return force_par_tonne_N * masse_t

def calculer_consommation_trajet(trajets_train, mission, df_gares, energy_params):
    """
    Calcule la consommation d'énergie pour la chronologie complète d'un train.
    Calcule le profil physique le plus rapide basé sur la vitesse moyenne horaire.
    """
    if not trajets_train:
        return {"total_kwh": 0, "batterie_log": [], "erreurs": []}

    params = energy_params
    type_materiel = mission.get("type_materiel", "diesel")
    masse_t = params.get("masse_tonne", 100)
    masse_kg = masse_t * 1000
    total_conso_brute_kwh = 0
    total_recup_kwh = 0
    erreurs = []
    log_batterie = []
    total_distance_km = 0
    total_conso_thermique_kwh = 0
    total_conso_electrique_kwh = 0

    is_batterie = type_materiel == "batterie"
    capacite_max_kwh = params.get("capacite_batterie_kwh", 600)
    facteur_charge_C = params.get("facteur_charge_C", 4.0)
    puissance_max_batterie_kw = capacite_max_kwh * facteur_charge_C
    niveau_batterie_kwh = capacite_max_kwh
    recup_pct = params.get("recuperation_pct", 65) / 100.0
    kwh_per_liter = params.get("kwh_per_liter_diesel", 10.0)
    facteur_aux_kwh_h = params.get("facteur_aux_kwh_h", 50.0)
    accel_ms2 = params.get("accel_ms2", 0.5)
    decel_ms2 = params.get("decel_ms2", 0.8)

    try:
        gares_info = df_gares.set_index('gare').to_dict('index')
    except KeyError:
        return {"total_kwh": 0, "batterie_log": [], "erreurs": ["Format du DataFrame des gares incorrect."]}

    if not trajets_train: return {"total_kwh": 0, "batterie_log": [], "erreurs": ["Trajets vides"]}

    v_precedente_kph = 0

    for i, trajet in enumerate(trajets_train):
        gare_depart_nom = trajet["origine"]
        gare_arrivee_nom = trajet["terminus"]
        duree_planifiee_h = max(0, (trajet["end"] - trajet["start"]).total_seconds() / 3600.0)
        duree_planifiee_sec = duree_planifiee_h * 3600.0

        info_depart = gares_info.get(gare_depart_nom)
        if not info_depart:
            erreurs.append(f"Gare départ {gare_depart_nom} non trouvée.")
            continue

        info_arrivee = gares_info.get(gare_arrivee_nom, {})
        distance_km = abs(info_arrivee.get('distance', info_depart['distance']) - info_depart['distance'])
        distance_m = distance_km * 1000

        conso_aux_segment_kwh = 0
        conso_moteur_kwh = 0
        recup_segment_kwh = 0
        source_moteur = "aucune"
        duree_physique_h = 0 # Durée REELLE du mouvement
        electrification_arret = "F"
        profil_physique = {} # Profil utilisé pour calcul conso

        if duree_planifiee_h > 0 and distance_km == 0: # Arrêt
            source_moteur = "arret"
            v_finale_kph_segment = 0
            # La durée physique est la durée de l'arrêt planifié
            duree_physique_h = duree_planifiee_h
            conso_aux_segment_kwh = facteur_aux_kwh_h * duree_physique_h
            profil_physique = {"accel": (0,0,0), "cruise": (0,0,0), "decel": (0,0,0)}

            if is_batterie:
                electrification_arret = info_arrivee.get("electrification", "F").upper()
                if electrification_arret.startswith("R"):
                    try:
                        puissance_source_kw = int(electrification_arret[1:])
                        puissance_recharge_nette_kw = min(puissance_source_kw, puissance_max_batterie_kw)
                        energie_rechargee = puissance_recharge_nette_kw * duree_physique_h
                        niveau_batterie_kwh += energie_rechargee
                        log_batterie.append((trajet["end"], niveau_batterie_kwh, f"Recharge {puissance_recharge_nette_kw:.0f}kW à {gare_arrivee_nom}"))
                    except ValueError:
                        erreurs.append(f"Format recharge invalide: {electrification_arret}")


        elif duree_planifiee_h > 0 and distance_km > 0: # Mouvement
            total_distance_km += distance_km
            # Les auxiliaires consomment pendant TOUTE la durée planifiée
            conso_aux_segment_kwh = facteur_aux_kwh_h * duree_planifiee_h

            v_initiale_kph = v_precedente_kph
            is_stop_after = False
            if i + 1 < len(trajets_train):
                trajet_suivant = trajets_train[i+1]
                if trajet_suivant["origine"] == trajet_suivant["terminus"] and trajet_suivant["origine"] == gare_arrivee_nom:
                    is_stop_after = True
            else: is_stop_after = True

            v_finale_kph_segment_target = 0 if is_stop_after else v_initiale_kph # Cible physique

            # --- Calcul Vitesse Croisière Implicite ---
            v_avg_implied_kph = (distance_m / duree_planifiee_sec) * 3.6 if duree_planifiee_sec > 0 else 0
            # Utiliser une vitesse de croisière légèrement supérieure à la moyenne, bornée
            v_cruise_implied_kph = max(v_avg_implied_kph * 1.1, v_initiale_kph, v_finale_kph_segment_target, 1.0)
            v_cruise_implied_kph = min(v_cruise_implied_kph, DEFAULT_V_MAX_KPH) # Borner par Vmax
            # --- Fin Calcul Vitesse Croisière ---

            v_finale_kph_segment_real = 0 if is_stop_after else v_cruise_implied_kph # Vitesse à la fin du mouvement

            # Obtenir le profil physique le plus RAPIDE possible pour cette Vcruise implicite
            profil_physique = get_physical_profile(
                 distance_m, v_initiale_kph, v_finale_kph_segment_real,
                 v_cruise_implied_kph, accel_ms2, decel_ms2
                 # Pas besoin de passer temps_planifie_sec ici
            )
            # Durée physique REELLE du mouvement (peut être < duree_planifiee_sec)
            duree_physique_sec = sum(t for _, t, _ in profil_physique.values())
            duree_physique_h = max(0, duree_physique_sec / 3600.0) # Assurer >= 0

            electrification = info_depart.get("electrification", "F").upper()
            rampe_pourmille = info_depart.get("rampe_section_a_venir", 0)
            is_catenary = electrification in ["C1500", "C25"]

            mode_traction = "thermique"
            # ... (logique mode_traction inchangée) ...
            if type_materiel == "electrique":
                mode_traction = "electrique" if is_catenary else "aucun"
                if not is_catenary: erreurs.append(f"Train elec sur section non élec ({gare_depart_nom} -> {gare_arrivee_nom})")
            elif type_materiel == "bimode":
                mode_traction = "electrique" if is_catenary else "thermique"
            elif type_materiel == "batterie":
                mode_traction = "electrique"
            elif type_materiel == "diesel":
                mode_traction = "thermique"

            if mode_traction == "electrique":
                rendement = params.get("rendement_electrique_pct", 88) / 100.0
                source_moteur = "electrique"
            elif mode_traction == "thermique":
                rendement = params.get("rendement_thermique_pct", 38) / 100.0
                source_moteur = "thermique"
            else: rendement = 0.01; source_moteur = "aucun"

            # Bilan énergétique Mouvement (basé sur profil_physique)
            if source_moteur != "aucun":
                e_moteur_J, e_recup_potential_J = 0, 0
                delta_h_m = distance_m * (rampe_pourmille / 1000.0)
                e_pente_J = masse_kg * GRAVITY_MS2 * delta_h_m
                v_start_ms = v_initiale_kph / 3.6
                v_end_ms = v_finale_kph_segment_real / 3.6
                e_cinetique_delta_J = 0.5 * masse_kg * (max(0, v_end_ms**2) - max(0, v_start_ms**2))

                e_resist_total_J = 0
                for phase, (d_m, t_s, v_avg_kph) in profil_physique.items():
                    if d_m > 0 and v_avg_kph > 0:
                        force_resist_N = _calculer_force_resistance_davis_N(v_avg_kph, masse_t, params)
                        e_resist_total_J += force_resist_N * d_m

                e_meca_nette_J = e_resist_total_J + e_pente_J + e_cinetique_delta_J
                if e_meca_nette_J > 0: e_moteur_J = e_meca_nette_J / rendement
                else: e_recup_potential_J = abs(e_meca_nette_J)

                conso_moteur_kwh = e_moteur_J / JOULES_PER_KWH
                recup_segment_kwh = (e_recup_potential_J * recup_pct) / JOULES_PER_KWH

            # La consommation moteur n'a lieu QUE pendant duree_physique_h
            total_conso_brute_kwh += conso_moteur_kwh
            total_recup_kwh += recup_segment_kwh
            v_finale_kph_segment = v_finale_kph_segment_real

        else: v_finale_kph_segment = v_precedente_kph

        # --- Bilan conso et Batterie ---
        # Auxiliaires déjà ajoutés pour la durée PLANIFIEE
        total_conso_brute_kwh += conso_aux_segment_kwh

        source_aux = "thermique"
        is_catenary_seg = gares_info.get(gare_depart_nom, {}).get("electrification", "F").upper() in ["C1500", "C25"]
        if type_materiel in ["electrique", "batterie"] or (type_materiel == "bimode" and is_catenary_seg):
            source_aux = "electrique"

        # Imputer la conso des auxiliaires (sur la durée planifiée)
        if source_aux == "electrique": total_conso_electrique_kwh += conso_aux_segment_kwh
        else: total_conso_thermique_kwh += conso_aux_segment_kwh

        # Imputer la conso moteur (sur la durée physique)
        if source_moteur == "electrique": total_conso_electrique_kwh += conso_moteur_kwh
        elif source_moteur == "thermique": total_conso_thermique_kwh += conso_moteur_kwh

        if is_batterie:
            # Conso = Moteur (pendant t_phys) + Aux (pendant t_planifié)
            # Si le train attend implicitement (t_planifié > t_phys),
            # la batterie doit fournir les Aux pendant ce temps d'attente.
            conso_totale_segment_kwh = conso_moteur_kwh + conso_aux_segment_kwh
            energie_rechargee_kwh = 0
            log_txt = ""

            if is_catenary_seg and distance_km > 0: # Recharge caténaire
                puissance_max_source_kw = 4000 if electrification == "C1500" else (6000 if electrification == "C25" else 0)
                # Puissance consommée MOYENNE pendant le mouvement réel
                puissance_consommee_moteur_kw = (conso_moteur_kwh / duree_physique_h) if duree_physique_h > 0 else 0
                puissance_consommee_aux_kw = facteur_aux_kwh_h # Auxiliaires constants
                puissance_consommee_tot_kw = puissance_consommee_moteur_kw + puissance_consommee_aux_kw

                puissance_recharge_dispo_source_kw = max(0, puissance_max_source_kw - puissance_consommee_tot_kw)
                puissance_recharge_nette_kw = min(puissance_recharge_dispo_source_kw, puissance_max_batterie_kw)
                # Recharger pendant la durée physique réelle du mouvement
                energie_rechargee_kwh = puissance_recharge_nette_kw * duree_physique_h

                # La batterie gagne recharge + récup. Elle ne fournit rien.
                niveau_batterie_kwh += energie_rechargee_kwh + recup_segment_kwh
                log_txt = f"Grid (Charge {puissance_recharge_nette_kw:.0f}kW) {gare_depart_nom}->{gare_arrivee_nom}"
            else: # Pas caténaire ou arrêt
                # La batterie fournit conso_moteur (pdt t_phys) + conso_aux (pdt t_planifié)
                niveau_batterie_kwh -= conso_totale_segment_kwh
                niveau_batterie_kwh += recup_segment_kwh
                if source_moteur == "arret":
                    if not (electrification_arret.startswith("R") and duree_planifiee_h > 0):
                        log_txt = f"Arrêt à {gare_depart_nom}"
                elif distance_km > 0:
                    log_txt = f"Batterie {gare_depart_nom}->{gare_arrivee_nom}"

            niveau_batterie_kwh = min(niveau_batterie_kwh, capacite_max_kwh)
            if log_txt: log_batterie.append((trajet["end"], niveau_batterie_kwh, log_txt))
            if niveau_batterie_kwh < -0.1:
                erreurs.append(f"Batterie vide! ({niveau_batterie_kwh:.1f} kWh) à {gare_arrivee_nom} {trajet['end']:%H:%M}")


        v_precedente_kph = v_finale_kph_segment

    # --- Bilan final ---
    total_litres_diesel = total_conso_thermique_kwh / kwh_per_liter if kwh_per_liter > 0 else 0

    return {
        "total_conso_brute_kwh": total_conso_brute_kwh,
        "total_recup_kwh": total_recup_kwh,
        "total_conso_nette_kwh": total_conso_brute_kwh - total_recup_kwh,
        "total_conso_electrique_kwh": total_conso_electrique_kwh,
        "total_conso_thermique_kwh": total_conso_thermique_kwh,
        "total_litres_diesel": total_litres_diesel,
        "total_distance_km": total_distance_km,
        "batterie_log": log_batterie,
        "erreurs": erreurs
    }

