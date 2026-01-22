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
        "masse_tonne": 50,
        "capacite_batterie_kwh": 800, # Basé sur l'analyse train actuel
        "facteur_charge_C": 4.0, # Facteur de charge max (XC)
        "recuperation_pct": 65, # Basé sur l'analyse

        # Coeff de Davis (Force en Newtons par tonne)
        "davis_A_N_t": 16.0  # Résistance mécanique (N/t)
        "davis_B_N_t_kph": 0.115, # Résistance roulements (N/t/kph)
        "davis_C_N_t_kph2": 0.0042, # Résistance aéro (N/t/kph^2)

        # Performance (caractéristiques du matériel)
        "accel_ms2": 0.5,
        "decel_ms2": 0.8,

        "facteur_aux_kwh_h": 20.0,

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

    # Accumulateurs globaux
    total_conso_brute_kwh = 0
    total_recup_kwh = 0
    total_conso_thermique_kwh = 0
    total_conso_electrique_kwh = 0
    total_distance_km = 0
    erreurs = []

    # Gestion Batterie
    is_batterie = type_materiel == "batterie"
    capacite_max_kwh = params.get("capacite_batterie_kwh", 600)
    facteur_charge_C = params.get("facteur_charge_C", 4.0)
    puissance_max_batterie_kw = capacite_max_kwh * facteur_charge_C
    niveau_batterie_kwh = capacite_max_kwh # Départ à 100%

    log_batterie = []

    recup_pct = params.get("recuperation_pct", 65) / 100.0
    kwh_per_liter = params.get("kwh_per_liter_diesel", 10.0)
    facteur_aux_kwh_h = params.get("facteur_aux_kwh_h", 50.0)
    accel_ms2 = params.get("accel_ms2", 0.5)
    decel_ms2 = params.get("decel_ms2", 0.8)

    try:
        gares_info = df_gares.set_index('gare').to_dict('index')
    except KeyError:
        return {"total_kwh": 0, "batterie_log": [], "erreurs": ["Format gares incorrect"]}

    # --- Helper: Gestion de l'arrêt (Recharge ou Décharge Aux) ---
    def simuler_arret(nom_gare, heure_debut, heure_fin, niveau_actuel, contexte="Arrêt"):
        duree_h = (heure_fin - heure_debut).total_seconds() / 3600.0
        if duree_h <= 0: return niveau_actuel

        info_gare = gares_info.get(nom_gare, {})
        electrification = info_gare.get("electrification", "F")
        puissance_infra_kw = _get_puissance_infra(electrification)

        # Conso auxiliaires
        conso_aux = facteur_aux_kwh_h * duree_h

        # Logique de recharge
        nouveau_niveau = niveau_actuel
        log_msg = ""

        # Si infrastructure de recharge disponible (Catenary ou Borne)
        if puissance_infra_kw > 0:
            # Les auxiliaires sont pris sur l'infra
            puissance_dispo_charge = max(0, puissance_infra_kw - facteur_aux_kwh_h)

            # Limite de puissance (kW)
            limite_puissance = min(puissance_dispo_charge, puissance_max_batterie_kw)
            raison_limite_puissance = "Infra" if puissance_dispo_charge < puissance_max_batterie_kw else "Capacité Charge"

            # Energie potentielle ajoutée
            energie_rechargee_potentielle = limite_puissance * duree_h

            # Limite de capacité (kWh)
            espace_dispo = capacite_max_kwh - niveau_actuel

            if energie_rechargee_potentielle >= espace_dispo:
                energie_rechargee_reelle = espace_dispo
                raison_limite = "Batterie Pleine"
            else:
                energie_rechargee_reelle = energie_rechargee_potentielle
                raison_limite = raison_limite_puissance

            nouveau_niveau += energie_rechargee_reelle

            # Log plus précis
            if energie_rechargee_reelle > 0.01:
                log_msg = f"{contexte} sous {electrification} (+{energie_rechargee_reelle:.1f} kWh). Limite: {raison_limite}"
            else:
                log_msg = f"{contexte} sous {electrification} (Maintenu à 100%)"

        else:
            # Pas d'infra : décharge auxiliaires
            nouveau_niveau -= conso_aux
            log_msg = f"{contexte} sans infra (-{conso_aux:.1f} kWh Aux)"

        # Bornage final
        nouveau_niveau = max(0, min(nouveau_niveau, capacite_max_kwh))

        # Ajouter au log
        pct = (nouveau_niveau / capacite_max_kwh) * 100
        log_batterie.append((heure_fin, nouveau_niveau, f"{pct:.1f}%", log_msg))

        return nouveau_niveau


    # Log initial
    ajouter_log_batt = lambda h, k, m: log_batterie.append((h, k, f"{(k/capacite_max_kwh)*100:.1f}%", m))
    ajouter_log_batt(trajets_train[0]["start"], niveau_batterie_kwh, "Départ Mission")

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

        # --- CAS 1 : ARRÊT EXPLICITE (Distance = 0) ---
        if distance_km == 0:
            if is_batterie:
                niveau_batterie_kwh = simuler_arret(gare_depart_nom, trajet["start"], trajet["end"], niveau_batterie_kwh, contexte="Stationnement")

        # --- CAS 2 : MOUVEMENT ---
        else:
            duree_planifiee_h = max(0, (trajet["end"] - trajet["start"]).total_seconds() / 3600.0)
            duree_planifiee_sec = duree_planifiee_h * 3600.0

            # Initialisation segment
            conso_aux_segment_kwh = facteur_aux_kwh_h * duree_planifiee_h
            total_distance_km += distance_km

            # Analyse Infra Segment (simplifiée au départ)
            electrification_dep = info_depart.get("electrification", "F").upper()
            is_catenary_segment = electrification_dep in ["C1500", "C25"]
            puissance_infra_kw = _get_puissance_infra(electrification_dep)

            # -- Calcul Profil Vitesse --
            v_initiale_kph = v_precedente_kph
            # Logique "Stop After"
            is_stop_after = True
            if i + 1 < len(trajets_train):
                ts = trajets_train[i+1]
                # Si le prochain trajet part de là où on arrive et qu'il y a continuité temporelle immédiate (pas d'arrêt long)
                # Mais attention, ici on veut savoir si on s'arrête physiquement (v=0).
                # Si le prochain segment est un "passage" (ex: gare non desservie ou passage direct), on ne s'arrête pas.
                # Dans notre modèle, chaque segment "origine->terminus" implique un arrêt si ce sont des gares distinctes
                # SAUF si explicitement modélisé autrement. Ici on assume arrêt si gares différentes.
                if ts["origine"] == ts["terminus"] and ts["origine"] == gare_arrivee_nom:
                     is_stop_after = True
                else:
                     # Est-ce un passage sans arrêt ? (Vérifier si le prochain part de notre arrivée)
                     # Pour simplifier dans ce modèle macro : Arrêt systématique sauf indication contraire.
                     is_stop_after = True

            v_finale_target = 0 if is_stop_after else v_initiale_kph
            v_avg_implied = (distance_m / duree_planifiee_sec) * 3.6 if duree_planifiee_sec > 0 else 0
            v_cruise = max(v_avg_implied * 1.1, v_initiale_kph, v_finale_target, 10.0)
            v_cruise = min(v_cruise, DEFAULT_V_MAX_KPH)

            profil = get_physical_profile(distance_m, v_initiale_kph, v_finale_target, v_cruise, accel_ms2, decel_ms2)
            duree_physique_sec = sum(t for _, t, _ in profil.values())
            duree_physique_h = duree_physique_sec / 3600.0

            # Vitesse finale réelle
            _, _, v_avg_decel = profil["decel"]
            v_finale_kph_segment = 0 if is_stop_after else v_cruise

            # -- Calcul Energies --
            rampe = info_depart.get("rampe_section_a_venir", 0)

            # Pente
            delta_h = distance_m * (rampe / 1000.0)
            e_pente_J = masse_kg * GRAVITY_MS2 * delta_h

            # Cinétique
            v_start_ms = v_initiale_kph / 3.6
            v_end_ms = v_finale_kph_segment / 3.6
            e_cin_J = 0.5 * masse_kg * (max(0, v_end_ms**2) - max(0, v_start_ms**2))

            # Résistance
            e_resist_J = 0
            for phase in profil.values():
                d, _, v_avg = phase
                if d > 0:
                    f_res = _calculer_force_resistance_davis_N(v_avg, masse_t, params)
                    e_resist_J += f_res * d

            e_total_necessaire_J = e_resist_J + e_pente_J + e_cin_J

            # Source & Rendements
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
            if source_actuelle == "electrique": total_conso_electrique_kwh += (conso_traction_kwh + conso_aux_segment_kwh)
            else: total_conso_thermique_kwh += (conso_traction_kwh + conso_aux_segment_kwh)
            total_conso_brute_kwh += (conso_traction_kwh + conso_aux_segment_kwh)

            # -- Logique Batterie (Mouvement) --
            if is_batterie:
                conso_totale_segment = conso_traction_kwh + conso_aux_segment_kwh
                if is_catenary_segment:
                    # Charge dynamique (sous caténaire)
                    # Puissance consommée par le train
                    p_conso_train = (conso_totale_segment / duree_physique_h) if duree_physique_h > 0 else 0
                    p_dispo_charge = max(0, puissance_infra_kw - p_conso_train)

                    p_target = min(p_dispo_charge, puissance_max_batterie_kw)
                    e_charge_potential = p_target * duree_physique_h

                    space = capacite_max_kwh - niveau_batterie_kwh
                    # On ajoute aussi la récup à la batterie
                    e_added = min(e_charge_potential + recup_possible_kwh, space + conso_totale_segment)
                    # Note : on simplifie ici. La récup est prioritaire. La charge complète le reste.

                    # Approche bilan net :
                    # La batterie ne se vide pas (conso prise sur caténaire).
                    # Elle se remplit avec (Charge + Recup).

                    gain_net = min(e_charge_potential + recup_possible_kwh, space)
                    niveau_batterie_kwh += gain_net

                    limite_txt = "Infra" if p_dispo_charge < puissance_max_batterie_kw else "Capacité Charge"
                    if math.isclose(niveau_batterie_kwh, capacite_max_kwh): limite_txt = "Batterie Pleine"

                    log_msg = f"Sous caténaire (+{gain_net:.1f} kWh). Limite: {limite_txt}"
                else:
                    # Décharge
                    niveau_batterie_kwh -= conso_totale_segment
                    niveau_batterie_kwh += recup_possible_kwh
                    log_msg = f"Sur batterie (-{conso_totale_segment:.1f} kWh, Recup +{recup_possible_kwh:.1f})"

                niveau_batterie_kwh = max(0, min(niveau_batterie_kwh, capacite_max_kwh))
                ajouter_log_batt(trajet["end"], niveau_batterie_kwh, log_msg)

                if niveau_batterie_kwh <= 0.1:
                    erreurs.append(f"Batterie vide à {gare_arrivee_nom}!")

            v_precedente_kph = v_finale_kph_segment

        # --- GESTION DES TEMPS D'ARRÊT ENTRE SEGMENTS (Recharge implicite) ---
        if is_batterie and i < len(trajets_train) - 1:
            next_trajet = trajets_train[i+1]
            gap_seconds = (next_trajet["start"] - trajet["end"]).total_seconds()

            # Si un trou significatif (> 1 min) existe entre l'arrivée ici et le départ suivant
            if gap_seconds > 60:
                niveau_batterie_kwh = simuler_arret(
                    trajet["terminus"],
                    trajet["end"],
                    next_trajet["start"],
                    niveau_batterie_kwh,
                    contexte="Attente/Terminus"
                )

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