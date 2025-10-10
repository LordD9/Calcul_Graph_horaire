# -*- coding: utf-8 -*-
"""
utils.py

Ce module regroupe des fonctions utilitaires "pures" (sans effet de bord et indépendantes de Streamlit)
qui sont utilisées à plusieurs endroits de l'application.
"""

from datetime import datetime

def trouver_mission_pour_od(origine_selectionnee, terminus_selectionne, toutes_les_missions):
    """
    Recherche une mission correspondant à une origine et une destination données.
    Gère également la recherche dans le sens inverse, en tenant compte des trajets asymétriques
    et en inversant les points de passage si nécessaire.

    Args:
        origine_selectionnee (str): La gare de départ.
        terminus_selectionne (str): La gare d'arrivée.
        toutes_les_missions (list): La liste des dictionnaires de missions.

    Returns:
        dict or None: La configuration de la mission trouvée, ou None si aucune ne correspond.
    """
    # Cherche une mission directe
    for mission in toutes_les_missions:
        if mission["origine"] == origine_selectionnee and mission["terminus"] == terminus_selectionne:
            return mission

    # Si pas de mission directe, cherche une mission inverse
    for mission in toutes_les_missions:
        if mission["origine"] == terminus_selectionne and mission["terminus"] == origine_selectionnee:
            # Cas d'un trajet retour asymétrique défini explicitement
            if mission.get("trajet_asymetrique", False):
                return {
                    "origine": origine_selectionnee,
                    "terminus": terminus_selectionne,
                    "frequence": mission["frequence"],
                    "temps_trajet": mission.get("temps_trajet_retour", mission["temps_trajet"]),
                    "temps_retournement_A": mission.get("temps_retournement_B", 10), # Inversion logique
                    "temps_retournement_B": mission.get("temps_retournement_A", 10),
                    "passing_points": mission.get("passing_points_retour", [])
                }
            # Cas d'un trajet retour symétrique (calculé par inversion)
            else:
                points_passage_inverses = []
                if isinstance(mission.get("passing_points"), list):
                    temps_total_aller = mission["temps_trajet"]
                    for pp in reversed(mission["passing_points"]):
                        points_passage_inverses.append({
                            "gare": pp["gare"],
                            "time_offset_min": temps_total_aller - pp["time_offset_min"]
                        })

                return {
                    "origine": origine_selectionnee,
                    "terminus": terminus_selectionne,
                    "frequence": mission["frequence"],
                    "temps_trajet": mission["temps_trajet"],
                    "temps_retournement_A": mission.get("temps_retournement_B", 10),
                    "temps_retournement_B": mission.get("temps_retournement_A", 10),
                    # CORRECTION: Il faut trier les points de passage inversés par leur nouvelle heure
                    "passing_points": sorted(points_passage_inverses, key=lambda x: x["time_offset_min"])
                }
    return None

def obtenir_temps_trajet_defaut_etape_manuelle(gare_depart, gare_arrivee, toutes_les_missions):
    """
    Estime un temps de trajet par défaut pour une étape manuelle.
    Il cherche d'abord une mission complète, puis un segment au sein d'une mission existante.

    Args:
        gare_depart (str): La gare de départ de l'étape.
        gare_arrivee (str): La gare d'arrivée de l'étape.
        toutes_les_missions (list): La liste des missions.

    Returns:
        int: Le temps de trajet estimé en minutes.
    """
    mission_correspondante = trouver_mission_pour_od(gare_depart, gare_arrivee, toutes_les_missions)
    if mission_correspondante:
        return mission_correspondante["temps_trajet"]

    # Si aucune mission directe n'existe, on cherche un segment dans une mission plus grande
    for mission in toutes_les_missions:
        chemin = [(mission["origine"], 0)] + [(pp["gare"], pp["time_offset_min"]) for pp in mission.get("passing_points", [])] + [(mission["terminus"], mission["temps_trajet"])]
        chemin.sort(key=lambda x: x[1])

        for i in range(len(chemin) - 1):
            if chemin[i][0] == gare_depart and chemin[i+1][0] == gare_arrivee:
                return int(chemin[i+1][1] - chemin[i][1])

    return 30 # Valeur par défaut si aucun segment n'est trouvé

def obtenir_temps_retournement_defaut(nom_gare, toutes_les_missions):
    """
    Récupère le temps de retournement configuré pour une gare donnée, qu'elle soit
    une origine (temps_retournement_A) ou un terminus (temps_retournement_B).

    Args:
        nom_gare (str): Le nom de la gare.
        toutes_les_missions (list): La liste des missions.

    Returns:
        int: Le temps de retournement configuré en minutes.
    """
    for mission in toutes_les_missions:
        if nom_gare == mission["origine"]:
            return mission.get("temps_retournement_A", 10)
        if nom_gare == mission["terminus"]:
            return mission.get("temps_retournement_B", 10)
    return 10 # Valeur par défaut
