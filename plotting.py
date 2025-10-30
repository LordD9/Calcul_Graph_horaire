# -*- coding: utf-8 -*-
"""
plotting.py - Version corrigée

Tracé du graphique horaire en respectant les temps RÉELS générés par core_logic.

Principe :
- En mode Standard : tracé linéaire simple entre start et end
- En mode Calcul Energie : tracé physique (accel/cruise/decel) basé sur la vitesse
  calculée pour respecter le temps réel du segment


"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec

# Import des fonctions de calcul physique
try:
    from energy_logic import find_implicit_v_cruise, get_physical_profile
except ImportError:
    find_implicit_v_cruise = None
    get_physical_profile = None


def creer_graphique_horaire(
    chronologie_trajets,
    df_gares,
    heure_debut_service,
    params_affichage,
    mode_calcul="Standard",
    missions_par_train=None,
    all_energy_params=None
):
    """
    Crée le graphique horaire (graphique espace-temps).

    Args:
        chronologie_trajets: Dict {id_train: [liste de trajets]}
                            Chaque trajet contient "start", "end", "origine", "terminus"
        df_gares: DataFrame des gares avec distances et infrastructure
        heure_debut_service: time object pour le début du service
        params_affichage: Dict avec 'duree_fenetre' et 'decalage_heure'
        mode_calcul: "Standard" ou "Calcul Energie"
        missions_par_train: Dict {id_train: mission} pour connaître le matériel
        all_energy_params: Dict {type_materiel: params} pour les calculs physiques
    """
    df_gares_triees = df_gares.sort_values("distance", ascending=True).reset_index(drop=True)
    gare_vers_distance = {row["gare"]: row["distance"] for _, row in df_gares_triees.iterrows()}

    # Création de la figure avec deux axes
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 20], wspace=0.05)
    ax_infra = fig.add_subplot(gs[0])
    ax_graph = fig.add_subplot(gs[1], sharey=ax_infra)

    # ========== DESSIN DU SCHÉMA D'INFRASTRUCTURE ==========
    ax_infra.set_ylim(
        df_gares_triees["distance"].min() - 5,
        df_gares_triees["distance"].max() + 5
    )
    ax_infra.set_xlim(-0.5, 2.5)
    ax_infra.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    ax_infra.get_xaxis().set_ticks([])
    ax_infra.get_yaxis().set_ticks([])

    diamond_height = 1.0
    diamond_width = 0.1
    d_count = 0

    # Dessin des segments de voie
    for i in range(len(df_gares_triees) - 1):
        g1 = df_gares_triees.iloc[i]
        g2 = df_gares_triees.iloc[i+1]

        start_y = g1['distance']
        if g1['infra'] == 'VE':
            start_y += diamond_height / 2
        end_y = g2['distance']
        if g2['infra'] == 'VE':
            end_y -= diamond_height / 2

        if g1['infra'] == 'D':
            d_count += 1
        is_double = (d_count % 2 == 1)

        if end_y > start_y:
            if is_double:
                ax_infra.plot([-0.1, -0.1], [start_y, end_y], color='black', linewidth=1.5)
                ax_infra.plot([0.1, 0.1], [start_y, end_y], color='black', linewidth=1.5)
            else:
                ax_infra.plot([0, 0], [start_y, end_y], color='black', linewidth=1.5)

    # Dessin des symboles de gares
    for i, gare in df_gares_triees.iterrows():
        dist = gare['distance']
        if gare['infra'] == 'VE':  # Voie d'évitement: losange
            x_coords = [0, diamond_width, 0, -diamond_width, 0]
            y_coords = [
                dist + diamond_height / 2,
                dist,
                dist - diamond_height / 2,
                dist,
                dist + diamond_height / 2
            ]
            ax_infra.plot(x_coords, y_coords, color='black', linewidth=1.5)
        elif gare['infra'] == 'D':  # Début/fin voie double
            ax_infra.plot([-0.2, 0.2], [dist, dist], color='gray', linewidth=1.5, linestyle='--')
        else:  # Simple arrêt
            ax_infra.plot([-0.15, 0.15], [dist, dist], color='black', linewidth=1.5)

        # Ajustement taille police selon espacement
        font_size = 9
        min_dist_voisin = float('inf')
        if i > 0:
            min_dist_voisin = min(min_dist_voisin, abs(dist - df_gares_triees.iloc[i-1]['distance']))
        if i < len(df_gares_triees) - 1:
            min_dist_voisin = min(min_dist_voisin, abs(df_gares_triees.iloc[i+1]['distance'] - dist))

        if min_dist_voisin < 4:
            font_size = 6
        elif min_dist_voisin < 8:
            font_size = 7.5

        ax_infra.text(0.4, gare['distance'], f"{gare['gare']}",
                     ha='left', va='center', fontsize=font_size)

    ax_infra.set_title("Infrastructure", fontsize=10)

    # ========== DESSIN DU GRAPHIQUE HORAIRE ==========
    colors = plt.get_cmap('tab20').colors

    # Déterminer si on peut utiliser le tracé physique
    use_physical_plotting = (
        mode_calcul == "Calcul Energie" and
        find_implicit_v_cruise is not None and
        get_physical_profile is not None and
        missions_par_train is not None and
        all_energy_params is not None
    )

    # Boucle par train
    for i, (id_train, trajets) in enumerate(sorted(chronologie_trajets.items())):
        trajets_tries = sorted(trajets, key=lambda t: t["start"])
        if not trajets_tries:
            continue

        couleur_train = colors[i % len(colors)]

        mission = None
        params = None
        if use_physical_plotting:
            mission = missions_par_train.get(id_train)
            if mission:
                params = all_energy_params.get(mission.get("type_materiel"))

        should_plot_physically = use_physical_plotting and mission and params
        v_precedente_kph = 0

        last_end_time = None
        last_end_dist = None
        first_segment = True

        for j, trajet in enumerate(trajets_tries):
            if trajet["origine"] not in gare_vers_distance or trajet["terminus"] not in gare_vers_distance:
                continue

            start_dist_km = gare_vers_distance[trajet["origine"]]
            end_dist_km = gare_vers_distance[trajet["terminus"]]

            # ========== CAS 0 : TROU (Temps d'attente/retournement) ==========
            if (j > 0 and last_end_time and trajet["start"] > last_end_time and
                start_dist_km == last_end_dist):
                # Ligne pointillée pour visualiser l'attente
                ax_graph.plot(
                    [last_end_time, trajet["start"]],
                    [last_end_dist, start_dist_km],
                    linestyle='--', color=couleur_train, alpha=0.7, marker='None'
                )
                # Point de fin d'attente
                ax_graph.plot(
                    [trajet["start"]], [start_dist_km],
                    linestyle='None', marker='o', markersize=4,
                    color=couleur_train, alpha=0.7
                )
                v_precedente_kph = 0

            # ========== CAS 1 : ARRÊT ==========
            if start_dist_km == end_dist_km and trajet["start"] < trajet["end"]:
                # Ligne pointillée horizontale
                ax_graph.plot(
                    [trajet["start"], trajet["end"]],
                    [start_dist_km, start_dist_km],
                    linestyle='--', color=couleur_train, alpha=0.7, marker='None'
                )
                # Point de fin d'arrêt
                ax_graph.plot(
                    [trajet["end"]], [start_dist_km],
                    linestyle='None', marker='o', markersize=4,
                    color=couleur_train, alpha=0.7
                )
                v_precedente_kph = 0

            # ========== CAS 2 : MOUVEMENT ==========
            elif start_dist_km != end_dist_km:
                v_start_kph = v_precedente_kph

                # Point de départ (uniquement pour le premier segment)
                if first_segment:
                    ax_graph.plot(
                        [trajet["start"]], [start_dist_km],
                        linestyle='None', marker='o', markersize=4,
                        color=couleur_train, alpha=0.8,
                        label=f"Train {id_train}"
                    )
                    first_segment = False

                # Déterminer si arrêt après ce segment
                is_explicit_stop_after = False
                if j + 1 < len(trajets_tries):
                    next_trajet = trajets_tries[j+1]
                    if ((next_trajet["origine"] == next_trajet["terminus"] and
                         next_trajet["origine"] == trajet["terminus"]) or
                        (next_trajet["start"] > trajet["end"] and
                         next_trajet["origine"] == trajet["terminus"])):
                        is_explicit_stop_after = True
                else:
                    is_explicit_stop_after = True

                if not should_plot_physically:
                    # ===== TRACÉ STANDARD : Ligne droite simple =====
                    ax_graph.plot(
                        [trajet["start"], trajet["end"]],
                        [start_dist_km, end_dist_km],
                        marker='None', color=couleur_train, linewidth=1.5,
                    )
                    # Point de fin
                    ax_graph.plot(
                        [trajet["end"]], [end_dist_km],
                        linestyle='None', marker='o', markersize=4,
                        color=couleur_train, alpha=0.8
                    )
                    v_precedente_kph = 0 if is_explicit_stop_after else 50

                else:
                    # ===== TRACÉ PHYSIQUE : Profil accel/cruise/decel =====

                    # Temps RÉEL alloué (incluant délais éventuels)
                    temps_reel_sec = (trajet["end"] - trajet["start"]).total_seconds()
                    distance_m = abs(end_dist_km - start_dist_km) * 1000
                    dist_sign = 1 if end_dist_km > start_dist_km else -1

                    # Vitesses cibles
                    v_end_kph_target = 0 if is_explicit_stop_after else v_start_kph

                    # Calcul de la vitesse de croisière optimale pour le temps réel
                    v_cruise_kph = find_implicit_v_cruise(
                        distance_m, v_start_kph, v_end_kph_target,
                        params['accel_ms2'], params['decel_ms2'], temps_reel_sec
                    )

                    # Vitesse finale réelle
                    v_end_kph_real = 0 if is_explicit_stop_after else v_cruise_kph

                    # Profil physique
                    profile = get_physical_profile(
                        distance_m, v_start_kph, v_end_kph_real, v_cruise_kph,
                        params['accel_ms2'], params['decel_ms2']
                    )

                    current_time = trajet["start"]
                    current_dist_km = start_dist_km

                    # Phase Accélération
                    (d_a, t_a, v_a) = profile['accel']
                    if t_a > 0.1:
                        end_time_a = current_time + timedelta(seconds=t_a)
                        end_dist_a = current_dist_km + (d_a / 1000 * dist_sign)
                        ax_graph.plot(
                            [current_time, end_time_a],
                            [current_dist_km, end_dist_a],
                            marker='None', color=couleur_train,
                            linewidth=1.5, alpha=0.8
                        )
                        current_time = end_time_a
                        current_dist_km = end_dist_a

                    # Phase Croisière
                    (d_c, t_c, v_c) = profile['cruise']
                    if t_c > 0.1:
                        end_time_c = current_time + timedelta(seconds=t_c)
                        end_dist_c = current_dist_km + (d_c / 1000 * dist_sign)
                        ax_graph.plot(
                            [current_time, end_time_c],
                            [current_dist_km, end_dist_c],
                            marker='None', color=couleur_train, linewidth=1.5
                        )
                        current_time = end_time_c
                        current_dist_km = end_dist_c

                    # Phase Décélération
                    (d_d, t_d, v_d) = profile['decel']
                    # S'assurer qu'on arrive exactement à l'heure prévue
                    end_time_plot = trajet["end"]
                    if end_time_plot > current_time:
                        ax_graph.plot(
                            [current_time, end_time_plot],
                            [current_dist_km, end_dist_km],
                            marker='None', color=couleur_train,
                            linewidth=1.5, alpha=0.8
                        )

                    # Point de fin du mouvement
                    ax_graph.plot(
                        [trajet["end"]], [end_dist_km],
                        linestyle='None', marker='o', markersize=4,
                        color=couleur_train, alpha=0.8
                    )

                    v_precedente_kph = v_end_kph_real

            # Mise à jour pour détection de "trou"
            last_end_time = trajet["end"]
            last_end_dist = end_dist_km

    # ========== CONFIGURATION FINALE DU GRAPHIQUE ==========
    ax_graph.tick_params(axis='y', which='both', left=False, labelleft=False)

    # Fenêtre temporelle
    dt_debut_fenetre = datetime.combine(
        datetime.today(), heure_debut_service
    ) + timedelta(hours=params_affichage['decalage_heure'])
    dt_fin_fenetre = dt_debut_fenetre + timedelta(hours=params_affichage['duree_fenetre'])
    ax_graph.set_xlim(dt_debut_fenetre, dt_fin_fenetre)

    # Configuration de l'axe temporel
    duree_heures = params_affichage['duree_fenetre']
    if duree_heures <= 2:
        ax_graph.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
        ax_graph.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    elif duree_heures <= 6:
        ax_graph.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_graph.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    else:
        ax_graph.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_graph.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 30]))

    ax_graph.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax_graph.get_xticklabels(), rotation=45, ha="right")
    ax_graph.set_xlabel("Heure")

    # Titre
    titre = f"Graphique horaire ({len(chronologie_trajets)} trains) - Fenêtre de {duree_heures:.1f}h"
    if use_physical_plotting:
        titre += " (Tracé physique)"
    ax_graph.set_title(titre)

    # Grille
    ax_graph.grid(True, which="both", axis="both", linestyle=":", alpha=0.6)

    # Légende
    handles, labels = ax_graph.get_legend_handles_labels()
    if handles:
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label and label not in unique_labels:
                unique_labels[label] = handle
        if unique_labels:
            ax_graph.legend(
                unique_labels.values(), unique_labels.keys(),
                loc='upper left', bbox_to_anchor=(1.01, 1), title="Trains"
            )

    fig.subplots_adjust(right=0.85)
    return fig
