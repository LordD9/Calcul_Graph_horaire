# -*- coding: utf-8 -*-
"""
plotting.py

Ce module est entièrement dédié à la création du graphique horaire avec Matplotlib.
Il prend les données calculées en entrée et retourne un objet Figure.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec

def creer_graphique_horaire(chronologie_trajets, df_gares, heure_debut_service, params_affichage):
    """
    Crée le graphique horaire (graphique espace-temps) à partir de la chronologie des trajets,
    en y ajoutant une représentation de l'infrastructure.
    """
    df_gares_triees = df_gares.sort_values("distance", ascending=True).reset_index(drop=True)
    gare_vers_distance = {row["gare"]: row["distance"] for _, row in df_gares_triees.iterrows()}

    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 20], wspace=0.05)
    ax_infra = fig.add_subplot(gs[0])
    ax_graph = fig.add_subplot(gs[1], sharey=ax_infra)

    # --- DESSIN DU SCHÉMA D'INFRASTRUCTURE (sur ax_infra) ---
    ax_infra.set_ylim(df_gares_triees["distance"].min() - 5, df_gares_triees["distance"].max() + 5)
    ax_infra.set_xlim(-0.5, 2.5) # Retour à une marge standard
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
        if g1['infra'] == 'T':
            start_y += diamond_height / 2
        end_y = g2['distance']
        if g2['infra'] == 'T':
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

    # Dessin des symboles de gares et des étiquettes
    for i, gare in df_gares_triees.iterrows():
        dist = gare['distance']
        if gare['infra'] == 'T': # Croisement: losange
            x_coords = [0, diamond_width, 0, -diamond_width, 0]
            y_coords = [dist + diamond_height / 2, dist, dist - diamond_height / 2, dist, dist + diamond_height / 2]
            ax_infra.plot(x_coords, y_coords, color='black', linewidth=1.5)
        elif gare['infra'] == 'D': # Début/fin voie double: barre fine et pointillée
             ax_infra.plot([-0.2, 0.2], [dist, dist], color='gray', linewidth=1.5, linestyle='--')
        else: # Simple arrêt: barre plus courte
            ax_infra.plot([-0.15, 0.15], [dist, dist], color='black', linewidth=1.5)

        # Logique pour ajuster dynamiquement la taille de la police
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

        ax_infra.text(0.4, gare['distance'], f"{gare['gare']}", ha='left', va='center', fontsize=font_size)

    ax_infra.set_title("Infrastructure", fontsize=10)


    # --- DESSIN DU GRAPHIQUE HORAIRE (sur ax_graph) ---
    colors = plt.get_cmap('tab20').colors
    for i, (id_train, trajets) in enumerate(sorted(chronologie_trajets.items())):
        trajets_tries = sorted(trajets, key=lambda t: t["start"])
        if not trajets_tries: continue
        couleur_train = colors[i % len(colors)]
        for j, trajet in enumerate(trajets_tries):
            if trajet["origine"] in gare_vers_distance and trajet["terminus"] in gare_vers_distance:
                ax_graph.plot(
                    [trajet["start"], trajet["end"]],
                    [gare_vers_distance[trajet["origine"]], gare_vers_distance[trajet["terminus"]]],
                    marker='o', markersize=4, color=couleur_train, linewidth=1.5,
                    label=f"Train {id_train}" if j == 0 else ""
                )
            if j < len(trajets_tries) - 1:
                prochain_trajet = trajets_tries[j+1]
                if trajet["terminus"] == prochain_trajet["origine"] and prochain_trajet["start"] > trajet["end"]:
                    pos_y = gare_vers_distance[trajet["terminus"]]
                    ax_graph.plot([trajet["end"], prochain_trajet["start"]], [pos_y, pos_y], linestyle='--', color=couleur_train, alpha=0.7)

    ax_graph.tick_params(axis='y', which='both', left=False, labelleft=False)

    dt_debut_fenetre = datetime.combine(datetime.today(), heure_debut_service) + timedelta(hours=params_affichage['decalage_heure'])
    dt_fin_fenetre = dt_debut_fenetre + timedelta(hours=params_affichage['duree_fenetre'])
    ax_graph.set_xlim(dt_debut_fenetre, dt_fin_fenetre)

    duree_heures = params_affichage['duree_fenetre']
    if duree_heures <= 2:
        ax_graph.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
        ax_graph.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    else:
        ax_graph.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_graph.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 30]))

    ax_graph.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax_graph.get_xticklabels(), rotation=45, ha="right")
    ax_graph.set_xlabel("Heure")

    ax_graph.set_title(f"Graphique horaire ({len(chronologie_trajets)} trains) - Fenêtre de {duree_heures}h")
    ax_graph.grid(True, which="both", axis="both", linestyle=":", alpha=0.6)

    handles, labels = ax_graph.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax_graph.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.01, 1))

    fig.subplots_adjust(right=0.85) # Ajuster pour la légende
    return fig

