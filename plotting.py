# -*- coding: utf-8 -*-
"""
plotting.py

Ce module est entièrement dédié à la création du graphique horaire avec Matplotlib.
Il prend les données calculées en entrée et retourne un objet Figure.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def creer_graphique_horaire(chronologie_trajets, df_gares, heure_debut_service, params_affichage):
    """
    Crée le graphique horaire (graphique espace-temps) à partir de la chronologie des trajets.

    Args:
        chronologie_trajets (dict): Dictionnaire des trajets, clé par ID de train.
        df_gares (pd.DataFrame): DataFrame des gares et de leurs distances.
        heure_debut_service (datetime.time): Heure de début de service pour caler l'axe des temps.
        params_affichage (dict): Dictionnaire contenant 'duree_fenetre' et 'decalage_heure'.

    Returns:
        matplotlib.figure.Figure: L'objet Figure contenant le graphique.
    """
    df_gares_triees = df_gares.sort_values("distance").reset_index(drop=True)
    gare_vers_distance = {row["gare"]: row["distance"] for _, row in df_gares_triees.iterrows()}

    fig, ax = plt.subplots(figsize=(17, 6))

    # CORRECTION: Utilisation de la nouvelle API de Matplotlib pour les couleurs
    # L'ancienne méthode plt.cm.get_cmap est obsolète.
    try:
        colors = plt.get_cmap('tab20').colors
    except AttributeError:
        # Fallback pour les versions plus anciennes de Matplotlib si nécessaire
        colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]

    # Itération sur chaque train pour tracer ses trajets
    for i, (id_train, trajets) in enumerate(sorted(chronologie_trajets.items())):
        trajets_tries = sorted(trajets, key=lambda t: t["start"])
        if not trajets_tries: continue

        couleur_train = colors[i % len(colors)]

        for j, trajet in enumerate(trajets_tries):
            # Tracer le segment de trajet (sillon)
            if trajet["origine"] in gare_vers_distance and trajet["terminus"] in gare_vers_distance:
                ax.plot(
                    [trajet["start"], trajet["end"]],
                    [gare_vers_distance[trajet["origine"]], gare_vers_distance[trajet["terminus"]]],
                    marker='o', markersize=4, color=couleur_train, linewidth=1.5,
                    label=f"Train {id_train}" if j == 0 else "" # Label uniquement pour le premier segment
                )

            # Tracer le temps d'arrêt/retournement en gare
            if j < len(trajets_tries) - 1:
                prochain_trajet = trajets_tries[j+1]
                if trajet["terminus"] == prochain_trajet["origine"] and prochain_trajet["start"] > trajet["end"]:
                    pos_y = gare_vers_distance[trajet["terminus"]]
                    ax.plot([trajet["end"], prochain_trajet["start"]], [pos_y, pos_y], linestyle='--', color=couleur_train, alpha=0.7)

    # Configuration de l'axe Y (Gares)
    ax.set_yticks(df_gares_triees["distance"])
    ax.set_yticklabels(f"{row['gare']} ({row['distance']:.0f} km)" for _, row in df_gares_triees.iterrows())
    ax.set_ylabel("Gares et Distance (km)")
    # CORRECTION: Suppression de l'inversion de l'axe Y pour avoir le km 0 en bas.
    # ax.invert_yaxis()

    # Configuration de l'axe X (Temps)
    dt_debut_fenetre = datetime.combine(datetime.today(), heure_debut_service) + timedelta(hours=params_affichage['decalage_heure'])
    dt_fin_fenetre = dt_debut_fenetre + timedelta(hours=params_affichage['duree_fenetre'])
    ax.set_xlim(dt_debut_fenetre, dt_fin_fenetre)

    # Formatage des graduations de l'axe du temps
    duree_heures = params_affichage['duree_fenetre']
    if duree_heures <= 2:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    elif duree_heures <= 6:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 30]))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Heure")

    # Finalisation du graphique (titre, grille, légende)
    ax.set_title(f"Graphique horaire ({len(chronologie_trajets)} trains) - Fenêtre de {duree_heures}h")
    ax.grid(True, which="both", axis="both", linestyle=":", alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

