# -*- coding: utf-8 -*-
"""
plotting.py
===========

Module de visualisation graphique.

Ce module est responsable de la génération des graphiques espace-temps (Marey) et autres visualisations.

Fonctions principales :
- `creer_graphique_horaire` : Génère le graphique espace-temps principal.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec

# Import des fonctions de calcul physique
try:
    from energy_logic import (
        find_implicit_v_cruise,
        find_v_cruise_no_decel,
        get_physical_profile,
    )
except ImportError:
    find_implicit_v_cruise = None
    find_v_cruise_no_decel = None
    get_physical_profile = None


def _sample_accel_parabola(t_total_sec, v_start_kph, accel_ms2, n_samples=20):
    """Échantillonne la position le long de la phase d'accélération.

    Position physique : d(t) = v_s*t + 0.5*a*t² (parabole). Pente instantanée
    = v_s + a*t, croît de v_s à v_cruise. Tracer comme une droite produit une
    pente constante = (v_s+v_cruise)/2, ce qui crée des ruptures de pente
    visibles aux transitions de phase. Échantillonner la parabole les
    supprime.
    """
    v_start_ms = max(0.0, v_start_kph / 3.6)
    a = max(0.0, accel_ms2)
    points = []
    for k in range(n_samples + 1):
        t = k * t_total_sec / n_samples
        d = v_start_ms * t + 0.5 * a * t * t
        points.append((t, d))
    return points


def _sample_decel_parabola(t_total_sec, v_cruise_kph, decel_ms2, n_samples=20):
    """Échantillonne la position le long de la phase de décélération.

    Position physique : d(t) = v_c*t - 0.5*d*t² (parabole). Pente instantanée
    = v_c - d*t, décroît de v_c à v_end. Continuité de pente à l'entrée de la
    décélération.
    """
    v_cruise_ms = max(0.0, v_cruise_kph / 3.6)
    d_ms2 = max(0.0, decel_ms2)
    points = []
    for k in range(n_samples + 1):
        t = k * t_total_sec / n_samples
        d = v_cruise_ms * t - 0.5 * d_ms2 * t * t
        points.append((t, d))
    return points


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
    Génère le graphique espace-temps (diagramme de Marey) des circulations.

    Args:
        chronologie_trajets (dict): Dictionnaire {train_id: [liste de trajets]}.
            Chaque trajet est un dict avec clés 'start', 'end', 'origine', 'terminus'.
        df_gares (pd.DataFrame): DataFrame des gares avec positions kilométriques et codes infra.
        heure_debut_service (time): Heure de début de l'axe temporel.
        params_affichage (dict): Paramètres de vue ('duree_fenetre', 'decalage_heure').
        mode_calcul (str, optional): "Standard" (lignes droites) ou "Calcul Energie" (profils physiques).
        missions_par_train (dict, optional): Mapping pour identifier la mission de chaque train.
        all_energy_params (dict, optional): Paramètres énergétiques pour le tracé physique.

    Returns:
        matplotlib.figure.Figure: La figure générée contenant le graphique.
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
        # Indicateur pour placer le label sur le tout premier segment tracé
        first_segment_to_label = True

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
                v_precedente_kph = 0

            # ========== CAS 1 : ARRÊT ==========
            if start_dist_km == end_dist_km and trajet["start"] < trajet["end"]:
                # Ligne pointillée horizontale
                ax_graph.plot(
                    [trajet["start"], trajet["end"]],
                    [start_dist_km, start_dist_km],
                    linestyle='--', color=couleur_train, alpha=0.7, marker='None'
                )
                v_precedente_kph = 0

            # ========== CAS 2 : MOUVEMENT ==========
            elif start_dist_km != end_dist_km:
                v_start_kph = v_precedente_kph

                # Préparation du label (uniquement pour le premier segment du train)
                label_arg = f"Train {id_train}" if first_segment_to_label else None

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
                        label=label_arg
                    )
                    # Si on a tracé, on a utilisé le label
                    if label_arg:
                        first_segment_to_label = False

                    v_precedente_kph = 0 if is_explicit_stop_after else 50

                else:
                    # ===== TRACÉ PHYSIQUE : Profil accel/cruise/decel =====

                    # Temps RÉEL alloué
                    temps_reel_sec = (trajet["end"] - trajet["start"]).total_seconds()
                    distance_m = abs(end_dist_km - start_dist_km) * 1000
                    dist_sign = 1 if end_dist_km > start_dist_km else -1

                    # Vitesse de croisière + vitesse de sortie.
                    # Si arrêt après ce segment, profil accel/cruise/decel
                    # classique (v_end_real = 0). Sinon, c'est une transition
                    # sans arrêt (point de passage) : pas de phase de
                    # décélération, sinon le train ralentirait visuellement
                    # jusqu'à 0 puis "sauterait" à v_cruise au segment suivant
                    # — c'est ce qui faisait apparaître un faux arrêt aux
                    # points de passage. v_end_real = v_cruise assure la
                    # continuité de pente au point de passage.
                    if is_explicit_stop_after:
                        v_cruise_kph = find_implicit_v_cruise(
                            distance_m, v_start_kph, 0,
                            params['accel_ms2'], params['decel_ms2'], temps_reel_sec
                        )
                        v_end_kph_real = 0
                    else:
                        v_cruise_kph = find_v_cruise_no_decel(
                            distance_m, v_start_kph,
                            params['accel_ms2'], temps_reel_sec
                        ) if find_v_cruise_no_decel is not None else find_implicit_v_cruise(
                            distance_m, v_start_kph, v_start_kph,
                            params['accel_ms2'], params['decel_ms2'], temps_reel_sec
                        )
                        v_end_kph_real = v_cruise_kph

                    # Profil physique (avec v_end = v_cruise → pas de phase
                    # de décélération dans _calculate_phases côté non-arrêt).
                    profile = get_physical_profile(
                        distance_m, v_start_kph, v_end_kph_real, v_cruise_kph,
                        params['accel_ms2'], params['decel_ms2']
                    )

                    (d_a, t_a, v_a) = profile['accel']
                    (d_c, t_c, v_c) = profile['cruise']
                    (d_d, t_d, v_d) = profile['decel']

                    # Vitesse réelle au début de la décélération.
                    # Trapézoïdal : = v_cruise. Triangulaire (planning serré
                    # → v_cruise clippée à v_max) : = v_peak < v_cruise.
                    # Dans les deux cas : v_peak = 2*v_avg_decel - v_end.
                    v_decel_start_kph = max(0.0, 2.0 * v_d - v_end_kph_real)

                    # Si la planification n'est pas physiquement faisable
                    # (temps physique > temps planifié), retomber sur une
                    # droite simple. Sans ça, la parabole de décélération
                    # déborde de la fenêtre et le snap final au planning crée
                    # une boucle visuelle (« pis de vache »).
                    t_phys_total = t_a + t_c + t_d
                    infeasible = t_phys_total > temps_reel_sec + 1.0

                    current_time = trajet["start"]
                    current_dist_km = start_dist_km

                    # Helper pour gérer le label sur le premier sous-segment visible
                    def get_label():
                        nonlocal first_segment_to_label
                        if first_segment_to_label:
                            first_segment_to_label = False
                            return f"Train {id_train}"
                        return None

                    if infeasible:
                        # Planning infaisable : ligne droite (vitesse moyenne
                        # nécessaire > v_max physique du matériel).
                        ax_graph.plot(
                            [trajet["start"], trajet["end"]],
                            [start_dist_km, end_dist_km],
                            marker='None', color=couleur_train,
                            linewidth=1.5, label=get_label()
                        )
                        v_precedente_kph = v_end_kph_real
                        last_end_time = trajet["end"]
                        last_end_dist = end_dist_km
                        continue

                    # Phase Accélération — rendu parabolique pour continuité
                    # de pente avec la phase de croisière.
                    if t_a > 0.1:
                        curve = _sample_accel_parabola(
                            t_a, v_start_kph, params['accel_ms2']
                        )
                        times_a = [current_time + timedelta(seconds=t) for t, _ in curve]
                        dists_a = [current_dist_km + (d / 1000) * dist_sign for _, d in curve]
                        ax_graph.plot(
                            times_a, dists_a,
                            marker='None', color=couleur_train,
                            linewidth=1.5, alpha=0.8,
                            label=get_label()
                        )
                        current_time = times_a[-1]
                        current_dist_km = dists_a[-1]

                    # Phase Croisière — droite (pente constante = v_cruise).
                    if t_c > 0.1:
                        end_time_c = current_time + timedelta(seconds=t_c)
                        end_dist_c = current_dist_km + (d_c / 1000 * dist_sign)
                        ax_graph.plot(
                            [current_time, end_time_c],
                            [current_dist_km, end_dist_c],
                            marker='None', color=couleur_train, linewidth=1.5,
                            label=get_label()
                        )
                        current_time = end_time_c
                        current_dist_km = end_dist_c

                    # Phase Décélération — rendu parabolique pour continuité
                    # de pente avec la phase de croisière. Sans cela, la pente
                    # passerait brutalement de v_cruise à v_cruise/2 (rupture
                    # visible au point de transition cruise→decel).
                    if t_d > 0.1:
                        curve = _sample_decel_parabola(
                            t_d, v_decel_start_kph, params['decel_ms2']
                        )
                        times_d = [current_time + timedelta(seconds=t) for t, _ in curve]
                        dists_d = [current_dist_km + (d / 1000) * dist_sign for _, d in curve]
                        # Force le dernier point à coller exactement au
                        # planning pour éviter toute dérive numérique.
                        times_d[-1] = trajet["end"]
                        dists_d[-1] = end_dist_km
                        ax_graph.plot(
                            times_d, dists_d,
                            marker='None', color=couleur_train,
                            linewidth=1.5, alpha=0.8,
                            label=get_label()
                        )
                    elif trajet["end"] > current_time:
                        # Pas de phase de décélération mais il reste du temps
                        # à couvrir (ex. arrondi ou planning légèrement >
                        # temps physique total) : on ferme par une droite.
                        ax_graph.plot(
                            [current_time, trajet["end"]],
                            [current_dist_km, end_dist_km],
                            marker='None', color=couleur_train,
                            linewidth=1.5, alpha=0.8,
                            label=get_label()
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

    # Configuration de l'axe temporel — quadrillage fin à 15 min systématique
    duree_heures = params_affichage['duree_fenetre']
    if duree_heures <= 2:
        ax_graph.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
        ax_graph.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    else:
        ax_graph.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax_graph.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))

    ax_graph.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax_graph.get_xticklabels(), rotation=45, ha="right")
    ax_graph.set_xlabel("Heure")

    # Titre
    titre = f"Graphique horaire ({len(chronologie_trajets)} trains) - Fenêtre de {duree_heures:.1f}h"
    if use_physical_plotting:
        titre += " (Tracé physique)"
    ax_graph.set_title(titre)

    # Fond alterné par tranche d'heure (style graphe horaire SNCF)
    band_start = dt_debut_fenetre.replace(minute=0, second=0, microsecond=0)
    if band_start > dt_debut_fenetre:
        band_start -= timedelta(hours=1)
    i = 0
    while band_start < dt_fin_fenetre:
        band_end = band_start + timedelta(hours=1)
        if i % 2 == 1:
            ax_graph.axvspan(
                max(band_start, dt_debut_fenetre),
                min(band_end, dt_fin_fenetre),
                facecolor='#c8d8ec', alpha=0.18, edgecolor='none', linewidth=0,
            )
        band_start = band_end
        i += 1

    # Grille : trait plein à l'heure, pointillé aux quarts d'heure
    ax_graph.set_axisbelow(True)
    ax_graph.grid(True, which='major', axis='x', linestyle='-', linewidth=0.8, color='#777777', alpha=0.85)
    ax_graph.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.8, color='#888888', alpha=0.75)
    ax_graph.grid(True, which='both',  axis='y', linestyle=':', linewidth=0.4, color='#aaaaaa', alpha=0.55)

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

def creer_graphique_batterie(batterie_log, train_id, soc_min_pct=20, soc_max_pct=95, terminaux_valides=None, max_c_rate=4.0):
    """
    Génère un graphique d'évolution du SoC (State of Charge) et de la puissance de charge pour un train.
    """
    if not batterie_log:
        return None

    if terminaux_valides is None:
        terminaux_valides = set()

    times = [x[0] for x in batterie_log]
    socs = []
    p_charges = []
    stats = []

    for x in batterie_log:
        try:
            val = float(x[2].strip('%'))
        except:
            val = 0.0
        socs.append(val)
        if len(x) > 4:
            p_charges.append(x[4])
            stats.append((x[5], x[6]))
        else:
            p_charges.append(0.0)
            stats.append((False, ""))

    # Création figure avec 2 subplots
    fig, (ax_soc, ax_pwr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.6})

    # --- 1. Graphique SoC ---
    ax_soc.plot(times, socs, color='#2ca02c', linewidth=2, label='SoC (%)')
    ax_soc.fill_between(times, socs, alpha=0.2, color='#2ca02c')

    ax_soc.axhline(y=soc_max_pct, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.8, label=f'SoC max {soc_max_pct}%')
    ax_soc.axhline(y=soc_min_pct, color='#d62728', linestyle='--', linewidth=1, alpha=0.8, label=f'SoC min {soc_min_pct}%')

    # Assignation de couleurs aux terminaux valides uniquement
    unique_terminuses = []
    for is_stat, nom_gare in stats:
        if is_stat and nom_gare and nom_gare in terminaux_valides and nom_gare not in unique_terminuses:
            unique_terminuses.append(nom_gare)
            
    cmap = plt.get_cmap('tab10')
    terminus_colors = {gare: cmap(i % 10) for i, gare in enumerate(unique_terminuses)}
    seen_terminuses = set()

    # Visualisation des terminus
    for i in range(1, len(times)):
        is_stat, nom_gare = stats[i]
        if is_stat and nom_gare and nom_gare in terminaux_valides:
            t_start = times[i-1]
            t_end = times[i]
            color = terminus_colors[nom_gare]
            
            label = f"Terminus {nom_gare}" if nom_gare not in seen_terminuses else None
            seen_terminuses.add(nom_gare)
            
            ax_soc.axvspan(t_start, t_end, color=color, alpha=0.3, label=label)

    ax_soc.set_ylim(-5, 105)
    ax_soc.set_ylabel('Batterie (%)')
    ax_soc.set_title(f'Profil de charge - Train {train_id}')
    ax_soc.grid(True, linestyle=':', alpha=0.6)
    
    # Formatage temporel pour ax_soc
    ax_soc.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_soc.tick_params(axis='x', labelbottom=True)
    plt.setp(ax_soc.get_xticklabels(), rotation=0, ha="center")
    
    # Légende pour ax_soc placée en dessous de ax_soc
    ax_soc.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize='small', frameon=False)

    # --- 2. Graphique Puissance de Charge ---
    ax_pwr.step(times, p_charges, where='pre', color='#ff7f0e', linewidth=1.5, label='Puissance (C)')
    ax_pwr.fill_between(times, p_charges, step='pre', alpha=0.2, color='#ff7f0e')
    
    # Ligne rouge continue pour puissance max
    ax_pwr.axhline(y=max_c_rate, color='red', linestyle='-', linewidth=1.5, label=f'Max ({max_c_rate}C)')

    # Pour avoir un axe bien cadré s'il y a très peu de puissance (ex: 0 à 1 C)
    max_p = max(p_charges) if p_charges else 0
    ax_pwr.set_ylim(-0.1, max(max_p * 1.2, max_c_rate * 1.2, 1.0))
    ax_pwr.set_ylabel('Charge (C)')
    ax_pwr.grid(True, linestyle=':', alpha=0.6)

    # Formatage temporel pour ax_pwr
    ax_pwr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax_pwr.get_xticklabels(), rotation=0, ha="center")

    # Légende pour ax_pwr placée en dessous de ax_pwr
    ax_pwr.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize='small', frameon=False)

    plt.tight_layout()
    # On ajoute de l'espace en bas pour la légende du graphique du bas
    plt.subplots_adjust(bottom=0.15)
    return fig