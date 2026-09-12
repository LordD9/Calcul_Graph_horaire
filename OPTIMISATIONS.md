# Pistes d'acceleration du calcul de graphique horaire

Ordre de grandeur : le cout est dans **l'optimisation de grille** (genetique / exhaustif / smart), pas dans le dessin matplotlib.

## Deja en place
- Cache `construire_horaire_mission` (core_logic).
- Parallelisme genetique (`use_parallel`).
- Recalcul energie **separe** de la grille (bouton + cache fingerprint materiel).

## Gains rapides (prochaines PR)
1. **Memoizer `evaluer_params_simulation`** sur un hash (missions + offsets + buffers). Le genetique reevalue des genomes proches.
2. **Smart progressive** : ne pas redescendre tout l'espace a chaque pas ; garder le meilleur offset et n'explorer qu'un voisinage.
3. **Pas adaptatif** : Fast 10 min puis raffiner 1 min seulement autour des meilleurs (deja l'idee smart ; elargir le voisinage conjoint cadencement+retournement — voir PLAN_optimisation_croisements.md).
4. **Energie** : vectoriser Davis sur le profil (numpy) au lieu d'integrer trop finement si dt trop petit.
5. **UI Streamlit** : le cache energie evite de refaire la physique a chaque widget (telechargement, expander).

## Ne pas faire en premier
- Reecrire le moteur evenementiel sans tests de non-regression des croisements.
- Exhaustif sur >3 missions (explosif par design).
