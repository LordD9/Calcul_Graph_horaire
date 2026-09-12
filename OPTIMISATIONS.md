# Pistes d'acceleration du calcul de graphique horaire

Ordre de grandeur : le cout est dans **l'optimisation de grille** (genetique / exhaustif / smart), pas dans le dessin matplotlib.

## Deja en place
- Cache `construire_horaire_mission` (core_logic) — rebranche, `reset_caches` fonctionne.
- Cache `evaluer_params_simulation` (params identiques, ex. affinement smart qui recroise la grille 5 min).
- Cache `_construire_durees_theoriques` (independant des offsets).
- Parallelisme genetique (`use_parallel`).
- Recalcul energie **separe** de la grille (branche energie).

## Gains rapides restants (plus risqués)
1. **Smart progressive voisinage** : ne pas redescendre tout l'espace a chaque pas — change le resultat, pas juste la vitesse.
2. **Pas adaptatif conjoint** cadencement+retournement (PLAN_optimisation_croisements.md).
3. **Energie** : Davis est deja 3 phases, pas une boucle seconde-par-seconde — gain faible.

## Ne pas faire en premier
- Reecrire le moteur evenementiel sans tests de non-regression des croisements.
- Exhaustif sur >3 missions (explosif par design).
