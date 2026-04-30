# Améliorer l'algorithme génétique : croisements par paire, pénalité usager, cache inter-générations

## Contexte

L'algorithme génétique de `optimisation_logic.py` optimise actuellement trois variables : décalage de cadence (`timing`), buffer de retournement (`turnaround_buffers`) et durées d'arrêt aux VE (`crossing`). Trois faiblesses limitent la qualité et la vitesse des solutions :

1. **Croisements peu stratégiques** : le génome encode une durée d'arrêt par gare VE et par mission, mais il ne décide pas *à quelle gare VE* deux trains opposés se croisent réellement. Le moteur (`SimulationEngine.solve_mission_schedule`, `core_logic.py:269-345`) prend toujours la prochaine VE en aval — ce choix peut être loin d'optimal.
2. **Coût usager non explicite** : `_score_chronologie_bruit` (`core_logic.py:930-958`) applique une pénalité quadratique sur les arrêts de croisement au-delà d'un seuil **codé en dur à 5 min**, indépendamment du seuil utilisateur (`OptimizationConfig.crossing_optimization.max_delay_minutes`). La pénalité ne reflète pas la dégradation du temps de parcours réel.
3. **Évaluation lente** : chaque génération recrée un `ProcessPoolExecutor` (`optimisation_logic.py:604`) et le `SolutionCache` (`optimisation_logic.py:188`) vit dans chaque worker — les doublons de génome (élitisme + crossovers redondants) sont réévalués à chaque génération.

**Objectif** : enrichir le génome avec un choix explicite de gare de croisement par paire de trains, brancher la pénalité de temps usager sur le seuil utilisateur, et ajouter un cache inter-générations dans le processus maître. L'UI est ajustée pour rendre l'optimisation des croisements obligatoire en mode génétique.

---

## Approche recommandée

### Partie A — Affectation par paire de trains dans le génome

#### A.1 Énumération déterministe des rencontres

Nouvelle fonction dans `core_logic.py` (à insérer après `_is_crossing_point`, vers la ligne 75) :

```python
def enumerer_rencontres(mission, df_gares, heure_debut, heure_fin,
                        reference_minute_str, frequence_par_heure):
    """
    Retourne la liste déterministe des rencontres aller×retour idéalisées
    (sans extension de croisement) pour une mission cadencée.
    Chaque entrée : {
        'aller_idx': int,     # index du départ aller dans la période (0..N-1)
        'retour_idx': int,    # index du départ retour dans la période
        'meeting_km': float,  # km où les trajectoires droites se croisent
        'candidate_ve': List[str],  # gares VE proches (±1 maille en amont/aval)
        'natural_ve': str,    # VE la plus proche du point de rencontre
    }
    """
```

Algorithme :
1. Calcule les départs aller idéalisés à partir de `reference_minute_str` (parsing existant ligne 604) et `frequence`.
2. Calcule les départs retour idéalisés via `min_turnaround` au terminus + `mission['turnaround_min']`.
3. Pour chaque (k_aller, k_retour) actif dans la fenêtre `[heure_debut, heure_fin]`, résout l'intersection des deux trajectoires linéaires (km vs temps) — formule fermée à partir des temps de parcours dans `construire_horaire_mission` (`core_logic.py:1118`).
4. Mappe `meeting_km` à la VE la plus proche en utilisant `df_gares['distance_km']` et le code infra.

Cette énumération est **rapide** (O(N×M) avec N,M ≈ 5–20) et **stable** : mêmes inputs → mêmes clés.

#### A.2 Forme étendue du génome

```python
genome = {
    'timing': {mission_id: minute_offset},                  # existant
    'turnaround_buffers': {mission_id: int},                # existant
    'crossing': {mission_id: {ve_gare: stop_duration}},     # existant (fallback)
    'crossing_pairs': {                                     # NOUVEAU
        f"{mission_id}|A{aller_k}|R{retour_k}": ve_gare_name
    },
}
```

Clés en chaîne (et non tuples) → JSON-sérialisable, indispensable pour le cache (Partie C).

#### A.3 Consommation par le moteur

Plomberie via `SimulationParams` (`optimisation_logic.py:70-99`) :
- Ajouter le champ `crossing_pair_assignments: Dict[str, str]` (clé canonique → ve_gare).
- `_evaluate_genome_worker` (lignes 319-357) le passe à `evaluer_params_simulation`.
- `executer_simulation_evenementielle` (`core_logic.py:547-927`) le passe à chaque appel `engine.solve_mission_schedule`.

Modification de `solve_mission_schedule` (`core_logic.py:211-345`) :
- Avant le bloc `target_idx` (ligne 272), consulte `crossing_pair_assignments[key]` où `key = f"{mission_id}|A{aller_idx_dans_periode}|R{retour_opposant_idx}"`.
- Si une assignation existe et que la VE assignée est dans la fenêtre `[i, terminus]`, **forcer** `target_idx` sur l'index de cette VE (au lieu de prendre la prochaine VE en aval).
- Calculer le `planned_stop_extension` nécessaire pour que la rencontre ait lieu sur la VE assignée (au lieu de simplement lire `crossing_strategy.stop_durations`).

Cascade de fallback :
1. Honorer l'assignation `crossing_pairs`.
2. Si extension > `max_delay_minutes` → fallback sur `crossing.stop_durations` (legacy).
3. Si toujours infaisable → fallback sur l'heuristique actuelle (prochaine VE en aval).
4. Marquer le trajet : `final_path[*]['crossing_assignment_violated'] = True` pour la pénalité douce (Partie B).

#### A.4 Initialisation, mutation, crossover

Dans `optimisation_logic.py:GeneticOptimizer` :
- `_initialize_population` (lignes 521-580) : après assemblage de chaque génome, appeler un nouveau helper `_seed_crossing_pairs(genome)` qui exécute l'énumérateur sur le `timing` du génome et tire la VE assignée parmi `candidate_ve` (biais 30 % vers `natural_ve`).
- `_crossover` (lignes 655-686) : ajouter un bloc miroir de la logique `crossing` actuelle, héritage clé-par-clé entre les parents.
- `_mutate` (lignes 688-736) : 25 % de probabilité par clé de tirer une nouvelle VE parmi les candidates.

#### A.5 Exemple — 1 mission, 2 trains/h, 3 VE

Mission M1 A→D, VE = {B, C}, fréquence = 2/h, `reference_minutes = "0,30"`, fenêtre 06:00–08:00.

Allers : A0=06:00, A1=06:30, A2=07:00, A3=07:30. Retours : R0=06:35, R1=07:05, R2=07:35, R3=08:05.

Rencontres idéalisées : `(A0,R0)≈06:18 près de B`, `(A1,R0)≈06:32 près de C`, `(A1,R1)≈06:48 près de B`, etc.

Génome possible :
```python
'crossing_pairs': {
    "M1|A0|R0": "B", "M1|A1|R0": "C",
    "M1|A1|R1": "B", "M1|A2|R1": "C",
    "M1|A2|R2": "B", "M1|A3|R2": "C", "M1|A3|R3": "B",
}
```

Une mutation `"M1|A0|R0": "C"` force A0 à pousser au-delà de B sans arrêt puis attendre à C — si l'extension reste ≤ `max_delay_minutes`, c'est respecté ; sinon fallback.

---

### Partie B — Pénalité explicite de dégradation usager

#### B.1 Branchement de `max_delay_minutes`

Modifier la signature de `_score_chronologie_bruit` (`core_logic.py:930`) et de `evaluer_configuration` (`core_logic.py:380-520`) :

```python
def _score_chronologie_bruit(chronologie, warnings, config=None):
    max_delay = (config.crossing_optimization.max_delay_minutes
                 if config and config.crossing_optimization
                 else 5)
    ...
```

Mettre à jour les sites d'appel dans `evaluer_params_simulation` (`optimisation_logic.py`, autour de la ligne 743) — `config` est déjà dans le scope.

#### B.2 Nouvelle forme de pénalité (remplace lignes 945-956)

```python
penalty_arrets_ligne = 0.0
violated_count = 0
if chronologie:
    for steps in chronologie.values():
        for step in steps:
            if step.get('crossing_assignment_violated'):
                violated_count += 1
            ext = step.get('crossing_extension_min', 0)
            if ext <= 0:
                continue
            if ext <= max_delay:
                penalty_arrets_ligne += ext * 50.0          # linéaire douce
            else:
                over = ext - max_delay
                penalty_arrets_ligne += max_delay * 50.0 + (over ** 2) * 800.0  # quadratique raide
penalty_arrets_ligne += violated_count * 1500
```

Le bonus Gini (`- avg_gini * 3000`) reste inchangé conformément à la consigne utilisateur.

---

### Partie C — Cache de génome inter-générations (processus maître)

Nouvelle classe dans `optimisation_logic.py` (à insérer après `SolutionCache`, ligne 188) :

```python
class GenomeCache:
    """Cache process-master, persistant sur toutes les générations d'un run."""
    def __init__(self, max_size=5000):
        self.cache = {}
        self.access = {}
        self.max_size = max_size

    def key(self, genome) -> str:
        canonical = json.dumps(genome, sort_keys=True, default=str)
        return hashlib.md5(canonical.encode()).hexdigest()

    def get(self, genome):
        k = self.key(genome)
        v = self.cache.get(k)
        if v is not None:
            self.access[k] = self.access.get(k, 0) + 1
        return v

    def put(self, genome, score, warnings, chronologie):
        if len(self.cache) >= self.max_size:
            evict = min(self.access, key=self.access.get)
            self.cache.pop(evict, None); self.access.pop(evict, None)
        k = self.key(genome)
        self.cache[k] = (score, warnings, chronologie)
        self.access[k] = 0
```

Intégration dans `_evaluate_population_parallel` (lignes 582-621) :
1. Avant la construction d'`args_list`, séparer `population` en `cached_hits` et `to_evaluate`.
2. Soumettre uniquement `to_evaluate` à `ProcessPoolExecutor`.
3. Après collecte des futures, écrire les nouveaux résultats dans `self.genome_cache`.
4. Concaténer `cached_hits + futures_results` avant retour.

Instanciation : `self.genome_cache = GenomeCache()` dans `GeneticOptimizer.__init__` après la ligne 391.

Gain attendu : 15–25 % de wall-time sur la durée totale (élitisme 10 % + doublons crossover en fin de convergence ≈ 20–30 % de hits).

---

### Partie D — UI : forcer l'optimisation de croisement en mode génétique

Dans `app.py` autour de la ligne 1015, remplacer le `st.checkbox` par :

```python
is_genetic = optimization_mode == "genetic"
enable_crossing_opt = st.checkbox(
    "Activer l'optimisation des croisements",
    value=True if is_genetic else False,
    disabled=is_genetic,
    help=("Verrouillé en mode génétique : l'algorithme optimise désormais "
          "les croisements par paire de trains."
          if is_genetic else
          "Prolonge stratégiquement les arrêts pour améliorer les croisements"),
)
```

Vérifier que la variable `enable_crossing_opt` alimente bien `CrossingOptimization(enabled=...)` à la ligne 1186 (déjà OK).

---

## Fichiers à modifier

| Fichier | Sections / lignes | Nature |
|---|---|---|
| `core_logic.py` | après ligne 75 (nouvelle fonction `enumerer_rencontres`) | Ajout |
| `core_logic.py` | 211–345 (`SimulationEngine.solve_mission_schedule`) | Modif moteur — chemin opt-in |
| `core_logic.py` | 547–927 (`executer_simulation_evenementielle`) | Plombage des assignations |
| `core_logic.py` | 930–958 (`_score_chronologie_bruit`) + 380–520 (`evaluer_configuration`) | Signature + nouvelle pénalité |
| `optimisation_logic.py` | 70–99 (`SimulationParams`) | Champ `crossing_pair_assignments` |
| `optimisation_logic.py` | 188 (après `SolutionCache`) | Ajout classe `GenomeCache` |
| `optimisation_logic.py` | 319–357 (`_evaluate_genome_worker`) | Passage des assignations |
| `optimisation_logic.py` | 378–391 (`__init__` GA) | Instancier `self.genome_cache` |
| `optimisation_logic.py` | 521–580 (`_initialize_population`) | Helper `_seed_crossing_pairs` |
| `optimisation_logic.py` | 582–621 (`_evaluate_population_parallel`) | Lookup cache avant submit |
| `optimisation_logic.py` | 655–736 (`_crossover` + `_mutate`) | Étendre aux `crossing_pairs` |
| `app.py` | 1015–1019 (checkbox) | Forçage mode génétique |

## Réutilisations existantes

- `_get_infra_at_gare` (`core_logic.py:34-54`) — déjà utilisé par `_identify_ve_gares` (`optimisation_logic.py:438`) pour lister les VE candidates.
- `construire_horaire_mission` non-cachée (`core_logic.py:1118`) — fournit les temps de parcours utilisés par l'énumérateur de rencontres.
- `_calculer_stats_homogeneite` (`core_logic.py`, appelé ligne 936) — Gini conservé tel quel.
- `SolutionCache` (`optimisation_logic.py:148-186`) — le `GenomeCache` réutilise le pattern LRU mais vit dans le maître.

---

## Vérification

1. **Lancer le benchmark de base avant modification** : `streamlit run app.py`, configurer un cas représentatif (10 gares dont 3 VE, 2 missions, 4 h de service, mode génétique 50×100), noter le temps total et le score final.
2. **Après Partie C uniquement** : relancer le même cas, attendre une réduction du temps total de 15–25 %, score identique au cent-millième près.
3. **Après Parties A+B** : relancer, vérifier visuellement sur le diagramme de Marey que :
   - Les croisements ont lieu aux VE attendues (lecture directe sur la chronologie).
   - Aucun arrêt > `max_delay_minutes` n'est introduit pour les rencontres résolues.
   - Le nombre de rames et la régularité (Gini) ne se dégradent pas.
4. **Test de non-régression** : passer en mode `simple` (cadre baseline) et vérifier que la sortie est inchangée — la baseline du génome 0 doit toujours produire le résultat utilisateur original.
5. **Test UI Partie D** : basculer entre modes `smart_progressive` et `genetic` dans la sidebar, confirmer que la case « Activer l'optimisation des croisements » devient cochée + grisée en mode génétique.

## Risques

- **Modification du moteur (Partie A.3) = risque le plus élevé** : `solve_mission_schedule` est tightly coupled. Mitigation : ajouter un flag `config.crossing_optimization.use_per_pair_assignment` (défaut `True` en génétique, `False` ailleurs) pour rendre le nouveau chemin opt-in pendant la mise au point.
- **Dérive d'énumération** : les rencontres réelles peuvent différer des idéalisées si turnaround_buffers > 0 — d'où la cascade de fallback.
- **Mémoire du cache** : 5000 entrées × chronologie peut être volumineux. Stocker uniquement `(score, warnings_dict, chronologie_compacte)` ; les éléments réémis comme élites seront re-simulés en sortie de GA si nécessaire.
- **Compatibilité sessions Streamlit antérieures** : tout accès à `genome['crossing_pairs']` doit utiliser `.get('crossing_pairs', {})`.
