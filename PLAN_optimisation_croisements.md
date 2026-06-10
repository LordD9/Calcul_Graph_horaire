# Plan d'implémentation — Optimisation des croisements (modes non-simples)

> **Contexte** : l'optimisation (smart_progressive, exhaustif, génétique) manque des solutions
> meilleures qui, en décalant l'horaire de départ retour (retournement plus long au terminus)
> ou en favorisant une voie d'évitement (VE) plutôt qu'une autre, donneraient un meilleur
> cadencement et des trajets plus courts (moins d'attente en ligne).
>
> **Exigence utilisateur** : les décalages au terminus sont souhaitables mais doivent rester
> **réguliers** — pas de départ retour à la minute 5, puis 8, puis 9 d'une heure à l'autre.
> → On ne décale pas par un buffer libre, on **cadence les retours sur une grille de minutes
> de référence**, comme les allers.

## Diagnostic (résumé)

| # | Problème | Localisation |
|---|---|---|
| D1 | Le score ne mesure pas le temps de parcours ; les attentes subies en conflit ne sont enregistrées nulle part dans la chronologie | `core_logic.py:1126` (`_score_chronologie_bruit`), branche conflit `core_logic.py:1052-1060` |
| D2 | Asymétrie : arrêt stratégique VE pénalisé 50 pts/min, attente subie gratuite → le génétique apprend à ne jamais utiliser les stratégies de croisement | `core_logic.py:1152` |
| D3 | Les buffers de retournement sont pénalisés directement (`+ sum(buffers)*2`), jamais récompensés | `optimisation_logic.py:912` |
| D4 | Les retours partent "dès que possible" : aucun levier pour les décaler individuellement, et le buffer s'applique aux deux terminus à la fois | `core_logic.py:1094-1114` |
| D5 | `crossing_pairs` (VE préférée) ne s'applique qu'aux allers ; le fallback est silencieux ; la pénalité `crossing_assignment_violated` est lue mais jamais écrite | `core_logic.py:866`, `core_logic.py:1146` |
| D6 | `smart_progressive` = descente coordonnée gloutonne en un seul passage → ne trouve jamais les mouvements conjoints (cadencement + retournement) | `optimisation_logic.py:955-1076` |
| D7 | Le gène `timing` écrase les `reference_minutes` multi-valeurs (remplace `"0,30"` par une valeur unique) ; incohérent avec `_seed_crossing_pairs` qui le traite en offset | `optimisation_logic.py:102-109` |
| D8 | Code mort : `SolutionScorer.score_solution` / `_evaluate_crossing_quality` inspectent des clés inexistantes ; la mutation ne peut jamais *ajouter* un gène `crossing` à un génome qui n'en a pas | `optimisation_logic.py:230-295`, `optimisation_logic.py:847-855` |

---

## Phase 1 — Refonte du score : créer le gradient manquant

**Objectif** : que "trajets plus courts" et "attente en ligne réduite" soient visibles et
récompensés, et que arrêt stratégique ≡ attente subie (même coût pour le même effet physique).

### 1.1 Tracer l'attente subie en conflit dans la chronologie

Fichier : `core_logic.py`, `executer_simulation_evenementielle`.

- Dans la branche conflit (`core_logic.py:1052-1060`), quand l'événement est re-planifié à
  `fin_conflit`, accumuler le délai dans `details["conflict_wait_min"]`
  (`+= (fin_conflit - heure_depart_reelle).total_seconds()/60`).
- Au moment où le bloc passe enfin (branche `not conflit`), si `conflict_wait_min > 0` :
  - si le train attendait dans une gare (index_etape > 0 ou départ de mission) : émettre une
    entrée d'arrêt `{"start": heure_arrivée_théorique_attente, "end": heure_depart_reelle,
    "origine": gare, "terminus": gare, "subi": True, "crossing_extension_min": wait}` —
    visuellement l'attente apparaîtra aussi sur le graphique de Marey (bonus de lisibilité) ;
  - a minima, stocker `"conflict_wait_min": wait` sur le premier segment du bloc pour le score.
- ⚠️ Ne PAS générer ces entrées pour les trajets fictifs (`is_trajet_fictif`).

### 1.2 Terme de temps de parcours excédentaire dans le score

Fichier : `core_logic.py`, `_score_chronologie_bruit` → nouvelle signature :

```python
def _score_chronologie_bruit(chronologie, warnings, max_arret_ligne_min=5,
                             durees_theoriques=None):
```

- `durees_theoriques` : dict `{mission_label: duree_min}` (ex. `"A → B": 42`), calculé une
  seule fois dans `evaluer_params_simulation` via `construire_horaire_mission(m, 'aller'/'retour',
  df_gares)[-1]['time_offset_min']` pour chaque mission active et chaque sens.
- Reconstruire chaque course depuis la chronologie : groupes de segments entre deux
  `is_mission_start=True` (même logique que `_calculer_stats_homogeneite`).
  `excès = (fin_dernier_segment − départ_premier_segment) − durée_théorique`, borné à ≥ 0.
- Nouveau terme : `penalty_temps_parcours = somme_excès_minutes * POIDS_EXCES` avec
  `POIDS_EXCES = 60.0` (point de départ ; ordre de grandeur : 10 min d'excès cumulé ≈ 600 pts,
  comparable à un déplacement de 0.2 sur le Gini moyen, nettement sous le coût d'une rame).
  Exposer la constante en haut de fichier pour calibration facile.

### 1.3 Symétriser arrêt stratégique / attente subie

- L'actuel barème `crossing_extension_min` (50/min puis quadratique au-delà de
  `max_arret_ligne_min`) **est supprimé en tant que pénalité séparée** : ces minutes sont déjà
  comptées dans l'excès de temps de parcours (1.2), au même tarif que l'attente subie.
- On garde uniquement la composante quadratique au-delà de `max_arret_ligne_min` (plafond
  utilisateur "Retard max acceptable") appliquée **aux deux types d'attente** (stratégique et
  subie), pour respecter la borne UI.

### 1.4 Supprimer la pénalité directe des buffers

Fichier : `optimisation_logic.py`, `evaluer_params_simulation` (`:912`) — supprimer :

```python
extra_delay = sum(params.turnaround_buffers.values()) * 2
score += extra_delay
```

Le coût d'un retournement long doit émerger de ses effets réels (nombre de rames, Gini,
temps de parcours), pas d'une taxe a priori.

### Validation Phase 1
- Mode `simple` inchangé (aucun paramètre d'optim) : mêmes chronologies qu'avant.
- Sur un scénario voie unique avec 1 VE : vérifier qu'un génome "arrêt stratégique 5 min à la VE"
  et un génome "attente subie 5 min au même endroit" ont désormais le même score.

---

## Phase 2 — Retours cadencés : décalage au terminus régulier par construction

**Objectif** : permettre de retarder le départ des retours pour optimiser les croisements,
tout en garantissant des départs aux **mêmes minutes chaque heure**.

### 2.1 Principe

On remplace le levier "buffer libre en minutes" (terminus B) par une **minute de référence
retour** par mission : `retour_offset ∈ {0..59}` (ou `None` = comportement actuel "ASAP").

Le départ retour devient : *premier créneau de la grille `{(retour_offset + k·intervalle) % 60}`
tel que `créneau ≥ arrivée + temps_retournement_B_min`*. L'intervalle est celui de la mission
(`60/frequence`). Régularité garantie par construction : tous les retours d'une mission partent
aux mêmes minutes, comme les allers.

Le buffer libre existant reste disponible en interne (compat), mais les optimiseurs ne
l'explorent plus pour le terminus B : ils explorent `retour_offset`.

### 2.2 Moteur événementiel

Fichier : `core_logic.py`, `executer_simulation_evenementielle`.

- Nouveau paramètre : `retour_reference_offsets: dict {mission_id: int} = None`.
- Dans `fin_mission` pour un aller (`core_logic.py:1096-1111`) :
  ```python
  heure_dispo_min = heure_arrivee_mission + timedelta(minutes=t_ret_min + buf)
  offset = retour_reference_offsets.get(mission_id)
  if offset is not None:
      heure_depart_retour = _prochain_creneau(heure_dispo_min, offset, intervalle_mission)
  else:
      heure_depart_retour = heure_dispo_min   # comportement actuel
  ```
  `_prochain_creneau(t, offset, intervalle)` : petite fonction pure (candidate pour `utils.py`),
  arrondit au prochain instant dont la minute appartient à la grille. Attention au passage
  d'heure et aux intervalles non diviseurs de 60 (ex. fréquence 1.5/h → grille glissante :
  générer les créneaux depuis `dt_debut` comme pour les allers, pas par modulo simple).
- L'intervalle de la mission doit être accessible dans `fin_mission` : le passer dans `details`
  ou le recalculer depuis `mission_cfg['frequence']`.
- En cas de conflit au départ du retour : retry à `fin_conflit` (comportement actuel). Le Gini
  et le terme de temps de parcours pénaliseront le créneau raté ; l'optimiseur convergera vers
  des offsets sans conflit. (Variante "sauter au créneau suivant" : non retenue en V1, plus
  intrusive.)
- `inject_from_terminus_2` (`core_logic.py:729-736`) : intégrer l'attente de créneau dans
  l'estimation `temps_avant_service` (prendre le pire cas : `+ intervalle`).

### 2.3 Exposition dans `SimulationParams` et les génomes

Fichier : `optimisation_logic.py`.

- `SimulationParams` : nouveau champ `retour_offsets: Dict[str, int] = None` + passage à
  `executer_simulation_evenementielle(retour_reference_offsets=...)` dans
  `evaluer_params_simulation`.
- **Génétique** — génome : nouvelle clé `'retour_offsets': {mid: int|None}` :
  - init population : 1/3 `None` (baseline ASAP), 1/3 multiples de 5, 1/3 aléatoire 0-59 ;
    le génome 0 (baseline) garde `None` partout ;
  - `_crossover` : héritage clé par clé comme `turnaround_buffers` ;
  - `_mutate` : 30 % de chance par mission — petit décalage ±5 (70 %) ou valeur aléatoire /
    retour à `None` (30 %).
- **smart_progressive** : nouvelle phase `('Retour cadencé', [('retour_offset', mid,
  [None] + list(range(0, 60, 5))) ...])` insérée après la phase Cadencement, **à la place** de
  la phase Retournement actuelle (le buffer libre B disparaît de l'exploration) ; l'Affinement
  final teste `range(0, 60)` sur cadencement **et** retour_offset.
- **Exhaustif** : remplacer `turnaround_range` par `retour_offset_range = [None, 0, 10, 20, 30,
  40, 50]` (coarse) dans le produit cartésien ; mêmes garde-fous de taille.

### 2.4 Buffers A/B séparés (correctif D4 résiduel)

Le retournement au terminus A (origine) n'a pas besoin de cadencement : les départs aller sont
déjà sur grille. Mais le buffer unique actuel s'applique aux deux bouts :

- `core_logic.py:1094` : remplacer `buf = turnaround_buffers.get(mission_id, 0)` par une
  lecture directionnelle — accepter soit un int (compat : appliqué aux deux), soit un dict
  `{"A": int, "B": int}`.
- Les optimiseurs n'explorent plus que le buffer A (valeurs petites `[0, 3, 5, 8, 10]`), le
  côté B étant couvert par `retour_offset`.

### Validation Phase 2
- Scénario 1 mission, fréquence 1/h, `retour_offset=15` : tous les départs retour à minute 15,
  toutes les heures (vérifier sur le graphique de Marey et l'export Excel).
- Vérifier l'indice d'homogénéité affiché ≈ 1.0 pour le sens retour.
- Fréquence 2/h : départs retour aux minutes {offset, offset+30}.

---

## Phase 3 — Choix du point de croisement : signal et couverture des retours

### 3.1 Écrire réellement `crossing_assignment_violated`

Fichier : `core_logic.py`, branche conflit avec `use_preferred_ve=True` (`core_logic.py:1042-1051`).

- Quand on re-tente avec `use_preferred_ve=False`, marquer `new_details["preferred_ve_failed"] = True`.
- Quand le bloc de repli passe, poser `"crossing_assignment_violated": True` sur le premier
  segment émis. La pénalité existante (`core_logic.py:1146,1156`, 1500 pts) devient active.
- **Recalibrer la pénalité à ~300 pts** : un fallback n'est pas une faute grave, c'est un
  signal "ce génome promet une VE qu'il ne tient pas" ; 1500 pts écraserait le terme Gini.

### 3.2 Étendre la VE préférée aux retours

Fichier : `core_logic.py:865-866` — supprimer la condition `trajet_spec == "aller"`.

- `crossing_pair_assignments` devient directionnel : `{mission_id: {"aller": [...], "retour":
  [...]}}`, avec compat liste simple = les deux sens.
- Génétique : `_seed_crossing_pairs` peuple les deux sens à partir de `enumerer_rencontres`
  (la VE naturelle d'une rencontre vaut pour les deux trains qui se croisent) ; la mutation
  `crossing_pairs` tire le sens au hasard.
- `enumerer_rencontres` (`core_logic.py:75`) : prendre en compte `retour_offset` quand il est
  fourni — les départs retour ne sont plus `t_a + duree_aller + t_ret_min` mais le prochain
  créneau de la grille (réutiliser `_prochain_creneau`). Sinon les rencontres idéalisées
  divergent des rencontres réelles et le seeding devient contre-productif.

### Validation Phase 3
- Scénario 2 VE : forcer `crossing_pairs` vers la VE n°2 en mode simple instrumenté, vérifier
  que les croisements s'y produisent dans les deux sens ; vérifier que le flag violated apparaît
  quand on force une VE impossible.

---

## Phase 4 — Sémantique du gène `timing` : offset, pas valeur absolue (D7)

Fichier : `optimisation_logic.py`, `SimulationParams.get_adjusted_reference_minutes` (`:102-109`).

```python
# AVANT : result[str(i)] = str(offset)                  # écrase "0,30" → "17"
# APRÈS : décalage de chaque minute du pattern d'origine
original = [int(x) for x in str(m.get('reference_minutes', '0')).split(',') if x.strip().isdigit()] or [0]
result[str(i)] = ",".join(str((v + offset) % 60) for v in sorted(original))
```

- Cohérent avec `_seed_crossing_pairs` (`:631`) qui fait déjà `(m + timing_offset) % 60`.
- L'espace de recherche `range(0, 60)` reste valable (offset 0 = baseline exacte).
- ⚠️ Mode `"simple"` de `optimiser_graphique_horaire` (`:1231-1239`) : il aplatit aussi les
  multi-valeurs (`refs[0]`) — le corriger de la même façon (passer le pattern complet,
  offset 0).

---

## Phase 5 — Capacité d'exploration des optimiseurs

### 5.1 smart_progressive : passes multiples + mouvements conjoints (D6)

Fichier : `optimisation_logic.py`, `_optimisation_smart_progressive`.

- **Boucler les phases 2 fois** (ou jusqu'à absence d'amélioration sur une passe complète).
  `total_steps` ×2 pour une barre de progression honnête.
- **Nouvelle phase conjointe** (avant l'Affinement) : grille grossière 2D par mission —
  `(cadencement ∈ range(0,60,10)) × (retour_offset ∈ [None,0,15,30,45])`, soit 30 essais/mission.
  C'est elle qui attrape les solutions "décaler le départ ET le retour ensemble".
- Garder `tol=1.0` pour l'acceptation, mais l'Affinement final passe à `tol=0.0`.

### 5.2 Génétique : la mutation peut créer des gènes (D8)

Fichier : `optimisation_logic.py`, `_mutate` (`:847-855`).

- 10 % de chance par mission **sans** gène `crossing` d'en créer un (durées tirées de
  `cross_choices` sur les VE de la mission — utiliser `self.crossing_points[mission_label]`
  plutôt que toutes les VE de la ligne).
- Idem pour `crossing_pairs` (déjà partiellement le cas) et `retour_offsets` (cf. 2.3).

### 5.3 Nettoyage du code mort (D8)

- Supprimer `SolutionScorer.score_solution` et `_evaluate_crossing_quality` (jamais appelés
  par le chemin réel ; les clés `crossings`/`crossing_extensions` n'existent pas). Garder
  `SolutionScorer.is_valid_solution`.
- Si on veut conserver une classe scorer, la réduire à un wrapper de
  `_score_chronologie_bruit` pour qu'il n'y ait **qu'une seule** fonction de coût.

---

## Ordre d'implémentation et dépendances

```
Phase 1 (score)          ── indépendante, à faire en premier (débloque le gradient)
Phase 2 (retours cadencés) ── dépend de 1 pour être correctement évaluée
Phase 3 (VE préférée)    ── dépend de 2.2 (_prochain_creneau) pour enumerer_rencontres
Phase 4 (timing offset)  ── indépendante, petite, peut se glisser n'importe quand
Phase 5 (exploration)    ── dépend de 2 et 3 (nouveaux gènes à explorer)
```

Chaque phase = un commit distinct, testé en lançant l'app (`streamlit run app.py`) sur :
1. un scénario voie unique 1 VE, 1 mission 1/h (cas pédagogique : le décalage retour doit
   supprimer l'attente en ligne au profit d'une attente au terminus, régulière) ;
2. un scénario 2 missions / 2+ VE (vérifier le choix de VE et l'absence de régression rames) ;
3. un scénario de la bibliothèque `scenarios/` en mode Calcul Energie (non-régression).

## Points de vigilance

- **Picklabilité** : les nouveaux champs de génome restent des dicts JSON-sérialisables
  (`GenomeCache.key` utilise `json.dumps(..., default=str)` ; `None` y est valide).
- **Baseline préservée** : le génome 0 / `_baseline_simulation_params()` doivent rester
  strictement équivalents au mode simple (tous les nouveaux paramètres à `None`/vide).
- **`heure_fin_service`** : borne les départs, pas les arrivées — le créneau retour choisi doit
  respecter `heure_depart_retour < dt_fin` (logique existante `core_logic.py:1104` conservée).
- **Calibration des poids** : après Phase 1, vérifier sur le scénario 1 que l'ordre des
  préférences est : violations ≫ rames ≫ (temps de parcours ↔ Gini) ≫ fallback VE. Ajuster
  `POIDS_EXCES` si le Gini domine trop (symptôme : l'optimiseur accepte de longues attentes
  en ligne pour des départs parfaitement réguliers).
- **UI** : aucun changement de schéma scénario JSON (les paramètres sont internes à
  l'optimiseur). Optionnel en fin de projet : afficher la minute de référence retour retenue
  dans le panneau de résultats.
