# Robustesse : Auray–Quiberon et rames de renfort

## Contrat

`evaluer_robustesse(..., reserve_gares=(), max_reserve_rames=1, activer_renforts=True)`

- `max_reserve_rames` est un entier positif ou nul, pas un booléen. Il borne les identités physiques supplémentaires utilisées sur tout le scénario, pas les courses ni les activations. Les scénarios 5/10/20 sont indépendants.
- Les gares sélectionnées sont des lieux de déploiement initial possibles. Plusieurs identités peuvent avoir le même lieu initial, sans dépasser le plafond global. Une identité conserve son numéro quand elle circule ailleurs.
- Les rames de rotation déjà disponibles et de matériel compatible sont essayées avant les réserves supplémentaires. À gain égal, une réserve déjà utilisée passe avant une nouvelle identité. Avec un plafond de 0, les redéploiements de rotation restent possibles.
- La disponibilité de rotation utilise la position et l'heure effectivement atteintes après le service précédent, son retournement, puis une marge de préparation de 5 min. Une suppression conserve la position réellement atteinte, sans trajet de retour inventé. Avant sa première mission, une rame est supposée stationnée à son origine initiale ; ses missions futures restent protégées.
- Les substitutions partent à l'heure voyageur prévue. Tout le scénario est rejoué, avec infrastructure et énergie. Une substitution ne peut supprimer ou retarder un autre service, y compris un train déjà en route. Aucune téléportation ni avance voyageur.
- La sécurité énergétique historique est conservée : un défaut batterie, même nominal et sur une autre rame, empêche de valider un renfort. Le matériel de la mission n'est pas transformé pour contourner une alerte.
- `activer_renforts=False` désactive les substitutions et renvoie `avec_reserve=None`. Ce paramètre et le plafond font partie de `fingerprint_robustesse`, donc invalident le cache UI.

Les résultats exposent `reserve_rames_utilisees` (nombre d'identités supplémentaires distinctes), `reserve_activations`, `redeploiements_rotation` et `rotation_refus`. Les enregistrements conservent `service_id` pour l'API et fournissent `service_label` pour les humains : `08:25 B → A`.

Chaque scénario expose aussi `candidates_evaluated` (point et score après correction énergétique), `points_ex_aequo` et `tie_count`. La réserve est évaluée pour le point représentatif affiché, pas pour chacun des autres ex æquo. L'UI le dit et permet de consulter tous les scores.

## Mesure du score

Retard cumulé = somme des retards d'arrivée des services opérés + attente du prochain service effectivement assuré dans la même mission et direction + 10 min par suppression.

À défaut de prochain départ observé dans l'horizon, l'hypothèse de fréquence/intervalle est affichée explicitement. Ce score n'est ni le seul retard du train incident, ni le temps de retour à l'horaire normal.

Libellés identiques dans les comparaisons et les détails : « Retard cumulé — Sans réserve » et « Retard cumulé — Avec réserve ».

## Preuve sur le scénario enregistré

Entrée inchangée : `scenarios/Bretagne/Auray-Quiberon/auray_quiberon_freq_heure_omnibus.json`.

Chargement par `load_scenario`, génération par `generer_tous_trajets_optimises` avec partage des rotations et stratégie simple, matériel batterie et paramètres énergétiques du fichier conservés.

66 candidats par amplitude : 33 départs et 33 arrivées au croisement de Plouharnel-Carnac. Ils couvrent 06:00 à 22:22, toutes les heures de 06 h à 22 h. Les identifiants des 33 départs correspondent exactement aux 33 services compilés, y compris la dernière mission de 22:00.

| Tranche horaire | Départs | Croisements |
| --- | ---: | ---: |
| 06 h | 2 | 1 |
| Chaque heure de 07 h à 21 h | 2 | 2 |
| 22 h | 1 | 2 |
| Total | 33 | 33 |

Tous les candidats passent par le rejeu et la correction de recharge avant classement. Aucun échantillonnage ni arrêt après les premiers trains.

| Incident | Maximum du score (min) | Départs ex æquo Quiberon → Auray | Nombre d'ex æquo | Maximum aux croisements | Score au départ de 06:00 |
| --- | ---: | --- | ---: | ---: | ---: |
| +5 min | 10 | 07:00, 08:00, …, 22:00 | 16 | 5 | 5 |
| +10 min | 25 | 07:00, 08:00, …, 21:00 | 15 | 15 | 15 |
| +20 min | 75 | 07:00, 08:00, …, 20:00 | 14 | 45 | 45 |

Le départ de 07:00 est donc le premier maximum dans l'ordre déterministe, pas un maximum unique, et pas le premier départ de la journée. Le départ de 06:00 est bien testé. Le motif horaire régulier explique les égalités ; la fin de journée raccourcit les cascades les plus tardives.

Pour le représentant 07:00, les services retardés donnent :

- +5 : départs 07:00 Quiberon → Auray (+5) et 06:49 Auray → Quiberon (+5), soit 10.
- +10 : les deux précédents (+10 chacun) et 07:49 Auray → Quiberon (+5), soit 25.
- +20 : 07:00 (+20), 07:49 (+15), 09:00 (+5), 06:49 (+20), 08:00 (+10), 08:49 (+5), soit 75.

Aucune suppression dans ces trois représentants. Le nominal batterie atteint déjà le seuil SoC de 20 % pour les deux rames. Ces alertes restent visibles : aucune recharge corrective prétendument réussie ni réserve fictive n'est annoncée. Les trois courbes « avec réserve » sont néanmoins réellement dessinées, même quand aucun renfort n'est admissible.

Le classement ne force pas le matin : un autre test physique, sans mock, place un vrai maximum unique à un croisement de 16:23. Score 88, contre au plus 82 pour les départs et 5 pour les points du matin.

## Reproduction du défaut de figure

Le code de `plotting.py` du commit `1fa2967` est exécuté avec une chronologie réellement rejouée, comprenant les identités `1` et `Réserve B #1`. Pour chacune des amplitudes 5/10/20, la trace réelle est :

```
File "plotting.py@1fa2967", line 189, in creer_graphique_horaire
    for i, (id_train, trajets) in enumerate(sorted(chronologie_trajets.items())):
TypeError: '<' not supported between instances of 'str' and 'int'
```

Le tri corrigé conserve les types et identités d'origine. Le même rejeu produit maintenant trois figures Matplotlib effectivement rendues et enregistrées en PNG. Le scénario de test à réserve déployée est distinct d'Auray : il ne change pas le matériel batterie du fichier utilisateur pour obtenir artificiellement une réserve.

## Vérification reproductible

```
python -m pytest -q
python -m pytest -q test_robustness_fleet.py test_robustness_audit.py test_robustness_plotting.py
```

Les nouveaux tests couvrent notamment plafond 0/1/2 et validation API, cache UI, priorité rotation, réutilisation, plusieurs identités au même lieu, disponibilité après suppression, matériel/énergie, absence de dégradation des missions suivantes, vrais rendus Matplotlib et audit exhaustif du fixture.
