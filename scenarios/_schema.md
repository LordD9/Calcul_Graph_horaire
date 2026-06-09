# Format des scénarios Chronofer

Un scénario = un fichier `.json` autonome contenant l'infrastructure, les missions
et les paramètres matériel d'une étude énergétique. Tous les scénarios sont chargés
dans le mode **Calcul Energie** via l'encart "Charger un scénario" en haut de page.

## Champs

### `schema_version` (entier, requis)

Version du schéma. Permet une migration future si la structure évolue.
Version courante : `1`.

### `metadata` (objet)

- `nom` (string) : nom affiché dans la liste des scénarios.
- `description` (string) : description libre.
- `auteur` (string) : auteur du scénario.
- `date_creation` (string ISO `YYYY-MM-DD`) : date de création.
- `date_modification` (string ISO `YYYY-MM-DD`) : dernière modification.
- `tags` (array de strings) : étiquettes pour filtrer (région, type de ligne, etc.).

### `service` (objet)

- `heure_debut` (string `HH:MM`) : heure de début de service.
- `heure_fin` (string `HH:MM`) : heure de fin (borne pour le départ des trains).

### `infrastructure.gares` (array d'objets, requis)

Liste des gares triées par distance croissante. Chaque gare :

- `gare` (string) : nom.
- `distance` (float, km) : distance depuis l'origine.
- `infra` (string) : `VE` (voie d'évitement), `F` (voie unique sans croisement),
  `D` (toggle voie double).
- `electrification` (string) : `C1500`, `C25` (caténaire), `R<kW>` (recharge statique,
  ex. `R500`), `F` (non électrifié).
- `rampe_section_a_venir` (float, ‰) : pente de la section qui suit cette gare
  dans le sens des distances croissantes (positive = montée).

### `missions` (array d'objets, requis)

Chaque mission :

- `origine`, `terminus` (string) : gares de la mission.
- `frequence` (float, trains/h).
- `temps_trajet` (int, min) : temps trajet aller planifié.
- `temps_trajet_retour` (int, min) : temps trajet retour planifié.
- `trajet_asymetrique` (bool) : si `false`, `temps_trajet_retour` est égal à `temps_trajet`.
- `temps_retournement_A`, `temps_retournement_B` (int, min) : temps de retournement
  aux terminus origine (A) et terminus (B).
- `reference_minutes` (string) : minutes de référence pour le cadencement, séparées
  par des virgules (ex. `"15,45"`).
- `type_materiel` (string) : `diesel`, `electrique`, `bimode`, `batterie`.
- `inject_from_terminus_2` (bool) : injection de trajets fictifs avant le début de
  service pour que les rames soient au terminus B à l'ouverture.
- `passing_points` (array) : points de passage aller. Chacun :
  - `gare` (string)
  - `time_offset_min` (int) : offset depuis le départ de l'origine
  - `arret_commercial` (bool)
  - `duree_arret_min` (int)
- `passing_points_retour` (array) : idem pour le sens retour.
- `pp_raw_text` (string) : saisie en lot originale (préservée pour fidélité UI).

### `energy_params` (objet)

Un sous-dictionnaire par `type_materiel` utilisé (`diesel`, `electrique`, `bimode`,
`batterie`). Les types absents sont initialisés via `get_default_energy_params()`
au chargement.

Champs par matériel : `masse_tonne`, `capacite_batterie_kwh`, `facteur_charge_C`,
`recuperation_pct`, `soc_min_pct`, `soc_max_pct`, `capacite_eol_pct`,
`simuler_fin_de_vie`, `davis_A_N_t`, `davis_B_N_t_kph`, `davis_C_N_t_kph2`,
`accel_ms2`, `decel_ms2`, `facteur_aux_kwh_h`, `rendement_thermique_pct`,
`rendement_electrique_pct`, `kwh_per_liter_diesel`.

## Alimentation de la bibliothèque

L'écriture dans ce dossier se fait **hors application** : l'utilisateur télécharge
un scénario depuis l'UI (bouton dans le menu d'export du graphe horaire) puis
dépose le fichier ici manuellement (via Box, git, etc.).

Sous-dossiers facultatifs pour catégoriser (région, type de ligne…). Le chargeur
explore récursivement.
