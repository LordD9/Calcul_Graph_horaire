# Plan : Scénarios pré-enregistrés (option du mode Calcul Energie)

## Objectif

Ajouter au **mode "Calcul Energie"** existant la possibilité de :
1. **Charger un scénario** (infrastructure + missions + paramètres matériel) depuis une bibliothèque locale de fichiers JSON.
2. **Télécharger** la configuration courante au format JSON (via le menu de téléchargement du graphe horaire) pour la recharger plus tard.
3. **Importer** un fichier JSON téléchargé précédemment (round-trip).

Pas de mode dédié : la fonctionnalité s'intègre comme deux encarts dans le flux du mode Calcul Energie (chargement en haut, téléchargement à côté des autres exports).

La **bibliothèque côté serveur** (`scenarios/`) est alimentée **hors-app** : l'utilisateur télécharge le JSON depuis l'UI, puis dépose le fichier dans le dossier `scenarios/` manuellement (via Box, git, etc.). Pas de bouton d'écriture serveur dans l'UI.

---

## 1. Structure des données

### 1.1 Format de fichier : JSON

Un scénario = un fichier `.json` autonome :
- Lisible en clair, diffable via git
- Pas de dépendance supplémentaire (déjà utilisé dans le projet)
- Sérialisation/désérialisation directes d'un dict Python
- Partage individuel possible (téléchargement / pièce jointe)

### 1.2 Arborescence

```
Calcul_Graph_horaire/
├── app.py
├── scenarios/                          # nouveau dossier (bibliothèque)
│   ├── _schema.md                      # documentation du format
│   ├── nimes_grau_du_roi.json
│   └── ligne_des_cevennes.json
```

Sous-dossiers facultatifs pour catégoriser. Le chargeur explore récursivement.

### 1.3 Schéma JSON

```json
{
  "schema_version": 1,
  "metadata": {
    "nom": "Nîmes – Le Grau-du-Roi",
    "description": "Ligne TER mono-voie avec voies d'évitement, simulation batterie",
    "auteur": "Cerema",
    "date_creation": "2026-06-09",
    "date_modification": "2026-06-09",
    "tags": ["TER", "batterie", "voie_unique"]
  },
  "service": {
    "heure_debut": "06:00",
    "heure_fin": "22:00"
  },
  "infrastructure": {
    "gares": [
      {
        "gare": "Nîmes",
        "distance": 0.0,
        "infra": "VE",
        "electrification": "C1500",
        "rampe_section_a_venir": 5.0
      },
      {
        "gare": "Vauvert",
        "distance": 20.0,
        "infra": "D",
        "electrification": "F",
        "rampe_section_a_venir": -3.0
      },
      {
        "gare": "Le Grau-du-Roi",
        "distance": 50.0,
        "infra": "VE",
        "electrification": "R500",
        "rampe_section_a_venir": 0.0
      }
    ]
  },
  "missions": [
    {
      "origine": "Nîmes",
      "terminus": "Le Grau-du-Roi",
      "frequence": 1.0,
      "temps_trajet": 45,
      "temps_trajet_retour": 45,
      "trajet_asymetrique": false,
      "temps_retournement_A": 10,
      "temps_retournement_B": 10,
      "reference_minutes": "15,45",
      "type_materiel": "batterie",
      "inject_from_terminus_2": false,
      "passing_points": [
        {"gare": "Vauvert", "time_offset_min": 20, "arret_commercial": true, "duree_arret_min": 2}
      ],
      "passing_points_retour": [
        {"gare": "Vauvert", "time_offset_min": 25, "arret_commercial": true, "duree_arret_min": 2}
      ],
      "pp_raw_text": "Vauvert;20;2;25;2"
    }
  ],
  "energy_params": {
    "batterie": {
      "masse_tonne": 50,
      "capacite_batterie_kwh": 800,
      "facteur_charge_C": 1.0,
      "recuperation_pct": 75,
      "soc_min_pct": 20,
      "soc_max_pct": 95,
      "capacite_eol_pct": 70,
      "simuler_fin_de_vie": false,
      "davis_A_N_t": 15.0,
      "davis_B_N_t_kph": 0.1,
      "davis_C_N_t_kph2": 0.0041,
      "accel_ms2": 1.0,
      "decel_ms2": 0.6,
      "facteur_aux_kwh_h": 43.0,
      "rendement_thermique_pct": 38,
      "rendement_electrique_pct": 88,
      "kwh_per_liter_diesel": 10.0
    }
  }
}
```

### 1.4 Notes sur le schéma

- **`schema_version`** : entier. Permet une migration future.
- **`energy_params`** : un sous-dict par `type_materiel` utilisé. Les types non utilisés sont initialisés via `get_default_energy_params()`.
- **`pp_raw_text`** : conservé pour fidélité de la saisie originale (l'utilisateur retrouve son texte tel quel).
- **`service`** : heures de début/fin de service, à intégrer dans `session_state` pour être auto-suffisant.
- Aucune sortie calculée (chronologie, conso) n'est sauvegardée — seules les **entrées**.

---

## 2. Module de gestion des scénarios

### 2.1 Nouveau fichier : `scenarios_manager.py`

```python
# scenarios_manager.py
SCENARIOS_DIR = Path(__file__).parent / "scenarios"
CURRENT_SCHEMA_VERSION = 1

def list_scenarios() -> list[dict]:
    """Retourne [{path, metadata}, ...] en parcourant scenarios/ récursivement."""

def load_scenario(path_or_filelike) -> dict:
    """Charge un scénario depuis un chemin local ou un file-like (st.file_uploader).
    Applique les migrations, valide la structure."""

def serialize_scenario(scenario: dict) -> bytes:
    """Sérialise en JSON utf-8 (indent=2, ensure_ascii=False) — pour download_button."""

def build_scenario_from_session(st_session) -> dict:
    """Compose un dict scénario à partir du session_state courant."""

def apply_scenario_to_session(scenario: dict, st_session) -> None:
    """Injecte le scénario dans st.session_state (gares, missions, energy_params, service)."""

def validate_scenario(scenario: dict) -> list[str]:
    """Retourne une liste d'erreurs/warnings (gares manquantes, format, etc.)."""

def migrate_scenario(scenario: dict) -> dict:
    """Applique les migrations si schema_version < CURRENT_SCHEMA_VERSION."""

def slugify(nom: str) -> str:
    """Convertit un nom en nom de fichier sûr (a-z, 0-9, -, _)."""
```

### 2.2 Validation à la lecture

- Présence des clés obligatoires (`infrastructure.gares`, `missions`)
- Chaque mission référence des gares qui existent
- Types matériels référencés ont leur entrée dans `energy_params` (sinon valeurs par défaut + warning)
- Distances triables, origines/terminus cohérents

### 2.3 Injection dans `session_state`

`session_state.gares` est un `pd.DataFrame`. `apply_scenario_to_session` reconstruit via `pd.DataFrame(scenario["infrastructure"]["gares"])` puis trie par distance.

---

## 3. UI — Intégration dans le mode "Calcul Energie"

### 3.1 Encart "Bibliothèque" en haut du mode énergie

Inséré juste après la sélection `mode_calcul == "Calcul Energie"`, **avant** la section 1 "Gares".

```python
if mode_calcul == "Calcul Energie":
    with st.expander("📚 Charger un scénario", expanded=(st.session_state.gares is None)):
        tab_biblio, tab_fichier = st.tabs(["Bibliothèque", "Importer un fichier"])

        with tab_biblio:
            scenarios = scenarios_manager.list_scenarios()
            if not scenarios:
                st.caption("Aucun scénario dans `scenarios/`. Téléchargez votre configuration "
                           "actuelle pour démarrer la bibliothèque.")
            else:
                col_sel, col_btn = st.columns([4, 1])
                with col_sel:
                    options = ["— Sélectionner —"] + [s["metadata"]["nom"] for s in scenarios]
                    choix = st.selectbox("Scénario", options, key="scenario_selector")
                with col_btn:
                    if st.button("Charger", disabled=(choix == "— Sélectionner —")):
                        sc = next(s for s in scenarios if s["metadata"]["nom"] == choix)
                        data = scenarios_manager.load_scenario(sc["path"])
                        scenarios_manager.apply_scenario_to_session(data, st.session_state)
                        st.success(f"Scénario '{choix}' chargé.")
                        st.rerun()

                if choix != "— Sélectionner —":
                    md = next(s["metadata"] for s in scenarios if s["metadata"]["nom"] == choix)
                    st.caption(
                        f"**{md['nom']}** — {md.get('description', '')}  \n"
                        f"Auteur : {md.get('auteur', '?')} · Modifié : {md.get('date_modification', '?')}  \n"
                        f"Tags : {', '.join(md.get('tags', []))}"
                    )

        with tab_fichier:
            uploaded = st.file_uploader("Fichier scénario (.json)", type=["json"],
                                        key="scenario_upload")
            if uploaded is not None:
                try:
                    data = scenarios_manager.load_scenario(uploaded)
                    if st.button("Appliquer ce fichier"):
                        scenarios_manager.apply_scenario_to_session(data, st.session_state)
                        st.success(f"Scénario importé : {data['metadata'].get('nom', '?')}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Fichier invalide : {e}")
```

### 3.2 Téléchargement du scénario — dans le menu de téléchargement du graphe

Localiser le bloc où sont déjà placés `st.download_button` pour le PNG/CSV/Excel du graphe horaire. Ajouter un bouton supplémentaire **conditionné au mode énergie** :

```python
# Dans le bloc des téléchargements existants du graphe horaire
if mode_calcul == "Calcul Energie" and st.session_state.gares is not None:
    scenario_dict = scenarios_manager.build_scenario_from_session(st.session_state)
    scenario_bytes = scenarios_manager.serialize_scenario(scenario_dict)
    default_name = scenarios_manager.slugify(
        scenario_dict["metadata"].get("nom") or "scenario"
    ) + ".json"
    st.download_button(
        "📥 Télécharger le scénario (.json)",
        data=scenario_bytes,
        file_name=default_name,
        mime="application/json",
        help="Enregistre l'infrastructure, les missions et les paramètres matériel "
             "dans un fichier rechargeable plus tard."
    )
```

Le nom et tags du scénario peuvent être saisis dans un petit champ optionnel à proximité, ou laissés vides (slug par défaut basé sur la date).

### 3.3 Métadonnées au moment du téléchargement

Pour que le JSON téléchargé soit identifiable dans la bibliothèque, proposer (juste avant le `download_button`) un petit formulaire optionnel :

```python
col_nom, col_tags = st.columns(2)
nom_dl = col_nom.text_input("Nom du scénario (optionnel)", key="scenario_dl_name")
tags_dl = col_tags.text_input("Tags (séparés par virgules)", key="scenario_dl_tags")

scenario_dict = scenarios_manager.build_scenario_from_session(st.session_state)
if nom_dl:
    scenario_dict["metadata"]["nom"] = nom_dl
if tags_dl:
    scenario_dict["metadata"]["tags"] = [t.strip() for t in tags_dl.split(",") if t.strip()]
```

Si l'utilisateur ne saisit rien, le `nom` est laissé vide et le fichier est nommé selon la convention par défaut (voir §8.4).

---

## 4. Étapes d'implémentation (ordre suggéré)

1. **Créer le dossier `scenarios/`** avec un `_schema.md` documentant le format et 1 scénario d'exemple à la main.
2. **Créer `scenarios_manager.py`** avec les fonctions de la section 2.1.
   - Commencer par `list_scenarios`, `load_scenario`, `validate_scenario`, `serialize_scenario`
   - Puis `apply_scenario_to_session`, `build_scenario_from_session`, `save_scenario_to_library`
   - Tests manuels en REPL
3. **Ajouter `heure_debut_service` / `heure_fin_service` dans `session_state`** et brancher les `time_input` existants dessus (pour que le scénario soit auto-suffisant).
4. **Ajouter l'encart "Charger un scénario"** (tabs Bibliothèque / Importer) en tête du bloc `mode_calcul == "Calcul Energie"`.
5. **Ajouter le bouton de téléchargement JSON** dans le menu d'export du graphe horaire (avec champs optionnels nom/tags).
6. **Tester le round-trip** : config énergie → download → dépôt manuel dans `scenarios/` → recharge depuis bibliothèque → vérifier l'identité des entrées.
7. **Documenter dans `CLAUDE.md`** : section "Bibliothèque de scénarios" + format du dossier `scenarios/`.

---

## 5. Cas limites et points d'attention

### 5.1 Cohérence gares/missions après chargement
Si une mission charge des passing_points référençant une gare absente de l'infrastructure (édition manuelle), `construire_horaire_mission` les ignorerait. La validation doit le détecter et avertir.

### 5.2 Reset du `session_state` à l'import
Avant d'appliquer un scénario, vider :
- `st.session_state.chronologie_calculee = None`
- `st.session_state.warnings_calcul = {}`
- `st.session_state.energy_errors = []`
- Clés UI temporaires (`tt{i}`, `tt_retour_{i}`, `_prev_pp_raw_{i}`, `pp_raw_{i}`, etc.)

→ Helper `_reset_session_for_scenario_load()` qui itère sur ces clés.

### 5.3 Clés de widgets dynamiques
Les widgets utilisent des clés indexées par numéro de mission (`tt0`, `tt1`, ...). Si le scénario chargé a moins de missions que la config précédente, les clés résiduelles peuvent perturber l'affichage. Le reset doit supprimer toutes les clés de cette forme.

### 5.4 Migrations de schéma
Quand `schema_version` augmentera, `migrate_scenario` applique les transformations en chaîne.
```python
MIGRATIONS = {1: migrate_v1_to_v2, 2: migrate_v2_to_v3}
```

### 5.5 Nom du fichier téléchargé
- Slugification du nom saisi (ASCII, `[a-z0-9_-]+`)
- Si nom vide : convention de repli (cf. §8.4)
- Encodage du contenu JSON en UTF-8 explicite avec `ensure_ascii=False` pour conserver les accents lisibles

### 5.6 Validation des types matériels
Si un scénario référence un `type_materiel` absent de `energy_params`, initialiser depuis `get_default_energy_params()` et émettre un warning UI.

### 5.7 Heure de début/fin de service

Actuellement dans `time_input` non synchronisés au session_state global. Ajouter `st.session_state.heure_debut_service` / `heure_fin_service` et inclure sous `service.heure_debut` / `service.heure_fin`.

**Profiter de cette intégration pour corriger la sémantique de `heure_fin`** : aujourd'hui dans `core_logic.py`, c'est mixte (borne sur le départ + arrivée). Intention attendue : *"on programme des départs jusqu'à `heure_fin`, et le trajet correspondant est tracé entièrement même si l'arrivée dépasse `heure_fin`"*.

Corrections à appliquer dans `core_logic.py` :

- **Ligne 933** (`if heure_arrivee_finale > engine.dt_fin: continue`) : à **supprimer**. Un bloc dont le départ est valide doit être tracé jusqu'à son arrivée, peu importe l'heure.
- **Ligne 1104** (`if heure_dispo_retour + timedelta(minutes=temps_retour) <= engine.dt_fin`) : à **remplacer** par `if heure_dispo_retour < engine.dt_fin` (seul le départ du retour doit être borné, pas son arrivée).
- **Lignes 749, 762, 907** : conservées (bornent bien le départ).

Vérifier aussi dans `optimisation_logic.py` qu'aucun calcul de score (ex. comptage de trains) ne dépend strictement de l'arrivée < `heure_fin`.

### 5.8 Fichier importé invalide
`load_scenario` doit lever une exception claire en cas de JSON malformé ou schéma incorrect. L'UI catch et affiche `st.error`.

---

## 6. Évolutions possibles (hors scope initial)

- **Comparaison de scénarios** : charger deux scénarios et afficher diff infrastructure/missions/conso
- **Catalogue distant** : pointer `SCENARIOS_DIR` vers un dossier réseau partagé
- **Templates partiels** : charger uniquement l'infrastructure (sans missions) ou uniquement les paramètres matériel
- **Snapshot avec résultats** : option pour inclure la chronologie/conso calculée dans le JSON (lecture seule, pour archivage)

---

## 7. Estimation

| Étape | Temps estimé |
|---|---|
| Création `scenarios/` + scénario d'exemple | 30 min |
| `scenarios_manager.py` (≈ 180 lignes) | 1 h 45 |
| Intégration `heure_debut/fin_service` dans session_state | 30 min |
| Encart "Charger un scénario" (tabs) | 45 min |
| Bouton téléchargement JSON + champs nom/tags | 30 min |
| Tests manuels (round-trip) + ajustements | 1 h |
| Documentation CLAUDE.md | 15 min |
| **Total** | **≈ 5 h** |

---

## 8. Décisions validées

1. ~~**Bouton "Ajouter à la bibliothèque"**~~ — **non**. Seul le téléchargement JSON est exposé ; la bibliothèque est alimentée hors-app (Box / dépôt manuel dans `scenarios/`).
2. **Heures de service dans le scénario** — **oui**. Ajout de `heure_debut_service` / `heure_fin_service` dans `session_state` et branchement des `time_input` existants. **Bonus** : corriger en même temps la sémantique de `heure_fin` (cf. §5.7) — borne pour le **départ** des trains, pas pour leur arrivée.
3. **Import par fichier (tab "Importer un fichier")** — **oui**. Permet aussi à un utilisateur de charger ses scénarios dans la version en ligne.
4. **Nom du téléchargement par défaut** — **option C** : `<slug>.json` si nom saisi par l'utilisateur, sinon `chronofer_scenario_YYYYMMDD_HHMM.json`.
5. **Format JSON** — confirmé.
