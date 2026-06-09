# -*- coding: utf-8 -*-
"""
scenarios_manager.py
====================

Gestion des scénarios pré-enregistrés (infrastructure + missions + paramètres
matériel) pour le mode "Calcul Energie".

Un scénario = un fichier JSON autonome décrit dans `scenarios/_schema.md`.
La bibliothèque côté serveur est alimentée hors-app (dépôt manuel dans
`scenarios/`). L'utilisateur télécharge ses configurations depuis l'UI et
peut aussi recharger un fichier JSON via l'onglet "Importer un fichier".
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import IO, Union

import pandas as pd

from energy_logic import get_default_energy_params


SCENARIOS_DIR = Path(__file__).parent / "scenarios"
CURRENT_SCHEMA_VERSION = 1

_MATERIEL_TYPES = ("diesel", "electrique", "bimode", "batterie")


# =============================================================================
# Découverte et lecture
# =============================================================================

def list_scenarios() -> list[dict]:
    """Parcourt `scenarios/` récursivement et retourne la liste des scénarios.

    Chaque entrée : {"path": Path, "metadata": dict}. Les fichiers invalides
    sont silencieusement ignorés.
    """
    if not SCENARIOS_DIR.exists():
        return []
    result = []
    for path in sorted(SCENARIOS_DIR.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            md = data.get("metadata", {}) or {}
            if "nom" not in md:
                md["nom"] = path.stem
            result.append({"path": path, "metadata": md})
        except (OSError, json.JSONDecodeError):
            continue
    return result


def load_scenario(source: Union[Path, str, IO]) -> dict:
    """Charge un scénario depuis un chemin ou un file-like (st.file_uploader).

    Applique les migrations puis valide. Lève ValueError sur erreur grave.
    """
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
    else:
        path = Path(source)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Le fichier JSON doit contenir un objet à la racine.")

    data = migrate_scenario(data)

    errors = validate_scenario(data)
    fatal = [e for e in errors if not e.startswith("⚠️")]
    if fatal:
        raise ValueError("Scénario invalide : " + " ; ".join(fatal))
    return data


def serialize_scenario(scenario: dict) -> bytes:
    """Sérialise un scénario en JSON UTF-8 indenté (pour st.download_button)."""
    txt = json.dumps(scenario, indent=2, ensure_ascii=False, default=_json_default)
    return txt.encode("utf-8")


# =============================================================================
# Construction et application
# =============================================================================

def build_scenario_from_session(st_session) -> dict:
    """Compose un dict scénario depuis le session_state Streamlit courant."""
    df_gares = st_session.get("gares")
    if df_gares is None:
        raise ValueError("Aucune infrastructure définie : session_state.gares est vide.")

    gares_records = df_gares.to_dict(orient="records")
    # Nettoyage : NaN -> None pour JSON propre
    for g in gares_records:
        for k, v in list(g.items()):
            if isinstance(v, float) and pd.isna(v):
                g[k] = None

    h_deb = st_session.get("heure_debut_service")
    h_fin = st_session.get("heure_fin_service")

    missions_used = st_session.get("missions") or []
    materiels_utilises = {
        m.get("type_materiel", "diesel") for m in missions_used if m.get("type_materiel")
    }
    energy_params_full = st_session.get("energy_params") or {}
    energy_params_filtered = {
        mat: dict(energy_params_full[mat])
        for mat in materiels_utilises
        if mat in energy_params_full
    }

    today_iso = datetime.now().strftime("%Y-%m-%d")

    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "metadata": {
            "nom": "",
            "description": "",
            "auteur": "",
            "date_creation": today_iso,
            "date_modification": today_iso,
            "tags": [],
        },
        "service": {
            "heure_debut": _time_to_str(h_deb),
            "heure_fin": _time_to_str(h_fin),
        },
        "infrastructure": {
            "gares": gares_records,
        },
        "missions": [dict(m) for m in missions_used],
        "energy_params": energy_params_filtered,
    }


def apply_scenario_to_session(scenario: dict, st_session) -> None:
    """Injecte un scénario chargé dans le session_state.

    Vide les états calculés et les clés UI temporaires des missions précédentes
    avant d'écraser gares/missions/energy_params/heures de service.
    """
    _reset_session_for_scenario_load(st_session)

    # Infrastructure -> DataFrame
    gares = scenario.get("infrastructure", {}).get("gares", [])
    df = pd.DataFrame(gares)
    if not df.empty and "distance" in df.columns:
        df = df.sort_values("distance").reset_index(drop=True)
    st_session["gares"] = df

    # Missions
    st_session["missions"] = [dict(m) for m in scenario.get("missions", [])]

    # Heures de service
    svc = scenario.get("service", {})
    h_deb = _parse_time(svc.get("heure_debut"))
    h_fin = _parse_time(svc.get("heure_fin"))
    if h_deb is not None:
        st_session["heure_debut_service"] = h_deb
    if h_fin is not None:
        st_session["heure_fin_service"] = h_fin

    # Paramètres énergie : merge avec les défauts pour les types absents
    energy_params = {}
    defaults = get_default_energy_params()
    sc_energy = scenario.get("energy_params", {}) or {}
    for mat in _MATERIEL_TYPES:
        merged = defaults.copy()
        if mat in sc_energy and isinstance(sc_energy[mat], dict):
            merged.update(sc_energy[mat])
        energy_params[mat] = merged
    st_session["energy_params"] = energy_params

    # Mode de calcul forcé : un scénario s'applique au mode Énergie
    st_session["mode_calcul"] = "Calcul Energie"


# =============================================================================
# Validation et migration
# =============================================================================

def validate_scenario(scenario: dict) -> list[str]:
    """Retourne une liste d'erreurs (fatales) et de warnings (préfixés ⚠️)."""
    errors = []
    if "schema_version" not in scenario:
        errors.append("Champ 'schema_version' manquant.")
    if "infrastructure" not in scenario or "gares" not in scenario.get("infrastructure", {}):
        errors.append("Champ 'infrastructure.gares' manquant.")
        return errors
    if "missions" not in scenario:
        errors.append("Champ 'missions' manquant.")
        return errors

    gares = scenario["infrastructure"]["gares"]
    if not isinstance(gares, list) or not gares:
        errors.append("'infrastructure.gares' doit être une liste non vide.")
        return errors

    gare_names = set()
    for i, g in enumerate(gares):
        if "gare" not in g or "distance" not in g:
            errors.append(f"Gare #{i+1} : 'gare' ou 'distance' manquant.")
            continue
        gare_names.add(g["gare"])

    for j, m in enumerate(scenario["missions"]):
        for champ in ("origine", "terminus"):
            v = m.get(champ)
            if v and v not in gare_names:
                errors.append(f"Mission #{j+1} : {champ} '{v}' absent de l'infrastructure.")
        for kind in ("passing_points", "passing_points_retour"):
            for pp in m.get(kind, []) or []:
                if pp.get("gare") and pp["gare"] not in gare_names:
                    errors.append(
                        f"⚠️ Mission #{j+1} : point de passage '{pp['gare']}' "
                        f"absent de l'infrastructure."
                    )
        mat = m.get("type_materiel")
        if mat and mat not in _MATERIEL_TYPES:
            errors.append(f"⚠️ Mission #{j+1} : type_materiel '{mat}' inconnu.")

    return errors


def migrate_scenario(scenario: dict) -> dict:
    """Applique les migrations en chaîne jusqu'à CURRENT_SCHEMA_VERSION."""
    v = scenario.get("schema_version", 1)
    # Pas de migration nécessaire pour le moment (v1 est la version initiale).
    # Quand v2 arrivera : while v < CURRENT_SCHEMA_VERSION: scenario = MIGRATIONS[v](scenario); v += 1
    if v > CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Scénario en version {v}, supérieure à la version supportée "
            f"({CURRENT_SCHEMA_VERSION}). Mettez à jour Chronofer."
        )
    scenario["schema_version"] = CURRENT_SCHEMA_VERSION
    return scenario


# =============================================================================
# Utilitaires
# =============================================================================

def slugify(nom: str) -> str:
    """Convertit un nom en slug ASCII sûr pour un nom de fichier."""
    if not nom:
        return ""
    # Décompose les accents et garde la base ASCII
    nfkd = unicodedata.normalize("NFKD", nom)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_str = ascii_str.encode("ascii", "ignore").decode("ascii")
    ascii_str = ascii_str.lower()
    ascii_str = re.sub(r"[^a-z0-9]+", "_", ascii_str).strip("_")
    return ascii_str


def default_download_filename(scenario: dict) -> str:
    """Nom de fichier proposé pour le téléchargement.

    Option C du plan : slug du nom si renseigné, sinon horodaté.
    """
    nom = scenario.get("metadata", {}).get("nom") or ""
    slug = slugify(nom)
    if slug:
        return slug + ".json"
    return "chronofer_scenario_" + datetime.now().strftime("%Y%m%d_%H%M") + ".json"


def _reset_session_for_scenario_load(st_session) -> None:
    """Vide les états calculés et les clés UI dépendant des missions."""
    for k in ("chronologie_calculee", "warnings_calcul", "stats_homogeneite"):
        if k in st_session:
            st_session[k] = None if k == "chronologie_calculee" else {}
    if "energy_errors" in st_session:
        st_session["energy_errors"] = []
    st_session["run_calculation"] = False

    # Suppression des clés UI dynamiques indexées par mission
    patterns = (
        re.compile(r"^tt\d+$"),
        re.compile(r"^tt_retour_\d+$"),
        re.compile(r"^pp_raw_\d+$"),
        re.compile(r"^_prev_pp_raw_\d+$"),
        re.compile(r"^saisie_pp_\d+$"),
        re.compile(r"^orig\d+$"),
        re.compile(r"^term\d+$"),
        re.compile(r"^freq\d+$"),
    )
    to_delete = [k for k in list(st_session.keys())
                 if any(p.match(k) for p in patterns)]
    for k in to_delete:
        try:
            del st_session[k]
        except KeyError:
            pass


def _time_to_str(t) -> str:
    """time / datetime / str -> 'HH:MM'."""
    if t is None:
        return "06:00"
    if isinstance(t, str):
        return t
    if isinstance(t, datetime):
        return t.strftime("%H:%M")
    if isinstance(t, dt_time):
        return t.strftime("%H:%M")
    return str(t)


def _parse_time(s) -> dt_time | None:
    if s is None:
        return None
    if isinstance(s, dt_time):
        return s
    if isinstance(s, str):
        try:
            return datetime.strptime(s.strip(), "%H:%M").time()
        except ValueError:
            return None
    return None


def _json_default(o):
    """Convertisseur pour json.dumps — gère time, datetime, pandas NaT."""
    if isinstance(o, (dt_time, datetime)):
        return o.isoformat()
    if pd.isna(o):
        return None
    raise TypeError(f"Object of type {type(o).__name__} not JSON serializable")
