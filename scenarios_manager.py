# -*- coding: utf-8 -*-
"""
scenarios_manager.py
====================

Gestion des scénarios pré-enregistrés (infrastructure + missions + paramètres
matériel) pour le mode "Calcul Energie".

Un scénario = un fichier JSON autonome décrit dans `scenarios/_schema.md`.
La bibliothèque côté serveur est alimentée hors-app (dépôt manuel dans
`scenarios/<région>/<ligne>/`). L'utilisateur télécharge ses configurations
depuis l'UI et peut aussi recharger un fichier JSON via l'onglet
« Importer un fichier ».
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

UNCLASSIFIED_REGION = "Non classé"
DEFAULT_LIGNE = "Général"

# Mapping paramètre énergétique -> (clé widget Streamlit, coerceur)
_ENERGY_WIDGET_MAP = (
    ("masse_tonne", "masse_{mat}", int),
    ("facteur_aux_kwh_h", "f_aux_{mat}", float),
    ("capacite_batterie_kwh", "cap_batt_{mat}", int),
    ("facteur_charge_C", "f_charge_c_{mat}", float),
    ("simuler_fin_de_vie", "eol_check_{mat}", bool),
    ("capacite_eol_pct", "eol_pct_{mat}", int),
    ("soc_min_pct", "soc_min_{mat}", int),
    ("soc_max_pct", "soc_max_{mat}", int),
    ("accel_ms2", "accel_{mat}", float),
    ("decel_ms2", "decel_{mat}", float),
    ("davis_A_N_t", "f_davis_a_{mat}", float),
    ("davis_B_N_t_kph", "f_davis_b_{mat}", float),
    ("davis_C_N_t_kph2", "f_davis_c_{mat}", float),
    ("rendement_thermique_pct", "rend_therm_{mat}", int),
    ("kwh_per_liter_diesel", "f_kwh_l_{mat}", float),
    ("rendement_electrique_pct", "rend_elec_{mat}", int),
    ("recuperation_pct", "recup_{mat}", int),
)


# =============================================================================
# Découverte et lecture
# =============================================================================

def classify_scenario_path(path: Path, metadata: dict | None = None) -> tuple[str, str]:
    """Déduit (région, ligne) depuis le chemin relatif à `scenarios/`.

    Convention : `scenarios/<région>/<ligne>/<fichier>.json`.
    Un seul niveau de dossier → région = ce dossier, ligne = « Général ».
    Fichier à la racine → métadonnées, sinon « Non classé » / « Général ».
    """
    metadata = metadata or {}
    try:
        rel = path.resolve().relative_to(SCENARIOS_DIR.resolve())
    except ValueError:
        rel = Path(path.name)
    parts = rel.parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return parts[0], metadata.get("ligne") or DEFAULT_LIGNE
    region = (metadata.get("region") or "").strip() or UNCLASSIFIED_REGION
    ligne = (metadata.get("ligne") or "").strip() or DEFAULT_LIGNE
    return region, ligne


def list_scenarios() -> list[dict]:
    """Parcourt `scenarios/` récursivement et retourne la liste des scénarios.

    Chaque entrée : path, rel_path, region, ligne, metadata.
    Les fichiers invalides sont silencieusement ignorés.
    """
    if not SCENARIOS_DIR.exists():
        return []
    result = []
    for path in sorted(SCENARIOS_DIR.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            md = dict(data.get("metadata", {}) or {})
            if "nom" not in md:
                md["nom"] = path.stem
            region, ligne = classify_scenario_path(path, md)
            try:
                rel_path = path.resolve().relative_to(SCENARIOS_DIR.resolve()).as_posix()
            except ValueError:
                rel_path = path.name
            result.append({
                "path": path,
                "rel_path": rel_path,
                "region": region,
                "ligne": ligne,
                "metadata": md,
            })
        except (OSError, json.JSONDecodeError):
            continue
    return result


def group_scenarios_by_region_line(scenarios: list[dict] | None = None) -> dict[str, dict[str, list[dict]]]:
    """Regroupe les scénarios en arbre {région: {ligne: [scénarios]}}.

    Les clés sont triées (sauf « Non classé », toujours en dernier).
    """
    if scenarios is None:
        scenarios = list_scenarios()
    tree: dict[str, dict[str, list[dict]]] = {}
    for s in scenarios:
        tree.setdefault(s["region"], {}).setdefault(s["ligne"], []).append(s)

    def _sort_label(label: str) -> tuple[int, str]:
        return (1 if label == UNCLASSIFIED_REGION else 0, label.casefold())

    ordered: dict[str, dict[str, list[dict]]] = {}
    for region in sorted(tree, key=_sort_label):
        ordered[region] = {
            ligne: tree[region][ligne]
            for ligne in sorted(tree[region], key=_sort_label)
        }
    return ordered


def load_scenario(source: Union[Path, str, IO, bytes]) -> dict:
    """Charge un scénario depuis un chemin, des bytes ou un file-like.

    Relit toujours la source (seek(0) sur les flux déjà consommés).
    Applique les migrations puis valide. Lève ValueError sur erreur grave.
    """
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source).decode("utf-8")
        data = json.loads(raw)
    elif hasattr(source, "read"):
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not (raw or "").strip():
            raise ValueError(
                "Le fichier scénario est vide (flux déjà consommé ou fichier invalide)."
            )
        data = json.loads(raw)
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
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
            "region": "",
            "ligne": "",
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

    Vide les états calculés et TOUTES les clés widgets liées au scénario
    précédent, puis réécrit gares / missions / energy_params / heures de
    service ET les clés widgets correspondantes. À appeler avant le rendu
    des widgets (callback ou début de script), sinon Streamlit ignore
    les nouvelles valeurs.
    """
    _reset_session_for_scenario_load(st_session)
    st_session["_scenario_ui_epoch"] = int(st_session.get("_scenario_ui_epoch") or 0) + 1

    # Infrastructure -> DataFrame
    gares = scenario.get("infrastructure", {}).get("gares", [])
    df = pd.DataFrame(gares)
    if not df.empty and "distance" in df.columns:
        df = df.sort_values("distance").reset_index(drop=True)
    st_session["gares"] = df

    # Synchronise le texte du formulaire "Liste des gares" avec l'infra chargée,
    # sinon le text_area continue d'afficher la valeur par défaut et un clic
    # involontaire sur "Valider" écraserait l'infra du scénario.
    st_session["gares_texte_input"] = format_gares_text(df)

    # Missions (copies indépendantes)
    missions = [dict(m) for m in scenario.get("missions", [])]
    st_session["missions"] = missions
    st_session["nombre_missions"] = max(len(missions), 1)

    # Pré-remplissage du mode "Saisie manuelle par lot" pour chaque mission.
    # Le mode "Interface Guidée" ne sait pas relire les passing_points (ses
    # widgets utilisent des valeurs par défaut indépendantes), donc on force le
    # mode bulk et on écrit le texte attendu par le parser de app.py.
    for idx, m in enumerate(missions):
        bulk_text = format_mission_to_bulk_text(m, mode_calcul="Calcul Energie")
        m["pp_raw_text"] = bulk_text
        _seed_mission_widget_keys(st_session, idx, m, bulk_text)

    # Heures de service
    svc = scenario.get("service", {})
    h_deb = _parse_time(svc.get("heure_debut"))
    h_fin = _parse_time(svc.get("heure_fin"))
    st_session["heure_debut_service"] = h_deb or dt_time(6, 0)
    st_session["heure_fin_service"] = h_fin or dt_time(22, 0)

    # Paramètres énergie : merge avec les défauts pour les types absents
    energy_params = {}
    defaults = get_default_energy_params()
    sc_energy = scenario.get("energy_params", {}) or {}
    for mat in _MATERIEL_TYPES:
        merged = defaults.copy()
        if mat in sc_energy and isinstance(sc_energy[mat], dict):
            merged.update(sc_energy[mat])
        energy_params[mat] = merged
        _seed_energy_widget_keys(st_session, mat, merged)
    st_session["energy_params"] = energy_params

    # Mode de calcul forcé : un scénario s'applique au mode Énergie
    st_session["mode_calcul"] = "Calcul Energie"
    st_session["mode_calcul_selector"] = "Calcul Energie"
    st_session["mode_generation"] = "Rotation optimisée"

    md = dict(scenario.get("metadata", {}) or {})
    st_session["_loaded_scenario_meta"] = md
    st_session["scenario_dl_name"] = md.get("nom") or ""
    st_session["scenario_dl_tags"] = ", ".join(md.get("tags") or [])
    st_session["scenario_dl_desc"] = md.get("description") or ""
    st_session["scenario_dl_region"] = md.get("region") or ""
    st_session["scenario_dl_ligne"] = md.get("ligne") or ""


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


def format_gares_text(df_gares) -> str:
    """Reconstruit le contenu du text_area "Liste des gares" à partir du DataFrame.

    Format Calcul Energie : ``nom;distance;infra;electrification;rampe``.
    Format Standard : ``nom;distance;infra``. Le mode est détecté via la
    présence de la colonne ``electrification``.
    """
    if df_gares is None or df_gares.empty:
        return ""
    is_energy = "electrification" in df_gares.columns
    lines = []
    for _, row in df_gares.iterrows():
        nom = str(row.get("gare", "")).strip()
        distance = row.get("distance", "")
        infra = row.get("infra") or ""
        if is_energy:
            electr = row.get("electrification") or "F"
            rampe = row.get("rampe_section_a_venir", 0)
            if pd.isna(rampe):
                rampe = 0
            lines.append(f"{nom};{distance};{infra};{electr};{rampe}")
        else:
            lines.append(f"{nom};{distance};{infra}")
    return "\n".join(lines)


def format_mission_to_bulk_text(mission: dict, mode_calcul: str = "Calcul Energie") -> str:
    """Convertit les points de passage d'une mission en texte pour le mode
    "Saisie manuelle par lot".

    Format Calcul Energie : ``gare;t_aller;arret_aller[;t_retour;arret_retour]``.
    Format Standard       : ``gare;t_aller[;t_retour]``.

    Les gares présentes dans l'aller et le retour sont fusionnées sur une seule
    ligne. Les gares présentes uniquement dans le retour (cas inhabituel) sont
    émises avec des zéros côté aller — l'utilisateur devra corriger à la main.
    """
    aller_pp = mission.get("passing_points") or []
    retour_pp = mission.get("passing_points_retour") or []
    retour_by_gare = {pp.get("gare"): pp for pp in retour_pp if pp.get("gare")}

    is_energy = mode_calcul == "Calcul Energie"
    lines = []
    used_retour = set()

    for pp in aller_pp:
        gare = pp.get("gare")
        if not gare:
            continue
        t_a = pp.get("time_offset_min", 0)
        ar_a = int(pp.get("duree_arret_min") or 0)
        ret = retour_by_gare.get(gare)
        if is_energy:
            if ret is not None:
                t_r = ret.get("time_offset_min", 0)
                ar_r = int(ret.get("duree_arret_min") or 0)
                lines.append(f"{gare};{t_a};{ar_a};{t_r};{ar_r}")
                used_retour.add(gare)
            else:
                lines.append(f"{gare};{t_a};{ar_a}")
        else:
            if ret is not None:
                t_r = ret.get("time_offset_min", 0)
                lines.append(f"{gare};{t_a};{t_r}")
                used_retour.add(gare)
            else:
                lines.append(f"{gare};{t_a}")

    for pp in retour_pp:
        gare = pp.get("gare")
        if not gare or gare in used_retour:
            continue
        t_r = pp.get("time_offset_min", 0)
        ar_r = int(pp.get("duree_arret_min") or 0)
        if is_energy:
            lines.append(f"{gare};0;0;{t_r};{ar_r}")
        else:
            lines.append(f"{gare};0;{t_r}")

    return "\n".join(lines)


def _seed_mission_widget_keys(st_session, idx: int, mission: dict, bulk_text: str) -> None:
    """Écrit les clés widgets d'une mission pour que Streamlit les affiche."""
    st_session[f"pp_raw_{idx}"] = bulk_text
    # _prev_pp_raw_{idx} = bulk_text empêche l'auto-update de tt{idx}
    # de s'exécuter au premier rendu (texte considéré "déjà appliqué").
    st_session[f"_prev_pp_raw_{idx}"] = bulk_text
    st_session[f"saisie_pp_{idx}"] = "Saisie manuelle par lot"

    origine = mission.get("origine")
    terminus = mission.get("terminus")
    if origine:
        st_session[f"orig{idx}"] = origine
    if terminus:
        st_session[f"term{idx}"] = terminus
    st_session[f"freq{idx}"] = float(mission.get("frequence", 1.0))
    st_session[f"tt{idx}"] = int(mission.get("temps_trajet", 45) or 45)
    st_session[f"tr_a_{idx}"] = int(mission.get("temps_retournement_A", 10) or 10)
    st_session[f"tr_b_{idx}"] = int(mission.get("temps_retournement_B", 10) or 10)
    st_session[f"type_mat_{idx}"] = mission.get("type_materiel") or "diesel"
    st_session[f"ref_mins{idx}"] = str(mission.get("reference_minutes", "0") or "0")
    st_session[f"inj_t2_{idx}"] = bool(mission.get("inject_from_terminus_2", False))
    st_session[f"asym_{idx}"] = bool(mission.get("trajet_asymetrique", False))
    t_retour = mission.get("temps_trajet_retour", mission.get("temps_trajet", 45))
    st_session[f"tt_retour_{idx}"] = int(t_retour or 45)
    st_session[f"n_pass_{idx}"] = len(mission.get("passing_points") or [])
    st_session[f"n_pass_retour_{idx}"] = len(mission.get("passing_points_retour") or [])


def _seed_energy_widget_keys(st_session, mat: str, params: dict) -> None:
    """Écrit les clés widgets des paramètres énergétiques d'un matériel."""
    for param_key, key_tpl, caster in _ENERGY_WIDGET_MAP:
        if param_key not in params:
            continue
        raw = params[param_key]
        try:
            value = caster(raw)
        except (TypeError, ValueError):
            continue
        st_session[key_tpl.format(mat=mat)] = value


def _reset_session_for_scenario_load(st_session) -> None:
    """Vide les états calculés et les clés UI dépendant du scénario précédent."""
    for k in ("chronologie_calculee", "warnings_calcul", "stats_homogeneite"):
        if k in st_session:
            st_session[k] = None if k == "chronologie_calculee" else {}
    if "energy_errors" in st_session:
        st_session["energy_errors"] = []
    st_session["run_calculation"] = False
    st_session["roulement_manuel"] = {}

    # Clés widgets « stables » (non indexées par mission). Il faut les
    # supprimer AVANT de les réécrire, sinon Streamlit peut restaurer
    # la valeur du run précédent et ignorer le nouveau scénario.
    stable_keys = (
        "heure_debut_service",
        "heure_fin_service",
        "gares_texte_input",
        "nombre_missions",
        "mode_calcul_selector",
        "mode_generation",
        "scenario_dl_name",
        "scenario_dl_tags",
        "scenario_dl_desc",
        "scenario_dl_region",
        "scenario_dl_ligne",
        "_loaded_scenario_meta",
    )
    for k in stable_keys:
        if k in st_session:
            try:
                del st_session[k]
            except KeyError:
                pass

    for mat in _MATERIEL_TYPES:
        for _param, key_tpl, _caster in _ENERGY_WIDGET_MAP:
            k = key_tpl.format(mat=mat)
            if k in st_session:
                try:
                    del st_session[k]
                except KeyError:
                    pass

    # Suppression des clés UI dynamiques indexées par mission. Sans ce nettoyage,
    # les valeurs persistées d'un précédent scénario écrasent les valeurs par
    # défaut lues depuis mission.get(...) et peuvent violer les bornes (min/max)
    # du nouveau scénario.
    patterns = (
        re.compile(r"^tt\d+$"),
        re.compile(r"^tt_retour_\d+$"),
        re.compile(r"^pp_raw_\d+$"),
        re.compile(r"^_prev_pp_raw_\d+$"),
        re.compile(r"^saisie_pp_\d+$"),
        re.compile(r"^orig\d+$"),
        re.compile(r"^term\d+$"),
        re.compile(r"^freq\d+$"),
        re.compile(r"^tr_a_\d+$"),
        re.compile(r"^tr_b_\d+$"),
        re.compile(r"^type_mat_\d+$"),
        re.compile(r"^ref_mins\d+$"),
        re.compile(r"^inj_t2_\d+$"),
        re.compile(r"^asym_\d+$"),
        re.compile(r"^n_pass_\d+$"),
        re.compile(r"^n_pass_retour_\d+$"),
        re.compile(r"^pp_gare_\d+_\d+$"),
        re.compile(r"^pp_tps_\d+_\d+$"),
        re.compile(r"^pp_arret_\d+_\d+$"),
        re.compile(r"^pp_duree_arret_\d+_\d+$"),
        re.compile(r"^pp_gare_retour_\d+_\d+$"),
        re.compile(r"^pp_tps_retour_\d+_\d+$"),
        re.compile(r"^pp_arret_r_\d+_\d+$"),
        re.compile(r"^pp_duree_arret_r_\d+_\d+$"),
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
