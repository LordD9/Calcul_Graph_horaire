# -*- coding: utf-8 -*-
"""Analyse de robustesse d'une chronologie sans relancer l'optimisation."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections import Counter
from copy import deepcopy
from datetime import datetime, time, timedelta

import pandas as pd

from core_logic import SimulationEngine, _get_infra_at_gare, _is_crossing_point
from energy_logic import (
    _get_puissance_infra,
    calculer_consommation_trajet,
    get_default_energy_params,
)


AMPLITUDES_RETARD_MIN = (5, 10, 20)
_EPSILON_MIN = 1e-7
_UNFOURNI = object()
_CLE_CHRONOLOGIE_NOMINALE = ("__chronologie_nominale__",)


def _point_croisement(infra_code):
    """Normalise le code avant d'appliquer le prédicat commun du moteur."""
    code = str(infra_code or "F").strip().upper()
    return _is_crossing_point("Terminus" if code == "TERMINUS" else code)


def _minutes(delta):
    return delta.total_seconds() / 60.0


def _mission_pour_label(label, missions):
    """Associe un label directionnel à une configuration de mission unique."""
    if not isinstance(label, str) or " → " not in label:
        raise ValueError(f"Label de mission non pris en charge : {label!r}")
    origine, terminus = (part.strip() for part in label.split(" → ", 1))
    correspondances = []
    for mission in missions:
        if mission.get("origine") == origine and mission.get("terminus") == terminus:
            correspondances.append((mission, "aller"))
        elif mission.get("origine") == terminus and mission.get("terminus") == origine:
            correspondances.append((mission, "retour"))
    if len(correspondances) != 1:
        if not correspondances:
            raise ValueError(f"Aucune mission ne correspond au trajet {label!r}")
        raise ValueError(f"Mission ambiguë pour le trajet {label!r}")
    return origine, terminus, correspondances[0][0], correspondances[0][1]


def _compiler_blocs(service, gares_map, infra_map):
    """Regroupe les segments jusqu'au prochain point où un croisement est possible."""
    rows = service["segments"]
    blocks = []
    current = []
    index = 0
    service["prefix_segments"] = []

    def finaliser():
        nonlocal current
        movements = [row for row in current if row["origine"] != row["terminus"]]
        if not movements:
            raise ValueError(f"Bloc sans circulation dans {service['service_id']}")
        stations = [movements[0]["origine"]] + [row["terminus"] for row in movements]
        first_idx = gares_map.get(stations[0])
        last_idx = gares_map.get(stations[-1])
        if first_idx is None or last_idx is None:
            raise ValueError(f"Gare inconnue dans le trajet {service['service_id']}")
        for row in movements:
            left = gares_map.get(row["origine"])
            right = gares_map.get(row["terminus"])
            if left is None or right is None:
                raise ValueError(f"Gare inconnue dans le trajet {service['service_id']}")
            lo, hi = sorted((left, right))
            for point in service["ordered_gare_names"][lo + 1:hi]:
                if _point_croisement(infra_map.get(point, "F")):
                    raise ValueError(
                        f"Le segment {row['origine']} → {row['terminus']} saute le point de croisement {point!r}"
                    )
        arrival_row = movements[-1]
        end_station = arrival_row["terminus"]
        release_row = next(
            (row for row in reversed(current)
             if row["origine"] == end_station and row["terminus"] == end_station),
            None,
        )
        planned_start = movements[0]["start"]
        planned_arrival = arrival_row["end"]
        planned_release = release_row["end"] if release_row else planned_arrival
        if planned_release < planned_arrival:
            raise ValueError(f"Arrêt négatif au point {end_station!r}")
        blocks.append({
            "rows": [dict(row) for row in current],
            "movements": movements,
            "start": planned_start,
            "arrival": planned_arrival,
            "release": planned_release,
            "start_station": movements[0]["origine"],
            "end_station": end_station,
            "idx_min": min(first_idx, last_idx),
            "idx_max": max(first_idx, last_idx),
            "crossing_at_end": _point_croisement(infra_map.get(end_station, "F")),
        })
        current = []

    while index < len(rows):
        row = rows[index]
        if row["origine"] == row["terminus"]:
            if not current and not blocks and row["origine"] == service["origine"]:
                service["prefix_segments"].append(dict(row))
                index += 1
                continue
            if not current or current[-1]["terminus"] != row["origine"]:
                raise ValueError(f"Arrêt sans circulation adjacente dans {service['service_id']}")
            current.append(row)
            if _point_croisement(infra_map.get(row["origine"], "F")):
                finaliser()
            index += 1
            continue

        if current:
            last_movement = next(
                (item for item in reversed(current) if item["origine"] != item["terminus"]),
                None,
            )
            if last_movement is None or last_movement["terminus"] != row["origine"]:
                raise ValueError(f"Itinéraire discontinu dans {service['service_id']}")
        current.append(row)
        if _point_croisement(infra_map.get(row["terminus"], "F")):
            while index + 1 < len(rows):
                following = rows[index + 1]
                if (following["origine"] == following["terminus"] == row["terminus"]):
                    current.append(following)
                    index += 1
                else:
                    break
            finaliser()
        index += 1

    if current:
        finaliser()
    if not blocks:
        raise ValueError(f"Mission sans segment de circulation : {service['service_id']}")
    if blocks[0]["start_station"] != service["origine"]:
        raise ValueError(f"Origine incohérente dans {service['service_id']}")
    if blocks[-1]["end_station"] != service["terminus"]:
        raise ValueError(f"Mission incomplète dans {service['service_id']}")
    for previous, following in zip(blocks, blocks[1:]):
        if previous["end_station"] != following["start_station"]:
            raise ValueError(f"Continuité des blocs invalide dans {service['service_id']}")
        if following["start"] < previous["release"]:
            raise ValueError(f"Blocs qui se chevauchent dans {service['service_id']}")
    service["blocks"] = blocks
    service["planned_start"] = blocks[0]["start"]
    service["planned_arrival"] = blocks[-1]["arrival"]
    service["planned_release"] = blocks[-1]["release"]


def _compiler_services(chronologie, df_gares, missions):
    if not isinstance(chronologie, dict):
        raise ValueError("La chronologie doit être un dictionnaire de rames")
    if not hasattr(df_gares, "columns") or not {"gare", "distance", "infra"}.issubset(df_gares.columns):
        raise ValueError("df_gares doit contenir les colonnes 'gare', 'distance' et 'infra'")
    if df_gares["gare"].duplicated().any():
        raise ValueError("Les noms de gares doivent être uniques")

    gares = df_gares.copy()
    try:
        gares["distance"] = pd.to_numeric(gares["distance"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Les distances de gares doivent être numériques") from exc
    if gares["distance"].isna().any() or not all(math.isfinite(float(value)) for value in gares["distance"]):
        raise ValueError("Les distances de gares doivent être finies")
    gares = gares.sort_values("distance").reset_index(drop=True)
    gare_names = list(gares["gare"])
    gares_map = {name: index for index, name in enumerate(gare_names)}
    infra_map = {row["gare"]: _get_infra_at_gare(df_gares, row["gare"])
                 for _, row in gares.iterrows()}
    services = []

    for train_id, entries in sorted(chronologie.items(), key=lambda item: str(item[0])):
        if not entries:
            continue
        normalized = []
        for position, original in enumerate(entries):
            if not isinstance(original, dict):
                raise ValueError(f"Segment invalide pour la rame {train_id!r}")
            row = dict(original)
            for key in ("start", "end", "origine", "terminus"):
                if key not in row:
                    raise ValueError(f"Champ {key!r} absent pour la rame {train_id!r}")
            if not isinstance(row["start"], datetime) or not isinstance(row["end"], datetime):
                raise ValueError("Les horaires doivent être des objets datetime")
            if row["start"].utcoffset() is not None or row["end"].utcoffset() is not None:
                raise ValueError("Les horaires timezone-aware ne sont pas pris en charge")
            if row["end"] < row["start"]:
                raise ValueError(f"Durée négative pour la rame {train_id!r}")
            row["_input_order"] = position
            normalized.append(row)
        normalized.sort(key=lambda row: (row["start"], row["end"], row["_input_order"]))

        groups = []
        current = None
        pending_prefix = []
        for row in normalized:
            if row.get("is_mission_start", False):
                if current is not None:
                    groups.append(current)
                current = pending_prefix + [row]
                pending_prefix = []
            elif current is None:
                if row["origine"] == row["terminus"]:
                    pending_prefix.append(row)
                else:
                    raise ValueError(
                        f"Segment sans is_mission_start avant toute mission pour la rame {train_id!r}"
                    )
            else:
                current.append(row)
        if current is not None:
            groups.append(current)
        if pending_prefix:
            raise ValueError(f"Attente sans mission pour la rame {train_id!r}")

        for ordinal, rows in enumerate(groups, start=1):
            label = rows[0].get("mission")
            origine, terminus, mission, direction = _mission_pour_label(label, missions or [])
            for row in rows:
                if row.get("mission") != label:
                    raise ValueError(f"Labels de mission mélangés dans la rame {train_id!r}")
                if row["start"] < rows[0]["start"]:
                    raise ValueError(f"Segments hors ordre dans la rame {train_id!r}")
            for previous, following in zip(rows, rows[1:]):
                if following["start"] < previous["end"]:
                    raise ValueError(f"Segments qui se chevauchent dans la rame {train_id!r}")
                if previous["terminus"] != following["origine"]:
                    raise ValueError(f"Itinéraire discontinu dans la rame {train_id!r}")
            travel = [row for row in rows if row["origine"] != row["terminus"]]
            if not travel or travel[0]["origine"] != origine:
                raise ValueError(f"Mission incomplète ou label incohérent : {label!r}")
            if any(row["origine"] not in gares_map or row["terminus"] not in gares_map for row in rows):
                raise ValueError(f"Gare absente de df_gares dans {label!r}")
            tronquee = False
            if travel[-1]["terminus"] != terminus:
                # La génération coupe la dernière mission d'une rame à la fin de
                # service : on l'évalue jusqu'à la dernière gare réellement
                # atteinte, à condition qu'elle se trouve sur le trajet du label.
                atteinte = travel[-1]["terminus"]
                derniere = ordinal == len(groups)
                if (not derniere or origine not in gares_map or terminus not in gares_map):
                    raise ValueError(f"Mission incomplète ou label incohérent : {label!r}")
                lo, hi = sorted((gares_map[origine], gares_map[terminus]))
                if not lo < gares_map[atteinte] < hi:
                    raise ValueError(f"Mission incomplète ou label incohérent : {label!r}")
                terminus = atteinte
                tronquee = True
            service_id = f"{type(train_id).__name__}:{train_id}#{ordinal}"
            clean_rows = []
            for row in rows:
                row.pop("_input_order", None)
                clean_rows.append(row)
            service = {
                "service_id": service_id,
                "train_id": train_id,
                "mission": label,
                "mission_config": mission,
                "direction": direction,
                "origine": origine,
                "terminus": terminus,
                "tronquee": tronquee,
                "material": mission.get("type_materiel", "diesel"),
                "segments": clean_rows,
                "ordered_gare_names": gare_names,
            }
            _compiler_blocs(service, gares_map, infra_map)
            turnaround = float(
                mission.get("temps_retournement_B" if direction == "aller" else "temps_retournement_A", 10) or 0
            )
            if not math.isfinite(turnaround) or turnaround < 0:
                raise ValueError(f"Temps de retournement invalide dans {label!r}")
            service["turnaround_min"] = turnaround
            services.append(service)

    source_sequences = {}
    for service in services:
        source_sequences.setdefault(service["train_id"], []).append(service)
    for train_id, sequence in source_sequences.items():
        sequence.sort(key=lambda service: (service["planned_start"], service["service_id"]))
        materials = {service["material"] for service in sequence}
        if len(materials) > 1:
            raise ValueError(f"Matériels différents pour la rame physique {train_id!r}")
        for previous, following in zip(sequence, sequence[1:]):
            if previous["terminus"] != following["origine"]:
                raise ValueError(
                    f"Aucun parcours de repositionnement entre {previous['service_id']} et {following['service_id']}"
                )
    return services, gares, infra_map


def _point_depart(service):
    return {
        "train_id": service["train_id"],
        "gare": service["origine"],
        "time": service["planned_start"],
        "service_id": service["service_id"],
        "mission": service["mission"],
        "kind": "depart",
        "block_index": 0,
    }


def _enumerer_points(services, infra_map):
    points = []
    seen = set()
    for service in sorted(services, key=lambda item: (item["planned_start"], item["service_id"])):
        departure = _point_depart(service)
        key = (service["service_id"], departure["gare"], departure["time"])
        if key not in seen:
            points.append(departure)
            seen.add(key)
        for index, block in enumerate(service["blocks"]):
            if not block["crossing_at_end"]:
                continue
            key = (service["service_id"], block["end_station"], block["arrival"])
            if key in seen:
                continue
            points.append({
                "train_id": service["train_id"],
                "gare": block["end_station"],
                "time": block["arrival"],
                "service_id": service["service_id"],
                "mission": service["mission"],
                "kind": "croisement",
                "block_index": index,
            })
            seen.add(key)
    return points


def _attente(service, gare, debut, fin, extension=None, perturbation=None):
    if fin <= debut:
        return None
    row = {
        "start": debut,
        "end": fin,
        "origine": gare,
        "terminus": gare,
        "mission": service["mission"],
        "is_mission_start": False,
        "service_id": service["service_id"],
        "train_id_original": service["train_id"],
    }
    if extension is not None and extension > 0:
        row["crossing_extension_min"] = extension
        row["subi"] = True
    if perturbation is not None and perturbation > 0:
        row["perturbation_min"] = perturbation
    return row


def _rejouer(services, df_gares, incident=None, delay_min=0, reserve_map=None,
             reserve_stations=(), turnaround_overrides=None):
    """Rejoue les blocs fixes dans une file dynamique, sans ajouter de missions."""
    reserve_map = dict(reserve_map or {})
    turnaround_overrides = dict(turnaround_overrides or {})
    engine = SimulationEngine(df_gares, time(0, 0), time(23, 59))
    services_by_id = {service["service_id"]: service for service in services}
    source_first = {}
    for service in services:
        prior = source_first.get(service["train_id"])
        if prior is None or service["planned_start"] < prior["planned_start"]:
            source_first[service["train_id"]] = service

    units = {}
    operations = {}
    for train_id, first in source_first.items():
        key = ("train", train_id)
        units[key] = {
            "display_id": train_id,
            "kind": "train",
            "location": first["origine"],
            "available_at": datetime.min,
            "material": first["material"],
        }
        operations[key] = []
    reserve_keys = {}
    used_display_ids = {str(train_id) for train_id in source_first}
    for index, gare in enumerate(reserve_stations, start=1):
        key = ("reserve", index)
        display_id = f"Réserve {gare} #{index}"
        while display_id in used_display_ids:
            display_id += "*"
        used_display_ids.add(display_id)
        reserve_keys[gare] = key
        units[key] = {
            "display_id": display_id,
            "kind": "reserve",
            "location": gare,
            "available_at": datetime.min,
            "material": None,
        }
        operations[key] = []

    assignment_errors = []
    for service_id, gare in reserve_map.items():
        service = services_by_id[service_id]
        reserve_key = reserve_keys.get(gare)
        if reserve_key is None:
            assignment_errors.append(f"Aucune réserve disponible à {gare!r}")
            continue
        unit = units[reserve_key]
        if unit["material"] is not None and unit["material"] != service["material"]:
            assignment_errors.append(f"Matériel incompatible pour la réserve de {gare}")
            continue
        unit["material"] = service["material"]
        operations[reserve_key].append(service)
    for service in services:
        if service["service_id"] not in reserve_map:
            operations[("train", service["train_id"])].append(service)

    for key in operations:
        operations[key].sort(key=lambda service: (service["planned_start"], service["service_id"]))

    event_heap = []
    event_counter = 0
    next_operation = {key: 0 for key in units}
    actual = {}
    output = {units[key]["display_id"]: [] for key in units}
    infra_violations = []
    physical_violations = list(assignment_errors)
    suppression_motifs = {}

    def push(event_time, kind, payload):
        nonlocal event_counter
        event_counter += 1
        heapq.heappush(event_heap, (event_time, event_counter, kind, payload))

    def schedule_next_operation(unit_key):
        position = next_operation[unit_key]
        if position >= len(operations[unit_key]):
            return
        service = operations[unit_key][position]
        unit = units[unit_key]
        event_time = max(service["planned_start"], unit["available_at"])
        push(event_time, "depart_service", {"unit_key": unit_key, "service_id": service["service_id"]})

    def supprimer_service(service, unit_key, motif, event_time):
        service_id = service["service_id"]
        state = actual.setdefault(service_id, {
            "departure": None, "arrival": None, "complete": False,
            "unit_key": unit_key, "blocks": {},
        })
        state["complete"] = False
        state["arrival"] = None
        state["suppressed"] = True
        state["suppression_motif"] = motif
        suppression_motifs[service_id] = motif
        unit = units[unit_key]
        if not state.get("blocks"):
            output[unit["display_id"]] = [
                row for row in output[unit["display_id"]]
                if row.get("service_id") != service_id
            ]
        unit["available_at"] = max(unit["available_at"], event_time)
        next_operation[unit_key] += 1
        schedule_next_operation(unit_key)

    for unit_key in units:
        schedule_next_operation(unit_key)

    while event_heap:
        event_time, _, event_kind, event = heapq.heappop(event_heap)
        service = services_by_id[event["service_id"]]
        unit_key = event["unit_key"]
        unit = units[unit_key]

        if event_kind == "depart_service":
            if unit["location"] != service["origine"]:
                motif = (
                    f"Trajectoire impossible : rame à {unit['location']!r}, "
                    f"départ prévu à {service['origine']!r}"
                )
                if unit["kind"] == "reserve":
                    physical_violations.append(f"{unit['display_id']} ({service['service_id']}) : {motif}")
                supprimer_service(service, unit_key, motif, event_time)
                continue

            ready_time = max(service["planned_start"], unit["available_at"])
            for row in service.get("prefix_segments", []):
                copied = dict(row)
                copied["service_id"] = service["service_id"]
                copied["train_id_original"] = service["train_id"]
                output[unit["display_id"]].append(copied)
            incident_departure = bool(
                incident
                and incident["service_id"] == service["service_id"]
                and incident["kind"] == "depart"
            )
            departure_time = ready_time + timedelta(minutes=delay_min if incident_departure else 0)
            actual[service["service_id"]] = {
                "departure": None,
                "arrival": None,
                "complete": False,
                "unit_key": unit_key,
                "ready_time": ready_time,
                "departure_hold_min": delay_min if incident_departure else 0,
                "blocks": {},
            }
            push(departure_time, "bloc", {
                "unit_key": unit_key,
                "service_id": service["service_id"],
                "block_index": 0,
                "earliest": departure_time,
                "wait_from": ready_time,
                "attempts": 0,
            })
            continue

        block_index = event["block_index"]
        block = service["blocks"][block_index]
        earliest = max(event["earliest"], block["start"])
        arrival = earliest + (block["arrival"] - block["start"])
        release = earliest + (block["release"] - block["start"])
        index_start = engine.gares_map.get(block["start_station"])
        index_end = engine.gares_map.get(block["end_station"])
        is_free, next_available = engine.check_segment_availability(
            min(index_start, index_end), max(index_start, index_end), earliest, release
        )

        if not is_free:
            point_attente_valide = _point_croisement(
                engine.infra_map.get(block["start_station"], "F")
            )
            if point_attente_valide and event["attempts"] < 500 and next_available is not None:
                retry_at = max(earliest, next_available)
                push(retry_at, "bloc", {
                    "unit_key": unit_key,
                    "service_id": service["service_id"],
                    "block_index": block_index,
                    "earliest": retry_at,
                    "wait_from": event["wait_from"],
                    "attempts": event["attempts"] + 1,
                })
                continue
            else:
                if not point_attente_valide:
                    motif = (
                        f"Conflit sur le bloc au départ de {block['start_station']!r} : "
                        "aucun point d'attente valide avant le bloc"
                    )
                else:
                    motif = (
                        f"Conflit persistant au point d'attente {block['start_station']!r} "
                        "sans créneau libre"
                    )
                supprimer_service(service, unit_key, motif, earliest)
                continue

        state = actual[service["service_id"]]
        shift = earliest - block["start"]
        if block_index == 0:
            state["departure"] = earliest
            if earliest > service["planned_start"]:
                wait = _attente(
                    service, service["origine"], service["planned_start"], earliest,
                    extension=max(0.0, _minutes(earliest - event["wait_from"])),
                    perturbation=state["departure_hold_min"] or None,
                )
                if wait:
                    output[unit["display_id"]].append(wait)
        elif earliest > event["wait_from"]:
            wait = _attente(
                service, block["start_station"], event["wait_from"], earliest,
                extension=_minutes(earliest - event["wait_from"]),
            )
            if wait:
                output[unit["display_id"]].append(wait)

        first_movement = True
        for row in block["rows"]:
            copied = dict(row)
            copied["start"] = row["start"] + shift
            copied["end"] = row["end"] + shift
            copied["is_mission_start"] = bool(
                block_index == 0 and first_movement
                and row["origine"] != row["terminus"]
            )
            if copied["is_mission_start"]:
                first_movement = False
            copied["service_id"] = service["service_id"]
            copied["train_id_original"] = service["train_id"]
            output[unit["display_id"]].append(copied)

        incident_crossing = bool(
            incident
            and incident["service_id"] == service["service_id"]
            and incident["kind"] == "croisement"
            and incident["block_index"] == block_index
        )
        hold_min = delay_min if incident_crossing else 0
        if hold_min:
            wait = _attente(
                service,
                block["end_station"],
                release,
                release + timedelta(minutes=hold_min),
                extension=hold_min,
                perturbation=hold_min,
            )
            if wait:
                output[unit["display_id"]].append(wait)
            release += timedelta(minutes=hold_min)

        engine.committed_schedules.append({
            "train_id": unit["display_id"],
            "service_id": service["service_id"],
            "path": [
                {"gare": block["start_station"], "index": index_start,
                 "arr": earliest, "dep": earliest},
                {"gare": block["end_station"], "index": index_end,
                 "arr": arrival, "dep": release},
            ],
        })
        state["blocks"][block_index] = {"departure": earliest, "arrival": arrival, "release": release}
        unit["location"] = block["end_station"]

        if block_index + 1 < len(service["blocks"]):
            following = service["blocks"][block_index + 1]
            scheduled_gap = max(timedelta(0), following["start"] - block["release"])
            next_earliest = max(following["start"], release + scheduled_gap)
            push(next_earliest, "bloc", {
                "unit_key": unit_key,
                "service_id": service["service_id"],
                "block_index": block_index + 1,
                "earliest": next_earliest,
                "wait_from": next_earliest,
                "attempts": 0,
            })
            continue

        state["arrival"] = arrival
        state["complete"] = True
        unit["location"] = service["terminus"]
        turnaround = service["turnaround_min"]
        if _minutes(release - service["planned_release"]) > _EPSILON_MIN:
            turnaround = 5.0
        initial_turnaround = turnaround
        requested_turnaround = _nombre_fini(
            turnaround_overrides.get(service["service_id"])
        )
        if (service["material"] == "batterie" and requested_turnaround is not None
                and requested_turnaround >= 0):
            if _puissance_charge_gare(df_gares, service["terminus"]) > 0:
                turnaround = max(turnaround, requested_turnaround)
        extra_turnaround = turnaround - initial_turnaround
        if extra_turnaround > _EPSILON_MIN:
            output[unit["display_id"]].append({
                "start": release,
                "end": release + timedelta(minutes=extra_turnaround),
                "origine": service["terminus"],
                "terminus": service["terminus"],
                "mission": service["mission"],
                "is_mission_start": False,
                "service_id": service["service_id"],
                "train_id_original": service["train_id"],
                "prolongation_retournement": True,
            })
        unit["available_at"] = release + timedelta(minutes=turnaround)
        next_operation[unit_key] += 1
        schedule_next_operation(unit_key)

    for unit_key, unit in units.items():
        output[unit["display_id"]].sort(
            key=lambda row: (row["start"], row["end"], row.get("service_id", ""))
        )

    for service in services:
        if service["service_id"] not in actual:
            motif = "Service non exécuté après une affectation impossible"
            actual[service["service_id"]] = {
                "departure": None, "arrival": None, "complete": False,
                "unit_key": None, "suppressed": True,
                "suppression_motif": motif,
            }
            suppression_motifs[service["service_id"]] = motif

    metric_rows = []
    total_delay = 0.0
    for service in services:
        state = actual.get(service["service_id"], {})
        arrival = state.get("arrival")
        if arrival is None:
            arrival_delay = None
        else:
            arrival_delay = max(0.0, _minutes(arrival - service["planned_arrival"]))
            total_delay += arrival_delay
        metric_rows.append({
            "service_id": service["service_id"],
            "train_id": service["train_id"],
            "mission": service["mission"],
            "depart_planifie": service["planned_start"],
            "depart_reel": state.get("departure"),
            "arrivee_planifiee": service["planned_arrival"],
            "arrivee_reelle": arrival,
            "retard_arrivee_min": arrival_delay,
            "complete": bool(state.get("complete")),
            "train_reel_id": units[state["unit_key"]]["display_id"] if state.get("unit_key") in units else None,
            "supprime": bool(state.get("suppressed")),
            "motif_suppression": state.get("suppression_motif"),
        })

    suppressions = []
    penalite_suppressions = 0.0
    for service in sorted(services, key=lambda item: (item["planned_start"], item["service_id"])):
        motif = suppression_motifs.get(service["service_id"])
        if motif is None:
            continue
        following = [
            (candidate, actual.get(candidate["service_id"], {}))
            for candidate in services
            if candidate["mission"] == service["mission"]
            and candidate["direction"] == service["direction"]
            and candidate["service_id"] != service["service_id"]
            and actual.get(candidate["service_id"], {}).get("complete")
            and actual.get(candidate["service_id"], {}).get("departure") is not None
            and actual[candidate["service_id"]]["departure"] > service["planned_start"]
        ]
        following.sort(key=lambda pair: (
            pair[1]["departure"], pair[0]["planned_start"], pair[0]["service_id"]
        ))
        hypothesis = None
        if following:
            next_service, next_state = following[0]
            attente = max(0.0, _minutes(next_state["departure"] - service["planned_start"]))
            service_suivant = next_service["service_id"]
        else:
            mission_services = sorted(
                (candidate for candidate in services
                 if candidate["mission"] == service["mission"]
                 and candidate["direction"] == service["direction"]),
                key=lambda item: (item["planned_start"], item["service_id"]),
            )
            try:
                frequency = float(service["mission_config"].get("frequence", 0) or 0)
            except (TypeError, ValueError):
                frequency = 0.0
            if math.isfinite(frequency) and frequency > 0:
                attente = 60.0 / frequency
                hypothesis = f"Fréquence mission {frequency:g} train/h"
            else:
                gaps = [
                    _minutes(right["planned_start"] - left["planned_start"])
                    for left, right in zip(mission_services, mission_services[1:])
                    if right["planned_start"] > left["planned_start"]
                ]
                if gaps:
                    attente = sorted(gaps)[len(gaps) // 2]
                    hypothesis = "Intervalle médian des départs planifiés de la mission"
                else:
                    attente = 60.0
                    hypothesis = "Fréquence et intervalle absents : hypothèse explicite de 1 train/h"
            service_suivant = None
        penalite = attente + 10.0
        penalite_suppressions += penalite
        suppression = {
            "service_id": service["service_id"],
            "motif": motif,
            "penalite_min": round(penalite, 6),
            "attente_min": round(attente, 6),
            "service_suivant_assure": service_suivant,
        }
        if hypothesis is not None:
            suppression["hypothese"] = hypothesis
        suppressions.append(suppression)

    return {
        "chronologie": {key: rows for key, rows in output.items() if rows},
        "retard_trains_min": round(total_delay, 6),
        "penalite_suppressions_min": round(penalite_suppressions, 6),
        "suppressions_count": len(suppressions),
        "suppressions": suppressions,
        "retard_cumule_min": round(total_delay + penalite_suppressions, 6),
        "services": metric_rows,
        "infra_violations": infra_violations,
        "physical_violations": physical_violations,
        "actual": actual,
        "units": units,
    }


def _parametres_materiel(energy_params, material):
    if energy_params is None:
        if material == "batterie":
            return None, True
        return get_default_energy_params(), False
    if not isinstance(energy_params, dict):
        if material == "batterie":
            return energy_params, False
        return get_default_energy_params(), True
    if isinstance(energy_params, dict) and "masse_tonne" in energy_params:
        return energy_params, False
    value = (energy_params or {}).get(material)
    if not value:
        if material == "batterie":
            return None, True
        return get_default_energy_params(), True
    return value, False


def _nombre_fini(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _puissance_charge_gare(df_gares, gare):
    if "electrification" not in df_gares.columns:
        return 0
    rows = df_gares.loc[df_gares["gare"] == gare, "electrification"]
    return _get_puissance_infra(rows.iloc[0]) if not rows.empty else 0


def _valider_parametres_batterie(params):
    if not isinstance(params, dict):
        return None, None, None, "paramètres batterie absents ou invalides"
    required = ("capacite_batterie_kwh", "soc_min_pct", "soc_max_pct")
    missing = [key for key in required if key not in params]
    if missing:
        return None, None, None, "paramètres batterie absents : " + ", ".join(missing)
    capacity = _nombre_fini(params["capacite_batterie_kwh"])
    threshold_pct = _nombre_fini(params["soc_min_pct"])
    max_soc_pct = _nombre_fini(params["soc_max_pct"])
    if capacity is None or capacity <= 0:
        return None, None, None, "capacité batterie invalide"
    if (threshold_pct is None or max_soc_pct is None
            or threshold_pct < 0 or max_soc_pct > 100
            or threshold_pct >= max_soc_pct):
        return None, None, None, "seuils de SoC batterie invalides"
    if params.get("simuler_fin_de_vie"):
        eol_pct = _nombre_fini(params.get("capacite_eol_pct"))
        if eol_pct is None or not 0 < eol_pct <= 100:
            return None, None, None, "capacité batterie en fin de vie invalide"
        capacity *= eol_pct / 100.0
    return capacity, threshold_pct, max_soc_pct, None


def _enregistrer_autonomie_non_verifiable(
    train_id, material, raison, trains, errors, reserve_errors,
    alertes_autonomie, alertes_autonomie_non_verifiable, train_errors=None,
):
    message = f"Train {train_id}: {raison}"
    if message not in errors:
        errors.append(message)
    if str(train_id).startswith("Réserve ") and message not in reserve_errors:
        reserve_errors.append(message)
    alert = f"{message}; autonomie batterie non vérifiable"
    alertes_autonomie.append(alert)
    alertes_autonomie_non_verifiable.append(alert)
    row_errors = list(train_errors or [])
    if message not in row_errors:
        row_errors.append(message)
    trains[str(train_id)] = {
        "type_materiel": material,
        "consommation_nette_kwh": None,
        "consommation_brute_kwh": None,
        "distance_km": None,
        "soc_initial_pct": None,
        "min_soc_pct": None,
        "soc_final_pct": None,
        "seuil_soc_min_pct": None,
        "autonomie_insuffisante": False,
        "autonomie_non_verifiable": True,
        "alertes_autonomie_non_verifiable": [alert],
        "alertes_autonomie": [alert],
        "marge_kwh": None,
        "marge_soc_pct": None,
        "energie_utilisable_au_dessus_seuil_kwh": None,
        "erreurs": row_errors,
    }


def _deficit_batterie_depuis_journal(battery_log, seuil_kwh):
    """Estime la profondeur de franchissement du seuil avant son bornage."""
    if not battery_log:
        return 0.0
    niveau = _nombre_fini(battery_log[0][1])
    if niveau is None:
        return 0.0
    deficit = 0.0
    for entry in battery_log[1:]:
        message = str(entry[3]) if len(entry) > 3 else ""
        match = re.search(r"ΔBatt:\s*([+-]?)([+-]?\d+(?:\.\d+)?)\s*kWh", message)
        if match:
            prefix, amount = match.groups()
            try:
                delta = float(amount) * (-1.0 if prefix == "-" else 1.0)
            except ValueError:
                delta = 0.0
            deficit = max(deficit, seuil_kwh - (niveau + delta))
        next_level = _nombre_fini(entry[1]) if len(entry) > 1 else None
        if next_level is not None:
            niveau = next_level
    return max(0.0, deficit)


def _analyser_energie(chronologie, services, df_gares, energy_params,
                      include_diagnostics=False):
    """Calcule les bilans physiques sur la chronologie nominale ou rejouée."""
    service_by_id = {service["service_id"]: service for service in services}
    service_by_source = {}
    for service in services:
        service_by_source.setdefault(service["train_id"], service)
    trains = {}
    errors = []
    minimum_soc = []
    reserve_errors = []
    autonomie_insuffisante = False
    alertes_autonomie = []
    autonomie_non_verifiable = False
    alertes_autonomie_non_verifiable = []
    for train_id, rows in (chronologie or {}).items():
        if not rows:
            continue
        service = None
        for row in rows:
            service = service_by_id.get(row.get("service_id"))
            if service is not None:
                break
        if service is None:
            service = service_by_source.get(train_id)
        if service is None:
            errors.append(f"Train {train_id}: mission non associée")
            continue
        params, params_missing = _parametres_materiel(energy_params, service["material"])
        is_battery = service["material"] == "batterie"
        battery_capacity = threshold_pct = max_soc_pct = None
        if is_battery:
            if params_missing:
                battery_issue = "paramètres absents pour batterie"
            else:
                battery_capacity, threshold_pct, max_soc_pct, battery_issue = (
                    _valider_parametres_batterie(params)
                )
            if (battery_issue or battery_capacity is None or threshold_pct is None
                    or max_soc_pct is None):
                _enregistrer_autonomie_non_verifiable(
                    train_id, service["material"],
                    battery_issue or "paramètres batterie invalides", trains, errors,
                    reserve_errors, alertes_autonomie,
                    alertes_autonomie_non_verifiable,
                )
                autonomie_non_verifiable = True
                continue
            assert battery_capacity is not None
            assert threshold_pct is not None
            assert max_soc_pct is not None
        elif params_missing:
            missing_message = f"Train {train_id}: paramètres absents pour {service['material']}"
            errors.append(missing_message)
            if str(train_id).startswith("Réserve "):
                reserve_errors.append(missing_message)
        if not isinstance(params, dict):
            params = get_default_energy_params()
        try:
            result = calculer_consommation_trajet(rows, service["mission_config"], df_gares, params)
        except Exception as exc:  # une erreur énergétique est rapportée, pas masquée
            if is_battery:
                _enregistrer_autonomie_non_verifiable(
                    train_id, service["material"],
                    f"calcul énergétique impossible ({exc})", trains, errors,
                    reserve_errors, alertes_autonomie,
                    alertes_autonomie_non_verifiable,
                )
                autonomie_non_verifiable = True
            else:
                message = f"Train {train_id}: calcul énergétique impossible ({exc})"
                errors.append(message)
                if str(train_id).startswith("Réserve "):
                    reserve_errors.append(message)
            continue
        if not isinstance(result, dict):
            if is_battery:
                _enregistrer_autonomie_non_verifiable(
                    train_id, service["material"],
                    "résultat de calcul énergétique invalide", trains, errors,
                    reserve_errors, alertes_autonomie,
                    alertes_autonomie_non_verifiable,
                )
                autonomie_non_verifiable = True
            else:
                errors.append(f"Train {train_id}: résultat de calcul énergétique invalide")
            continue
        raw_train_errors = result.get("erreurs") or []
        if not isinstance(raw_train_errors, (list, tuple)):
            raw_train_errors = ["liste d'erreurs de calcul invalide"]
        train_errors = [str(error) for error in raw_train_errors]
        errors.extend(f"Train {train_id}: {error}" for error in train_errors)
        if str(train_id).startswith("Réserve "):
            reserve_errors.extend(f"Train {train_id}: {error}" for error in train_errors)

        if is_battery:
            seuil_errors = [
                error for error in train_errors
                if "Batterie au seuil minimum" in error
            ]
            autres_erreurs = [error for error in train_errors if error not in seuil_errors]
            battery_issue = None
            if autres_erreurs:
                battery_issue = "erreur(s) de calcul : " + "; ".join(autres_erreurs)
            result_fields = (
                "total_conso_nette_kwh", "total_conso_brute_kwh",
                "total_distance_km",
            )
            result_values = {}
            for field in result_fields:
                if field not in result:
                    battery_issue = battery_issue or f"résultat incomplet ({field} absent)"
                    continue
                value = _nombre_fini(result[field])
                if value is None:
                    battery_issue = battery_issue or f"résultat invalide ({field})"
                else:
                    result_values[field] = value
            battery_log = result.get("batterie_log")
            if not isinstance(battery_log, (list, tuple)) or not battery_log:
                battery_issue = battery_issue or "journal de batterie absent ou vide"
                soc_values = []
            else:
                soc_values = []
                for entry in battery_log:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                        battery_issue = battery_issue or "journal de batterie invalide"
                        break
                    value = _nombre_fini(entry[1])
                    if (value is None or value < 0
                            or value > battery_capacity * max_soc_pct / 100.0 + 1e-6):
                        battery_issue = battery_issue or "SoC invalide dans le journal de batterie"
                        break
                    soc_values.append(value)
                if not battery_issue and not soc_values:
                    battery_issue = "journal de batterie sans valeurs de SoC"
            if battery_issue:
                _enregistrer_autonomie_non_verifiable(
                    train_id, service["material"], battery_issue, trains, errors,
                    reserve_errors, alertes_autonomie,
                    alertes_autonomie_non_verifiable, train_errors,
                )
                autonomie_non_verifiable = True
                continue

            initial_soc = soc_values[0] / battery_capacity * 100.0
            min_train_soc = min(soc_values) / battery_capacity * 100.0
            final_soc = soc_values[-1] / battery_capacity * 100.0
            threshold_kwh = battery_capacity * threshold_pct / 100.0
            margin_kwh = max(0.0, min(soc_values) - threshold_kwh)
            margin_soc_pct = max(0.0, min_train_soc - threshold_pct)
            train_insufficient = min_train_soc <= threshold_pct + 1e-6
            if seuil_errors and not train_insufficient:
                _enregistrer_autonomie_non_verifiable(
                    train_id, service["material"],
                    "journal incohérent avec le signalement du seuil batterie",
                    trains, errors, reserve_errors, alertes_autonomie,
                    alertes_autonomie_non_verifiable, train_errors,
                )
                autonomie_non_verifiable = True
                continue
            if train_insufficient:
                autonomie_insuffisante = True
                alertes_autonomie.append(
                    f"Train {train_id}: SoC minimal {min_train_soc:.2f}% "
                    f"au seuil soc_min_pct {threshold_pct:.2f}%"
                )
            minimum_soc.append(min_train_soc)
            train_result = {
                "type_materiel": service["material"],
                "consommation_nette_kwh": result_values["total_conso_nette_kwh"],
                "consommation_brute_kwh": result_values["total_conso_brute_kwh"],
                "distance_km": result_values["total_distance_km"],
                "soc_initial_pct": initial_soc,
                "min_soc_pct": min_train_soc,
                "soc_final_pct": final_soc,
                "seuil_soc_min_pct": threshold_pct,
                "autonomie_insuffisante": train_insufficient,
                "autonomie_non_verifiable": False,
                "alertes_autonomie_non_verifiable": [],
                "alertes_autonomie": [],
                "marge_kwh": margin_kwh,
                "marge_soc_pct": margin_soc_pct,
                "energie_utilisable_au_dessus_seuil_kwh": margin_kwh,
                "erreurs": train_errors,
            }
            if include_diagnostics:
                train_result["_deficit_batterie_kwh"] = _deficit_batterie_depuis_journal(
                    battery_log, threshold_kwh
                )
            trains[str(train_id)] = train_result
            continue

        capacity = float(params.get("capacite_batterie_kwh", 0) or 0)
        if params.get("simuler_fin_de_vie"):
            capacity *= float(params.get("capacite_eol_pct", 80)) / 100.0
        battery_log = result.get("batterie_log") or []
        soc_values = [float(entry[1]) for entry in battery_log if len(entry) > 1]
        initial_soc = soc_values[0] / capacity * 100.0 if capacity > 0 and soc_values else None
        min_train_soc = min(soc_values) / capacity * 100.0 if capacity > 0 and soc_values else None
        final_soc = soc_values[-1] / capacity * 100.0 if capacity > 0 and soc_values else None
        if min_train_soc is not None:
            minimum_soc.append(min_train_soc)
        trains[str(train_id)] = {
            "type_materiel": service["material"],
            "consommation_nette_kwh": float(result.get("total_conso_nette_kwh", 0.0)),
            "consommation_brute_kwh": float(result.get("total_conso_brute_kwh", 0.0)),
            "distance_km": float(result.get("total_distance_km", 0.0)),
            "soc_initial_pct": initial_soc,
            "min_soc_pct": min_train_soc,
            "soc_final_pct": final_soc,
            "seuil_soc_min_pct": None,
            "autonomie_insuffisante": False,
            "autonomie_non_verifiable": False,
            "alertes_autonomie_non_verifiable": [],
            "alertes_autonomie": [],
            "marge_kwh": None,
            "marge_soc_pct": None,
            "energie_utilisable_au_dessus_seuil_kwh": None,
            "erreurs": train_errors,
        }
    return {
        "consommation_nette_kwh": (
            None if any(row["consommation_nette_kwh"] is None for row in trains.values())
            else round(sum(row["consommation_nette_kwh"] for row in trains.values()), 6)
        ),
        "min_soc_pct": (
            None if autonomie_non_verifiable
            else min(minimum_soc) if minimum_soc else None
        ),
        "par_train": trains,
        "erreurs": errors,
        "erreurs_reserve": reserve_errors,
        "autonomie_insuffisante": autonomie_insuffisante,
        "alertes_autonomie": alertes_autonomie,
        "autonomie_non_verifiable": autonomie_non_verifiable,
        "alertes_autonomie_non_verifiable": alertes_autonomie_non_verifiable,
    }


def _cle_chronologie_energie(chronologie):
    return tuple(
        (
            str(train_id),
            str(row.get("service_id", "")),
            str(row.get("mission", "")),
            row.get("start"),
            row.get("end"),
            row.get("origine"),
            row.get("terminus"),
        )
        for train_id, rows in sorted(
            (chronologie or {}).items(), key=lambda item: str(item[0])
        )
        for row in rows or []
    )


def _cle_train_dans_cle_energie(key, train_id):
    train_id = str(train_id)
    return tuple(row for row in key if row[0] == train_id)


def _analyser_energie_cache(chronologie, services, df_gares, energy_params,
                            energy_cache, include_diagnostics=True):
    key = _cle_chronologie_energie(chronologie)
    if key not in energy_cache:
        energy_cache[key] = _analyser_energie(
            chronologie, services, df_gares, energy_params,
            include_diagnostics=include_diagnostics,
        )
    return energy_cache[key]


def _retour_normal(services, metrics, incident, delay_min, infra_violations=()):
    if infra_violations:
        return None
    by_id = {row["service_id"]: row for row in metrics}
    impact_times = [incident["time"] + timedelta(minutes=delay_min)]
    impact_times.extend(
        row["arrivee_reelle"] for row in metrics
        if row["complete"]
        and (row["retard_arrivee_min"] or 0.0) > _EPSILON_MIN
        and row["arrivee_reelle"] is not None
    )
    impact_times.extend(
        row.get("depart_planifie") or next(
            service["planned_start"] for service in services
            if service["service_id"] == row["service_id"]
        )
        for row in metrics if row.get("supprime")
    )
    event_time = max(impact_times)

    proofs = []
    for service in services:
        row = by_id.get(service["service_id"])
        if row is None or not row["complete"] or row["depart_reel"] is None:
            continue
        if service["planned_start"] < event_time:
            continue
        if abs(_minutes(row["depart_reel"] - service["planned_start"])) > _EPSILON_MIN:
            continue
        if (row["retard_arrivee_min"] or 0.0) > _EPSILON_MIN:
            continue
        proofs.append((service, row))
    proofs.sort(key=lambda pair: (pair[0]["planned_start"], pair[0]["service_id"]))
    for proof_service, _ in proofs:
        later = [
            (service, by_id.get(service["service_id"]))
            for service in services
            if service["planned_start"] >= proof_service["planned_start"]
        ]
        if all(
            row is not None
            and row["complete"]
            and not row.get("supprime")
            and row["depart_reel"] is not None
            and abs(_minutes(row["depart_reel"] - service["planned_start"])) <= _EPSILON_MIN
            and (row["retard_arrivee_min"] or 0.0) <= _EPSILON_MIN
            for service, row in later
        ):
            return max(0.0, _minutes(proof_service["planned_start"] - incident["time"]))
    return None


def _bilan_batterie_valide_suffisant(train):
    return bool(
        train
        and train.get("type_materiel") == "batterie"
        and not train.get("autonomie_insuffisante")
        and not train.get("autonomie_non_verifiable")
        and _nombre_fini(train.get("min_soc_pct")) is not None
        and _nombre_fini(train.get("seuil_soc_min_pct")) is not None
    )


def _ligne_nominale_pour_train(train_id, replay, services, baseline_energy):
    rows = baseline_energy.get("par_train", {})
    baseline = rows.get(str(train_id))
    if baseline is not None:
        return baseline
    if not str(train_id).startswith("Réserve "):
        return None
    metric_by_id = {row["service_id"]: row for row in replay.get("services", [])}
    sources = {
        str(service["train_id"])
        for service in services
        if metric_by_id.get(service["service_id"], {}).get("train_reel_id") == train_id
    }
    source_rows = [rows[source] for source in sources if source in rows]
    if len(source_rows) == 1:
        return source_rows[0]
    return None


def _alertes_batterie_nouvelles(energy, replay, services, baseline_energy):
    alerts = {}
    for train_id, train in energy.get("par_train", {}).items():
        if (train.get("type_materiel") != "batterie"
                or not train.get("autonomie_insuffisante")):
            continue
        baseline = _ligne_nominale_pour_train(
            train_id, replay, services, baseline_energy
        )
        if _bilan_batterie_valide_suffisant(baseline):
            alerts[str(train_id)] = train
    return alerts


def _insuffisances_batterie(energy):
    return {
        str(train_id)
        for train_id, train in energy.get("par_train", {}).items()
        if train.get("type_materiel") == "batterie"
        and train.get("autonomie_insuffisante")
    }


def _deficit_train(train):
    deficit = _nombre_fini(train.get("_deficit_batterie_kwh"))
    if deficit is None:
        return 0.001 if train.get("autonomie_insuffisante") else 0.0
    return max(0.0, deficit)


def _retournement_effectif(service, replay):
    state = replay.get("actual", {}).get(service["service_id"], {})
    last_block = state.get("blocks", {}).get(len(service["blocks"]) - 1)
    if not state.get("complete") or last_block is None:
        return None
    if _minutes(last_block["release"] - service["planned_release"]) > _EPSILON_MIN:
        return 5.0
    return service["turnaround_min"]


def _limite_charge_retournement(service, replay, services, df_gares, energy_params):
    puissance_infra = _puissance_charge_gare(df_gares, service["terminus"])
    if service["material"] != "batterie" or puissance_infra <= 0:
        return None
    params, missing = _parametres_materiel(energy_params, "batterie")
    if missing or not isinstance(params, dict):
        return None
    capacity, threshold_pct, max_soc_pct, issue = _valider_parametres_batterie(params)
    if issue or capacity is None or threshold_pct is None or max_soc_pct is None:
        return None
    aux_kw = _nombre_fini(params.get("facteur_aux_kwh_h", 50.0))
    aux_rate_pct = _nombre_fini(params.get("taux_aux_terminus_pct", 100.0))
    charge_c = _nombre_fini(params.get("facteur_charge_C", 4.0))
    if aux_kw is None:
        return None
    if aux_rate_pct is None:
        aux_rate_pct = 100.0
    if charge_c is None:
        return None
    aux_kw *= max(0.0, min(100.0, aux_rate_pct)) / 100.0
    charge_kw = min(max(0.0, puissance_infra - aux_kw), capacity * max(0.0, charge_c))
    if charge_kw <= 0:
        return None
    usable_kwh = capacity * (max_soc_pct - threshold_pct) / 100.0
    max_charge_minutes = math.ceil(usable_kwh * 60.0 / charge_kw)
    state = replay.get("actual", {}).get(service["service_id"], {})
    last_block = state.get("blocks", {}).get(len(service["blocks"]) - 1)
    if last_block is None:
        return None
    horizon_end = max(item["planned_release"] for item in services)
    horizon_minutes = math.floor(_minutes(horizon_end - last_block["release"]))
    if horizon_minutes <= 0:
        return None
    return max(0, min(max_charge_minutes, horizon_minutes))


def _replay_correction_admissible(original, trial):
    original_rows = {row["service_id"]: row for row in original["services"]}
    trial_rows = {row["service_id"]: row for row in trial["services"]}
    if set(original_rows) != set(trial_rows):
        return False
    if {sid for sid, row in original_rows.items() if row["complete"]} != {
        sid for sid, row in trial_rows.items() if row["complete"]
    }:
        return False
    if {sid for sid, row in original_rows.items() if row.get("supprime")} != {
        sid for sid, row in trial_rows.items() if row.get("supprime")
    }:
        return False
    if Counter(trial["infra_violations"]) - Counter(original["infra_violations"]):
        return False
    if Counter(trial["physical_violations"]) - Counter(original["physical_violations"]):
        return False
    for service_id, row in trial_rows.items():
        departure = row.get("depart_reel")
        planned = row.get("depart_planifie")
        original_departure = original_rows[service_id].get("depart_reel")
        if departure is not None and planned is not None:
            if _minutes(departure - planned) < -_EPSILON_MIN:
                return False
        if (departure is not None and original_departure is not None
                and _minutes(departure - original_departure) < -_EPSILON_MIN):
            return False
    return True


def _energies_suffisantes_nominales(baseline_energy, services, df_gares):
    return any(
        service["material"] == "batterie"
        and _puissance_charge_gare(df_gares, service["terminus"]) > 0
        and _bilan_batterie_valide_suffisant(
            baseline_energy.get("par_train", {}).get(str(service["train_id"]))
        )
        for service in services
    )


def _liste_recharges_prolongees(services, overrides, initial_turnarounds, replay=None):
    service_by_id = {service["service_id"]: service for service in services}
    records = []
    for service_id, retained in overrides.items():
        service = service_by_id[service_id]
        initial = (
            _retournement_effectif(service, replay)
            if replay is not None else None
        )
        if initial is None:
            initial = initial_turnarounds[service_id]
        if retained <= initial + _EPSILON_MIN:
            continue
        records.append({
            "service_id": service_id,
            "train_id": service["train_id"],
            "gare": service["terminus"],
            "retournement_initial_min": round(initial, 6),
            "retournement_retenu_min": round(retained, 6),
            "prolongation_min": round(retained - initial, 6),
            "motif": "suppression d'une nouvelle alerte batterie",
        })
    records.sort(key=lambda item: (
        service_by_id[item["service_id"]]["planned_start"], item["service_id"]
    ))
    return records


def _minimiser_retournements(replay, energy, services, df_gares, incident,
                             delay_min, baseline_energy, energy_params,
                             reserve_map, reserve_stations, energy_cache,
                             overrides, initial_turnarounds, original_replay):
    overrides = dict(overrides)
    current_replay = replay
    current_energy = energy
    service_by_id = {service["service_id"]: service for service in services}
    for _ in range(max(1, len(overrides) + 1)):
        changed = False
        for service_id in sorted(
            overrides,
            key=lambda sid: (service_by_id[sid]["planned_start"], sid),
        ):
            service = service_by_id[service_id]
            retained = overrides[service_id]
            minimum = max(initial_turnarounds[service_id], service["turnaround_min"])
            for target in range(math.ceil(minimum - _EPSILON_MIN), math.ceil(retained)):
                trial_overrides = dict(overrides)
                trial_overrides[service_id] = float(target)
                trial = _rejouer(
                    services, df_gares, incident, delay_min, reserve_map,
                    reserve_stations, turnaround_overrides=trial_overrides,
                )
                if not _replay_correction_admissible(original_replay, trial):
                    continue
                trial_energy = _analyser_energie_cache(
                    trial["chronologie"], services, df_gares, energy_params,
                    energy_cache,
                )
                if (not _insuffisances_batterie(trial_energy).issubset(
                        _insuffisances_batterie(current_energy))
                        or _alertes_batterie_nouvelles(
                            trial_energy, trial, services, baseline_energy
                        )):
                    continue
                overrides = trial_overrides
                current_replay = trial
                current_energy = trial_energy
                changed = True
                break
        if not changed:
            break
    return current_replay, current_energy, overrides


def _corriger_retournements(replay, services, df_gares, incident, delay_min,
                            baseline_energy, energy_params, reserve_map=None,
                            reserve_stations=(), energy_cache=None):
    """Ajoute le minimum de recharge de retournement corrigeant les alertes neuves."""
    initial_return = _retour_normal(
        services, replay["services"], incident, delay_min, replay["infra_violations"]
    )
    energy_cache = energy_cache if energy_cache is not None else {}
    reserve_map = dict(reserve_map or {})
    if not _energies_suffisantes_nominales(baseline_energy, services, df_gares):
        return replay, None, [], initial_return
    nominal_key = energy_cache.get(_CLE_CHRONOLOGIE_NOMINALE)
    battery_train_ids = [
        train_id for train_id, train in baseline_energy.get("par_train", {}).items()
        if train.get("type_materiel") == "batterie"
    ]
    replay_key = _cle_chronologie_energie(replay["chronologie"])
    if (not reserve_map and nominal_key is not None and battery_train_ids
            and all(
                _cle_train_dans_cle_energie(replay_key, train_id)
                == _cle_train_dans_cle_energie(nominal_key, train_id)
                for train_id in battery_train_ids
            )):
        return replay, None, [], initial_return

    energy = _analyser_energie_cache(
        replay["chronologie"], services, df_gares, energy_params, energy_cache
    )
    alerts = _alertes_batterie_nouvelles(energy, replay, services, baseline_energy)
    if not alerts:
        return replay, energy, [], initial_return

    original_replay = replay
    original_energy = energy
    current_replay = replay
    current_energy = energy
    overrides = {}
    initial_turnarounds = {}
    best_complete = None
    max_iterations = max(1, len(services) * 2)

    for _ in range(max_iterations):
        current_alerts = _alertes_batterie_nouvelles(
            current_energy, current_replay, services, baseline_energy
        )
        if not current_alerts:
            if overrides:
                current_replay, current_energy, overrides = _minimiser_retournements(
                    current_replay, current_energy, services, df_gares, incident,
                    delay_min, baseline_energy, energy_params, reserve_map,
                    reserve_stations, energy_cache, overrides,
                    initial_turnarounds, original_replay,
                )
            return current_replay, current_energy, _liste_recharges_prolongees(
                services, overrides, initial_turnarounds, current_replay
            ), initial_return

        current_deficits = {
            train_id: _deficit_train(train)
            for train_id, train in current_alerts.items()
        }
        current_all_insufficient = _insuffisances_batterie(current_energy)
        metric_by_id = {row["service_id"]: row for row in current_replay["services"]}
        service_by_id = {service["service_id"]: service for service in services}
        solution = None
        partial = None
        for service in services:
            metric = metric_by_id.get(service["service_id"], {})
            train_id = str(metric.get("train_reel_id") or "")
            if (not metric.get("complete") or train_id not in current_alerts
                    or service["material"] != "batterie"):
                continue
            if not any(
                later["complete"]
                and str(later.get("train_reel_id") or "") == train_id
                and later["service_id"] != service["service_id"]
                and next(item for item in services
                         if item["service_id"] == later["service_id"])["planned_start"]
                > service["planned_start"]
                for later in current_replay["services"]
            ):
                continue
            limit = _limite_charge_retournement(
                service, current_replay, services, df_gares, energy_params
            )
            if limit is None:
                continue
            max_extra = limit
            effective = _retournement_effectif(service, current_replay)
            if effective is None:
                continue
            lower = effective
            current_target = overrides.get(service["service_id"])
            if current_target is not None:
                lower = max(lower, current_target)
            upper = math.floor(min(
                lower + max_extra,
                _minutes(max(item["planned_release"] for item in services)
                         - current_replay["actual"][service["service_id"]]["blocks"][
                             len(service["blocks"]) - 1
                         ]["release"]),
            ))
            if current_target is None:
                first_target = math.ceil(lower - _EPSILON_MIN)
                if first_target <= effective + _EPSILON_MIN:
                    first_target = math.floor(effective + _EPSILON_MIN) + 1
            else:
                first_target = math.floor(current_target + _EPSILON_MIN) + 1
            if first_target > upper:
                continue

            for target in range(first_target, upper + 1):
                trial_overrides = dict(overrides)
                trial_overrides[service["service_id"]] = float(target)
                trial = _rejouer(
                    services, df_gares, incident, delay_min, reserve_map,
                    reserve_stations, turnaround_overrides=trial_overrides,
                )
                if not _replay_correction_admissible(original_replay, trial):
                    continue
                trial_energy = _analyser_energie_cache(
                    trial["chronologie"], services, df_gares, energy_params,
                    energy_cache,
                )
                trial_alerts = _alertes_batterie_nouvelles(
                    trial_energy, trial, services, baseline_energy
                )
                trial_all_insufficient = _insuffisances_batterie(trial_energy)
                if not trial_all_insufficient.issubset(current_all_insufficient):
                    continue
                if not set(trial_alerts).issubset(current_alerts):
                    continue
                if any(
                    trial_alerts[train].get("_deficit_batterie_kwh", 0.0)
                    > current_deficits.get(train, 0.0) + 1e-5
                    for train in trial_alerts
                ):
                    continue
                added = sum(
                    max(0.0, value - initial_turnarounds.get(
                        service_key,
                        effective if service_key == service["service_id"] else (
                            _retournement_effectif(
                                service_by_id[service_key], original_replay
                            ) or 0.0
                        ),
                    ))
                    for service_key, value in trial_overrides.items()
                )
                if not trial_alerts:
                    candidate = (added, service["planned_start"], service["service_id"],
                                 trial, trial_energy, trial_overrides, target, effective)
                    if solution is None or candidate[:3] < solution[:3]:
                        solution = candidate
                    break
                trial_deficit = sum(_deficit_train(row) for row in trial_alerts.values())
                current_deficit = sum(current_deficits.values())
                final_gain = sum(
                    max(0.0,
                        (_nombre_fini(trial_alerts[train].get("soc_final_pct")) or 0.0)
                        - (_nombre_fini(current_alerts[train].get("soc_final_pct")) or 0.0)
                    )
                    for train in current_alerts if train in trial_alerts
                )
                cleared = len(current_alerts) - len(trial_alerts)
                if (cleared <= 0 and current_deficit - trial_deficit <= 1e-5
                        and final_gain <= _EPSILON_MIN):
                    continue
                added_increment = max(
                    1.0,
                    target - max(effective, current_target or effective),
                )
                progress = max(current_deficit - trial_deficit, final_gain)
                candidate = (
                    len(trial_alerts), -round(progress / added_increment, 6),
                    round(trial_deficit, 6), added, service["planned_start"],
                    service["service_id"], trial, trial_energy, trial_overrides,
                    target, effective,
                )
                if partial is None or candidate[:6] < partial[:6]:
                    partial = candidate

        if solution is not None:
            (added, _, service_id, complete_replay, complete_energy,
             complete_overrides, _, effective) = solution
            complete_initials = dict(initial_turnarounds)
            complete_initials.setdefault(service_id, effective)
            if best_complete is None or added < best_complete[0] - _EPSILON_MIN:
                best_complete = (
                    added, complete_replay, complete_energy,
                    complete_overrides, complete_initials,
                )

        if (partial is not None
                and (solution is None or partial[3] < solution[0] - _EPSILON_MIN)):
            (_, _, _, _, _, service_id, current_replay, current_energy,
             overrides, _, effective) = partial
            initial_turnarounds.setdefault(service_id, effective)
        elif solution is None:
            break
        else:
            break

    if _alertes_batterie_nouvelles(
        current_energy, current_replay, services, baseline_energy
    ):
        if best_complete is not None:
            _, complete_replay, complete_energy, complete_overrides, complete_initials = best_complete
            complete_replay, complete_energy, complete_overrides = (
                _minimiser_retournements(
                    complete_replay, complete_energy, services, df_gares,
                    incident, delay_min, baseline_energy, energy_params,
                    reserve_map, reserve_stations, energy_cache,
                    complete_overrides, complete_initials, original_replay,
                )
            )
            return complete_replay, complete_energy, _liste_recharges_prolongees(
                services, complete_overrides, complete_initials, complete_replay
            ), initial_return
        return original_replay, original_energy, [], initial_return
    return current_replay, current_energy, _liste_recharges_prolongees(
        services, overrides, initial_turnarounds, current_replay
    ), initial_return


def _energie_publique(energy):
    result = deepcopy(energy)
    for train in result.get("par_train", {}).values():
        train.pop("_deficit_batterie_kwh", None)
    return result


def _format_result(replay, services, incident, delay_min, reserve_map,
                   baseline_energy, df_gares, energy_params, energy=None,
                   recharges_prolongees=(),
                   retour_normal_sans_recharge_min=_UNFOURNI):
    energy = _energie_publique(
        energy if energy is not None else _analyser_energie(
            replay["chronologie"], services, df_gares, energy_params
        )
    )
    baseline_energy = _energie_publique(baseline_energy)
    baseline_trains = baseline_energy["par_train"]
    for train_id, train in energy["par_train"].items():
        baseline = baseline_trains.get(train_id)

        def delta(field):
            value = train.get(field)
            baseline_value = baseline.get(field) if baseline is not None else None
            if value is None or baseline_value is None:
                return None
            return round(value - baseline_value, 6)

        train["delta_vs_nominal"] = {
            "distance_km": delta("distance_km"),
            "consommation_nette_kwh": delta("consommation_nette_kwh"),
            "marge_kwh": delta("marge_kwh"),
            "marge_soc_pct": delta("marge_soc_pct"),
        }
    delta_conso_nette = None
    if (energy["consommation_nette_kwh"] is not None
            and baseline_energy["consommation_nette_kwh"] is not None):
        delta_conso_nette = round(
            energy["consommation_nette_kwh"]
            - baseline_energy["consommation_nette_kwh"], 6
        )
    battery = {
        "baseline": baseline_energy,
        "scenario": energy,
        "delta_conso_nette_kwh": delta_conso_nette,
        "delta_kwh": delta_conso_nette,
        "min_soc_pct": energy["min_soc_pct"],
        "soc_min_pct": energy["min_soc_pct"],
        "autonomie_insuffisante": energy["autonomie_insuffisante"],
        "alertes_autonomie": list(energy["alertes_autonomie"]),
        "autonomie_non_verifiable": energy["autonomie_non_verifiable"],
        "alertes_autonomie_non_verifiable": list(
            energy["alertes_autonomie_non_verifiable"]
        ),
        "baseline_autonomie_non_verifiable": baseline_energy[
            "autonomie_non_verifiable"
        ],
        "alertes_autonomie_non_verifiable_baseline": list(
            baseline_energy["alertes_autonomie_non_verifiable"]
        ),
        "erreurs": list(energy["erreurs"]),
        "erreurs_reserve": list(energy["erreurs_reserve"]),
    }
    reserve_activations = []
    by_id = {row["service_id"]: row for row in replay["services"]}
    for service_id, gare in reserve_map.items():
        service = next(service for service in services if service["service_id"] == service_id)
        row = by_id[service_id]
        reserve_activations.append({
            "service_id": service_id,
            "train_id": service["train_id"],
            "reserve_train_id": row["train_reel_id"],
            "gare": gare,
            "mission": service["mission"],
            "depart_planifie": service["planned_start"],
            "depart_reel": row["depart_reel"],
            "gain_cumule_min": None,
        })
    reserve_activations.sort(key=lambda item: (item["depart_planifie"], item["service_id"]))
    return {
        "chronologie": replay["chronologie"],
        "retard_cumule_min": replay["retard_cumule_min"],
        "retard_trains_min": replay["retard_trains_min"],
        "penalite_suppressions_min": replay["penalite_suppressions_min"],
        "suppressions_count": replay["suppressions_count"],
        "suppressions": replay["suppressions"],
        "retour_normal_min": _retour_normal(
            services, replay["services"], incident, delay_min, replay["infra_violations"]
        ),
        "retour_normal_sans_recharge_min": (
            _retour_normal(
                services, replay["services"], incident, delay_min,
                replay["infra_violations"],
            )
            if retour_normal_sans_recharge_min is _UNFOURNI
            else retour_normal_sans_recharge_min
        ),
        "recharges_prolongees": list(recharges_prolongees),
        "reserve_activations": reserve_activations,
        "reserve_refus": [],
        "battery": battery,
        "infra_violations": replay["infra_violations"],
        "physical_violations": replay["physical_violations"],
        "arrivals": replay["services"],
    }


def _etendre_rotation_reserve(services, reserve_map, service, gare):
    """Transfère les rotations nécessaires jusqu'au retour du train d'origine."""
    trial_map = dict(reserve_map)
    trial_map[service["service_id"]] = gare
    sequence = sorted(
        (item for item in services if item["train_id"] == service["train_id"]),
        key=lambda item: (item["planned_start"], item["service_id"]),
    )
    position = next(i for i, item in enumerate(sequence) if item["service_id"] == service["service_id"])
    source_location = sequence[0]["origine"]
    for previous in sequence[:position]:
        if previous["service_id"] not in reserve_map:
            source_location = previous["terminus"]

    reserve_location = service["terminus"]
    for following in sequence[position + 1:]:
        if following["origine"] == source_location:
            break
        existing_station = trial_map.get(following["service_id"])
        if existing_station is not None:
            if existing_station == gare and following["origine"] == reserve_location:
                reserve_location = following["terminus"]
            continue
        if following["origine"] != reserve_location:
            break
        trial_map[following["service_id"]] = gare
        reserve_location = following["terminus"]
    return trial_map


def _evaluuer_avec_reserves(services, df_gares, incident, delay_min, stations,
                            no_reserve, baseline_energy, energy_params,
                            current_energy=None, current_recharges=(),
                            current_return_normal_sans_recharge=_UNFOURNI,
                            energy_cache=None):
    current_map = {}
    current_replay = no_reserve
    reserve_refus = []
    activation_gains = {}
    if not stations:
        return None
    energy_cache = energy_cache if energy_cache is not None else {}
    if current_energy is None:
        current_energy = _analyser_energie_cache(
            current_replay["chronologie"], services, df_gares,
            energy_params, energy_cache,
        )
    current_recharges = list(current_recharges)
    if current_return_normal_sans_recharge is _UNFOURNI:
        current_return_normal_sans_recharge = _retour_normal(
            services, current_replay["services"], incident, delay_min,
            current_replay["infra_violations"],
        )

    sorted_services = sorted(services, key=lambda service: (service["planned_start"], service["service_id"]))
    for service in sorted_services:
        if service["service_id"] == incident["service_id"] or service["service_id"] in current_map:
            continue
        if service["planned_start"] < incident["time"]:
            continue
        state = next((row for row in current_replay["services"]
                      if row["service_id"] == service["service_id"]), None)
        if not state or (not state["complete"] and not state.get("supprime")):
            continue
        if not state.get("supprime") and (
            (state["retard_arrivee_min"] or 0.0) <= _EPSILON_MIN
            and (
                state["depart_reel"] is None
                or abs(_minutes(state["depart_reel"] - state["depart_planifie"]))
                <= _EPSILON_MIN
            )
        ):
            continue

        best = None
        for gare in stations:
            if service["service_id"] in current_map:
                continue
            trial_map = _etendre_rotation_reserve(services, current_map, service, gare)
            trial = _rejouer(services, df_gares, incident, delay_min, trial_map, stations)
            if trial["infra_violations"] or trial["physical_violations"]:
                reasons = []
                if trial["infra_violations"]:
                    reasons.append("Conflit d'infrastructure non résolu")
                if trial["physical_violations"]:
                    reasons.append("Réserve indisponible physiquement ou matériel incompatible")
                reserve_refus.append({
                    "service_id": service["service_id"], "gare": gare,
                    "raisons": reasons,
                })
                continue
            trial, trial_energy, trial_recharges, trial_return_normal = (
                _corriger_retournements(
                    trial, services, df_gares, incident, delay_min,
                    baseline_energy, energy_params, trial_map, stations,
                    energy_cache,
                )
            )
            if trial_energy is None:
                trial_energy = _analyser_energie_cache(
                    trial["chronologie"], services, df_gares, energy_params,
                    energy_cache,
                )
            raisons = []
            if trial["infra_violations"]:
                raisons.append("Conflit d'infrastructure non résolu")
            if trial["physical_violations"]:
                raisons.append("Réserve indisponible physiquement ou matériel incompatible")
            if raisons:
                reserve_refus.append({"service_id": service["service_id"], "gare": gare, "raisons": raisons})
                continue
            trial_state = next(row for row in trial["services"]
                               if row["service_id"] == service["service_id"])
            newly_assigned = set(trial_map) - set(current_map)
            departures = {row["service_id"]: row["depart_reel"] for row in trial["services"]}
            if not trial_state["complete"] or any(
                departures.get(service_id) != next(
                    item["planned_start"] for item in services if item["service_id"] == service_id
                )
                for service_id in newly_assigned
            ):
                reserve_refus.append({
                    "service_id": service["service_id"], "gare": gare,
                    "raisons": ["Une rotation de réserve ne peut pas prendre son départ planifié"],
                })
                continue
            if (trial_energy["erreurs"] or trial_energy["autonomie_insuffisante"]
                    or trial_energy["autonomie_non_verifiable"]):
                raisons_energie = list(trial_energy["erreurs"])
                raisons_energie.extend(
                    alert for alert in trial_energy["alertes_autonomie"]
                    if alert not in raisons_energie
                )
                raisons_energie.extend(
                    alert for alert in trial_energy["alertes_autonomie_non_verifiable"]
                    if alert not in raisons_energie
                )
                reserve_refus.append({
                    "service_id": service["service_id"], "gare": gare,
                    "raisons": raisons_energie,
                })
                continue
            gain = current_replay["retard_cumule_min"] - trial["retard_cumule_min"]
            if gain <= _EPSILON_MIN:
                reserve_refus.append({
                    "service_id": service["service_id"], "gare": gare,
                    "raisons": ["Aucun gain strict sur les retards cumulés"],
                })
                continue
            candidate = (
                gain, gare, trial, trial_map, trial_energy,
                trial_recharges, trial_return_normal,
            )
            if best is None or candidate[0] > best[0] + _EPSILON_MIN:
                best = candidate
        if best is not None:
            (gain, _, trial_replay, trial_map, trial_energy,
             trial_recharges, trial_return_normal) = best
            after = next(row for row in trial_replay["services"]
                         if row["service_id"] == service["service_id"])
            before_suppression_penalty = next(
                (item["penalite_min"] for item in current_replay["suppressions"]
                 if item["service_id"] == service["service_id"]),
                0.0,
            )
            after_suppression_penalty = next(
                (item["penalite_min"] for item in trial_replay["suppressions"]
                 if item["service_id"] == service["service_id"]),
                0.0,
            )
            before_delay = (state["retard_arrivee_min"] or 0.0) + before_suppression_penalty
            after_delay = (after["retard_arrivee_min"] or 0.0) + after_suppression_penalty
            activation_gains[service["service_id"]] = {
                "gain_cumule_min": round(gain, 6),
                "retard_evite_min": round(max(0.0, before_delay - after_delay), 6),
            }
            current_replay = trial_replay
            current_map = trial_map
            current_energy = trial_energy
            current_recharges = trial_recharges
            current_return_normal_sans_recharge = trial_return_normal

    formatted = _format_result(
        current_replay, services, incident, delay_min, current_map,
        baseline_energy, df_gares, energy_params, current_energy,
        current_recharges, current_return_normal_sans_recharge,
    )
    formatted["reserve_refus"] = reserve_refus
    for activation in formatted["reserve_activations"]:
        activation.update(activation_gains.get(activation["service_id"], {}))
    return formatted


def _resultat_sans_reserve(replay, services, incident, delay_min,
                           baseline_energy, df_gares, energy_params, energy=None,
                           recharges_prolongees=(),
                           retour_normal_sans_recharge_min=_UNFOURNI):
    return _format_result(
        replay, services, incident, delay_min, {},
        baseline_energy, df_gares, energy_params, energy,
        recharges_prolongees, retour_normal_sans_recharge_min,
    )


def evaluer_robustesse(chronologie, df_gares, missions, energy_params=None,
                       reserve_gares=()):
    """Évalue 5/10/20 min de retard sur le graphe fourni, sans le réoptimiser.

    Chaque scénario applique un incident unique au départ d'une mission ou à
    l'arrivée sur un point de croisement. Le point retenu maximise la somme des
    retards d'arrivée, comptée une seule fois par mission complète.
    """
    services, gares, infra_map = _compiler_services(chronologie or {}, df_gares, missions or [])
    if not services:
        return {"scenarios": [
            {"delay_min": value, "point": None,
             "sans_reserve": {"chronologie": {}, "retard_cumule_min": 0.0,
                              "retard_trains_min": 0.0,
                              "penalite_suppressions_min": 0.0,
                              "suppressions_count": 0, "suppressions": [],
                              "retour_normal_min": None, "reserve_activations": [],
                              "retour_normal_sans_recharge_min": None,
                              "recharges_prolongees": [],
                              "battery": {"baseline": {}, "scenario": {},
                                          "delta_conso_nette_kwh": 0.0,
                                          "delta_kwh": 0.0,
                                          "min_soc_pct": None, "erreurs": [],
                                          "soc_min_pct": None, "erreurs_reserve": [],
                                          "autonomie_insuffisante": False,
                                          "alertes_autonomie": []},
                              "reserve_refus": []},
             "avec_reserve": None}
            for value in AMPLITUDES_RETARD_MIN
        ], "candidates_count": 0}

    points = _enumerer_points(services, infra_map)
    nominal = _rejouer(services, gares)
    if (nominal["infra_violations"] or nominal["physical_violations"]
            or nominal["suppressions"]):
        raise ValueError("La chronologie nominale contient un conflit ou un déplacement impossible")
    if nominal["retard_cumule_min"] > _EPSILON_MIN:
        raise ValueError("La chronologie nominale ne respecte pas ses horaires de référence")
    if any(
        not row["complete"] or row["depart_reel"] is None
        or abs(_minutes(row["depart_reel"] - row["depart_planifie"])) > _EPSILON_MIN
        for row in nominal["services"]
    ):
        raise ValueError("La chronologie nominale comporte un départ retardé")

    baseline_energy = _analyser_energie(chronologie, services, gares, energy_params)
    stations = []
    for gare in reserve_gares or ():
        if gare not in infra_map:
            raise ValueError(f"Gare de réserve absente de df_gares : {gare!r}")
        if gare not in stations:
            stations.append(gare)

    energy_cache = {
        _CLE_CHRONOLOGIE_NOMINALE: _cle_chronologie_energie(chronologie)
    }
    scenarios = []
    for amplitude in AMPLITUDES_RETARD_MIN:
        best_point = None
        best_replay = None
        best_energy = None
        best_recharges = []
        best_return_normal = None
        best_score = -math.inf
        for point in points:
            candidate_replay = _rejouer(services, gares, point, amplitude)
            (candidate_replay, candidate_energy, candidate_recharges,
             candidate_return_normal) = _corriger_retournements(
                candidate_replay, services, gares, point, amplitude,
                baseline_energy, energy_params, energy_cache=energy_cache,
            )
            score = candidate_replay["retard_cumule_min"]
            if score > best_score + _EPSILON_MIN:
                best_score = score
                best_point = point
                best_replay = candidate_replay
                best_energy = candidate_energy
                best_recharges = candidate_recharges
                best_return_normal = candidate_return_normal
        if best_point is None:
            scenarios.append({"delay_min": amplitude, "point": None,
                              "sans_reserve": None, "avec_reserve": None})
            continue
        sans_reserve = _resultat_sans_reserve(
            best_replay, services, best_point, amplitude,
            baseline_energy, gares, energy_params, best_energy,
            best_recharges, best_return_normal,
        )
        avec_reserve = _evaluuer_avec_reserves(
            services, gares, best_point, amplitude, stations,
            best_replay, baseline_energy, energy_params,
            best_energy, best_recharges, best_return_normal, energy_cache,
        )
        public_point = {key: value for key, value in best_point.items() if key != "block_index"}
        scenarios.append({
            "delay_min": amplitude,
            "point": public_point,
            "sans_reserve": sans_reserve,
            "avec_reserve": avec_reserve,
        })
    return {"scenarios": scenarios, "candidates_count": len(points)}


def fingerprint_robustesse(chronologie, df_gares, missions, energy_params=None,
                           reserve_gares=()):
    """Empreinte SHA-256 stable des entrées de l'analyse de robustesse."""
    skeleton = []
    for train_id, rows in sorted((chronologie or {}).items(), key=lambda item: str(item[0])):
        for row in sorted(rows or [], key=lambda item: (str(item.get("start")), str(item.get("end")))):
            skeleton.append({
                "train_id": str(train_id),
                "start": row.get("start"),
                "end": row.get("end"),
                "origine": row.get("origine"),
                "terminus": row.get("terminus"),
                "mission": row.get("mission"),
                "is_mission_start": row.get("is_mission_start", False),
            })
    gares = []
    if hasattr(df_gares, "columns"):
        cols = [name for name in (
            "gare", "distance", "infra", "rampe", "rampe_section_a_venir", "electrification"
        ) if name in df_gares.columns]
        if cols:
            gares = df_gares[cols].sort_values("gare").to_dict(orient="records")
    payload = {
        "chronologie": skeleton,
        "gares": gares,
        "missions": missions or [],
        "energy_params": energy_params,
        "reserve_gares": list(reserve_gares or ()),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
