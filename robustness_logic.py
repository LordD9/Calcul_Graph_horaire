# -*- coding: utf-8 -*-
"""Analyse de robustesse d'une chronologie sans relancer l'optimisation."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from datetime import datetime, time, timedelta

import pandas as pd

from core_logic import SimulationEngine, _get_infra_at_gare, _is_crossing_point
from energy_logic import calculer_consommation_trajet, get_default_energy_params


AMPLITUDES_RETARD_MIN = (5, 10, 20)
_EPSILON_MIN = 1e-7


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
            if not travel or travel[0]["origine"] != origine or travel[-1]["terminus"] != terminus:
                raise ValueError(f"Mission incomplète ou label incohérent : {label!r}")
            if any(row["origine"] not in gares_map or row["terminus"] not in gares_map for row in rows):
                raise ValueError(f"Gare absente de df_gares dans {label!r}")
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
             reserve_stations=()):
    """Rejoue les blocs fixes dans une file dynamique, sans ajouter de missions."""
    reserve_map = dict(reserve_map or {})
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

    for unit_key in units:
        schedule_next_operation(unit_key)

    while event_heap:
        event_time, _, event_kind, event = heapq.heappop(event_heap)
        service = services_by_id[event["service_id"]]
        unit_key = event["unit_key"]
        unit = units[unit_key]

        if event_kind == "depart_service":
            if unit["location"] != service["origine"]:
                physical_violations.append(
                    f"{unit['display_id']} à {unit['location']!r}, départ prévu à {service['origine']!r} "
                    f"({service['service_id']})"
                )
                actual[service["service_id"]] = {
                    "departure": None, "arrival": None, "complete": False,
                    "unit_key": unit_key,
                }
                next_operation[unit_key] += 1
                schedule_next_operation(unit_key)
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
            if not _point_croisement(engine.infra_map.get(block["start_station"], "F")):
                infra_violations.append({
                    "service_id": service["service_id"],
                    "train_id": service["train_id"],
                    "gare": block["start_station"],
                    "time": earliest,
                    "reason": "Conflit sur un bloc sans possibilité d'attente au départ",
                })
            elif event["attempts"] < 500 and next_available is not None:
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
                infra_violations.append({
                    "service_id": service["service_id"],
                    "train_id": service["train_id"],
                    "gare": block["start_station"],
                    "time": earliest,
                    "reason": "Aucun créneau libre trouvé après 500 tentatives",
                })

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
            "path": [
                {"gare": block["start_station"], "index": index_start,
                 "arr": earliest, "dep": earliest},
                {"gare": block["end_station"], "index": index_end,
                 "arr": arrival, "dep": release},
            ],
        })
        state["blocks"][block_index] = {"departure": earliest, "arrival": arrival, "release": release}

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
            turnaround = max(5.0, turnaround)
        unit["available_at"] = release + timedelta(minutes=turnaround)
        next_operation[unit_key] += 1
        schedule_next_operation(unit_key)

    for unit_key, unit in units.items():
        output[unit["display_id"]].sort(
            key=lambda row: (row["start"], row["end"], row.get("service_id", ""))
        )

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
        })

    return {
        "chronologie": {key: rows for key, rows in output.items() if rows},
        "retard_cumule_min": round(total_delay, 6),
        "services": metric_rows,
        "infra_violations": infra_violations,
        "physical_violations": physical_violations,
        "actual": actual,
        "units": units,
    }


def _parametres_materiel(energy_params, material):
    if energy_params is None:
        return get_default_energy_params(), False
    if isinstance(energy_params, dict) and "masse_tonne" in energy_params:
        return energy_params, False
    value = (energy_params or {}).get(material)
    if not value:
        return get_default_energy_params(), True
    return value, False


def _analyser_energie(chronologie, services, df_gares, energy_params):
    """Calcule les bilans physiques sur la chronologie nominale ou rejouée."""
    service_by_id = {service["service_id"]: service for service in services}
    service_by_source = {}
    for service in services:
        service_by_source.setdefault(service["train_id"], service)
    trains = {}
    errors = []
    minimum_soc = []
    reserve_errors = []
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
        if params_missing:
            missing_message = f"Train {train_id}: paramètres absents pour {service['material']}"
            errors.append(missing_message)
            if str(train_id).startswith("Réserve "):
                reserve_errors.append(missing_message)
        try:
            result = calculer_consommation_trajet(rows, service["mission_config"], df_gares, params)
        except Exception as exc:  # une erreur énergétique est rapportée, pas masquée
            message = f"Train {train_id}: calcul énergétique impossible ({exc})"
            errors.append(message)
            if str(train_id).startswith("Réserve "):
                reserve_errors.append(message)
            continue
        train_errors = [str(error) for error in (result.get("erreurs") or [])]
        errors.extend(f"Train {train_id}: {error}" for error in train_errors)
        if str(train_id).startswith("Réserve "):
            reserve_errors.extend(f"Train {train_id}: {error}" for error in train_errors)
        capacity = float(params.get("capacite_batterie_kwh", 0) or 0)
        if params.get("simuler_fin_de_vie"):
            capacity *= float(params.get("capacite_eol_pct", 80)) / 100.0
        battery_log = result.get("batterie_log") or []
        soc_values = [float(entry[1]) for entry in battery_log if len(entry) > 1]
        initial_soc = soc_values[0] / capacity * 100.0 if capacity > 0 and soc_values else None
        min_train_soc = min(soc_values) / capacity * 100.0 if capacity > 0 and soc_values else None
        final_soc = soc_values[-1] / capacity * 100.0 if capacity > 0 and soc_values else None
        margin_kwh = None
        margin_soc_pct = None
        if service["material"] == "batterie" and capacity > 0 and min_train_soc is not None:
            threshold_pct = float(params.get("soc_min_pct", 20))
            threshold_kwh = capacity * threshold_pct / 100.0
            # Marge au point de SoC le plus bas, au-dessus du seuil d'exploitation.
            margin_kwh = max(0.0, min(soc_values) - threshold_kwh)
            margin_soc_pct = max(0.0, min_train_soc - threshold_pct)
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
            "marge_kwh": margin_kwh,
            "marge_soc_pct": margin_soc_pct,
            "energie_utilisable_au_dessus_seuil_kwh": margin_kwh,
            "erreurs": train_errors,
        }
    return {
        "consommation_nette_kwh": round(sum(row["consommation_nette_kwh"] for row in trains.values()), 6),
        "min_soc_pct": min(minimum_soc) if minimum_soc else None,
        "par_train": trains,
        "erreurs": errors,
        "erreurs_reserve": reserve_errors,
    }


def _retour_normal(services, metrics, incident, delay_min, infra_violations=()):
    if infra_violations:
        return None
    by_id = {row["service_id"]: row for row in metrics}
    delayed = [row for row in metrics
               if row["complete"] and (row["retard_arrivee_min"] or 0.0) > _EPSILON_MIN]
    event_time = incident["time"] + timedelta(minutes=delay_min)
    if delayed:
        last_late_arrival = max(row["arrivee_reelle"] for row in delayed)
        event_time = max(event_time, last_late_arrival)
    else:
        last_late_arrival = incident["time"]

    proofs = []
    for service in services:
        row = by_id[service["service_id"]]
        if not row["complete"] or row["depart_reel"] is None:
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
        later = [row for row in metrics
                 if row["complete"] and row["depart_planifie"] >= proof_service["planned_start"]]
        if all(
            row["depart_reel"] is not None
            and abs(_minutes(row["depart_reel"] - row["depart_planifie"])) <= _EPSILON_MIN
            and (row["retard_arrivee_min"] or 0.0) <= _EPSILON_MIN
            for row in later
        ):
            return max(0.0, _minutes(proof_service["planned_start"] - incident["time"]))
    return None


def _format_result(replay, services, incident, delay_min, reserve_map,
                   baseline_energy, df_gares, energy_params):
    energy = _analyser_energie(replay["chronologie"], services, df_gares, energy_params)
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
    battery = {
        "baseline": baseline_energy,
        "scenario": energy,
        "delta_conso_nette_kwh": round(
            energy["consommation_nette_kwh"] - baseline_energy["consommation_nette_kwh"], 6
        ),
        "delta_kwh": round(
            energy["consommation_nette_kwh"] - baseline_energy["consommation_nette_kwh"], 6
        ),
        "min_soc_pct": energy["min_soc_pct"],
        "soc_min_pct": energy["min_soc_pct"],
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
        "retour_normal_min": _retour_normal(
            services, replay["services"], incident, delay_min, replay["infra_violations"]
        ),
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
                            no_reserve, baseline_energy, energy_params):
    current_map = {}
    current_replay = no_reserve
    reserve_refus = []
    activation_gains = {}
    if not stations:
        return None

    sorted_services = sorted(services, key=lambda service: (service["planned_start"], service["service_id"]))
    for service in sorted_services:
        if service["service_id"] == incident["service_id"] or service["service_id"] in current_map:
            continue
        if service["planned_start"] < incident["time"]:
            continue
        state = next((row for row in current_replay["services"]
                      if row["service_id"] == service["service_id"]), None)
        if not state or not state["complete"]:
            continue
        if (state["retard_arrivee_min"] or 0.0) <= _EPSILON_MIN and (
            state["depart_reel"] is None
            or abs(_minutes(state["depart_reel"] - state["depart_planifie"])) <= _EPSILON_MIN
        ):
            continue

        best = None
        for gare in stations:
            if service["service_id"] in current_map:
                continue
            trial_map = _etendre_rotation_reserve(services, current_map, service, gare)
            trial = _rejouer(services, df_gares, incident, delay_min, trial_map, stations)
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
            trial_energy = _analyser_energie(trial["chronologie"], services, df_gares, energy_params)
            if trial_energy["erreurs"]:
                reserve_refus.append({
                    "service_id": service["service_id"], "gare": gare,
                    "raisons": list(trial_energy["erreurs"]),
                })
                continue
            gain = current_replay["retard_cumule_min"] - trial["retard_cumule_min"]
            if gain <= _EPSILON_MIN:
                reserve_refus.append({
                    "service_id": service["service_id"], "gare": gare,
                    "raisons": ["Aucun gain strict sur les retards cumulés"],
                })
                continue
            candidate = (gain, gare, trial, trial_map, trial_energy)
            if best is None or candidate[0] > best[0] + _EPSILON_MIN:
                best = candidate
        if best is not None:
            gain, _, trial_replay, trial_map, _ = best
            after = next(row for row in trial_replay["services"]
                         if row["service_id"] == service["service_id"])
            before_delay = state["retard_arrivee_min"] or 0.0
            after_delay = after["retard_arrivee_min"] or 0.0
            activation_gains[service["service_id"]] = {
                "gain_cumule_min": round(gain, 6),
                "retard_evite_min": round(max(0.0, before_delay - after_delay), 6),
            }
            current_replay = trial_replay
            current_map = trial_map

    formatted = _format_result(
        current_replay, services, incident, delay_min, current_map,
        baseline_energy, df_gares, energy_params,
    )
    formatted["reserve_refus"] = reserve_refus
    for activation in formatted["reserve_activations"]:
        activation.update(activation_gains.get(activation["service_id"], {}))
    return formatted


def _resultat_sans_reserve(replay, services, incident, delay_min,
                           baseline_energy, df_gares, energy_params):
    return _format_result(
        replay, services, incident, delay_min, {},
        baseline_energy, df_gares, energy_params,
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
                              "retour_normal_min": None, "reserve_activations": [],
                              "battery": {"baseline": {}, "scenario": {},
                                          "delta_conso_nette_kwh": 0.0,
                                          "delta_kwh": 0.0,
                                          "min_soc_pct": None, "erreurs": [],
                                          "soc_min_pct": None, "erreurs_reserve": []},
                              "reserve_refus": []},
             "avec_reserve": None}
            for value in AMPLITUDES_RETARD_MIN
        ], "candidates_count": 0}

    points = _enumerer_points(services, infra_map)
    nominal = _rejouer(services, gares)
    if nominal["infra_violations"] or nominal["physical_violations"]:
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

    scenarios = []
    for amplitude in AMPLITUDES_RETARD_MIN:
        best_point = None
        best_replay = None
        best_score = -math.inf
        for point in points:
            candidate_replay = _rejouer(services, gares, point, amplitude)
            score = candidate_replay["retard_cumule_min"]
            if score > best_score + _EPSILON_MIN:
                best_score = score
                best_point = point
                best_replay = candidate_replay
        if best_point is None:
            scenarios.append({"delay_min": amplitude, "point": None,
                              "sans_reserve": None, "avec_reserve": None})
            continue
        # Rejouer le pire point pour produire les chronologies, bilans et diagnostics.
        replay = _rejouer(services, gares, best_point, amplitude)
        sans_reserve = _resultat_sans_reserve(
            replay, services, best_point, amplitude,
            baseline_energy, gares, energy_params,
        )
        avec_reserve = _evaluuer_avec_reserves(
            services, gares, best_point, amplitude, stations,
            replay, baseline_energy, energy_params,
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
