# -*- coding: utf-8 -*-
"""Chronomètre l'optimisation réelle d'un scénario, mode par mode.

python outils/bench_optimisation.py [scenario.json] [modes...] > resultats.json
"""
import json
import random
import sys
import time as chrono
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_logic import reset_caches
from optimisation_logic import CrossingOptimization, OptimizationConfig, optimiser_graphique_horaire
from scenarios_manager import load_scenario

DEFAUT = "scenarios/Nouvelle-Aquitaine/Bayonne-Saint-Jean-Pied-de-Port/bayonne_sjpp_2032_croisement_itxassou.json"


def _heure(texte):
    h, m = texte.split(":")
    return time(int(h), int(m))


def mesurer(chemin, mode, graine=0):
    scenario = load_scenario(chemin)
    gares = pd.DataFrame(scenario["infrastructure"]["gares"])
    service = scenario["service"]
    random.seed(graine)
    np.random.seed(graine)
    reset_caches()
    config = OptimizationConfig(mode=mode, crossing_optimization=CrossingOptimization(enabled=True))
    debut = chrono.perf_counter()
    chronologie, warnings, stats = optimiser_graphique_horaire(
        scenario["missions"], gares, _heure(service["heure_debut"]), _heure(service["heure_fin"]), config)
    duree = chrono.perf_counter() - debut
    return {
        "mode": mode, "secondes": round(duree, 2), "score": stats.get("best_score"),
        "rames": len(chronologie), "violations_infra": len(warnings.get("infra_violations", [])),
        "segments": sum(len(v) for v in chronologie.values()),
    }


if __name__ == "__main__":
    chemin = (sys.argv[1] if len(sys.argv) > 1 else "") or DEFAUT
    modes = sys.argv[2:] or ["simple", "fast", "smart_progressive", "exhaustif", "genetic"]
    print(json.dumps([mesurer(chemin, m) for m in modes], indent=1))
