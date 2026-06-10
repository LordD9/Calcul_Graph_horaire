# -*- coding: utf-8 -*-
"""
optimisation_logic.py
=====================

Module d'optimisation avancée des horaires ferroviaires.

Ce module implémente des algorithmes pour trouver la meilleure grille horaire respectant les contraintes :
- Minimisation du nombre de rames nécessaires.
- Maximisation de la régularité (cadencement).
- Gestion stricte des croisements sur voie unique.

Algorithmes disponibles :
- **Algorithme Génétique (`GeneticOptimizer`)** : Recherche heuristique parallèle pour explorer l'espace des solutions.
- **Recherche Exhaustive** : Pour les petits problèmes.
- **Stratégies Progressives** : Affinement successif du pas de temps.

Classes Principales :
- `GeneticOptimizer` : Cœur de l'optimisation génétique.
- `OptimizationConfig` : Paramètres de configuration (taille population, mutations, etc.).
- `SolutionScorer` : Fonction de coût évaluant la qualité d'une grille.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import hashlib
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CrossingOptimization:
    enabled: bool = False
    max_delay_minutes: int = 15
    penalty_per_minute: float = 2.0

@dataclass
class CrossingStrategy:
    mission_id: str
    stop_durations: Dict[str, int]
    priority: float
    max_acceptable_delay: int

    def to_dict(self):
        return {
            'mission_id': self.mission_id,
            'stop_durations': self.stop_durations,
            'priority': self.priority,
            'max_acceptable_delay': self.max_acceptable_delay
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            mission_id=data['mission_id'],
            stop_durations=data['stop_durations'],
            priority=data['priority'],
            max_acceptable_delay=data['max_acceptable_delay']
        )

@dataclass
class SimulationParams:
    cadencements: Dict[str, int]
    turnaround_buffers: Dict[str, int]
    crossing_stop_durations: Dict[str, Dict[str, int]]
    crossing_pair_assignments: Dict[str, List[str]] = None  # mission_id → preferred VE list
    retour_offsets: Dict[str, int] = None  # mission_id → minute de référence du départ retour

    def get_turnaround_buffers(self, missions):
        # Le buffer exploré par les optimiseurs s'applique au terminus A (origine).
        # Le timing du retour (terminus B) est géré par retour_offsets (cadencé),
        # plus régulier qu'un buffer libre. D'où la forme directionnelle {"A","B"}.
        result = {}
        for i, m in enumerate(missions):
            mid = f"M{i+1}"
            buffer = self.turnaround_buffers.get(mid, self.turnaround_buffers.get(str(i), 0))
            if buffer != 0:
                result[mid] = {"A": buffer, "B": 0}
        return result

    def get_retour_offsets(self, missions):
        """{mission_id: int} pour les missions ayant un offset retour défini."""
        if not self.retour_offsets:
            return {}
        result = {}
        for i, m in enumerate(missions):
            mid = f"M{i+1}"
            off = self.retour_offsets.get(mid, self.retour_offsets.get(str(i)))
            if off is not None:
                result[mid] = off
        return result

    def get_crossing_strategies(self, missions, df_gares):
        from core_logic import _get_infra_at_gare
        result = {}
        for i, m in enumerate(missions):
            mid = f"M{i+1}"
            key = mid if mid in self.crossing_stop_durations else str(i)
            stop_durations = self.crossing_stop_durations.get(key, {})
            if stop_durations:
                result[mid] = CrossingStrategy(
                    mission_id=mid,
                    stop_durations=stop_durations,
                    priority=0.5,
                    max_acceptable_delay=15,
                )
        return result

    def get_adjusted_reference_minutes(self, missions):
        """Décale CHAQUE minute du pattern d'origine par l'offset de cadencement.

        ``cadencements[mid]`` est un OFFSET (0-59), pas une minute absolue : offset 0
        reproduit le pattern d'origine, et les patterns multi-valeurs ("0,30") sont
        préservés (→ "5,35" pour offset 5). Cohérent avec _seed_crossing_pairs.
        """
        result = {}
        for i, m in enumerate(missions):
            mid = f"M{i+1}"
            offset = self.cadencements.get(mid, self.cadencements.get(str(i), None))
            if offset is not None:
                original = [int(x) for x in str(m.get('reference_minutes', '0')).split(',')
                            if x.strip().lstrip('-').isdigit()] or [0]
                result[str(i)] = ",".join(str(v) for v in sorted({(o + offset) % 60 for o in original}))
        return result

@dataclass
class OptimizationConfig:
    """Configuration générale de l'optimisation."""
    mode: str = "smart_progressive"  # Options: "simple", "fast", "smart_progressive", "exhaustif", "genetic"
    crossing_optimization: CrossingOptimization = None
    
    # Paramètres génétiques optimisés
    population_size: int = 50  # Réduit de 100
    generations: int = 100     # Réduit de 150
    mutation_rate: float = 0.20  # Augmenté pour plus d'exploration
    crossover_rate: float = 0.85  # Augmenté légèrement
    elitism_ratio: float = 0.10   # Réduit pour plus de diversité
    early_stop_generations: int = 15  # Réduit de 20
    
    # Nouveaux paramètres d'optimisation
    adaptive_mutation: bool = True  # Mutation adaptative
    tournament_size: int = 3
    use_parallel: bool = True
    num_workers: int = None
    use_cache: bool = True
    timeout_per_eval: int = 60  # Timeout réduit à 20s
    
    # NOUVEAU : Optimisation des temps de retournement
    optimize_turnaround: bool = False  # Activer l'optimisation des temps de retournement
    turnaround_min_buffer: int = 0     # Minutes à ajouter au minimum utilisateur (par défaut : utiliser le minimum)
    turnaround_max_buffer: int = 30    # Maximum de minutes supplémentaires autorisées
    
    def __post_init__(self):
        if self.crossing_optimization is None:
            self.crossing_optimization = CrossingOptimization()
        if self.num_workers is None:
            self.num_workers = max(1, mp.cpu_count() - 1)


# =============================================================================
# CACHE INTELLIGENT
# =============================================================================

class SolutionCache:
    """Cache optimisé pour les solutions."""
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_key(self, missions, genome=None):
        """Génère une clé de cache compacte."""
        missions_str = json.dumps(sorted([
            (m['origine'], m['terminus'], m.get('frequence', 0))
            for m in missions
        ]))
        if genome:
            timing_key = tuple(sorted(genome['timing'].items()))
            crossing_key = tuple(sorted(
                (k, tuple(sorted(v['stop_durations'].items()))) 
                for k, v in genome.get('crossing', {}).items()
            ))
            key_str = f"{missions_str}:{timing_key}:{crossing_key}"
        else:
            key_str = missions_str
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key):
        result = self.cache.get(key)
        if result:
            self.access_count[key] = self.access_count.get(key, 0) + 1
        return result
    
    def put(self, key, value):
        # Si cache plein, supprimer les entrées les moins utilisées
        if len(self.cache) >= self.max_size:
            min_key = min(self.access_count, key=self.access_count.get)
            del self.cache[min_key]
            del self.access_count[min_key]
        
        self.cache[key] = value
        self.access_count[key] = 0

_solution_cache = SolutionCache()


class GenomeCache:
    """Cache process-master persistant sur toutes les générations d'un run."""
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
            self.cache.pop(evict, None)
            self.access.pop(evict, None)
        k = self.key(genome)
        self.cache[k] = (score, warnings, chronologie)
        self.access[k] = 0


# =============================================================================
# SYSTÈME DE SCORING OPTIMISÉ
# =============================================================================

class SolutionScorer:
    """Validation des solutions de l'algo génétique.

    Le scoring réel passe par ``_score_chronologie_bruit`` (core_logic), unique
    fonction de coût du projet. Cette classe ne conserve que le test de validité
    (absence de violation d'infrastructure) utilisé par l'optimiseur.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def is_valid_solution(self, warnings: Dict) -> bool:
        """Vérifie si solution valide (aucune violation d'infrastructure)."""
        return len(warnings.get("infra_violations", [])) == 0


# =============================================================================
# WORKER PARALLÈLE OPTIMISÉ
# =============================================================================

def _evaluate_genome_worker(args):
    """Worker optimisé — utilise evaluer_params_simulation au lieu de generer_tous_trajets_optimises."""
    import sys
    import io
    
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    
    try:
        genome, missions, df_gares, heure_debut, heure_fin, allow_sharing, config_dict = args
        config = OptimizationConfig(**config_dict)

        cadencements = genome.get('timing', {})
        turnaround_buffers = genome.get('turnaround_buffers', {})
        crossing_stop_durations = genome.get('crossing', {})
        crossing_pair_assignments = genome.get('crossing_pairs', {})
        retour_offsets = genome.get('retour_offsets', {})

        params = SimulationParams(
            cadencements=cadencements,
            turnaround_buffers=turnaround_buffers,
            crossing_stop_durations=crossing_stop_durations,
            crossing_pair_assignments=crossing_pair_assignments,
            retour_offsets=retour_offsets,
        )

        score, chronologie, warnings, stats = evaluer_params_simulation(
            params, missions, df_gares, heure_debut, heure_fin, allow_sharing, config=config
        )

        result = (genome, chronologie, warnings, score)

        if config.use_cache:
            cache_key = _solution_cache.get_key(missions, genome)
            _solution_cache.put(cache_key, result)

        return result

    except Exception as e:
        return (args[0], {}, {"infra_violations": [], "other": [str(e)]}, float('inf'))
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


# =============================================================================
# ALGORITHME GÉNÉTIQUE OPTIMISÉ
# =============================================================================

class GeneticOptimizer:
    """
    Optimiseur basé sur un algorithme génétique pour la planification ferroviaire.
    
    Cet optimiseur explore l'espace des horaires de départ (décalage de cadencement) 
    et des stratégies de croisement (temps d'arrêt aux évitements) pour minimiser 
    la fonction de coût définie par `SolutionScorer`.
    
    Caractéristiques :
    - Mutation adaptative (ajuste le taux selon la stagnation).
    - Évaluation parallèle (multiprocessing).
    - Gestion explicite des stratégies de croisement.
    """
    
    def __init__(self, missions, df_gares, heure_debut, heure_fin, 
                 config: OptimizationConfig, scorer: SolutionScorer, allow_sharing: bool):
        self.missions = missions
        self.df_gares = df_gares
        self.heure_debut = heure_debut
        self.heure_fin = heure_fin
        self.config = config
        self.scorer = scorer
        self.allow_sharing = allow_sharing
        
        self.search_space = self._build_search_space()
        self.crossing_points = self._identify_crossing_points()
        self.best_score_history = []
        self.current_mutation_rate = config.mutation_rate
        self.genome_cache = GenomeCache()
    
    def _identify_crossing_points(self) -> Dict[str, List[str]]:
        """Identifie les gares permettant le croisement pour chaque mission."""
        from core_logic import _get_infra_at_gare
        crossing_points = {}
        
        gares_list = self.df_gares['gare'].tolist()
        for mission in self.missions:
            mission_id = f"{mission['origine']}→{mission['terminus']}"
            points = []
            
            try:
                idx_orig = gares_list.index(mission['origine'])
                idx_term = gares_list.index(mission['terminus'])
                start, end = min(idx_orig, idx_term), max(idx_orig, idx_term)
                
                for idx in range(start, end + 1):
                    gare = gares_list[idx]
                    if _get_infra_at_gare(self.df_gares, gare) == 'VE':
                        points.append(gare)
            except:
                pass
            
            crossing_points[mission_id] = points
        
        return crossing_points
    
    def _build_search_space(self) -> Dict:
        """Construit l'espace de recherche (offsets de cadencement par mission).

        Les clés utilisent le format "M{i+1}" (ex: "M1", "M2") pour être compatibles
        avec SimulationParams.get_adjusted_reference_minutes et get_turnaround_buffers.

        ``default = 0`` car le gène timing est un OFFSET appliqué au pattern
        d'origine de la mission : offset 0 = pattern inchangé (baseline).
        """
        space = {}
        for i, mission in enumerate(self.missions):
            if mission.get('frequence', 0) <= 0:
                continue
            mid = f"M{i+1}"
            space[mid] = {'default': 0, 'range': (0, 59)}
        return space
    
    def _identify_ve_gares(self):
        """Retourne la liste de toutes les gares VE (voies d'évitement)."""
        from core_logic import _get_infra_at_gare
        ve_list = []
        for _, row in self.df_gares.iterrows():
            if _get_infra_at_gare(self.df_gares, row['gare']) == 'VE':
                ve_list.append(row['gare'])
        return ve_list
    
    def optimize(self, progress_callback: Optional[Callable] = None) -> Tuple[Dict, Dict, Dict]:
        """
        Exécute la boucle principale d'optimisation génétique.
        
        Args:
            progress_callback (callable, optional): Fonction pour rapporter l'avancement.

        Returns:
            tuple: (Meilleure solution (génome), Warnings associés, Stats de l'algo).
        """
        population = self._initialize_population()
        best_solution, best_warnings = None, None
        best_score = float('inf')
        generations_without_improvement = 0
        
        for generation in range(self.config.generations):
            # Mutation adaptative
            if self.config.adaptive_mutation and generation > 10:
                if generations_without_improvement > 5:
                    self.current_mutation_rate = min(0.4, self.current_mutation_rate * 1.2)
                else:
                    self.current_mutation_rate = max(0.1, self.current_mutation_rate * 0.95)
            
            # Évaluation parallèle
            evaluated_population = self._evaluate_population_parallel(population)
            valid_solutions = [x for x in evaluated_population if self.scorer.is_valid_solution(x[2])]
            
            if not valid_solutions:
                # Réinitialisation partielle si pas de solutions valides
                new_pop = self._initialize_population()
                population = new_pop[:len(population)//2] + population[len(population)//2:]
                continue
            
            valid_solutions.sort(key=lambda x: x[3])
            current_best = valid_solutions[0]
            
            if current_best[3] < best_score:
                best_score = current_best[3]
                best_solution = current_best[1]
                best_warnings = current_best[2]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            self.best_score_history.append(best_score)
            
            if progress_callback:
                progress_callback(
                    generation + 1, 
                    self.config.generations, 
                    best_score, 
                    len(best_solution) if best_solution else 0, 
                    0
                )
            
            # Arrêt anticipé
            if generations_without_improvement >= self.config.early_stop_generations:
                break
            
            # Nouvelle génération
            population = self._create_next_generation(valid_solutions)
        
        if best_solution is None:
            return {}, {"infra_violations": [], "other": ["Aucune solution valide trouvée"]}, \
                   {'mode': 'genetic', 'error': 'No valid solution found'}
        
        return best_solution, best_warnings, {
            'mode': 'genetic', 
            'generations': generation + 1, 
            'final_score': best_score,
            'population_size': self.config.population_size, 
            'best_score_history': self.best_score_history
        }
    
    def _initialize_population(self) -> List[Dict]:
        """Initialise la population avec cadencement, turnaround et croisement.

        Le premier génome représente la baseline utilisateur (équivalent mode simple),
        garantissant que l'algo génétique n'est jamais pire que la config d'entrée.
        """
        population = []

        # Plage de buffers adaptée à l'optim des croisements.
        crossing_enabled = self.config.crossing_optimization and self.config.crossing_optimization.enabled
        if crossing_enabled:
            max_buf = max(15, getattr(self.config, 'turnaround_max_buffer', 30))
            buf_choices = [0, 0, 0, 3, 5, 8, 10, 15, max_buf // 2, max_buf]
            max_delay = max(5, self.config.crossing_optimization.max_delay_minutes)
            cross_wide = [0, 2, 3, 5] + list(range(7, max_delay + 1, 2))
        else:
            buf_choices = [0, 0, 0, 3, 5, 8, 10]
            cross_wide = [0, 2, 3, 5]

        # Génome 0 = baseline utilisateur : timing/turnaround/crossing vides + retour
        # ASAP (offset None) → core_logic reproduit exactement le mode simple.
        baseline_genome = {'timing': {}, 'turnaround_buffers': {}, 'crossing': {},
                           'crossing_pairs': {}, 'retour_offsets': {}}
        for mission_id in self.search_space.keys():
            baseline_genome['turnaround_buffers'][mission_id] = 0
            baseline_genome['retour_offsets'][mission_id] = None
        population.append(baseline_genome)

        for i in range(1, self.config.population_size):
            genome = {'timing': {}, 'turnaround_buffers': {}, 'crossing': {},
                      'crossing_pairs': {}, 'retour_offsets': {}}

            for mission_id, info in self.search_space.items():
                if i < self.config.population_size // 4:
                    genome['timing'][mission_id] = info['default']
                elif i < self.config.population_size // 2:
                    genome['timing'][mission_id] = random.randrange(0, 60, 5)
                else:
                    genome['timing'][mission_id] = random.randint(0, 59)

                if i < self.config.population_size // 4:
                    genome['turnaround_buffers'][mission_id] = 0
                else:
                    genome['turnaround_buffers'][mission_id] = random.choice(buf_choices)

                # Offset retour : ~1/3 ASAP (None), 1/3 sur grille de 5 min, 1/3 libre.
                r = random.random()
                if i < self.config.population_size // 4 or r < 0.34:
                    genome['retour_offsets'][mission_id] = None
                elif r < 0.67:
                    genome['retour_offsets'][mission_id] = random.randrange(0, 60, 5)
                else:
                    genome['retour_offsets'][mission_id] = random.randint(0, 59)

            ve_list = self._identify_ve_gares()
            if ve_list and random.random() < 0.6:
                for j, mission in enumerate(self.missions):
                    if mission.get('frequence', 0) <= 0:
                        continue
                    mid = f"M{j+1}"
                    if random.random() < 0.7:
                        strategy_type = (i + j) % 3
                        if strategy_type == 0:
                            stop_durations = {ve: random.choices([0, 0, 2], weights=[0.6, 0.3, 0.1])[0] for ve in ve_list}
                        elif strategy_type == 1:
                            stop_durations = {ve: random.choice(cross_wide) for ve in ve_list}
                        else:
                            stop_durations = {ve: 0 for ve in ve_list}
                        genome['crossing'][mid] = stop_durations

            if crossing_enabled:
                self._seed_crossing_pairs(genome)

            population.append(genome)
        return population

    def _seed_crossing_pairs(self, genome):
        """Peuple crossing_pairs à partir des rencontres aller×retour idéalisées."""
        from core_logic import enumerer_rencontres

        crossing_pairs = {}
        for j, mission in enumerate(self.missions):
            if mission.get('frequence', 0) <= 0:
                continue
            mid = f"M{j+1}"
            timing_offset = genome.get('timing', {}).get(mid, 0)
            turnaround_buf = genome.get('turnaround_buffers', {}).get(mid, 0)
            retour_offset = genome.get('retour_offsets', {}).get(mid)

            ref_str = mission.get("reference_minutes", "0")
            try:
                original_minutes = [int(m.strip()) for m in ref_str.split(',') if m.strip().isdigit()]
                adjusted_ref_str = ",".join(str((m + timing_offset) % 60) for m in original_minutes)
            except Exception:
                adjusted_ref_str = ref_str

            try:
                rencontres = enumerer_rencontres(
                    mission, self.df_gares,
                    self.heure_debut, self.heure_fin,
                    adjusted_ref_str, mission['frequence'],
                    turnaround_buffer=turnaround_buf,
                    retour_offset=retour_offset,
                )
            except Exception:
                rencontres = []

            if not rencontres:
                continue

            # Aggréger les VE naturelles et candidates
            ve_counts = {}
            candidate_ves = set()
            for r in rencontres:
                ve = r['natural_ve']
                ve_counts[ve] = ve_counts.get(ve, 0) + 1
                for c in r['candidate_ve']:
                    candidate_ves.add(c)

            if not ve_counts:
                continue

            sorted_ves = sorted(ve_counts.keys(), key=lambda v: ve_counts[v], reverse=True)
            preferred = []
            if sorted_ves:
                # 70 % vers VE la plus naturelle, 30 % vers une candidate alternative
                if random.random() < 0.7:
                    preferred.append(sorted_ves[0])
                elif candidate_ves:
                    preferred.append(random.choice(list(candidate_ves)))
            if len(sorted_ves) > 1 and random.random() < 0.3:
                preferred.append(sorted_ves[1])

            crossing_pairs[mid] = list(set(preferred))

        genome['crossing_pairs'] = crossing_pairs
    
    def _evaluate_population_parallel(self, population: List[Dict]) -> List[Tuple]:
        """Évaluation parallèle optimisée."""
        config_dict = {
            'mode': self.config.mode,
            'population_size': self.config.population_size,
            'generations': self.config.generations,
            'mutation_rate': self.current_mutation_rate,
            'crossover_rate': self.config.crossover_rate,
            'elitism_ratio': self.config.elitism_ratio,
            'use_parallel': False,
            'num_workers': 1,
            'use_cache': self.config.use_cache,
            'timeout_per_eval': self.config.timeout_per_eval,
            # Propager l'optim. des croisements aux workers : sans ça, le worker
            # recrée un CrossingOptimization désactivé et le plafond d'arrêt
            # retombe à 5 min au lieu du max_delay utilisateur (pénalité faussée).
            'crossing_optimization': self.config.crossing_optimization,
            'turnaround_max_buffer': self.config.turnaround_max_buffer,
        }
        
        # Séparer les génomes déjà dans le cache des autres
        cached_results = []
        to_evaluate = []
        for genome in population:
            hit = self.genome_cache.get(genome)
            if hit is not None:
                score, warnings, chronologie = hit
                cached_results.append((genome, chronologie, warnings, score))
            else:
                to_evaluate.append(genome)

        args_list = [
            (genome, self.missions, self.df_gares, self.heure_debut,
             self.heure_fin, self.allow_sharing, config_dict)
            for genome in to_evaluate
        ]

        new_results = []
        if args_list:
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = {executor.submit(_evaluate_genome_worker, args): args for args in args_list}

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.config.timeout_per_eval)
                        new_results.append(result)
                        genome_res, chron, warn, score = result
                        if score < float('inf'):
                            self.genome_cache.put(genome_res, score, warn, chron)
                    except TimeoutError:
                        new_results.append((futures[future][0], {},
                                           {"infra_violations": [], "other": ["Timeout"]},
                                           float('inf')))
                    except Exception as e:
                        new_results.append((futures[future][0], {},
                                           {"infra_violations": [], "other": [str(e)]},
                                           float('inf')))

        return cached_results + new_results
    
    def _create_next_generation(self, valid_solutions: List[Tuple]) -> List[Dict]:
        """Crée la nouvelle génération avec opérateurs génétiques optimisés."""
        new_population = []
        num_elite = int(self.config.population_size * self.config.elitism_ratio)
        
        # Élitisme
        for i in range(num_elite):
            new_population.append(deepcopy(valid_solutions[i][0]))
        
        # Génération du reste
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate and len(valid_solutions) >= 2:
                parent1 = self._tournament_selection(valid_solutions)
                parent2 = self._tournament_selection(valid_solutions)
                child = self._crossover(parent1, parent2)
            else:
                child = deepcopy(self._tournament_selection(valid_solutions))
            
            if random.random() < self.current_mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, valid_solutions: List[Tuple]) -> Dict:
        """Sélection par tournoi."""
        tournament = random.sample(valid_solutions, 
                                  min(self.config.tournament_size, len(valid_solutions)))
        winner = min(tournament, key=lambda x: x[3])
        return deepcopy(winner[0])
    
    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        """Croisement à deux points — inclut turnaround_buffers et crossing_pairs."""
        child = {'timing': {}, 'turnaround_buffers': {}, 'crossing': {},
                 'crossing_pairs': {}, 'retour_offsets': {}}

        # Timing — union des clés des deux parents pour éviter KeyError sur le génome baseline
        all_timing_keys = set(p1.get('timing', {}).keys()) | set(p2.get('timing', {}).keys())
        mission_ids = sorted(all_timing_keys)
        space_default = self.search_space  # {mid: {'default': x}}
        if len(mission_ids) > 2:
            point1 = random.randint(0, len(mission_ids) - 1)
            point2 = random.randint(point1, len(mission_ids) - 1)
            for i, mid in enumerate(mission_ids):
                fallback = space_default.get(mid, {}).get('default', 0)
                v1 = p1.get('timing', {}).get(mid, fallback)
                v2 = p2.get('timing', {}).get(mid, fallback)
                child['timing'][mid] = v2 if point1 <= i <= point2 else v1
        else:
            for mid in mission_ids:
                fallback = space_default.get(mid, {}).get('default', 0)
                v1 = p1.get('timing', {}).get(mid, fallback)
                v2 = p2.get('timing', {}).get(mid, fallback)
                child['timing'][mid] = random.choice([v1, v2])

        # Turnaround buffers
        for mid in mission_ids:
            buf1 = p1.get('turnaround_buffers', {}).get(mid, 0)
            buf2 = p2.get('turnaround_buffers', {}).get(mid, 0)
            child['turnaround_buffers'][mid] = random.choice([buf1, buf2])

        # Crossing strategies
        all_crossing_keys = set(p1.get('crossing', {}).keys()) | set(p2.get('crossing', {}).keys())
        for key in all_crossing_keys:
            if key in p1.get('crossing', {}) and key in p2.get('crossing', {}):
                child['crossing'][key] = random.choice([p1['crossing'][key], p2['crossing'][key]])
            elif key in p1.get('crossing', {}):
                child['crossing'][key] = p1['crossing'][key]
            elif key in p2.get('crossing', {}):
                child['crossing'][key] = p2['crossing'][key]

        # Crossing pairs — héritage clé par clé
        all_cp_keys = set(p1.get('crossing_pairs', {}).keys()) | set(p2.get('crossing_pairs', {}).keys())
        for key in all_cp_keys:
            v1 = p1.get('crossing_pairs', {}).get(key, [])
            v2 = p2.get('crossing_pairs', {}).get(key, [])
            child['crossing_pairs'][key] = random.choice([v1, v2])

        # Retour offsets — héritage clé par clé (None = ASAP, valide)
        for mid in mission_ids:
            o1 = p1.get('retour_offsets', {}).get(mid)
            o2 = p2.get('retour_offsets', {}).get(mid)
            child['retour_offsets'][mid] = random.choice([o1, o2])

        return child
    
    def _mutate(self, genome: Dict) -> Dict:
        """Mutation avec intensité variable — inclut turnaround_buffers."""
        mutated = deepcopy(genome)

        # Mutation timing (30% des gènes)
        for mid in mutated['timing']:
            if random.random() < 0.3:
                if random.random() < 0.7:
                    current = mutated['timing'][mid]
                    mutated['timing'][mid] = max(0, min(59, current + random.randint(-5, 5)))
                else:
                    mutated['timing'][mid] = random.randint(0, 59)

        # Mutation turnaround buffers (40% de chance)
        crossing_enabled = self.config.crossing_optimization and self.config.crossing_optimization.enabled
        if crossing_enabled:
            max_buf = max(15, getattr(self.config, 'turnaround_max_buffer', 30))
            buf_choices = [0, 0, 3, 5, 8, 10, 15, max_buf // 2, max_buf]
            max_delay = max(5, self.config.crossing_optimization.max_delay_minutes)
            cross_choices = [0, 0, 2, 3, 5] + list(range(7, max_delay + 1, 2))
        else:
            buf_choices = [0, 0, 3, 5, 8, 10]
            cross_choices = [0, 0, 2, 3, 5]

        if 'turnaround_buffers' not in mutated:
            mutated['turnaround_buffers'] = {}
        for mid in list(mutated.get('timing', {}).keys()):
            if random.random() < 0.4:
                current_buf = mutated['turnaround_buffers'].get(mid, 0)
                if random.random() < 0.5:
                    delta_choices = [-3, -1, 0, 2, 5]
                    if crossing_enabled:
                        delta_choices.extend([8, 10])
                    mutated['turnaround_buffers'][mid] = max(0, current_buf + random.choice(delta_choices))
                else:
                    mutated['turnaround_buffers'][mid] = random.choice(buf_choices)

        # Mutation crossing (30% de chance)
        ve_list = self._identify_ve_gares()
        if random.random() < 0.3 and ve_list:
            for mid in list(mutated.get('crossing', {}).keys()):
                if random.random() < 0.4:
                    sd = mutated['crossing'][mid]
                    for gare in list(sd.keys()):
                        if random.random() < 0.5:
                            sd[gare] = random.choice(cross_choices)

        # Création d'un gène crossing pour une mission qui n'en a pas (10 %/mission).
        # Sans ça, un génome démarré sans stratégie de croisement ne pourrait jamais
        # en acquérir une par mutation (gène inatteignable).
        if 'crossing' not in mutated:
            mutated['crossing'] = {}
        if ve_list:
            for mid in self.search_space.keys():
                if mid not in mutated['crossing'] and random.random() < 0.1:
                    mutated['crossing'][mid] = {ve: random.choice(cross_choices) for ve in ve_list}

        # Mutation crossing_pairs (25 % de probabilité par mission)
        if 'crossing_pairs' not in mutated:
            mutated['crossing_pairs'] = {}
        if ve_list:
            for mid in list(mutated.get('timing', {}).keys()):
                if random.random() < 0.25:
                    current = list(mutated['crossing_pairs'].get(mid, []))
                    if current and random.random() < 0.4:
                        # Retirer une VE aléatoirement
                        current = [v for v in current if random.random() > 0.5] or current
                    else:
                        # Ajouter une VE candidate
                        new_ve = random.choice(ve_list)
                        if new_ve not in current:
                            current = current + [new_ve]
                    mutated['crossing_pairs'][mid] = current[:3]  # cap à 3 VE préférées

        # Mutation retour_offsets (30 % par mission) — décale le départ retour
        # cadencé. Peut activer (None→valeur) ou désactiver (→None) le cadencement.
        if 'retour_offsets' not in mutated:
            mutated['retour_offsets'] = {}
        for mid in list(mutated.get('timing', {}).keys()):
            if random.random() < 0.3:
                current = mutated['retour_offsets'].get(mid)
                roll = random.random()
                if roll < 0.2:
                    mutated['retour_offsets'][mid] = None  # repasser en ASAP
                elif current is not None and roll < 0.8:
                    # Petit décalage autour de la valeur courante (reste régulier).
                    mutated['retour_offsets'][mid] = (current + random.randint(-5, 5)) % 60
                else:
                    mutated['retour_offsets'][mid] = random.randint(0, 59)

        return mutated


# =============================================================================
# MODE EXHAUSTIF (INCHANGÉ)
# =============================================================================

def _construire_durees_theoriques(missions, df_gares):
    """{label_mission: duree_min} pour chaque mission active et chaque sens.

    Sert de référence au terme de temps de parcours du score. Indépendant des
    cadencements/offsets (qui ne changent que les heures de départ, pas les temps
    de marche), donc calculable une fois sur les missions d'origine.
    """
    from core_logic import construire_horaire_mission
    durees = {}
    for m in missions:
        if m.get('frequence', 0) <= 0:
            continue
        h_aller = construire_horaire_mission(m, 'aller', df_gares)
        if h_aller:
            durees[f"{m['origine']} → {m['terminus']}"] = h_aller[-1].get('time_offset_min', 0)
        h_retour = construire_horaire_mission(m, 'retour', df_gares)
        if h_retour:
            durees[f"{m['terminus']} → {m['origine']}"] = h_retour[-1].get('time_offset_min', 0)
    return durees


def evaluer_params_simulation(params, missions, df_gares, heure_debut, heure_fin, allow_sharing=True, config=None):
    from core_logic import executer_simulation_evenementielle, _calculer_stats_homogeneite, _score_chronologie_bruit
    from datetime import time as dt_time

    adjusted_ref = params.get_adjusted_reference_minutes(missions)
    modified_missions = []
    for i, m in enumerate(missions):
        m_copy = dict(m)
        if str(i) in adjusted_ref:
            m_copy['reference_minutes'] = adjusted_ref[str(i)]
        # Ne pas modifier les temps de retournement ici — turnaround_buffers est passé
        # séparément à executer_simulation_evenementielle pour éviter le double-comptage.
        modified_missions.append(m_copy)

    turn_bufs = params.get_turnaround_buffers(missions)
    cross_strats = params.get_crossing_strategies(missions, df_gares)
    pair_assignments = params.crossing_pair_assignments or {}
    retour_offsets = params.get_retour_offsets(missions)

    chronologie, warnings, stats = executer_simulation_evenementielle(
        modified_missions, df_gares, heure_debut, heure_fin,
        allow_sharing=allow_sharing,
        turnaround_buffers=turn_bufs,
        crossing_strategies=cross_strats,
        adjusted_reference_minutes=adjusted_ref,
        crossing_pair_assignments=pair_assignments,
        retour_reference_offsets=retour_offsets,
    )

    max_arret = (config.crossing_optimization.max_delay_minutes
                 if config and config.crossing_optimization and config.crossing_optimization.enabled
                 else 5)
    durees_theoriques = _construire_durees_theoriques(missions, df_gares)
    # NB : plus de pénalité directe sur les buffers de retournement. Leur coût
    # émerge désormais de leurs effets réels (nombre de rames, Gini, temps de
    # parcours) via le score ci-dessous.
    score = _score_chronologie_bruit(
        chronologie, warnings, max_arret_ligne_min=max_arret,
        durees_theoriques=durees_theoriques,
    )

    return score, chronologie, warnings, stats


def _baseline_simulation_params():
    """Crée un SimulationParams vide qui reproduit exactement le mode simple.

    Les cadencements restent vides pour que core_logic utilise les `reference_minutes`
    originales de la mission (préserve les patterns multi-valeurs comme "0,30").
    """
    return SimulationParams(
        cadencements={},
        turnaround_buffers={},
        crossing_stop_durations={},
    )


def _build_turnaround_range(config):
    """Retourne la plage de buffers de retournement à explorer.

    Quand l'optimisation des croisements est activée, on étend la plage pour permettre
    au retournement prolongé de dissoudre les conflits (alternative aux arrêts VE).
    """
    base = [0, 3, 5, 8, 10]
    if config and config.crossing_optimization and config.crossing_optimization.enabled:
        max_buf = max(15, getattr(config, 'turnaround_max_buffer', 30))
        extended = list(range(15, max_buf + 1, 5))
        return base + [v for v in extended if v not in base]
    return base


def _build_crossing_range(config):
    """Durées d'arrêt à tester sur les gares VE, bornées par max_delay_minutes."""
    if config and config.crossing_optimization and config.crossing_optimization.enabled:
        max_delay = max(5, config.crossing_optimization.max_delay_minutes)
        values = [0, 2, 3, 5]
        extras = list(range(7, max_delay + 1, 2))
        return values + [v for v in extras if v not in values]
    return [0, 2, 3, 5]


def _optimisation_smart_progressive(missions, df_gares, heure_debut, heure_fin,
                                    allow_sharing=True, config=None, progress_callback=None):
    if config is None:
        config = OptimizationConfig(mode='smart_progressive')

    # IDs uniquement pour les missions actives, dans l'ordre d'index (pas de slice).
    active_mission_ids = [f"M{i+1}" for i, m in enumerate(missions) if m.get('frequence', 0) > 0]

    # Gares VE disponibles par mission (fix: missions passées pour peupler le dict).
    ve_gares = _identifier_points_croisement(df_gares, missions)

    # --- BASELINE = mode simple : évaluée en premier, sert de référence à battre ---
    best_params = _baseline_simulation_params()
    best_score, best_chronologie, best_warnings, _ = evaluer_params_simulation(
        best_params, missions, df_gares, heure_debut, heure_fin, allow_sharing=allow_sharing, config=config
    )

    # Seuil minimal de gain pour remplacer la baseline : évite la dérive sur des
    # améliorations marginales qui perturbent le graphique sans bénéfice réel.
    tol = 1.0
    def _accept(new_score, current_score, trial_has_violations, best_has_violations):
        # Toujours accepter si on résout des violations d'infrastructure
        if best_has_violations and not trial_has_violations:
            return True
        if trial_has_violations and not best_has_violations:
            return False
        return new_score < current_score - tol

    best_has_violations = len(best_warnings.get("infra_violations", [])) > 0

    turnaround_vals = _build_turnaround_range(config)
    crossing_vals = _build_crossing_range(config)
    # Offsets de départ retour testés : None (ASAP, baseline) + grille de 5 min.
    retour_offset_vals = [None] + list(range(0, 60, 5))

    # Phases « mono-paramètre » répétées sur plusieurs passes (une amélioration sur
    # un paramètre peut en débloquer une sur un autre au tour suivant).
    main_phases = [
        ('Cadencement', [('cadencement', mid, list(range(0, 60, 5))) for mid in active_mission_ids]),
        # Retour cadencé : minute de référence du départ retour → départs réguliers
        # (remplace l'ancien buffer libre au terminus B).
        ('Retour cadencé', [('retour_offset', mid, retour_offset_vals) for mid in active_mission_ids]),
        # Retournement = buffer au terminus A (origine) uniquement.
        ('Retournement', [('turnaround', mid, turnaround_vals) for mid in active_mission_ids]),
    ]

    # Phase croisement : essayer différentes durées d'arrêt aux gares VE.
    crossing_steps = [
        ('crossing', mid, ve, crossing_vals)
        for mid in active_mission_ids
        for ve in ve_gares.get(mid, [])
    ]
    if crossing_steps:
        main_phases.append(('Croisement', crossing_steps))

    # Phase conjointe : grille 2D grossière (cadencement × offset retour) par mission.
    # Indispensable pour les solutions exigeant un mouvement simultané des deux.
    conjoint_grid = [(c, r) for c in range(0, 60, 10) for r in [None, 0, 15, 30, 45]]
    conjoint_phase = ('Conjoint', [('conjoint', mid, conjoint_grid) for mid in active_mission_ids])

    # Affinement : balayage fin du cadencement ET de l'offset retour.
    affinement_phase = ('Affinement', (
        [('cadencement', mid, list(range(0, 60))) for mid in active_mission_ids]
        + [('retour_offset', mid, [None] + list(range(0, 60))) for mid in active_mission_ids]
    ))

    NUM_PASSES = 2

    # Total = NUM_PASSES passes des phases mono-paramètre + conjoint + affinement.
    # (Borne haute : l'arrêt anticipé entre passes peut faire terminer plus tôt.)
    main_steps_per_pass = sum(sum(len(step[-1]) for step in ps) for _, ps in main_phases)
    total_steps = (NUM_PASSES * main_steps_per_pass
                   + sum(len(step[-1]) for step in conjoint_phase[1])
                   + sum(len(step[-1]) for step in affinement_phase[1]))

    # Phase Résolution : préparée mais seulement activée si des violations persistent.
    # On ne l'ajoute au total_steps qu'au moment de la déclencher (progress bar honnête).
    resolution_steps = []
    if config.crossing_optimization and config.crossing_optimization.enabled:
        max_buf = max(15, getattr(config, 'turnaround_max_buffer', 30))
        max_delay = max(5, config.crossing_optimization.max_delay_minutes)
        resolution_steps = [
            ('turnaround_force', mid, list(range(max(5, max_buf - 10), max_buf + 1, 3)))
            for mid in active_mission_ids
        ]
        for mid in active_mission_ids:
            for ve in ve_gares.get(mid, []):
                resolution_steps.append(('crossing_force', mid, ve, list(range(max(2, max_delay - 5), max_delay + 1, 2))))

    steps_done = 0

    def _run_phase(phase_name, phase_steps):
        nonlocal best_params, best_score, best_chronologie, best_warnings, best_has_violations, steps_done
        for step in phase_steps:
            step_type = step[0]
            mid = step[1]
            values = step[-1]

            for val in values:
                trial_params = SimulationParams(
                    cadencements=dict(best_params.cadencements),
                    turnaround_buffers=dict(best_params.turnaround_buffers),
                    crossing_stop_durations={k: dict(v) for k, v in best_params.crossing_stop_durations.items()},
                    retour_offsets=dict(best_params.retour_offsets or {}),
                )

                if step_type in ('cadencement',):
                    trial_params.cadencements[mid] = val
                elif step_type in ('turnaround', 'turnaround_force'):
                    # Buffer appliqué au terminus A (origine) ; le terminus B est
                    # géré par retour_offset (cf. get_turnaround_buffers).
                    trial_params.turnaround_buffers[mid] = val
                elif step_type == 'retour_offset':
                    # val peut être None (= retour ASAP, baseline).
                    trial_params.retour_offsets[mid] = val
                elif step_type == 'conjoint':
                    # Mouvement 2D simultané (cadencement, offset retour) : attrape
                    # les solutions « décaler le départ ET le retour ensemble » que
                    # la descente coordonnée mono-paramètre rate.
                    cad_val, roff_val = val
                    trial_params.cadencements[mid] = cad_val
                    trial_params.retour_offsets[mid] = roff_val
                elif step_type in ('crossing', 'crossing_force'):
                    ve_gare = step[2]
                    if mid not in trial_params.crossing_stop_durations:
                        trial_params.crossing_stop_durations[mid] = {}
                    trial_params.crossing_stop_durations[mid][ve_gare] = val

                score, chrono, warns, _ = evaluer_params_simulation(
                    trial_params, missions, df_gares, heure_debut, heure_fin,
                    allow_sharing=allow_sharing, config=config,
                )
                trial_has_violations = len(warns.get("infra_violations", [])) > 0

                if _accept(score, best_score, trial_has_violations, best_has_violations):
                    best_score = score
                    best_chronologie = chrono
                    best_warnings = warns
                    best_params = trial_params
                    best_has_violations = trial_has_violations

                steps_done += 1
                if progress_callback and total_steps > 0:
                    progress_callback(steps_done, total_steps, best_score, len(best_chronologie), 0)

    # Passes successives des phases mono-paramètre, avec arrêt anticipé dès qu'une
    # passe complète n'améliore plus le score (au-delà de la tolérance).
    for pass_idx in range(NUM_PASSES):
        score_avant_passe = best_score
        for phase_name, phase_steps in main_phases:
            _run_phase(phase_name, phase_steps)
        if best_score >= score_avant_passe - tol:
            break  # passe complète sans gain : inutile de répéter

    # Phase conjointe (une fois) : mouvements 2D cadencement × offset retour.
    _run_phase(*conjoint_phase)

    # Affinement final : balayage fin, tolérance nulle pour capter les petits gains.
    tol = 0.0
    _run_phase(*affinement_phase)

    # Phase Résolution : ne tourne que si des violations persistent OU si aucune solution sans violation n'a été trouvée
    if resolution_steps and best_has_violations:
        _run_phase('Résolution', resolution_steps)

    return best_chronologie, best_warnings, {
        'mode': 'smart_progressive',
        'best_score': best_score if best_score != float('inf') else None,
        'steps_evaluated': steps_done,
        'baseline_preserved': (not best_params.cadencements and not best_params.turnaround_buffers
                               and not best_params.crossing_stop_durations
                               and not any(v is not None for v in (best_params.retour_offsets or {}).values())),
    }


def _identifier_points_croisement(df_gares, missions=None):
    """Retourne {mission_id: [gares_VE]} pour toutes les missions actives.

    Le bug d'origine (boucle sur un dict vide) a été corrigé : toutes les gares VE
    de la ligne sont associées à chaque mission active.
    """
    from core_logic import _get_infra_at_gare
    ve_stations = []
    for _, row in df_gares.iterrows():
        if _get_infra_at_gare(df_gares, row['gare']) == 'VE':
            ve_stations.append(row['gare'])

    if not ve_stations or missions is None:
        return {}

    result = {}
    for i, m in enumerate(missions):
        if m.get('frequence', 0) > 0:
            result[f"M{i+1}"] = ve_stations
    return result


def optimize_exhaustive(missions, df_gares, heure_debut, heure_fin, config,
                       scorer, allow_sharing, progress_callback):
    """Mode exhaustif — exploration complète cadencement + turnaround + croisement."""
    from itertools import product

    active_missions = [(i, m) for i, m in enumerate(missions) if m.get('frequence', 0) > 0]
    if not active_missions:
        params = _baseline_simulation_params()
        score, chrono, warns, stats = evaluer_params_simulation(params, missions, df_gares, heure_debut, heure_fin, allow_sharing, config=config)
        return chrono, warns, {'mode': 'exhaustif', 'combinations_tested': 1}

    mission_ids = [f"M{i+1}" for i, _ in active_missions]

    # Baseline : évaluée en premier pour fournir un point de départ valide
    best_params = _baseline_simulation_params()
    best_score, best_chronologie, best_warnings, _ = evaluer_params_simulation(
        best_params, missions, df_gares, heure_debut, heure_fin, allow_sharing, config=config
    )

    cadence_range = list(range(0, 60, 5))
    # On explore l'offset de départ retour (cadencé) plutôt que le buffer libre
    # au terminus B : départs réguliers garantis. None = retour ASAP (baseline).
    retour_offset_range = [None, 0, 10, 20, 30, 40, 50]

    if len(active_missions) > 3:
        cadence_range = list(range(0, 60, 10))
        retour_offset_range = [None, 0, 20, 40]

    combos = list(product(cadence_range, retour_offset_range, repeat=len(active_missions)))
    if len(combos) > 100000:
        cadence_range = list(range(0, 60, 10))
        combos = list(product(cadence_range, retour_offset_range, repeat=len(active_missions)))

    for idx, combo in enumerate(combos):
        cadencements = {}
        retour_offsets = {}
        for j, (m_idx, m) in enumerate(active_missions):
            mid = f"M{m_idx+1}"
            cadencements[mid] = combo[2*j]
            retour_offsets[mid] = combo[2*j+1]

        params = SimulationParams(
            cadencements=cadencements,
            turnaround_buffers={},
            crossing_stop_durations={},
            retour_offsets=retour_offsets,
        )

        score, chrono, warns, stats = evaluer_params_simulation(
            params, missions, df_gares, heure_debut, heure_fin, allow_sharing, config=config
        )

        if score < best_score:
            best_score = score
            best_chronologie = chrono
            best_warnings = warns
            best_params = params

        if progress_callback:
            progress_callback(idx+1, len(combos), best_score, len(chrono) if chrono else 0, 0)

    if not best_chronologie or best_params is None:
        return {}, {"infra_violations": [], "other": ["Echec exhaustif"]}, {'mode': 'exhaustif', 'combinations_tested': len(combos)}

    # Balayage croisement : sur la meilleure configuration trouvée, tester des durées d'arrêt aux VE.
    ve_gares = _identifier_points_croisement(df_gares, missions)
    crossing_combos_tested = 0
    crossing_durs = _build_crossing_range(config)[1:]  # on exclut 0 ici (déjà testé par défaut)
    if ve_gares and crossing_durs:
        n_crossing = sum(
            len(crossing_durs) * len(ve_gares.get(mid, []))
            for mid in mission_ids
        )
        total_with_crossing = len(combos) + n_crossing
        for mid in mission_ids:
            for ve in ve_gares.get(mid, []):
                for dur in crossing_durs:
                    trial_crossing = {k: dict(v) for k, v in best_params.crossing_stop_durations.items()}
                    if mid not in trial_crossing:
                        trial_crossing[mid] = {}
                    trial_crossing[mid][ve] = dur
                    trial = SimulationParams(
                        cadencements=dict(best_params.cadencements),
                        turnaround_buffers=dict(best_params.turnaround_buffers),
                        crossing_stop_durations=trial_crossing,
                        retour_offsets=dict(best_params.retour_offsets or {}),
                    )
                    score, chrono, warns, stats = evaluer_params_simulation(
                        trial, missions, df_gares, heure_debut, heure_fin, allow_sharing, config=config
                    )
                    crossing_combos_tested += 1
                    if score < best_score:
                        best_score = score
                        best_chronologie = chrono
                        best_warnings = warns
                        best_params = trial
                    if progress_callback:
                        progress_callback(
                            len(combos) + crossing_combos_tested,
                            total_with_crossing,
                            best_score,
                            len(best_chronologie) if best_chronologie else 0,
                            0,
                        )

    return best_chronologie, best_warnings, {
        'mode': 'exhaustif',
        'best_score': best_score,
        'combinations_tested': len(combos) + crossing_combos_tested,
    }


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def optimiser_graphique_horaire(missions, df_gares, heure_debut, heure_fin,
                               config, allow_sharing=True, progress_callback=None):
    """Point d'entrée principal de l'optimisation — tous les modes utilisent le moteur événementiel."""
    
    if config.mode == "genetic":
        optimizer = GeneticOptimizer(missions, df_gares, heure_debut, heure_fin,
                                    config, SolutionScorer(config), allow_sharing)
        return optimizer.optimize(progress_callback)
    elif config.mode == "exhaustif":
        scorer = SolutionScorer(config)
        return optimize_exhaustive(missions, df_gares, heure_debut, heure_fin,
                                  config, scorer, allow_sharing, progress_callback)
    elif config.mode == "simple":
        # Cadencements vides → le moteur utilise les reference_minutes d'origine de
        # chaque mission, en préservant les patterns multi-valeurs ("0,30"). (Avant,
        # on forçait refs[0], ce qui aplatissait les patterns à plusieurs sillons/h.)
        params = SimulationParams(cadencements={}, turnaround_buffers={}, crossing_stop_durations={})
        score, chrono, warns, stats = evaluer_params_simulation(
            params, missions, df_gares, heure_debut, heure_fin, allow_sharing, config=config
        )
        return chrono, warns, {'mode': 'simple', 'description': 'Simulation directe avec paramètres utilisateur', 'best_score': score}
    else:
        return _optimisation_smart_progressive(
            missions, df_gares, heure_debut, heure_fin,
            allow_sharing=allow_sharing, config=config, progress_callback=progress_callback
        )
