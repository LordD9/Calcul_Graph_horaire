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

    def get_turnaround_buffers(self, missions):
        result = {}
        for i, m in enumerate(missions):
            mid = f"M{i+1}"
            buffer = self.turnaround_buffers.get(mid, self.turnaround_buffers.get(str(i), 0))
            if buffer != 0:
                result[mid] = buffer
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
        result = {}
        for i, m in enumerate(missions):
            mid = f"M{i+1}"
            offset = self.cadencements.get(mid, self.cadencements.get(str(i), None))
            if offset is not None:
                result[str(i)] = str(offset)
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


# =============================================================================
# SYSTÈME DE SCORING OPTIMISÉ
# =============================================================================

class SolutionScorer:
    """Système de scoring avec calculs optimisés."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def score_solution(self, chronologie: Dict, warnings: Dict) -> float:
        """Calcul de score optimisé avec focus sur la régularité et les croisements globaux."""
        # 1. Violations Infra (pénalité critique)
        infra_violations = len(warnings.get("infra_violations", []))
        
        # 2. Composantes du score
        num_rames = len(chronologie)
        other_warnings = len(warnings.get("other", []))
        
        # 3. Pénalité pour trajets annulés/retards
        cancelled_trips = sum(1 for w in warnings.get("other", []) if "annulé" in w.lower() or "impossible" in w.lower())
        
        # 4. Calcul des retards (extensions de croisement)
        total_delay = 0
        for _, trajets in chronologie.items():
            for trajet in trajets:
                # Vérifier si le trajet a subi des extensions pour croisement
                if 'crossing_extensions' in trajet:
                    for ext in trajet['crossing_extensions']:
                        total_delay += ext.get('duration', 0)
        
        # 5. Régularité (calcul vectorisé) - AUGMENTATION DU POIDS
        regularity_penalty = self._calculate_regularity_fast(chronologie)
        
        # 6. Score des croisements - bonus pour croisements optimaux
        crossing_quality = self._evaluate_crossing_quality(chronologie)
        
        # Score composite avec pondérations optimisées
        # AUGMENTATION du poids de régularité de 1000 à 5000 pour favoriser les graphiques réguliers
        score = (num_rames * 2000 +                    # Minimiser le nombre de rames
                other_warnings * 3000 +                 # Autres avertissements
                cancelled_trips * 15000 +               # Pénalité élevée pour trajets annulés
                total_delay * self.config.crossing_optimization.penalty_per_minute if self.config.crossing_optimization.enabled else 0 +  # Pénalité pour retards
                regularity_penalty * 5000 +             # Régularité (AUGMENTÉ de 1000 à 5000)
                infra_violations * 50000 -              # Violations critiques
                crossing_quality * 500                  # Bonus pour bons croisements
                )
        
        return score
    
    def _evaluate_crossing_quality(self, chronologie: Dict) -> float:
        """Évalue la qualité des croisements - bonus pour croisements optimaux sur voie double."""
        if not chronologie:
            return 0.0
        
        quality_score = 0.0
        
        # Analyser tous les croisements effectués
        for train_id, trajets in chronologie.items():
            for trajet in trajets:
                # Vérifier les croisements planifiés
                if 'crossings' in trajet:
                    for crossing in trajet['crossings']:
                        # Bonus si croisement sur voie double (pas d'arrêt nécessaire)
                        if crossing.get('on_double_track', False):
                            quality_score += 2.0
                        # Bonus moindre si croisement avec arrêt minimal
                        elif crossing.get('stop_duration', 0) <= 2:
                            quality_score += 1.0
                        # Pénalité légère si arrêt long
                        elif crossing.get('stop_duration', 0) > 5:
                            quality_score -= 0.5
        
        return quality_score
    
    def is_valid_solution(self, warnings: Dict) -> bool:
        """Vérifie si solution valide."""
        return len(warnings.get("infra_violations", [])) == 0
    
    def _calculate_regularity_fast(self, chronologie: Dict) -> float:
        """Calcul rapide de régularité avec coefficient de Gini (cohérent avec core_logic)."""
        if not chronologie:
            return 0.0
        
        missions_horaires = defaultdict(list)
        for _, trajets in chronologie.items():
            for trajet in trajets:
                mission_key = f"{trajet['origine']} → {trajet['terminus']}"
                missions_horaires[mission_key].append(trajet['start'].timestamp())
        
        total_penalty = 0.0
        for horaires_ts in missions_horaires.values():
            if len(horaires_ts) < 2:
                continue
            
            # Calcul vectorisé des intervalles
            horaires_array = np.array(sorted(horaires_ts))
            intervalles = np.diff(horaires_array) / 60.0  # Minutes
            intervalles = intervalles[intervalles > 0.1]
            
            if len(intervalles) == 0:
                total_penalty += 1.0
                continue
            
            # Calcul du coefficient de Gini (cohérent avec calculer_indice_homogeneite)
            n = len(intervalles)
            intervalles_sorted = np.sort(intervalles)
            somme_ponderee = np.sum((np.arange(n) + 1) * intervalles_sorted)
            somme_totale = np.sum(intervalles_sorted)
            
            if somme_totale == 0:
                total_penalty += 1.0
                continue
            
            gini = (2.0 * somme_ponderee) / (n * somme_totale) - (n + 1.0) / n
            # On veut pénaliser l'hétérogénéité, donc on utilise Gini directement
            # (Gini élevé = intervalles hétérogènes = mauvais)
            total_penalty += max(0.0, gini)
        
        return total_penalty


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
        # genome['crossing'] est déjà au format {mission_id: {gare_ve: durée}}
        crossing_stop_durations = genome.get('crossing', {})

        params = SimulationParams(
            cadencements=cadencements,
            turnaround_buffers=turnaround_buffers,
            crossing_stop_durations=crossing_stop_durations,
        )

        score, chronologie, warnings, stats = evaluer_params_simulation(
            params, missions, df_gares, heure_debut, heure_fin, allow_sharing
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
        """Construit l'espace de recherche (plages de minutes possibles pour chaque mission).

        Les clés utilisent le format "M{i+1}" (ex: "M1", "M2") pour être compatibles
        avec SimulationParams.get_adjusted_reference_minutes et get_turnaround_buffers.
        """
        space = {}
        for i, mission in enumerate(self.missions):
            if mission.get('frequence', 0) <= 0:
                continue
            mid = f"M{i+1}"
            try:
                minutes_ref = [int(m.strip()) for m in mission.get("reference_minutes", "0").split(',')
                              if m.strip().isdigit()] or [0]
            except:
                minutes_ref = [0]
            space[mid] = {'default': minutes_ref[0], 'range': (0, 59)}
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

        # Génome 0 = baseline utilisateur : timing/turnaround/crossing vides
        # → core_logic utilisera les reference_minutes originales de chaque mission.
        baseline_genome = {'timing': {}, 'turnaround_buffers': {}, 'crossing': {}}
        for mission_id in self.search_space.keys():
            baseline_genome['turnaround_buffers'][mission_id] = 0
        population.append(baseline_genome)

        for i in range(1, self.config.population_size):
            genome = {'timing': {}, 'turnaround_buffers': {}, 'crossing': {}}

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

            population.append(genome)
        return population
    
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
            'timeout_per_eval': self.config.timeout_per_eval
        }
        
        args_list = [
            (genome, self.missions, self.df_gares, self.heure_debut, 
             self.heure_fin, self.allow_sharing, config_dict) 
            for genome in population
        ]
        
        results = []
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {executor.submit(_evaluate_genome_worker, args): args for args in args_list}
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.config.timeout_per_eval)
                    results.append(result)
                except TimeoutError:
                    # Timeout: score infini
                    results.append((futures[future][0], {}, 
                                  {"infra_violations": [], "other": ["Timeout"]}, 
                                  float('inf')))
                except Exception as e:
                    results.append((futures[future][0], {}, 
                                  {"infra_violations": [], "other": [str(e)]}, 
                                  float('inf')))
        
        return results
    
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
        """Croisement à deux points — inclut turnaround_buffers."""
        child = {'timing': {}, 'turnaround_buffers': {}, 'crossing': {}}

        # Timing
        mission_ids = list(p1['timing'].keys())
        if len(mission_ids) > 2:
            point1 = random.randint(0, len(mission_ids) - 1)
            point2 = random.randint(point1, len(mission_ids) - 1)
            for i, mid in enumerate(mission_ids):
                child['timing'][mid] = p2['timing'][mid] if point1 <= i <= point2 else p1['timing'][mid]
        else:
            for mid in mission_ids:
                child['timing'][mid] = random.choice([p1['timing'][mid], p2['timing'][mid]])

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
        # genome['crossing'] = {mission_id: {gare_ve: durée}}
        ve_list = self._identify_ve_gares()
        if random.random() < 0.3 and ve_list:
            for mid in list(mutated.get('crossing', {}).keys()):
                if random.random() < 0.4:
                    sd = mutated['crossing'][mid]
                    for gare in list(sd.keys()):
                        if random.random() < 0.5:
                            sd[gare] = random.choice(cross_choices)

        return mutated


# =============================================================================
# MODE EXHAUSTIF (INCHANGÉ)
# =============================================================================

def evaluer_params_simulation(params, missions, df_gares, heure_debut, heure_fin, allow_sharing=True):
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

    chronologie, warnings, stats = executer_simulation_evenementielle(
        modified_missions, df_gares, heure_debut, heure_fin,
        allow_sharing=allow_sharing,
        turnaround_buffers=turn_bufs,
        crossing_strategies=cross_strats,
        adjusted_reference_minutes=adjusted_ref,
    )

    score = _score_chronologie_bruit(chronologie, warnings)
    extra_delay = sum(params.turnaround_buffers.values()) * 5
    score += extra_delay

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
        best_params, missions, df_gares, heure_debut, heure_fin, allow_sharing=allow_sharing
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

    phases = [
        ('Cadencement', [('cadencement', mid, list(range(0, 60, 5))) for mid in active_mission_ids]),
        ('Retournement', [('turnaround', mid, turnaround_vals) for mid in active_mission_ids]),
    ]

    # Phase croisement : essayer différentes durées d'arrêt aux gares VE.
    crossing_steps = [
        ('crossing', mid, ve, crossing_vals)
        for mid in active_mission_ids
        for ve in ve_gares.get(mid, [])
    ]
    if crossing_steps:
        phases.append(('Croisement', crossing_steps))

    phases.append(('Affinement', [('cadencement', mid, list(range(0, 60))) for mid in active_mission_ids]))

    # step[-1] est toujours la liste de valeurs, quelle que soit la longueur du tuple.
    total_steps = sum(sum(len(step[-1]) for step in phase_steps) for _, phase_steps in phases)

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
                )

                if step_type in ('cadencement',):
                    trial_params.cadencements[mid] = val
                elif step_type in ('turnaround', 'turnaround_force'):
                    trial_params.turnaround_buffers[mid] = val
                elif step_type in ('crossing', 'crossing_force'):
                    ve_gare = step[2]
                    if mid not in trial_params.crossing_stop_durations:
                        trial_params.crossing_stop_durations[mid] = {}
                    trial_params.crossing_stop_durations[mid][ve_gare] = val

                score, chrono, warns, _ = evaluer_params_simulation(
                    trial_params, missions, df_gares, heure_debut, heure_fin,
                    allow_sharing=allow_sharing,
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

    for phase_name, phase_steps in phases:
        _run_phase(phase_name, phase_steps)

    # Phase Résolution : ne tourne que si des violations persistent OU si aucune solution sans violation n'a été trouvée
    if resolution_steps and best_has_violations:
        _run_phase('Résolution', resolution_steps)

    return best_chronologie, best_warnings, {
        'mode': 'smart_progressive',
        'best_score': best_score if best_score != float('inf') else None,
        'steps_evaluated': steps_done,
        'baseline_preserved': not best_params.cadencements and not best_params.turnaround_buffers and not best_params.crossing_stop_durations,
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
        score, chrono, warns, stats = evaluer_params_simulation(params, missions, df_gares, heure_debut, heure_fin, allow_sharing)
        return chrono, warns, {'mode': 'exhaustif', 'combinations_tested': 1}

    mission_ids = [f"M{i+1}" for i, _ in active_missions]

    # Baseline : évaluée en premier pour fournir un point de départ valide
    best_params = _baseline_simulation_params()
    best_score, best_chronologie, best_warnings, _ = evaluer_params_simulation(
        best_params, missions, df_gares, heure_debut, heure_fin, allow_sharing
    )

    cadence_range = list(range(0, 60, 5))
    # Plage turnaround étendue si optim. croisement activée
    if config.crossing_optimization and config.crossing_optimization.enabled:
        max_buf = max(15, getattr(config, 'turnaround_max_buffer', 30))
        turnaround_range = [0, 5, 10, max(15, max_buf // 2), max_buf]
    else:
        turnaround_range = [0, 5, 10]

    if len(active_missions) > 3:
        cadence_range = list(range(0, 60, 10))
        turnaround_range = turnaround_range[:3]

    combos = list(product(cadence_range, turnaround_range, repeat=len(active_missions)))
    if len(combos) > 100000:
        cadence_range = list(range(0, 60, 10))
        combos = list(product(cadence_range, turnaround_range, repeat=len(active_missions)))

    for idx, combo in enumerate(combos):
        cadencements = {}
        turnaround_buffers = {}
        for j, (m_idx, m) in enumerate(active_missions):
            mid = f"M{m_idx+1}"
            cadencements[mid] = combo[2*j]
            turnaround_buffers[mid] = combo[2*j+1]

        params = SimulationParams(
            cadencements=cadencements,
            turnaround_buffers=turnaround_buffers,
            crossing_stop_durations={},
        )

        score, chrono, warns, stats = evaluer_params_simulation(
            params, missions, df_gares, heure_debut, heure_fin, allow_sharing
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
    if ve_gares:
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
                    )
                    score, chrono, warns, stats = evaluer_params_simulation(
                        trial, missions, df_gares, heure_debut, heure_fin, allow_sharing
                    )
                    crossing_combos_tested += 1
                    if score < best_score:
                        best_score = score
                        best_chronologie = chrono
                        best_warnings = warns
                        best_params = trial

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
        params = SimulationParams(cadencements={}, turnaround_buffers={}, crossing_stop_durations={})
        for i, m in enumerate(missions):
            mid = f"M{i+1}"
            refs = [int(x.strip()) for x in str(m.get('reference_minutes', '0')).split(',') if x.strip().isdigit()]
            params.cadencements[mid] = refs[0] if refs else 0
        score, chrono, warns, stats = evaluer_params_simulation(
            params, missions, df_gares, heure_debut, heure_fin, allow_sharing
        )
        return chrono, warns, {'mode': 'simple', 'description': 'Simulation directe avec paramètres utilisateur', 'best_score': score}
    else:
        return _optimisation_smart_progressive(
            missions, df_gares, heure_debut, heure_fin,
            allow_sharing=allow_sharing, config=config, progress_callback=progress_callback
        )
