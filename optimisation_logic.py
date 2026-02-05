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
    """Configuration pour l'optimisation des croisements."""
    enabled: bool = False
    max_delay_minutes: int = 15
    penalty_per_minute: float = 2.0

@dataclass
class CrossingStrategy:
    """Stratégie de croisement pour une mission donnée."""
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
    """Worker optimisé avec timeout et suppression des outputs."""
    import sys
    import io
    
    # Redirection complète des outputs
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    
    try:
        genome, missions, df_gares, heure_debut, heure_fin, allow_sharing, config_dict = args
        config = OptimizationConfig(**config_dict)
        
        # Check cache d'abord
        cache_key = _solution_cache.get_key(missions, genome)
        cached = _solution_cache.get(cache_key)
        if cached and config.use_cache:
            return cached
        
        from core_logic import generer_tous_trajets_optimises
        
        adjusted_missions = deepcopy(missions)
        crossing_strategies_dict = {}
        
        for mission_id, offset in genome['timing'].items():
            for mission in adjusted_missions:
                if f"{mission['origine']}→{mission['terminus']}" == mission_id:
                    mission['reference_minutes'] = str(offset)
                    if mission_id in genome.get('crossing', {}):
                        crossing_strategies_dict[f"Mission_{adjusted_missions.index(mission)}"] = \
                            CrossingStrategy.from_dict(genome['crossing'][mission_id])
                    break
        
        chronologie, warnings, _ = generer_tous_trajets_optimises(
            adjusted_missions, df_gares, heure_debut, heure_fin,
            allow_sharing=allow_sharing,
            search_strategy='fast',  # Mode rapide
            progress_callback=None,
            crossing_strategies=crossing_strategies_dict
        )
        
        scorer = SolutionScorer(config)
        score = scorer.score_solution(chronologie, warnings)
        result = (genome, chronologie, warnings, score)
        
        # Mise en cache
        if config.use_cache:
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
        """Construit l'espace de recherche (plages de minutes possibles pour chaque mission)."""
        space = {}
        for mission in self.missions:
            mission_id = f"{mission['origine']}→{mission['terminus']}"
            try:
                minutes_ref = [int(m.strip()) for m in mission.get("reference_minutes", "0").split(',') 
                              if m.strip().isdigit()] or [0]
            except: 
                minutes_ref = [0]
            space[mission_id] = {'default': minutes_ref[0], 'range': (0, 59)}
        return space
    
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
        """Initialise la population avec diversité et stratégies de croisement intelligentes."""
        population = []
        
        for i in range(self.config.population_size):
            genome = {'timing': {}, 'crossing': {}}
            
            # Timing: mélange de valeurs par défaut et aléatoires
            for mission_id, info in self.search_space.items():
                if i < self.config.population_size // 4:
                    # 25% utilisent valeurs par défaut
                    genome['timing'][mission_id] = info['default']
                elif i < self.config.population_size // 2:
                    # 25% utilisent des valeurs par pas de 5
                    genome['timing'][mission_id] = random.randrange(0, 60, 5)
                else:
                    # 50% complètement aléatoires
                    genome['timing'][mission_id] = random.randint(0, 59)
                
                # Stratégies de croisement optimisées
                ve_list = self.crossing_points.get(mission_id, [])
                if ve_list and random.random() < 0.6:  # 60% de chance d'avoir une stratégie
                    # Trois types de stratégies initiales pour diversité
                    strategy_type = i % 3
                    
                    if strategy_type == 0:
                        # Stratégie "croisement rapide" - favorise arrêts courts
                        stop_durations = {ve: random.choices([0, 0, 2], weights=[0.6, 0.3, 0.1])[0] for ve in ve_list}
                        priority = random.uniform(0.5, 0.8)
                        max_delay = random.choice([10, 15])
                    elif strategy_type == 1:
                        # Stratégie "flexible" - permet arrêts moyens
                        stop_durations = {ve: random.choices([0, 2, 3, 5], weights=[0.4, 0.3, 0.2, 0.1])[0] for ve in ve_list}
                        priority = random.uniform(0.4, 0.6)
                        max_delay = random.choice([10, 15, 20])
                    else:
                        # Stratégie "sans croisement planifié" - tout à 0
                        stop_durations = {ve: 0 for ve in ve_list}
                        priority = random.uniform(0.3, 0.5)
                        max_delay = 10
                    
                    genome['crossing'][mission_id] = CrossingStrategy(
                        mission_id=mission_id,
                        stop_durations=stop_durations,
                        priority=priority,
                        max_acceptable_delay=max_delay
                    ).to_dict()

            
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
        """Croisement à deux points."""
        child = {'timing': {}, 'crossing': {}}
        
        # Croisement timing
        mission_ids = list(p1['timing'].keys())
        if len(mission_ids) > 2:
            point1 = random.randint(0, len(mission_ids) - 1)
            point2 = random.randint(point1, len(mission_ids) - 1)
            
            for i, mid in enumerate(mission_ids):
                if point1 <= i <= point2:
                    child['timing'][mid] = p2['timing'][mid]
                else:
                    child['timing'][mid] = p1['timing'][mid]
        else:
            for mid in mission_ids:
                child['timing'][mid] = random.choice([p1['timing'][mid], p2['timing'][mid]])
        
        # Croisement stratégies de croisement
        all_mission_ids = set(p1.get('crossing', {}).keys()) | set(p2.get('crossing', {}).keys())
        for mid in all_mission_ids:
            if mid in p1.get('crossing', {}) and mid in p2.get('crossing', {}):
                child['crossing'][mid] = random.choice([p1['crossing'][mid], p2['crossing'][mid]])
            elif mid in p1.get('crossing', {}):
                child['crossing'][mid] = p1['crossing'][mid]
            elif mid in p2.get('crossing', {}):
                child['crossing'][mid] = p2['crossing'][mid]
        
        return child
    
    def _mutate(self, genome: Dict) -> Dict:
        """Mutation avec intensité variable et optimisation des croisements."""
        mutated = deepcopy(genome)
        
        # Mutation timing (30% des gènes)
        for mid in mutated['timing']:
            if random.random() < 0.3:
                # 70% mutation locale (±5min), 30% mutation globale
                if random.random() < 0.7:
                    current = mutated['timing'][mid]
                    mutated['timing'][mid] = max(0, min(59, current + random.randint(-5, 5)))
                else:
                    mutated['timing'][mid] = random.randint(0, 59)
        
        # Mutation crossing - optimisée pour favoriser croisements efficaces
        if random.random() < 0.3:  # Augmenté de 0.2 à 0.3 pour plus d'exploration
            for mid in list(mutated.get('crossing', {}).keys()):
                if random.random() < 0.4:  # Augmenté de 0.3 à 0.4
                    # Stratégie de mutation intelligente des durées d'arrêt
                    for ve in mutated['crossing'][mid]['stop_durations']:
                        if random.random() < 0.5:
                            # Favoriser les arrêts courts (0, 2, 3) pour minimiser les retards
                            # tout en permettant des croisements efficaces
                            mutated['crossing'][mid]['stop_durations'][ve] = random.choices(
                                [0, 0, 2, 3, 5, 8],  # Favorise 0 (2 fois plus probable)
                                weights=[0.35, 0.35, 0.15, 0.10, 0.04, 0.01]
                            )[0]
                    
                    # Mutation de la priorité et du délai acceptable
                    if random.random() < 0.3:
                        mutated['crossing'][mid]['priority'] = max(0.1, min(0.9, 
                            mutated['crossing'][mid]['priority'] + random.uniform(-0.2, 0.2)))
                    
                    if random.random() < 0.3:
                        mutated['crossing'][mid]['max_acceptable_delay'] = random.choice([5, 10, 15, 20])
        
        # Parfois, ajouter ou supprimer une stratégie de croisement
        if random.random() < 0.15:  # 15% de chance
            available_missions = list(self.crossing_points.keys())
            if available_missions:
                mission_id = random.choice(available_missions)
                ve_list = self.crossing_points.get(mission_id, [])
                
                if mission_id in mutated.get('crossing', {}) and random.random() < 0.3:
                    # Supprimer une stratégie existante
                    del mutated['crossing'][mission_id]
                elif ve_list and mission_id not in mutated.get('crossing', {}):
                    # Ajouter une nouvelle stratégie avec paramètres optimisés
                    if 'crossing' not in mutated:
                        mutated['crossing'] = {}
                    mutated['crossing'][mission_id] = CrossingStrategy(
                        mission_id=mission_id,
                        stop_durations={ve: random.choices([0, 0, 2, 3], weights=[0.5, 0.5, 0.3, 0.2])[0] for ve in ve_list},
                        priority=random.uniform(0.4, 0.7),  # Priorités moyennes à hautes
                        max_acceptable_delay=random.choice([10, 15])  # Délais raisonnables
                    ).to_dict()
        
        return mutated


# =============================================================================
# MODE EXHAUSTIF (INCHANGÉ)
# =============================================================================

def optimize_exhaustive(missions, df_gares, heure_debut, heure_fin, config, 
                       scorer, allow_sharing, progress_callback):
    """Mode exhaustif pour petits problèmes - CORRIGÉ."""
    from core_logic import generer_tous_trajets_optimises
    from itertools import product
    
    # CORRECTION : Identifier TOUTES les missions avec retour, pas seulement certaines
    mission_retours = []
    for m in missions:
        if m.get('frequence', 0) > 0:
            # Créer une liste de cadencements possibles pour chaque mission
            # Utiliser un pas fin (chaque minute) pour l'exhaustif
            mission_retours.append((
                f"{m['origine']}→{m['terminus']}", 
                list(range(0, 60, 1))  # CORRECTION : Tester chaque minute
            ))
    
    if not mission_retours:
        chrono, warns, _ = generer_tous_trajets_optimises(
            missions, df_gares, heure_debut, heure_fin, allow_sharing=allow_sharing
        )
        return chrono, warns, {'mode': 'exhaustif', 'combinations_tested': 1}
    
    # CORRECTION : Générer TOUTES les combinaisons possibles
    all_combos = list(product(*[v for _, v in mission_retours]))
    
    # Limitation de sécurité : si trop de combinaisons, réduire le pas
    if len(all_combos) > 100000:
        # Réduire au pas de 5 minutes si trop de combinaisons
        mission_retours = [(mid, list(range(0, 60, 5))) for mid, _ in mission_retours]
        all_combos = list(product(*[v for _, v in mission_retours]))
    
    best_res, best_score = (None, None, None), float('inf')
    
    for idx, combo in enumerate(all_combos):
        adj_missions = deepcopy(missions)
        # CORRECTION : Appliquer les cadencements à TOUTES les missions
        for i, ((mid, _), val) in enumerate(zip(mission_retours, combo)):
            for m in adj_missions:
                if f"{m['origine']}→{m['terminus']}" == mid:
                    m['reference_minutes'] = str(val)
        
        # CORRECTION : Utiliser la stratégie 'simple' qui donne de bons résultats
        chrono, warns, _ = generer_tous_trajets_optimises(
            adj_missions, df_gares, heure_debut, heure_fin, 
            allow_sharing=allow_sharing, search_strategy='simple'  # Utilise la logique simple optimisée
        )
        
        if not scorer.is_valid_solution(warns):
            if progress_callback:
                progress_callback(idx+1, len(all_combos), best_score, 0, 0)
            continue
        
        score = scorer.score_solution(chrono, warns)
        if score < best_score:
            best_score = score
            best_res = (chrono, warns)
        
        if progress_callback:
            progress_callback(idx+1, len(all_combos), best_score, len(chrono) if chrono else 0, 0)
    
    if best_res[0] is None:
        return {}, {"infra_violations": [], "other": ["Echec exhaustif"]}, {'mode': 'exhaustif', 'combinations_tested': len(all_combos)}
    return best_res[0], best_res[1], {'mode': 'exhaustif', 'best_score': best_score, 'combinations_tested': len(all_combos)}


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def optimiser_graphique_horaire(missions, df_gares, heure_debut, heure_fin, 
                               config, allow_sharing=True, progress_callback=None):
    """Point d'entrée principal de l'optimisation."""
    scorer = SolutionScorer(config)
    
    # Redirection du mode "smart" vers "smart_progressive"
    if config.mode == "smart":
        config.mode = "smart_progressive"
    
    if config.mode == "genetic":
        optimizer = GeneticOptimizer(missions, df_gares, heure_debut, heure_fin, 
                                    config, scorer, allow_sharing)
        return optimizer.optimize(progress_callback)
    elif config.mode == "exhaustif":
        return optimize_exhaustive(missions, df_gares, heure_debut, heure_fin, 
                                  config, scorer, allow_sharing, progress_callback)
    elif config.mode == "simple":
        # Mode simple : utilise la logique de base sans optimisation avancée
        # L'utilisateur contrôle directement via les temps de retournement configurés
        from core_logic import generer_tous_trajets_optimises
        c, w, _ = generer_tous_trajets_optimises(
            missions, df_gares, heure_debut, heure_fin, 
            allow_sharing=allow_sharing, progress_callback=progress_callback,
            search_strategy='simple'  # Mode simple sans recherche
        )
        return c, w, {'mode': 'simple', 'description': 'Simulation directe avec paramètres utilisateur'}
    else:  # smart_progressive, fast, ou autres
        from core_logic import generer_tous_trajets_optimises
        c, w, _ = generer_tous_trajets_optimises(
            missions, df_gares, heure_debut, heure_fin, 
            allow_sharing=allow_sharing, progress_callback=progress_callback,
            search_strategy=config.mode
        )
        return c, w, {'mode': config.mode}
