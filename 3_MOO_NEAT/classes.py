import numpy as np
import time
from collections import Counter
import time 
import os
import numpy as np
from deap import tools
from collections import Counter
from game_utils import (
    extract_objects_info, construct_input_vector,
    update_coverage, manhattan_distance 
)

from config import (
    LIMITED_ACTIONS, GHOST_SCORE, INITIAL_LIVES,
    MAX_EXPECTED_SCORE, N_GENERATIONS, POPULATION_SIZE, creator
)

class NEATIndividual:
    """Wrapper class to make NEAT genomes compatible with DEAP's NSGA-II"""
    def __init__(self, genome):
        self.genome = genome
        self.fitness = creator.FitnessMulti()
        
    def __getattr__(self, name):
        """Delegate unknown attributes to the genome"""
        return getattr(self.genome, name)
    
    def dominates(self, other):
        """Implementation of dominance comparison for NSGA-II"""
        if hasattr(self.fitness, 'values') and hasattr(other.fitness, 'values'):
            return all(self.fitness.values[i] >= other.fitness.values[i] 
                      for i in range(len(self.fitness.values))) and \
                   any(self.fitness.values[i] > other.fitness.values[i] 
                      for i in range(len(self.fitness.values)))
        return False
    
    def copy(self):
        """Create a copy of the individual"""
        new_ind = NEATIndividual(self.genome)
        new_ind.fitness = creator.FitnessMulti(self.fitness.values)
        return new_ind

class GameStateMetrics:
    """Track and analyse game state metrics"""
    def __init__(self, resolution_info=None):
        # Resolution info
        self.screen_width = resolution_info['width'] if resolution_info else 160
        self.screen_height = resolution_info['height'] if resolution_info else 210

        # Core metrics
        self.total_score = 0
        self.initial_lives = INITIAL_LIVES
        self.current_lives = INITIAL_LIVES
        self.frames_survived = 0
        self.ghost_points = 0
        self.pellet_points = 0
        self.total_points_from_actions = 0
        self.scoring_actions = 0
        self.deaths = 0
        
        # Action tracking
        self.action_counts = {i: 0 for i in range(len(LIMITED_ACTIONS))}
        
        # Ghost tracking
        self.ghost_distance_sum = 0.0
        self.ghost_distance_list = []
        
        # Score tracking
        self.score_segments = []
        self.score_change_frames = []
        
        # Pill tracking
        self.initial_pill_count = None
        self.current_pill_count = None
        self.initial_powerpill_count = None
        self.current_powerpill_count = None
        
        # Movement tracking
        self.visited_positions = []
        self.powerpill_consumption_frames = []
        self.ghost_proximity_history = []
        self.coverage_ratio = 0.0

        self.junction_points = []
        self.times_at_junction = 0
        self.decision_quality = 0.0
        self.hybrid_rewards = 0.0
        self.ghost_avoidance_score = 0.0
        self.exploration_score = 0.0
    
    def update(self, reward, lives, frame_number, action, player_pos,
               ghost_positions, pill_positions, powerpill_positions):
         # Basic updates
        self.total_score += reward
        self.frames_survived = frame_number
        self.action_counts[action] += 1
        
        if reward >= GHOST_SCORE:
            self.ghost_points += reward
        else:
            self.pellet_points += reward
        
        if reward > 0:
            self.scoring_actions += 1
            self.total_points_from_actions += reward
            self.score_change_frames.append(frame_number) # for later analysis
        
        if lives < self.current_lives:
            self.deaths += 1
        self.current_lives = lives

        if frame_number % 100 == 0:
            self.score_segments.append(self.total_score)
        
        # Track pill counts
        total_pills_current = len(pill_positions) + len(powerpill_positions)
        if self.initial_pill_count is None:
            self.initial_pill_count = total_pills_current
        self.current_pill_count = total_pills_current
        
        if self.initial_powerpill_count is None:
            self.initial_powerpill_count = len(powerpill_positions)
        self.current_powerpill_count = len(powerpill_positions)
        
        # New metric updates
        if player_pos is not None:
            self.visited_positions.append(player_pos)
            
            # Track ghost proximity
            if ghost_positions:
                distances = [manhattan_distance(player_pos, g) for g in ghost_positions]
                min_dist = min(distances)
                self.ghost_distance_sum += min_dist
                self.ghost_distance_list.append(min_dist)
                self.ghost_proximity_history.append((frame_number, min_dist))
        
        # Track power pill consumption
        prev_powerpill_count = self.current_powerpill_count
        if prev_powerpill_count > len(powerpill_positions):
            self.powerpill_consumption_frames.append(frame_number)
        pass
    
    def get_metrics(self):
         # Calculate derived metrics
        if self.ghost_distance_list:
            avg_ghost_distance = np.mean(self.ghost_distance_list)
            var_ghost_distance = np.var(self.ghost_distance_list)
            frequency_low = sum(1 for d in self.ghost_distance_list if d < 20) / len(self.ghost_distance_list)
        else:
            avg_ghost_distance = 0
            var_ghost_distance = 0
            frequency_low = 0
        
        return {
            # Screen dimensions
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,

            # Existing metrics
            'total_score': self.total_score,
            'frames_survived': self.frames_survived,
            'deaths': self.deaths,
            'score_segments': self.score_segments,
            'ghost_points': self.ghost_points,
            'pellet_points': self.pellet_points,
            'scoring_actions': self.scoring_actions,
            'total_points_from_actions': self.total_points_from_actions,
            'current_lives': self.current_lives,
            'score_change_frames': self.score_change_frames,
            'avg_ghost_distance': avg_ghost_distance,
            'var_ghost_distance': var_ghost_distance,
            'frequency_low_ghost_distance': frequency_low,
            'initial_pill_count': self.initial_pill_count,
            'current_pill_count': self.current_pill_count,
            'initial_powerpill_count': self.initial_powerpill_count,
            'current_powerpill_count': self.current_powerpill_count,
            
            # New metrics
            'visited_positions': self.visited_positions,
            'powerpill_consumption_frames': self.powerpill_consumption_frames, 
            'ghost_proximity_history': self.ghost_proximity_history,
            'coverage_ratio': self.coverage_ratio,

            'junction_points': self.junction_points,
            'times_at_junction': self.times_at_junction,
            'decision_quality': self.decision_quality,
            'hybrid_rewards': self.hybrid_rewards,
            'ghost_avoidance_score': self.ghost_avoidance_score,
            'exploration_score': self.exploration_score
        }
        

class NEATAnalytics:
    """Track and analyse NEAT's evolution progress"""
    def __init__(self):
        self.generation_stats = []
        self.network_evolution = []
        self.best_genomes = []
        self.species_stats = []
        self.game_recordings = []
    
    def record_generation(self, generation, population, species):
        """Record comprehensive statistics for each generation"""
        gen_stats = {
            'generation': generation,
            'timestamp': time.time(),
            
            # Population Stats
            'population_size': len(population),
            'species_count': len(species.species),
            'avg_fitness': np.mean([g.fitness for g in population.values()]),
            'max_fitness': max(g.fitness for g in population.values()),
            
            # Game Performance
            'scores': [g.total_score for g in population.values()],
            'avg_score': np.mean([g.total_score for g in population.values()]),
            'max_score': max(g.total_score for g in population.values()),
            
            # Multi-objective Stats
            'pareto_front_size': len(tools.sortNondominated(population.values(), len(population))[0]),
            'objective_values': [(g.fitness.values if hasattr(g, 'fitness') else (0,0,0)) 
                               for g in population.values()]
        }
        self.generation_stats.append(gen_stats)
        
        # Record best genome's network structure
        best_genome = max(population.values(), key=lambda g: g.fitness)
        network_stats = self.analyse_network(best_genome)
        self.network_evolution.append({
            'generation': generation,
            'network_stats': network_stats,
            'genome': best_genome
        })
        
        # Record species statistics
        species_stats = {
            'generation': generation,
            'species_sizes': [len(s.members) for s in species.species.values()],
            'species_fitness': [s.fitness for s in species.species.values()],
            'species_ages': [s.age for s in species.species.values()]
        }
        self.species_stats.append(species_stats)
        pass
    
    def analyse_network(self, genome):
        """Analyse neural network topology and parameters"""
        return {
            'node_count': len(genome.nodes),
            'connection_count': len(genome.connections),
            'enabled_connections': sum(1 for conn in genome.connections.values() if conn.enabled),
            'hidden_nodes': len([n for n in genome.nodes if n not in genome.input_keys + genome.output_keys]),
            'activation_functions': Counter(node.activation for node in genome.nodes.values()),
            'weight_stats': {
                'mean': np.mean([c.weight for c in genome.connections.values()]),
                'std': np.std([c.weight for c in genome.connections.values()]),
                'min': min(c.weight for c in genome.connections.values()),
                'max': max(c.weight for c in genome.connections.values())
            }
        }
        pass
    
    # def record_game(self, genome, game_data):
    #     """Record a complete game playthrough"""
    #     self.game_recordings.append({
    #         'genome': genome,
    #         'network_structure': self.analyse_network(genome),
    #         'game_data': game_data
    #     })
    #     pass
    
    # def save_analytics(self, filename):
    #     """Save all analytics data to pickle file"""
    #     data = {
    #         'generation_stats': self.generation_stats,
    #         'network_evolution': self.network_evolution,
    #         'species_stats': self.species_stats,
    #         'game_recordings': self.game_recordings,
    #         'config': {
    #             'population_size': POPULATION_SIZE,
    #             'num_generations': N_GENERATIONS,
    #             'network_params': {
    #                 'num_inputs': 40,
    #                 'num_outputs': 4,
    #                 'initial_connections': 0
    #             }
    #         }
    #     }
    #     pass

class DiversityManager:
    def __init__(self, population_size, target_species=10):
        self.population_size = population_size
        self.target_species = target_species
        self.performance_history = []
        self.diversity_history = []
        self.best_performers = []
        self.stagnation_counter = 0
        self.diversity_threshold = 0.1
        
    def update_stats(self, species_stats, generation_stats):
        """Update diversity and performance statistics"""
        # Check if generation_stats is a set and convert it properly
        if isinstance(generation_stats, set):
            # Create a default structure if a set
            current_max = 0
            self.performance_history.append(current_max)
            
            # Create a default diversity measure
            diversity_measure = {
                'species_count': 0,
                'genetic_distance_mean': 0,
                'genetic_distance_std': 0
            }
            if species_stats and isinstance(species_stats, dict):
                if 'species' in species_stats and 'count' in species_stats['species']:
                    diversity_measure['species_count'] = species_stats['species']['count']
                if 'genetic_distance' in species_stats:
                    if 'mean' in species_stats['genetic_distance']:
                        diversity_measure['genetic_distance_mean'] = species_stats['genetic_distance']['mean']
                    if 'std' in species_stats['genetic_distance']:
                        diversity_measure['genetic_distance_std'] = species_stats['genetic_distance']['std']
            
            self.diversity_history.append(diversity_measure)
        else:
            # Original code
            current_max = generation_stats['max_score']
            self.performance_history.append(current_max)
            
            # Track diversity
            diversity_measure = {
                'species_count': species_stats['species']['count'],
                'genetic_distance_mean': species_stats['genetic_distance']['mean'],
                'genetic_distance_std': species_stats['genetic_distance']['std']
            }
            self.diversity_history.append(diversity_measure)
        
        # Check for stagnation
        if len(self.performance_history) > 5:
            recent_max = max(self.performance_history[-5:])
            if current_max <= recent_max:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                
    def should_increase_diversity(self):
        """Check if diversity needs to be increased"""
        if not self.diversity_history:
            return False
            
        current_diversity = self.diversity_history[-1]
        
        # Check for low species count
        if current_diversity['species_count'] < self.target_species * 0.5:
            return True
            
        # Check for low genetic diversity
        if current_diversity['genetic_distance_std'] < self.diversity_threshold:
            return True
            
        # Check for stagnation
        if self.stagnation_counter > 10:
            return True
            
        return False
        
    def get_mutation_adjustments(self):
        """Get adjusted mutation rates based on current state"""
        if not self.should_increase_diversity():
            return None
            
        # Increase mutation rates when diversity is low
        return {
            'weight_mutate_rate': 0.95,
            'weight_mutate_power': 1.0,
            'conn_add_prob': 0.7,
            'node_add_prob': 0.5
        }
        
    def preserve_best_performers(self, population, generation_stats):
        """Preserve best performing individuals"""
        best_score = generation_stats['max_score']
        
        # Find best performer from current generation
        best_genome = max(population.values(), 
                        key=lambda g: g.fitness if hasattr(g, 'fitness') else 0)
        
        # Add to best performers if good enough
        if not self.best_performers or best_score >= max(g.fitness for g in self.best_performers):
            self.best_performers.append(best_genome)
            # Keep only top 5 best performers
            self.best_performers = sorted(self.best_performers,
                                    key=lambda g: g.fitness if hasattr(g, 'fitness') else 0,
                                    reverse=True)[:5]
            
    def get_preserved_genomes(self):
        """Get preserved genomes for reintroduction"""
        return self.best_performers.copy()