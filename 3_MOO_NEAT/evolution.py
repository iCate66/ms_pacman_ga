import neat
from deap import tools, creator
import numpy as np
import pickle
import itertools
import traceback
from game_utils import manhattan_distance
import os
import json
import random
import copy

from config import (
    N_GENERATIONS, DIVERSITY_THRESHOLD, NOVELTY_WEIGHT,
    ARCHIVE_SIZE, ELITE_SIZE, K_NEAREST_NEIGHBORS,
    CHECKPOINT_DIR, logger, MAX_POPULATION_SIZE, POPULATION_GROWTH_RATE
)
from classes import NEATIndividual, DiversityManager
from evaluation import parallel_eval_genomes

# Global trackers for the best-ever agents across all generations
BEST_EVER = {
    'overall': {
        'genome': None,
        'score': 0,
        'generation': 0,
        'island': None,
        'seed': None
    },
    'progression': {
        'genome': None,
        'score': 0,
        'generation': 0,
        'island': None,
        'seed': None
    },
    'survival': {
        'genome': None,
        'score': 0,
        'generation': 0,
        'island': None,
        'seed': None
    },
    'efficiency': {
        'genome': None,
        'score': 0,
        'generation': 0,
        'island': None,
        'seed': None
    }
}

# Create islands
def create_island_populations(config, num_islands=5):
    """Create multiple isolated populations (islands)"""
    islands = []
    for i in range(num_islands):
        island_pop = neat.Population(config)
        islands.append({
            'population': island_pop,
            'best_fitness': 0,
            'stagnation_counter': 0,
            'last_improvement': 0
        })
    return islands

def migrate_between_islands_enhanced(islands, config, migration_rate=0.08, migration_interval=4):
    """Enhanced migration between islands with better diversity preservation"""
    if len(islands) < 2:
        return islands
    
    # Keep track of best genomes before migration
    pre_migration_best = {}
    for i, island in enumerate(islands):
        pop = island['population'].population
        best_genome = max(pop.values(), key=lambda g: getattr(g, 'total_score', 0) if hasattr(g, 'total_score') else 0)
        pre_migration_best[i] = (best_genome, getattr(best_genome, 'total_score', 0))
    
    # Convert junction_points to lists before migration
    for island in islands:
        for genome_id, genome in island['population'].population.items():
            if hasattr(genome, 'metrics') and 'junction_points' in genome.metrics:
                if isinstance(genome.metrics['junction_points'], set):
                    genome.metrics['junction_points'] = list(genome.metrics['junction_points'])
    
    # For each island, select individuals to migrate
    for i, source_island in enumerate(islands):
        # Skip migration if this island is underperforming
        if source_island['stagnation_counter'] > 5:
            continue
            
        source_pop = source_island['population'].population
        source_genomes = list(source_pop.values())
        
        # Skip if too few genomes to work with
        if len(source_genomes) < 2:
            continue
        
        # Helper function for genome scoring
        def get_score(genome):
            if hasattr(genome, 'total_score') and genome.total_score is not None:
                return genome.total_score
            if hasattr(genome, 'fitness') and genome.fitness is not None:
                return genome.fitness
            return 0
        
        # Sort genomes by score
        source_genomes.sort(key=get_score, reverse=True)
        
        # Select migrants using mixed strategy
        migrants = []
        num_migrants = int(len(source_genomes) * migration_rate)
        num_random = max(1, int(num_migrants * 0.3))  # 30% random individuals
        
        # Add top performers (50% of migrants)
        elite_count = max(1, (num_migrants - num_random) // 2)
        migrants.extend(source_genomes[:elite_count])
        
        # Add tournament selection winners (20% of migrants)
        tournament_count = num_migrants - elite_count - num_random
        remaining_genomes = [g for g in source_genomes[elite_count:] if g not in migrants]
        
        if remaining_genomes and tournament_count > 0:
            for _ in range(tournament_count):
                candidates = random.sample(remaining_genomes, min(3, len(remaining_genomes)))
                winner = max(candidates, key=get_score)
                migrants.append(winner)
                remaining_genomes = [g for g in remaining_genomes if g != winner]
                if not remaining_genomes:
                    break
        
        # Add random new genomes (30% of migrants)
        for _ in range(num_random):
            new_genome = neat.DefaultGenome(0)
            new_genome.configure_new(config.genome_config)
            for _ in range(3):  # Add a few mutations
                new_genome.mutate(config.genome_config)
            migrants.append(new_genome)
        
        # Determine target islands
        other_islands = list(range(len(islands)))
        other_islands.remove(i)
        if not other_islands or not migrants:
            continue
        
        # Calculate score differences to determine migration paths
        island_scores = []
        source_score = source_island['best_fitness']
        for idx in other_islands:
            target_pop = islands[idx]['population'].population
            if target_pop:
                max_score = max([get_score(g) for g in target_pop.values()], default=0)
                island_scores.append((idx, max_score))
            else:
                island_scores.append((idx, 0))
        
        # Sort by score difference (prefer more different islands)
        island_scores.sort(key=lambda x: abs(x[1] - source_score), reverse=True)
        
        # Distribute migrants across islands
        migrants_per_island = max(1, len(migrants) // len(other_islands))
        remaining_migrants = migrants.copy()
        
        for target_idx, _ in island_scores:
            # Assign migrants to this island
            island_migrants = remaining_migrants[:migrants_per_island]
            remaining_migrants = remaining_migrants[migrants_per_island:]
            
            if not island_migrants:
                continue
                
            target_island = islands[target_idx]
            target_pop = target_island['population'].population
            
            # Add migrants to target population
            for migrant in island_migrants:
                migrant_copy = neat.DefaultGenome(0)
                
                # Handle fitness for NEAT's assertion
                original_fitness = None
                if hasattr(migrant, 'multi_fitness'):
                    original_fitness = migrant.multi_fitness
                migrant.fitness = 1.0  # Temporary dummy fitness
                
                # Copy the genome
                migrant_copy.configure_crossover(migrant, migrant, config.genome_config)
                
                # Restore original fitness
                if original_fitness is not None:
                    migrant.multi_fitness = original_fitness
                    migrant_copy.multi_fitness = original_fitness
                migrant.fitness = None  # Remove temporary fitness
                
                # Copy other important attributes
                if hasattr(migrant, 'total_score'):
                    migrant_copy.total_score = migrant.total_score
                if hasattr(migrant, 'metrics'):
                    migrant_copy.metrics = copy.deepcopy(migrant.metrics) if isinstance(migrant.metrics, dict) else migrant.metrics
                if hasattr(migrant, 'objective_values'):
                    migrant_copy.objective_values = copy.deepcopy(migrant.objective_values)
                if hasattr(migrant, 'is_elite'):
                    migrant_copy.is_elite = migrant.is_elite
                
                # Apply mild mutation only to non-elite genomes
                if not hasattr(migrant, 'is_elite') or not migrant.is_elite:
                    if random.random() < 0.3:  # 30% chance of mutation
                        original_rate = config.genome_config.weight_mutate_rate
                        config.genome_config.weight_mutate_rate *= 0.5  # Reduced rate
                        migrant_copy.mutate(config.genome_config)
                        config.genome_config.weight_mutate_rate = original_rate
                
                # Add to target population
                new_id = max(target_pop.keys()) + 1 if target_pop else 1
                migrant_copy.key = new_id
                target_pop[new_id] = migrant_copy
    
    # Ensure best genomes are preserved after migration
    for i, (best_genome, best_score) in pre_migration_best.items():
        if best_score <= 0:  # Skip if best score is invalid
            continue
            
        island = islands[i]
        pop = island['population'].population
        current_best = max(pop.values(), key=lambda g: getattr(g, 'total_score', 0) if hasattr(g, 'total_score') else 0)
        current_score = getattr(current_best, 'total_score', 0)
        
        # If migration resulted in losing the best genome, add it back
        if current_score < best_score * 0.9:
            logger.info(f"Re-adding best genome to island {i} after migration (score: {best_score})")
            new_id = max(pop.keys()) + 1
            best_copy = neat.DefaultGenome(new_id)
            
            # Handle fitness for NEAT's assertion
            original_fitness = None
            if hasattr(best_genome, 'multi_fitness'):
                original_fitness = best_genome.multi_fitness
            best_genome.fitness = 1.0  # Temporary dummy fitness
            
            # Copy the genome
            best_copy.configure_crossover(best_genome, best_genome, config.genome_config)
            
            # Restore original fitness
            if original_fitness is not None:
                best_genome.multi_fitness = original_fitness
                best_copy.multi_fitness = original_fitness
            best_genome.fitness = None  # Remove temporary fitness
            
            # Copy all relevant attributes
            if hasattr(best_genome, 'total_score'):
                best_copy.total_score = best_genome.total_score
            if hasattr(best_genome, 'metrics'):
                best_copy.metrics = copy.deepcopy(best_genome.metrics) if isinstance(best_genome.metrics, dict) else best_genome.metrics
            if hasattr(best_genome, 'objective_values'):
                best_copy.objective_values = copy.deepcopy(best_genome.objective_values)
            if hasattr(best_genome, 'is_elite'):
                best_copy.is_elite = best_genome.is_elite
            
            # Add to population
            pop[new_id] = best_copy
    
    return islands

# Calculate k-nearest neighbour distances
def calculate_behavioral_distance(behaviour1, behaviour2):
    """Enhanced behavioral distance calculation that considers more factors"""
    # Basic position-based distance (using existing code)
    pos1 = set(behaviour1.get('visited_positions', []))
    pos2 = set(behaviour2.get('visited_positions', []))
    
    if not pos1 or not pos2:
        return 1.0
    
    # Jaccard distance for positions
    intersection = len(pos1.intersection(pos2))
    union = len(pos1.union(pos2))
    position_diff = 1 - (intersection / union if union > 0 else 0)
    
    # Action sequence similarity
    action_counts1 = behaviour1.get('action_counts', {0:0, 1:0, 2:0, 3:0})
    action_counts2 = behaviour2.get('action_counts', {0:0, 1:0, 2:0, 3:0})
    
    total_actions1 = sum(action_counts1.values()) or 1
    total_actions2 = sum(action_counts2.values()) or 1
    
    # Normalise action distributions
    action_dist1 = [action_counts1.get(i, 0) / total_actions1 for i in range(4)]
    action_dist2 = [action_counts2.get(i, 0) / total_actions2 for i in range(4)]
    
    # Calculate action distribution distance
    action_diff = sum(abs(a1 - a2) for a1, a2 in zip(action_dist1, action_dist2)) / 4
    
    # Ghost interaction patterns
    ghost_avoid1 = behaviour1.get('ghost_avoidance_score', 0)
    ghost_avoid2 = behaviour2.get('ghost_avoidance_score', 0)
    ghost_diff = abs(ghost_avoid1 - ghost_avoid2)
    
    # Power pill strategy
    pp_strat1 = behaviour1.get('ghost_points', 0) / max(1, behaviour1.get('total_score', 1))
    pp_strat2 = behaviour2.get('ghost_points', 0) / max(1, behaviour2.get('total_score', 1))
    pp_diff = abs(pp_strat1 - pp_strat2)
    
    # Decision points (junctions)
    junction_visits1 = behaviour1.get('times_at_junction', 0) / max(1, len(behaviour1.get('visited_positions', [])))
    junction_visits2 = behaviour2.get('times_at_junction', 0) / max(1, len(behaviour2.get('visited_positions', [])))
    junction_diff = abs(junction_visits1 - junction_visits2)
    
    # Combined behavioral distance with weighted components
    return (0.3 * position_diff +     # Spatial coverage (reduced weight)
            0.2 * action_diff +       # Action patterns
            0.2 * ghost_diff +        # Ghost interaction
            0.15 * pp_diff +          # Power pill strategy
            0.15 * junction_diff)     # Junction decision making

def calculate_novelty(genome, population, archive):
    """Calculate novelty score for a genome"""
    if not hasattr(genome, 'metrics'):
        return 0.0
    
    all_behaviours = []
    for g in population:
        if hasattr(g, 'metrics'):
            all_behaviours.append(g.metrics)
    all_behaviours.extend(archive)
    
    if not all_behaviours:
        return 1.0
    
    # Calculate k-nearest neighbour distances
    k = min(15, len(all_behaviours))
    distances = []
    for behaviour in all_behaviours:
        dist = calculate_behavioral_distance(genome.metrics, behaviour)
        distances.append(dist)
    
    distances.sort()
    return sum(distances[:k]) / k

# To centralise population update logic
# def update_population(pop, selected, genome_indexer):
#     """Update population with selected individuals and force proper speciation"""
#     # Create new population dictionary
#     new_pop = {}
    
#     # Assign new IDs and update population
#     for i, ind in enumerate(selected):
#         genome = ind.genome
#         new_id = next(genome_indexer)
#         genome.key = new_id
#         new_pop[new_id] = genome
    
#     # Clear existing species
#     pop.species.species = {}
    
#     # Update population
#     pop.population = new_pop
    
#     # Force respeciation with adjusted parameters
#     pop.config.species_set_config.compatibility_threshold = 2.5
#     pop.species.speciate(pop.config, pop.population, generation=0)
    
#     # Check if necessary to adjust speciation threshold
#     num_species = len(pop.species.species)
#     if num_species < 5:  # Too few species
#         pop.config.species_set_config.compatibility_threshold *= 0.9
#     elif num_species > 20:  # Too many species
#         pop.config.species_set_config.compatibility_threshold *= 1.1
    
#     return pop

# def log_generation_stats(generation, population, scores, config):
#     """Log detailed statistics for each generation"""
#     stats = calculate_population_statistics(population, config)
    
#     # Basic statistics
#     logger.info(f"\nGeneration {generation}")
#     logger.info(f"Population size: {len(population.population)}")
    
#     # Score statistics
#     avg_score = np.mean(scores) if scores else 0
#     max_score = max(scores) if scores else 0
#     min_score = min(scores) if scores else 0
#     logger.info(f"Score stats - Avg: {avg_score:.1f}, Max: {max_score:.1f}, Min: {min_score:.1f}")
    
#     # Multi-objective fitness statistics
#     progression_values = []
#     survival_values = []
#     efficiency_values = []
    
#     for genome in population.population.values():
#         if hasattr(genome, 'multi_fitness') and genome.multi_fitness.values:
#             progression_values.append(genome.multi_fitness.values[0])
#             survival_values.append(genome.multi_fitness.values[1])
#             efficiency_values.append(genome.multi_fitness.values[2])
    
#     if progression_values:
#         logger.info("\nObjective Stats:")
#         logger.info(f"Progression - Avg: {np.mean(progression_values):.3f}, "
#                    f"Max: {max(progression_values):.3f}, "
#                    f"Min: {min(progression_values):.3f}")
#         logger.info(f"Survival    - Avg: {np.mean(survival_values):.3f}, "
#                    f"Max: {max(survival_values):.3f}, "
#                    f"Min: {min(survival_values):.3f}")
#         logger.info(f"Efficiency  - Avg: {np.mean(efficiency_values):.3f}, "
#                    f"Max: {max(efficiency_values):.3f}, "
#                    f"Min: {min(efficiency_values):.3f}")
    
#     # Genetic diversity statistics
#     if stats:
#         logger.info(f"\nMean genetic distance {stats['genetic_distance']['mean']:.3f}, " +
#                    f"standard deviation {stats['genetic_distance']['std']:.3f}")
        
#         # Species information
#         logger.info(f"Number of species: {stats['species']['count']}")
#         if stats['species']['sizes']:
#             logger.info(f"Species size stats - Avg: {stats['species']['avg_size']:.1f}, " +
#                        f"Max: {stats['species']['max_size']}, " +
#                        f"Min: {stats['species']['min_size']}")
        
#         # Log species size distribution
#         size_counts = {}
#         for size in stats['species']['sizes']:
#             size_counts[size] = size_counts.get(size, 0) + 1
#         logger.info(f"Species size distribution: {dict(sorted(size_counts.items()))}")

def calculate_population_statistics(population, config):
    """Calculate detailed statistics for the population"""
    if not population:
        return None
        
    # Convert population to list of genomes if it's a dictionary
    if isinstance(population, dict):
        genomes = list(population.values())
    else:
        genomes = list(population.population.values())
    
    # Add multi-objective statistics
    fitness_stats = {
        'progression': [],
        'survival': [],
        'efficiency': []
    }
    
    for genome in genomes:
        if hasattr(genome, 'multi_fitness') and genome.multi_fitness.values:
            fitness_stats['progression'].append(genome.multi_fitness.values[0])
            fitness_stats['survival'].append(genome.multi_fitness.values[1])
            fitness_stats['efficiency'].append(genome.multi_fitness.values[2])
    
    # Genetic distance calculations
    distances = []
    for i in range(len(genomes)):
        for j in range(i + 1, len(genomes)):
            try:
                distance = genomes[i].distance(genomes[j], config.genome_config)
                distances.append(distance)
            except Exception as e:
                logger.debug(f"Error calculating distance: {e}")
                continue
    
    mean_distance = np.mean(distances) if distances else 0
    std_distance = np.std(distances) if distances else 0
    
    # Species statistics
    if hasattr(population, 'species'):
        species = population.species.species
    else:
        species = {s.key: s for s in population.species.species.values()}
    
    species_sizes = [len(s.members) for s in species.values()]
    
    # Calculate species fitness
    species_fitness = []
    for s in species.values():
        species_members = [m for m in s.members if hasattr(m, 'fitness')]
        if species_members:
            species_fitness.append(np.mean([m.fitness for m in species_members]))
    
    return {
        'genetic_distance': {
            'mean': mean_distance,
            'std': std_distance
        },
        'species': {
            'count': len(species),
            'sizes': species_sizes,
            'avg_size': np.mean(species_sizes) if species_sizes else 0,
            'max_size': max(species_sizes) if species_sizes else 0,
            'min_size': min(species_sizes) if species_sizes else 0,
            'fitness': {
                'mean': np.mean(species_fitness) if species_fitness else 0,
                'max': max(species_fitness) if species_fitness else 0,
                'min': min(species_fitness) if species_fitness else 0
            }
        },
        'fitness_stats': fitness_stats
    }

def save_checkpoint_with_islands(gen, islands, config, generation_history, 
                               diversity_manager, elite_pool, current_best_score, 
                               diversity_metrics, best_ever=None):
    """Save a checkpoint of the current evolution state with island model"""
    # Convert any sets in junction_points to lists before saving
    for island in islands:
        for genome_id, genome in island['population'].population.items():
            if hasattr(genome, 'metrics') and 'junction_points' in genome.metrics:
                if isinstance(genome.metrics['junction_points'], set):
                    genome.metrics['junction_points'] = list(genome.metrics['junction_points'])
    
    # Get best genomes by objective from all islands
    best_genomes_by_objective = {}
    try:
        with open('best_genomes_by_objective.txt', 'r') as f:
            for line in f:
                parts = line.strip().split(': ', 1)
                if len(parts) == 2:
                    objective = parts[0]
                    data_str = parts[1]
                    
                    # Parse values from data string
                    data = {}
                    for item in data_str.split(', '):
                        key, value = item.split('=', 1)
                        data[key] = value
                    
                    best_genomes_by_objective[objective] = data
    except Exception as e:
        logger.error(f"Error reading best genomes file: {e}")
    
    checkpoint = {
        'generation': gen,
        'islands': islands,
        'config': config,
        'generation_history': generation_history,
        'diversity_manager': diversity_manager,
        'elite_pool': elite_pool,
        'current_best_score': current_best_score,
        'diversity_metrics': diversity_metrics,
        'best_genomes_by_objective': best_genomes_by_objective,
        'best_ever': best_ever
    }
    
    filename = f'neat_checkpoint_gen_{gen}.pkl'
    try:
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Saved checkpoint at generation {gen}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

# Save final results pkl for analysis
def save_final_results(results, filename='neat_final_results.pkl'):
    """Save final results to a pickle file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        # Try alternative save location
        backup_filename = f'backup_{filename}'
        try:
            with open(backup_filename, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Results saved to backup file: {backup_filename}")
        except Exception as e2:
            logger.error(f"Failed to save backup: {e2}")

# # load a checkpoint and restore the evolution state
# def load_checkpoint(filename):
#     """Load a checkpoint and restore the evolution state"""
#     with open(filename, 'rb') as f:
#         checkpoint = pickle.load(f)
    
#     # Restore NEAT configuration
#     config = checkpoint['config']
    
#     # Create new population with restored config and population
#     pop = neat.Population(config, checkpoint['population'])
    
#     # Restore species
#     pop.species = checkpoint['species']
    
#     # Restore genome indexer if it exists
#     if checkpoint.get('genome_indexer') is not None:
#         pop.genome_indexer = checkpoint['genome_indexer']
    
#     # Return all restored components
#     return (pop, 
#             checkpoint['generation'], 
#             checkpoint['generation_history'],
#             checkpoint['diversity_manager'])


def run_neat_parallel(config_path):
    """Run NEAT with parallel evaluation, island model, and NSGA-II selection"""
    # Import global tracker
    global BEST_EVER

    # --- Phase 0: Initialisation ---
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # # mutation operators test
    # def test_mutation_operators(config):
    #     """Test that structural mutation operators are working"""
    #     logger.info("Testing mutation operators...")
        
    #     # Create a test genome with minimal structure
    #     test_genome = neat.DefaultGenome(0)
    #     test_genome.configure_new(config.genome_config)
        
    #     # Add a node
    #     original_nodes = len(test_genome.nodes)
    #     test_genome.mutate_add_node(config.genome_config)
    #     new_nodes = len(test_genome.nodes)
        
    #     # Add connections
    #     original_conns = len(test_genome.connections)
    #     for _ in range(5):  # Try multiple times
    #         test_genome.mutate_add_connection(config.genome_config)
    #     new_conns = len(test_genome.connections)
        
    #     logger.info(f"Mutation test: Nodes {original_nodes} -> {new_nodes}, Connections {original_conns} -> {new_conns}")
    #     if new_nodes <= original_nodes:
    #         logger.error("NODE MUTATION NOT WORKING!!!!!")
    #     if new_conns <= original_conns:
    #         logger.error("CONNECTION MUTATION NOT WORKING!!!!!")
    # # Call the test function before creating island populations
    # test_mutation_operators(config)
    
    # Create island populations
    num_islands = 5
    islands = create_island_populations(config, num_islands)

    # Bootstrapping to introduce complexity
    logger.info("Bootstrapping some networks with initial complexity")
    for island_idx, island in enumerate(islands):
        pop = island['population']
        population_size = len(pop.population)
        
        # For 30% of genomes, force add hidden nodes and connections
        bootstrap_count = int(population_size * 0.3)  # 30% of population
        bootstrapped_genomes = 0
        
        for genome_id, genome in list(pop.population.items())[:bootstrap_count]:
            # Add 3-5 hidden nodes
            hidden_nodes_to_add = random.randint(3, 5)
            for _ in range(hidden_nodes_to_add):
                genome.mutate_add_node(config.genome_config)
            
            # Add 8-15 connections
            connections_to_add = random.randint(8, 15)
            for _ in range(connections_to_add):
                genome.mutate_add_connection(config.genome_config)
            
            bootstrapped_genomes += 1
        
        logger.info(f"Bootstrapped {bootstrapped_genomes} genomes in Island {island_idx+1}")

    # Initialise trackers and managers
    # diversity_manager = DiversityManager(config.pop_size)
    diversity_manager = DiversityManager(config.pop_size, target_species=15)  # Increased target species
    generation_history = []
    elite_pool = []
    absolute_best_genomes = []  # Track best genomes across ALL generations
    current_best_score = 0
    genome_indexer = itertools.count(start=1)
    diversity_metrics = {
        'genetic_distance_history': [],
        'species_count_history': [],
        'mutation_adjustments_made': 0,
        'preserved_genomes_used': 0,
        'improvement_after_diversity': 0
    }
    
    for gen in range(N_GENERATIONS):
        logger.info(f"\nGeneration {gen}")
        
        # Process each island separately
        all_scores = []
        all_species_stats = []

        # Variables to track if mutation rates were modified
        mutation_rates_modified = False
        original_mutation_rate = None
        original_node_add = None

        # Apply adaptive mutation rates based on stagnation
        # If no improvement for several generations, increase mutation
        stagnant = all(island['stagnation_counter'] > 3 for island in islands)
        if stagnant and gen > 10:
            logger.info("Detected stagnation - increasing mutation rates temporarily")
            original_mutation_rate = config.genome_config.weight_mutate_rate
            original_node_add = config.genome_config.node_add_prob
            
            # Temporarily increase mutation rates
            config.genome_config.weight_mutate_rate = min(0.98, original_mutation_rate * 1.5)
            config.genome_config.node_add_prob = min(0.9, original_node_add * 1.5)
            
            # Track adjustment
            diversity_metrics['mutation_adjustments_made'] += 1
            mutation_rates_modified = True
        
        for island_idx, island in enumerate(islands):
            pop = island['population']
            logger.info(f"Processing Island {island_idx+1}/{num_islands}")
            
            # --- Phase 1: Population Evaluation ---
            parallel_eval_genomes(list(pop.population.items()), config)
            
            # Collect island stats
            scores = [genome.total_score for genome in pop.population.values() 
                     if hasattr(genome, 'total_score')]
            all_scores.extend(scores)

            # Calculate average complexity stats for this island
            avg_hidden = sum(len([n for n in g.nodes.keys() 
                                if n not in config.genome_config.input_keys and 
                                n not in config.genome_config.output_keys]) 
                            for g in pop.population.values()) / max(1, len(pop.population))

            avg_connections = sum(len([c for c in g.connections.values() if c.enabled]) 
                                for g in pop.population.values()) / max(1, len(pop.population))

            # Get best network complexity
            best_genome = max(pop.population.values(), 
                             key=lambda g: g.total_score if hasattr(g, 'total_score') else 0)
            best_hidden = len([n for n in best_genome.nodes.keys() 
                              if n not in config.genome_config.input_keys and 
                              n not in config.genome_config.output_keys])
            best_connections = len([c for c in best_genome.connections.values() if c.enabled])
            
            logger.info(f"Island {island_idx+1} Structure - Avg: {avg_hidden:.2f} hidden, {avg_connections:.2f} connections | Best: {best_hidden} hidden, {best_connections} connections")
            
            # species
            species_stats = calculate_population_statistics(pop, config)
            all_species_stats.append(species_stats)
            
            gen_stats = {
                'avg_score': np.mean(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
                'min_score': min(scores) if scores else 0,
                'population_size': len(pop.population)
            }
            
            # Update island status and track best performers
            current_best = max(pop.population.values(), 
                              key=lambda g: g.total_score if hasattr(g, 'total_score') else 0)
            current_score = getattr(current_best, 'total_score', 0)
            
            # Update island tracking
            if current_score > island['best_fitness']:
                island['best_fitness'] = current_score
                island['last_improvement'] = gen
                island['stagnation_counter'] = 0
            else:
                island['stagnation_counter'] += 1
            
            # Update best-ever agents (once per island)
            update_best_ever_agents(pop.population.values(), gen, island_idx, config)
            
            # --- Phase 2: Diversity Management for this island ---
            diversity_manager.update_stats(species_stats, gen_stats)
            
            # Update global elite pool
            if current_score > current_best_score:
                current_best_score = current_score
                elite_pool.append(current_best)
                if len(elite_pool) > ELITE_SIZE:
                    elite_pool.sort(key=lambda g: g.total_score if hasattr(g, 'total_score') else 0,
                                    reverse=True)
                    elite_pool = elite_pool[:ELITE_SIZE]
            
            # --- Phase 3: Island Population Evolution ---
            # Create selection pool with appropriate diversity
            population_list = []
            for genome_id, genome in pop.population.items():
                neat_ind = NEATIndividual(genome)
                
                # Ensure all individuals have valid, complete fitness values
                if hasattr(genome, 'multi_fitness') and hasattr(genome.multi_fitness, 'values') and len(genome.multi_fitness.values) == 3:
                    neat_ind.fitness = genome.multi_fitness
                else:
                    # Create a default fitness with all three objectives
                    # Use 0.0 for missing values
                    neat_ind.fitness = creator.FitnessMulti((0.0, 0.0, 0.0))
                
                population_list.append(neat_ind)

            # Maintaining preserved genomes
            preserved_genomes = diversity_manager.get_preserved_genomes()
            for genome in preserved_genomes:
                elite_ind = NEATIndividual(genome)
                
                # Ensure valid fitness
                if hasattr(genome, 'multi_fitness') and hasattr(genome.multi_fitness, 'values') and len(genome.multi_fitness.values) == 3:
                    elite_ind.fitness = genome.multi_fitness
                else:
                    elite_ind.fitness = creator.FitnessMulti((0.0, 0.0, 0.0))
                
                population_list.append(elite_ind)

            # Maintaining elite pool
            for elite in elite_pool:
                elite_ind = NEATIndividual(elite)
                
                # Ensure valid fitness
                if hasattr(elite, 'multi_fitness') and hasattr(elite.multi_fitness, 'values') and len(elite.multi_fitness.values) == 3:
                    elite_ind.fitness = elite.multi_fitness
                else:
                    elite_ind.fitness = creator.FitnessMulti((0.0, 0.0, 0.0))
                
                population_list.append(elite_ind)

            # Maintaining random individuals
            if diversity_manager.should_increase_diversity():
                num_random = max(5, int(len(pop.population) * 0.1))
                for _ in range(num_random):
                    new_genome = neat.DefaultGenome(next(genome_indexer))
                    new_genome.configure_new(config.genome_config)
                    # Add random mutations to make viable
                    for _ in range(5):
                        new_genome.mutate(config.genome_config)
                    
                    random_ind = NEATIndividual(new_genome)
                    # Always set a default fitness for random individuals
                    random_ind.fitness = creator.FitnessMulti((0.0, 0.0, 0.0))
                    population_list.append(random_ind)
                
            # Calculate island population size
            target_pop_size = min(
                MAX_POPULATION_SIZE // num_islands,
                int(len(pop.population) * POPULATION_GROWTH_RATE)
            )
            
            # Perform NSGA-II selection for this island
            selected = tools.selNSGA2(population_list, target_pop_size)
            
            # Create new population for this island
            new_pop = {}
            for neat_ind in selected:
                genome = neat_ind.genome
                new_key = next(genome_indexer)
                genome.key = new_key
                new_pop[new_key] = genome
                
            # Clear species before updating population
            pop.species.species = {}
                
            # Update island population
            pop.population = new_pop
                
            # Perform speciation with clean slate
            pop.species.speciate(config, pop.population, gen)
            
            # Log island statistics
            logger.info(f"Island {island_idx+1} - Scores: Avg={gen_stats['avg_score']:.1f}, Max={gen_stats['max_score']:.1f}")

            # If this is a stagnant island, try injection of novel genomes
            if island['stagnation_counter'] > 7:
                logger.info(f"Island {island_idx+1} is stagnant - injecting novel genomes")
                
                # Create 5 completely new random genomes
                for _ in range(5):
                    new_genome = neat.DefaultGenome(next(genome_indexer))
                    new_genome.configure_new(config.genome_config)
                    pop.population[new_genome.key] = new_genome
                
                # Reset stagnation counter
                island['stagnation_counter'] = 0

        # Elite preservation: after processing all islands in each generation, collect all genomes
        all_genomes = []
        for island in islands:
            all_genomes.extend(island['population'].population.values())

        # Find current best genomes
        current_best_genomes = sorted(all_genomes, 
                                key=lambda g: getattr(g, 'total_score', 0) if hasattr(g, 'total_score') else 0,
                                reverse=True)[:ELITE_SIZE]

        # Update absolute best collection
        for genome in current_best_genomes:
            if genome.total_score > 0:  # Only consider valid scores
                # Check if this genome is better than the worst in the collection
                if not absolute_best_genomes or genome.total_score > min([g.total_score for g in absolute_best_genomes]):
                    # Create a deep copy to preserve this genome exactly as it is now
                    genome_copy = neat.DefaultGenome(0)
                    
                    # Store original multi-fitness and set temporary fitness to pass NEAT's assertion
                    original_multi_fitness = None
                    if hasattr(genome, 'multi_fitness'):
                        original_multi_fitness = genome.multi_fitness
                        genome.fitness = 1.0
                    
                    # Perform the crossover (passes assertion)
                    genome_copy.configure_crossover(genome, genome, config.genome_config)
                    
                    # Restore original multi-fitness and remove temporary fitness
                    if original_multi_fitness is not None:
                        genome.multi_fitness = original_multi_fitness
                        genome_copy.multi_fitness = copy.deepcopy(original_multi_fitness)
                        genome.fitness = None
                    
                    # Copy all other relevant attributes
                    if hasattr(genome, 'total_score'):
                        genome_copy.total_score = genome.total_score
                    if hasattr(genome, 'metrics'):
                        genome_copy.metrics = copy.deepcopy(genome.metrics)
                    if hasattr(genome, 'objective_values'):
                        genome_copy.objective_values = copy.deepcopy(genome.objective_values)
                    if hasattr(genome, 'seed'):
                        genome_copy.seed = genome.seed
                    
                    # Flag as an elite genome
                    genome_copy.is_elite = True
                    
                    absolute_best_genomes.append(genome_copy)

        # Keep only top ELITE_SIZE across all generations
        if absolute_best_genomes:
            absolute_best_genomes = sorted(absolute_best_genomes,
                                        key=lambda g: g.total_score,
                                        reverse=True)[:ELITE_SIZE]

        # Ensure these are preserved in next generation by adding to each island
        if absolute_best_genomes and gen < N_GENERATIONS - 1:  # Not needed in final generation
            logger.info(f"Preserving {len(absolute_best_genomes)} elite genomes (best score: {absolute_best_genomes[0].total_score})")
            
            for i, island in enumerate(islands):
                # Distribute elites evenly across islands
                elites_per_island = max(1, len(absolute_best_genomes) // len(islands))
                start_idx = i * elites_per_island
                end_idx = min(start_idx + elites_per_island, len(absolute_best_genomes))
                
                for elite_idx in range(start_idx, end_idx):
                    if elite_idx < len(absolute_best_genomes):
                        best = absolute_best_genomes[elite_idx]
                        
                        # Add exact copy to population
                        best_copy = neat.DefaultGenome(next(genome_indexer))
                        
                        # Store original multi-fitness and set temporary fitness to pass NEAT's assertion
                        original_multi_fitness = None
                        if hasattr(best, 'multi_fitness'):
                            original_multi_fitness = best.multi_fitness
                            best.fitness = 1.0
                        
                        # Perform the crossover (passes assertion)
                        best_copy.configure_crossover(best, best, config.genome_config)
                        
                        # Restore original multi-fitness and remove temporary fitness
                        if original_multi_fitness is not None:
                            best.multi_fitness = original_multi_fitness
                            best_copy.multi_fitness = copy.deepcopy(original_multi_fitness)
                            best.fitness = None
                        
                        # Copy all attributes
                        if hasattr(best, 'total_score'):
                            best_copy.total_score = best.total_score
                        if hasattr(best, 'metrics'):
                            best_copy.metrics = copy.deepcopy(best.metrics)
                        if hasattr(best, 'objective_values'):
                            best_copy.objective_values = copy.deepcopy(best.objective_values)
                        if hasattr(best, 'seed'):
                            best_copy.seed = best.seed
                        
                        # Mark as elite to protect from modification
                        best_copy.is_elite = True
                        
                        # Add to island population
                        island['population'].population[best_copy.key] = best_copy
            
        # --- Phase 4: Cross-Island Operations ---
        # Perform migration every X generations
        if gen % 10 == 0 and gen > 0:
            islands = migrate_between_islands_enhanced(islands, 
                                config,  
                                migration_rate=0.08, 
                                migration_interval=4)
            logger.info("Performed migration between islands")

        # Reset mutation rates if they were temporarily increased
        if mutation_rates_modified:
            config.genome_config.weight_mutate_rate = original_mutation_rate
            config.genome_config.node_add_prob = original_node_add
            logger.info("Reset mutation rates to original values")


         # Add overall stats after processing all islands
        logger.info("\nOverall Network Complexity:")
        for island_idx, island in enumerate(islands):
            pop = island['population']
            all_genomes = list(pop.population.values())
            
            # Skip if no genomes
            if not all_genomes:
                continue
                
            # Count non-zero complexity genomes
            complex_genomes = [g for g in all_genomes if len([n for n in g.nodes.keys() 
                                                           if n not in config.genome_config.input_keys and 
                                                           n not in config.genome_config.output_keys]) > 0]
            
            complex_pct = (len(complex_genomes) / len(all_genomes)) * 100 if all_genomes else 0
            logger.info(f"  Island {island_idx+1}: {len(complex_genomes)}/{len(all_genomes)} genomes ({complex_pct:.1f}%) have hidden nodes")
            

        # --- Phase 5: Statistics and Checkpointing ---
        # Calculate overall stats
        overall_avg_score = np.mean(all_scores) if all_scores else 0
        overall_max_score = max(all_scores) if all_scores else 0
        
        # Log overall statistics
        logger.info(f"Overall - Avg: {overall_avg_score:.1f}, Max: {overall_max_score:.1f}")
        
        # Get best score across all islands
        island_best_scores = [island['best_fitness'] for island in islands]
        generation_best_score = max(island_best_scores)

        # Extract objective values from all islands
        all_progression_values = []
        all_survival_values = []
        all_efficiency_values = []

        for island_idx, island in enumerate(islands):
            pop = island['population']
            for genome in pop.population.values():
                if hasattr(genome, 'multi_fitness') and hasattr(genome.multi_fitness, 'values'):
                    if len(genome.multi_fitness.values) >= 3:
                        all_progression_values.append(genome.multi_fitness.values[0])
                        all_survival_values.append(genome.multi_fitness.values[1])
                        all_efficiency_values.append(genome.multi_fitness.values[2])
        
        # Store generation data with enhanced statistics including objective stats
        generation_data = {
            'generation': gen,
            'avg_score': overall_avg_score,
            'max_score': overall_max_score,
            'island_scores': [island['best_fitness'] for island in islands],
            'island_stagnation': [island['stagnation_counter'] for island in islands],
            'num_islands': num_islands,
            'objective_stats': {
                'progression': {
                    'mean': np.mean(all_progression_values) if all_progression_values else 0,
                    'max': max(all_progression_values) if all_progression_values else 0,
                    'min': min(all_progression_values) if all_progression_values else 0
                },
                'survival': {
                    'mean': np.mean(all_survival_values) if all_survival_values else 0,
                    'max': max(all_survival_values) if all_survival_values else 0,
                    'min': min(all_survival_values) if all_survival_values else 0
                },
                'efficiency': {
                    'mean': np.mean(all_efficiency_values) if all_efficiency_values else 0,
                    'max': max(all_efficiency_values) if all_efficiency_values else 0,
                    'min': min(all_efficiency_values) if all_efficiency_values else 0
                }
            }
        }
        generation_history.append(generation_data)

        # Log objective statistics along with other stats
        logger.info(f"Objective stats across all islands:")
        if all_progression_values:
            logger.info(f"  Progression - Avg: {np.mean(all_progression_values):.3f}, Max: {max(all_progression_values):.3f}")
            logger.info(f"  Survival    - Avg: {np.mean(all_survival_values):.3f}, Max: {max(all_survival_values):.3f}")
            logger.info(f"  Efficiency  - Avg: {np.mean(all_efficiency_values):.3f}, Max: {max(all_efficiency_values):.3f}")

        # Save checkpoint at regular intervals
        if gen % 10 == 0 or gen == N_GENERATIONS - 1:
            save_checkpoint_with_islands(gen, islands, config, generation_history, 
                                       diversity_manager, elite_pool, current_best_score, 
                                       diversity_metrics, BEST_EVER)

    # Get best population across all islands
    best_island = max(islands, key=lambda island: island['best_fitness'])
    best_pop = best_island['population']
    
    # Ensure all junction_points are converted from sets to lists
    for island in islands:
        for genome_id, genome in island['population'].population.items():
            if hasattr(genome, 'metrics') and 'junction_points' in genome.metrics:
                if isinstance(genome.metrics['junction_points'], set):
                    genome.metrics['junction_points'] = list(genome.metrics['junction_points'])
    
    # Convert best_ever junction_points too
    for objective, data in BEST_EVER.items():
        if data['genome'] and hasattr(data['genome'], 'metrics') and 'junction_points' in data['genome'].metrics:
            if isinstance(data['genome'].metrics['junction_points'], set):
                data['genome'].metrics['junction_points'] = list(data['genome'].metrics['junction_points'])
    
    # Create and add the best agents summary
    best_agents_summary = create_best_agents_summary()

    final_results = {
        'generation_history': generation_history,
        'elite_pool': elite_pool,
        'absolute_best_genomes': absolute_best_genomes,  
        'final_islands': islands,
        'best_population': best_pop.population,
        'best_species': best_pop.species,
        'config': config,
        'best_score': current_best_score,
        'diversity_metrics': diversity_metrics,
        'best_ever': BEST_EVER,
        'best_agents_summary': best_agents_summary
    }

    # Save final results
    save_final_results(final_results, 'neat_final_results.pkl')

    # Also save a standalone JSON for easy reference
    with open('best_agents_replay_info.json', 'w') as f:
        json.dump(best_agents_summary, f, indent=2)

    logger.info(f"Best agents saved for easy replay:")
    logger.info(f"  Overall best: Score={BEST_EVER['overall']['score']}, Seed={BEST_EVER['overall']['seed']}")
    logger.info(f"  Best progression: Score={BEST_EVER['progression']['score']}, Seed={BEST_EVER['progression']['seed']}")
    logger.info(f"  Best survival: Score={BEST_EVER['survival']['score']}, Seed={BEST_EVER['survival']['seed']}")
    logger.info(f"  Best efficiency: Score={BEST_EVER['efficiency']['score']}, Seed={BEST_EVER['efficiency']['seed']}")
        
    # Find the absolute best genome across all islands
    all_genomes = []
    for island in islands:
        all_genomes.extend(island['population'].population.values())

    best_genome = max(all_genomes, 
                    key=lambda g: getattr(g, 'total_score', 0) if hasattr(g, 'total_score') else 0)

    best_score = getattr(best_genome, 'total_score', 0)
    logger.info(f"Found absolute best genome with score: {best_score}")

    # Validate against BEST_EVER
    if best_score > BEST_EVER['overall']['score']:
        logger.warning(f"Found better genome ({best_score}) than tracked best ({BEST_EVER['overall']['score']})")
        # Update BEST_EVER with the better genome
        update_best_ever_agents([best_genome], N_GENERATIONS, None)

    # Check absolute_best_genomes as well
    if absolute_best_genomes:
        absolute_best_score = absolute_best_genomes[0].total_score
        if absolute_best_score > BEST_EVER['overall']['score']:
            logger.warning(f"Found better preserved genome ({absolute_best_score}) than tracked best ({BEST_EVER['overall']['score']})")

    # Ensure final results contain the absolute best
    final_results['absolute_best_genome'] = absolute_best_genomes[0] if absolute_best_genomes and absolute_best_genomes[0].total_score > BEST_EVER['overall']['score'] else BEST_EVER['overall']['genome']
    final_results['absolute_best_score'] = max([best_score, BEST_EVER['overall']['score']] + ([absolute_best_genomes[0].total_score] if absolute_best_genomes else []))

    # Make sure best gets returned
    return best_genome.population if hasattr(best_genome, 'population') else island['population'], None, final_results
        

def update_best_ever_agents(population, generation, island_idx=None, config=None):
    """Update the best-ever agents for all objectives"""
    global BEST_EVER
    
    # Check for best overall (by total score)
    current_best = max(population, key=lambda g: getattr(g, 'total_score', 0))
    current_score = getattr(current_best, 'total_score', 0)
    
    if current_score > BEST_EVER['overall']['score']:
        best_copy = neat.DefaultGenome(0)
        best_copy.configure_crossover(current_best, current_best, config.genome_config)
        
        # Copy metrics and other relevant attributes
        if hasattr(current_best, 'metrics'):
            best_copy.metrics = current_best.metrics.copy() if isinstance(current_best.metrics, dict) else current_best.metrics
        if hasattr(current_best, 'total_score'):
            best_copy.total_score = current_best.total_score
            
        BEST_EVER['overall']['genome'] = best_copy
        BEST_EVER['overall']['score'] = current_score
        BEST_EVER['overall']['generation'] = generation
        BEST_EVER['overall']['island'] = island_idx
        
        # Get the seed from metrics
        if hasattr(current_best, 'metrics') and 'seed' in current_best.metrics:
            BEST_EVER['overall']['seed'] = current_best.metrics['seed']
        
        logger.info(f"New best-ever overall agent! Score: {current_score}, Genome ID: {current_best.key}, Generation: {generation}, Island: {island_idx+1}, Seed: {BEST_EVER['overall']['seed']}")
        
    # Check for best progression (objective 0)
    progression_candidates = [g for g in population if hasattr(g, 'multi_fitness') 
                           and hasattr(g.multi_fitness, 'values')
                           and len(g.multi_fitness.values) > 0]
    
    if progression_candidates:
        best_prog = max(progression_candidates, key=lambda g: g.multi_fitness.values[0])
        prog_score = best_prog.multi_fitness.values[0]
        
        if prog_score > BEST_EVER['progression']['score']:
            BEST_EVER['progression']['genome'] = best_prog
            BEST_EVER['progression']['score'] = prog_score
            BEST_EVER['progression']['generation'] = generation
            BEST_EVER['progression']['island'] = island_idx
            
            if hasattr(best_prog, 'metrics') and 'seed' in best_prog.metrics:
                BEST_EVER['progression']['seed'] = best_prog.metrics['seed']
            
            logger.info(f"New best-ever progression agent! Score: {prog_score}, Genome ID: {best_prog.key}, Generation: {generation}, Island: {island_idx}, Seed: {BEST_EVER['progression']['seed']}")
    
    # Check for best survival (objective 1)
    survival_candidates = [g for g in population if hasattr(g, 'multi_fitness') 
                         and hasattr(g.multi_fitness, 'values')
                         and len(g.multi_fitness.values) > 1]
    
    if survival_candidates:
        best_surv = max(survival_candidates, key=lambda g: g.multi_fitness.values[1])
        surv_score = best_surv.multi_fitness.values[1]
        
        if surv_score > BEST_EVER['survival']['score']:
            BEST_EVER['survival']['genome'] = best_surv
            BEST_EVER['survival']['score'] = surv_score
            BEST_EVER['survival']['generation'] = generation
            BEST_EVER['survival']['island'] = island_idx
            
            if hasattr(best_surv, 'metrics') and 'seed' in best_surv.metrics:
                BEST_EVER['survival']['seed'] = best_surv.metrics['seed']
            
            logger.info(f"New best-ever survival agent! Score: {surv_score}, Genome ID: {best_surv.key}, Generation: {generation}, Island: {island_idx}, Seed: {BEST_EVER['survival']['seed']}")
    
    # Check for best efficiency (objective 2)
    efficiency_candidates = [g for g in population if hasattr(g, 'multi_fitness') 
                           and hasattr(g.multi_fitness, 'values')
                           and len(g.multi_fitness.values) > 2]
    
    if efficiency_candidates:
        best_eff = max(efficiency_candidates, key=lambda g: g.multi_fitness.values[2])
        eff_score = best_eff.multi_fitness.values[2]
        
        if eff_score > BEST_EVER['efficiency']['score']:
            BEST_EVER['efficiency']['genome'] = best_eff
            BEST_EVER['efficiency']['score'] = eff_score
            BEST_EVER['efficiency']['generation'] = generation
            BEST_EVER['efficiency']['island'] = island_idx
            
            if hasattr(best_eff, 'metrics') and 'seed' in best_eff.metrics:
                BEST_EVER['efficiency']['seed'] = best_eff.metrics['seed']
            
            logger.info(f"New best-ever efficiency agent! Score: {eff_score}, Genome ID: {best_eff.key}, Generation: {generation}, Island: {island_idx}, Seed: {BEST_EVER['efficiency']['seed']}")

def create_best_agents_summary():
    """Create a summary of the best agents for easy reference"""
    summary = {
        "best_agents": {
            "overall": {
                "score": BEST_EVER['overall']['score'],
                "genome_id": getattr(BEST_EVER['overall']['genome'], 'key', None),
                "generation": BEST_EVER['overall']['generation'],
                "island": BEST_EVER['overall']['island'],
                "seed": BEST_EVER['overall']['seed']
            },
            "progression": {
                "score": BEST_EVER['progression']['score'],
                "genome_id": getattr(BEST_EVER['progression']['genome'], 'key', None),
                "generation": BEST_EVER['progression']['generation'],
                "island": BEST_EVER['progression']['island'],
                "seed": BEST_EVER['progression']['seed']
            },
            "survival": {
                "score": BEST_EVER['survival']['score'],
                "genome_id": getattr(BEST_EVER['survival']['genome'], 'key', None),
                "generation": BEST_EVER['survival']['generation'], 
                "island": BEST_EVER['survival']['island'],
                "seed": BEST_EVER['survival']['seed']
            },
            "efficiency": {
                "score": BEST_EVER['efficiency']['score'],
                "genome_id": getattr(BEST_EVER['efficiency']['genome'], 'key', None),
                "generation": BEST_EVER['efficiency']['generation'],
                "island": BEST_EVER['efficiency']['island'],
                "seed": BEST_EVER['efficiency']['seed']
            }
        },
        "reproduction_info": {
            "config_path": "neat_config.txt",
            "network_inputs": 40,  # Updated from your enhanced input representation
            "network_outputs": 4
        }
    }
    
    # Save JSON summary in separate file for easy access
    try:
        with open('best_agents_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving best agents summary: {e}")
    
    return summary