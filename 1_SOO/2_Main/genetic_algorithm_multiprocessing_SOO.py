import gymnasium as gym 
import ale_py # 
# from gymnasium.vector import AsyncVectorEnv # type: ignore
import numpy as np # 
import random
import multiprocessing
import logging
import pickle
from deap import base, creator, tools, algorithms 
import pandas as pd 
import time
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# System-specific parameters
NUM_PROCESSES = 6  # Leave 2 cores for system and overhead on an 8 core system; select 'None' for automatic selection based on system running
GENERATIONS = 1000
POPULATION_SIZE = 200  # baseline 100; testing resource scaling
SEQUENCE_LENGTH = 2000  # baseline 300; testing longer action sequences (or less to save memory)
MUTATION_RATE = 0.1  # baseline 0.05; for better exploration
CROSSOVER_RATE = 0.9 # baseline 0.7; testing more genetic mixing
TOURNAMENT_SIZE = 5 # baseline 3; Selection Pressure Test: Testing higher pressure

# Consistent action space
LIMITED_ACTIONS = [1, 2, 3, 4]  # NOOP, UP, RIGHT, LEFT, DOWN

def create_env():
    """Create a Ms. Pac-Man environment"""
    return gym.make("ALE/MsPacman-v5", render_mode="rgb_array")

def evaluate_individual(individual, generation, seed=None):
    """
    Evaluate a single individual across multiple environments
    """
    try:
        # Create a single environment
        env = create_env()

        # Use provided seed or generate a new one
        if seed is None:
            seed = random.randint(0, 1000000)
        
        individual_list = list(individual)
        observation, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0
        
        max_steps = 5000
        no_reward_counter = 0
        max_no_reward_steps = 1000 # score of at least a pellet 
        last_total_reward = 0
        
        try:
            while (not terminated and not truncated and 
                   steps < max_steps and no_reward_counter < max_no_reward_steps):
                
                action_idx = individual_list[steps % len(individual_list)] % len(LIMITED_ACTIONS)
                action = LIMITED_ACTIONS[action_idx]
                
                observation, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                
                if total_reward == last_total_reward:
                    no_reward_counter += 1
                else:
                    no_reward_counter = 0
                    last_total_reward = total_reward
                
                steps += 1
                
        finally:
            env.close()
        
        logger.info(f"Individual fitness: {total_reward}, Steps: {steps}, "
                   f"Generation: {generation}")
        return total_reward, seed
    
    except Exception as e:
        logger.error(f"Error in evaluate_individual: {e}")
        return 0.0, seed

def evaluate_population(population, generation, num_processes):
    """Evaluate entire population using a process pool"""
    with multiprocessing.Pool(processes=num_processes) as pool:
        eval_func = partial(evaluate_individual, generation=generation)
        results = pool.map(eval_func, population)

        # Unpack fitness values and seeds
        fitness_values, seeds = zip(*results)
        
        # Assign fitness values and store seeds as attributes
        for ind, fit, seed in zip(population, fitness_values, seeds):
            ind.fitness.values = (fit,)
            ind.seed = seed
    
    return population

def setup_genetic_algorithm():
    """Setup DEAP genetic algorithm components"""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_action", lambda: random.randint(0, len(LIMITED_ACTIONS) - 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_action, n=SEQUENCE_LENGTH)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_RATE)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    
    return toolbox

def run_genetic_algorithm(toolbox, pop_size=POPULATION_SIZE, generations=GENERATIONS, num_processes=None):
    """Run genetic algorithm optimisation"""
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    random.seed(42)
    np.random.seed(42)

    population = toolbox.population(n=POPULATION_SIZE)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else 0)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    start_time = time.time()
    time_stats = {
        'generation': [],
        'avg_time': [],
        'max_time': [],
    }

    # Initial population evaluation
    population = evaluate_population(population, 0, NUM_PROCESSES)

    # Added variables to track best overall individuals
    best_ever_fitness = float('-inf')
    best_ever_individual = None
    best_ever_seed = None
    top_10_individuals = []
    top_10_fitness_values = []
    
    # Evolution loop
    for gen in range(generations):
        # Select and clone offspring
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Parallel evaluation of offspring
        offspring = evaluate_population(offspring, gen + 1, NUM_PROCESSES)
        
        # Replace population
        population[:] = offspring

        # After evaluation, check for new best individuals
        for ind in population:
            if ind.fitness.values[0] > best_ever_fitness:
                best_ever_fitness = ind.fitness.values[0]
                best_ever_individual = toolbox.clone(ind)
                best_ever_seed = ind.seed
                
                # Update top 10 list
                top_10_individuals.append(toolbox.clone(ind))
                top_10_fitness_values.append(ind.fitness.values[0])
                
                # Sort and keep only top 10
                sorted_pairs = sorted(zip(top_10_fitness_values, top_10_individuals), 
                                   key=lambda x: x[0], reverse=True)
                top_10_fitness_values, top_10_individuals = zip(*sorted_pairs[:10])
                top_10_fitness_values = list(top_10_fitness_values)
                top_10_individuals = list(top_10_individuals)
        
        # Record statistics
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(offspring), **record)
        
        # Time tracking
        gen_time = time.time() - start_time
        time_stats['generation'].append(gen)
        time_stats['avg_time'].append(gen_time / (gen + 1))
        time_stats['max_time'].append(gen_time)
        
        # Save intermediate results every 100 generations
        if (gen + 1) % 100 == 0:
            results = {
                'generation': gen,
                'population': population,
                'logbook': logbook,
                'best_ind': tools.selBest(population, k=1)[0],
                'best_individual': best_ever_individual,
                'best_seed': best_ever_seed,
                'best_individuals': tools.selBest(population, k=10),
                'top_10_individuals': top_10_individuals,
                'top_10_fitness_values': top_10_fitness_values,  # Match the max values
                'best_fitness_values': [ind.fitness.values[0] for ind in tools.selBest(population, k=10)],
                'generation_data': pd.DataFrame({
                    'Generation': logbook.select("gen"),
                    'Max Fitness': logbook.select("max"),
                    'Avg Fitness': logbook.select("avg"),
                    'Avg Time Taken (s)': time_stats['avg_time'],
                    'Max Time Taken (s)': time_stats['max_time']
                })
            }
            # Create results DataFrame
            data = {
                'Generation': logbook.select("gen"),
                'Max Fitness': logbook.select("max"),
                'Avg Fitness': logbook.select("avg"),
                'Avg Time Taken (s)': time_stats['avg_time'],
                'Max Time Taken (s)': time_stats['max_time']
            }
            with open(f'ms_pacman_ga_results_gen_{gen+1}.pkl', 'wb') as f:
                pickle.dump(results, f)

            df = pd.DataFrame(data)
            df.to_csv(f'ms_pacman_generation_data_gen_{gen+1}.csv', index=False)
    
    # Save final results
    results = {
        'final_population': population,
        'logbook': logbook,
        'best_ind': tools.selBest(population, k=1)[0],
        'best_individual': best_ever_individual,
        'best_seed': best_ever_seed,
        'best_individuals': tools.selBest(population, k=10),
        'top_10_individuals': top_10_individuals,
        'top_10_fitness_values': top_10_fitness_values,  # Match the max values
        'best_fitness_values': [ind.fitness.values[0] for ind in tools.selBest(population, k=10)],
        'generation_data': pd.DataFrame({
            'Generation': logbook.select("gen"),
            'Max Fitness': logbook.select("max"),
            'Avg Fitness': logbook.select("avg"),
            'Avg Time Taken (s)': time_stats['avg_time'],
            'Max Time Taken (s)': time_stats['max_time']
        })
    }
    # Create results DataFrame
    data = {
        'Generation': logbook.select("gen"),
        'Max Fitness': logbook.select("max"),
        'Avg Fitness': logbook.select("avg"),
        'Avg Time Taken (s)': time_stats['avg_time'],
        'Max Time Taken (s)': time_stats['max_time']
    }
    
    with open('ms_pacman_ga_results_final.pkl', 'wb') as f:
        pickle.dump(results, f)

    df = pd.DataFrame(data)
    df.to_csv(f'ms_pacman_generation_data_gen_final.csv', index=False)
    
    return results

def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    logger.info(f"Starting optimization with:")
    logger.info(f"No. generations: {GENERATIONS}")
    logger.info(f"Population size: {POPULATION_SIZE}")
    logger.info(f"Number of processes: {NUM_PROCESSES}")
    logger.info(f"Sequence length: {SEQUENCE_LENGTH}")
    
    toolbox = setup_genetic_algorithm()
    results = run_genetic_algorithm(toolbox)
    
    print("Optimisation Complete. Results saved to ms_pacman_ga_results_final.pkl")
    print(f"Best individual seed: {results['best_seed']}")
    data = results['generation_data']
    print(f"Best individual fitness: {data['Max Fitness'].max()}")

if __name__ == "__main__":
    main()