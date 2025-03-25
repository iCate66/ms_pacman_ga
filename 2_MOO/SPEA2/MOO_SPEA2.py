import gymnasium as gym
import ale_py
import numpy as np
import random
import multiprocessing
import logging
import pickle
from deap import base, creator, tools, algorithms
from functools import partial
from scipy.spatial.distance import pdist

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
NUM_PROCESSES = 6 # set depending on system capability
GENERATIONS = 1000
POPULATION_SIZE = 300
SEQUENCE_LENGTH = 10000  # Increased to allow for full level completion
MUTATION_RATE = 0.05   
CROSSOVER_RATE = 0.7   
LIMITED_ACTIONS = [1, 2, 3, 4]  # UP, RIGHT, DOWN, LEFT

# Score thresholds for Ms. Pac-Man first level
GHOST_SCORE = 200
MAX_EXPECTED_FRAMES = SEQUENCE_LENGTH # Match to avoid agent repeating actions before reaching the end of the episode

class GameStateMetrics:
    """Track gameplay metrics based on state observations."""
    def __init__(self):
        # Basic game state
        self.total_score = 0
        self.initial_lives = 3
        self.current_lives = 3
        self.frames_survived = 0

        # Progression tracking
        self.score_segments = []
        self.score_change_frames = []
        self.consecutive_scoring_frames = 0
        self.longest_scoring_streak = 0
        self.highest_score_achieved = 0
      
        # Efficiency tracking
        self.ghost_points = 0
        self.pellet_points = 0
        self.total_points_from_actions = 0
        self.scoring_actions = 0
      
        # Performance tracking
        self.deaths = 0
        self.action_counts = {i: 0 for i in range(len(LIMITED_ACTIONS))}
      
    def update(self, reward, lives, frame_number, action):
        # Basic updates based on current game state
        self.total_score += reward
        self.frames_survived = frame_number
        self.action_counts[action] += 1
      
        # Track scoring patterns
        if reward > 0:
            self.score_change_frames.append(frame_number)
            self.consecutive_scoring_frames += 1
            self.longest_scoring_streak = max(self.longest_scoring_streak,
                                           self.consecutive_scoring_frames)
            self.scoring_actions += 1
            self.total_points_from_actions += reward
          
            # Track ghost points
            if reward >= GHOST_SCORE:
                self.ghost_points += reward
            else:
                self.pellet_points += reward
        else:
            self.consecutive_scoring_frames = 0

        # Update highest score
        if self.total_score > self.highest_score_achieved:
            self.highest_score_achieved = self.total_score
      
        # Track life losses and segment data
        if lives < self.current_lives:
            self.deaths += 1
        self.current_lives = lives
      
        # Record score segments every 100 frames
        if frame_number % 100 == 0:
            self.score_segments.append(self.total_score)

    def get_metrics(self):
        """Calculate comprehensive performance metrics"""
        return {
            'total_score': self.total_score,
            'frames_survived': self.frames_survived,
            'deaths': self.deaths,
            'score_segments': self.score_segments,
            'ghost_points': self.ghost_points,
            'pellet_points': self.pellet_points,
            'scoring_actions': self.scoring_actions,
            'total_points_from_actions': self.total_points_from_actions,
            'longest_scoring_streak': self.longest_scoring_streak,
            'current_lives': self.current_lives,
            'score_change_frames': self.score_change_frames
        }

def evaluate_individual(individual, generation=None, seed=None):
    """Evaluate an individual using three-objective system"""
    try:
        env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
        if seed is None:
            seed = random.randint(0, 1000000)
      
        # Get basic metrics
        metrics = GameStateMetrics()
        observation, info = env.reset(seed=seed)
        env.action_space.seed(seed)
      
        terminated = False
        truncated = False
        steps = 0
      
        try:
            while not (terminated or truncated) and steps < SEQUENCE_LENGTH: # and steps < 10000
                # Get current action
                action_idx = individual[steps % len(individual)] % len(LIMITED_ACTIONS)
                action = LIMITED_ACTIONS[action_idx]
              
                # Execute action
                observation, reward, terminated, truncated, info = env.step(action)

                # Update metrics
                metrics.update(reward, info['lives'], info['frame_number'], action_idx)
                steps += 1

        finally:
            env.close()

        # Calculate three objectives
        # Calculate each objective using dedicated functions
      
        progression_objective = calculate_progress(metrics)
        survival_objective = calculate_survival(metrics)
        efficiency_objective = calculate_efficiency(metrics)
      
        # # # Debug logging
        # # logger.info(f"evaluate_individual normalised score_objective: {progression_objective}")

        if generation is not None:
            logger.info(f"Gen {generation} - Progress: {progression_objective:.3f}, "
                       f"Survival: {survival_objective:.3f}, "
                       f"Efficiency: {efficiency_objective:.3f}, "
                       f"Steps: {steps}, "
                       f"Score: {metrics.total_score:.0f} ")
      
        return (progression_objective, survival_objective, efficiency_objective), metrics.get_metrics(), seed
  
    except Exception as e:
        logger.error(f"Error in evaluate_individual: {e}")
        return (0.0, 0.0, 0.0), None, seed
  
def calculate_progress(metrics):
  
    # Calculate pellet collection progress - direct measure of level completion
    # A typical Ms. Pac-Man level has around 240 regular pellets worth 10 points each
    pellet_progress = min(metrics.pellet_points / 2400, 1.0)

    # Base score progress as overall progress indicator - general measure of game progress
    base_score_progress = min(metrics.total_score / MAX_EXPECTED_FRAMES, 1.0)
  
    # Score consistency calculation for steady progress measurement 
    score_consistency = 0
    if len(metrics.score_segments) > 1:
        score_increases = [b - a for a, b in zip(metrics.score_segments[:-1],
                                                 metrics.score_segments[1:])]
        avg_score_increase = sum(score_increases) / max(1, len(score_increases))
        score_consistency = min(avg_score_increase / 20, 1.0)
  
    # Streak quality as measure of execution and sustained successful gameplay
    streak_quality = min(metrics.longest_scoring_streak / 100, 1.0)
  
    # Weighting that emphasises pellet collection
    progression_objective = (
        0.4 * pellet_progress +      # Emphasise level completion requirement
        0.3 * base_score_progress +  # Maintain overall progress importance
        0.2 * score_consistency +    # Keep steady progress component
        0.1 * streak_quality         # Retain execution quality measure
    )
  
    return progression_objective
    
def calculate_survival(metrics):
   
    # Survival time component - how long the agent lasts
    survival_time_ratio = metrics.frames_survived / MAX_EXPECTED_FRAMES
  
    # Active gameplay - considers chunks of time (every 100 frames) and checks if scoring occurred in each chunk
    chunk_size = 100
    total_chunks = max(1, metrics.frames_survived // chunk_size)
    scoring_chunks = 0
  
    # Group score_change_frames into chunks
    for i in range(total_chunks):
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size
        # If scoring happened in this chunk, count it as active
        if any(chunk_start <= frame < chunk_end for frame in metrics.score_change_frames):
            scoring_chunks += 1
  
    active_gameplay = scoring_chunks / total_chunks  # Active gameplay ratio
   
    # Survival objective that emphasises basic survival, longevity and sustained active gameplay
    survival_objective = (0.2 * (metrics.current_lives / metrics.initial_lives) +  # Basic survival
                          0.6 * survival_time_ratio +                              # Time survived
                          0.2 * active_gameplay)                                   # Scoring activity
  
    return survival_objective

def calculate_efficiency(metrics):

    # Rewarding high scoring actions
    avg_points_per_action = metrics.total_points_from_actions / max(1, metrics.scoring_actions)

    # Derived means of determining ratio of ghost points 
    ghost_hunting_ratio = metrics.ghost_points / max(1, metrics.total_score)

    # Efficiency objective that considers weighted scoring actions and effective ghost hunting
    efficiency_objective = (
        min(avg_points_per_action / 200, 1.0) +
        ghost_hunting_ratio
    ) / 2
  
    return efficiency_objective

def evaluate_population(population, generation, num_processes):
    """Evaluate entire population using process pool"""
    with multiprocessing.Pool(processes=num_processes) as pool:
        eval_func = partial(evaluate_individual, generation=generation)
        results = pool.map(eval_func, population)
      
        # Process results and collect generation metrics
        generation_metrics = {
            'generation': generation,
            'population_size': len(population),
            'objective_values': [],
            'metrics': []
        }
      
        for ind, (fitness_values, metrics, seed) in zip(population, results):

            # # Debug logging
            # logger.info(f"Before assigning fitness - values: {fitness_values}")

            ind.fitness.values = fitness_values
            # # Debug logging
            # logger.info(f"After assigning fitness - ind.fitness.values: {ind.fitness.values}")
            ind.metrics = metrics
            ind.seed = seed
          
            generation_metrics['objective_values'].append(fitness_values)
            # # Debug logging
            # logger.info(f"Stored in objective_values: {generation_metrics['objective_values'][-1]}")
            if metrics:
                metrics['seed'] = seed
                generation_metrics['metrics'].append(metrics)
      
        # Calculate generation-level metrics
        if generation_metrics['objective_values']:
            objective_array = np.array(generation_metrics['objective_values'])
          
            # Calculate spread of solutions
            if len(objective_array) > 1:
                distances = pdist(objective_array)
                generation_metrics['solution_spread'] = np.mean(distances)
                generation_metrics['solution_std'] = np.std(distances)
          
            # Store objective statistics
            generation_metrics['objective_stats'] = {
                'mean': np.mean(objective_array, axis=0),
                'std': np.std(objective_array, axis=0),
                'max': np.max(objective_array, axis=0),
                'min': np.min(objective_array, axis=0)
            }
      
        return population, generation_metrics
  
def analyse_playstyle(individual):
    """
    Analyse gameplay patterns to identify strategic approaches and adaptivity in agent behaviour.
    Provides insights into how agents balance survival, progression, and efficiency
    throughout their gameplay session.
    """
    if not hasattr(individual, 'metrics') or individual.metrics is None:
        if isinstance(individual, dict) and 'metrics' in individual:
            metrics = individual['metrics']
        else:
            return {'playstyle': 'Unknown', 'score': 0}
    else:
        metrics = individual.metrics

    # Temporal analysis - how strategy evolves during gameplay
    frames_survived = metrics.get('frames_survived', 0)
    score_segments = metrics.get('score_segments', [])
    score_change_frames = metrics.get('score_change_frames', [])
  
    # Analyse scoring patterns over time
    early_game = []
    mid_game = []
    late_game = []
  
    # Divide gameplay into thirds for temporal analysis
    for frame in score_change_frames:
        if frame < frames_survived / 3:
            early_game.append(frame)
        elif frame < 2 * frames_survived / 3:
            mid_game.append(frame)
        else:
            late_game.append(frame)
  
    # Calculate phase-specific activity levels
    early_activity = len(early_game) / max(1, frames_survived / 3)
    mid_activity = len(mid_game) / max(1, frames_survived / 3)
    late_activity = len(late_game) / max(1, frames_survived / 3)
  
    # Calculate strategic components
    survival_time_ratio = frames_survived / MAX_EXPECTED_FRAMES  
    ghost_points = metrics.get('ghost_points', 0)
    total_score = metrics.get('total_score', 0)
    ghost_hunting_ratio = ghost_points / max(1, total_score)
  
   # Analyse strategic adaptivity
    strategy_shifts = 0
    activity_threshold = 0.2
    if abs(early_activity - mid_activity) > activity_threshold:
        strategy_shifts += 1
    if abs(mid_activity - late_activity) > activity_threshold:
        strategy_shifts += 1
      
    # Determine primary and secondary strategic focuses
    survival_focus = survival_time_ratio > 0.6
    progression_focus = total_score > MAX_EXPECTED_FRAMES/2  # Half of our new baseline
    efficiency_focus = ghost_hunting_ratio > 0.3
  
    # Classify core strategy
    if survival_time_ratio > 0.8 and total_score > 4000:
        base_style = "Elite"
    elif strategy_shifts >= 2:
        base_style = "Adaptive"
    elif survival_focus and progression_focus:
        base_style = "Balanced"
    elif survival_focus:
        base_style = "Conservative"
    elif efficiency_focus and ghost_hunting_ratio > 0.5:
        base_style = "Hunter"
    elif progression_focus:
        base_style = "Progressive"
    else:
        base_style = "Developing"
  
    # Add strategic modifiers
    modifiers = []
    if early_activity > mid_activity + activity_threshold:
        modifiers.append("Early Aggression")
    if late_activity > early_activity + activity_threshold:
        modifiers.append("Late Game Scaling")
    if strategy_shifts >= 2:
        modifiers.append("Highly Adaptive")
  
    # Combine base style with modifiers
    full_style = f"{base_style}" + (f" ({', '.join(modifiers)})" if modifiers else "")
  
    # Create comprehensive analysis
    analysis = {
        'playstyle': full_style,
        'score': total_score,
        'temporal_analysis': {
            'early_game_activity': early_activity,
            'mid_game_activity': mid_activity,
            'late_game_activity': late_activity,
            'strategy_shifts': strategy_shifts
        },
        'strategic_focus': {
            'survival_emphasis': survival_time_ratio,
            'progression_emphasis': total_score / MAX_EXPECTED_FRAMES,  
            'efficiency_emphasis': ghost_hunting_ratio
        },
        'gameplay_patterns': {
            'survival_time': frames_survived,
            'scoring_consistency': {
                'early': early_activity,
                'mid': mid_activity,
                'late': late_activity
            },
            'adaptivity': strategy_shifts
        }
    }
  
    return analysis

def setup_spea2():
    """Setup DEAP SPEA2 components with tournament selection"""
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    # Define the tournament size 
    toolbox.tournament_size = 5  
    
    toolbox.register("attr_action", lambda: random.randint(0, len(LIMITED_ACTIONS) - 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_action, n=SEQUENCE_LENGTH)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_RATE)
    toolbox.register("select", tools.selSPEA2)
    toolbox.register("tournament_select", tools.selTournament, tournsize=toolbox.tournament_size)
    
    return toolbox

def run_spea2(toolbox, pop_size=POPULATION_SIZE, archive_size=POPULATION_SIZE//2, 
              generations=GENERATIONS, num_processes=NUM_PROCESSES):
    """Run SPEA2 optimisation with enhanced data collection"""
    random.seed(42)
    np.random.seed(42)

    # Add tracking for max score agent
    max_score_agent = {
        'individual': None,
        'seed': None,
        'fitness_values': None,
        'metrics': None,
        'generation': None,
        'playstyle': None,
        'total_score': 0
    }
    
    population = toolbox.population(n=pop_size)
    archive = []
    
    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals'] + stats.fields
    
    # Evolution tracking
    generation_history = []
    archive_history = []
    
    # Initial evaluation
    population, gen_metrics = evaluate_population(population, 0, num_processes)

    # Check for max score in initial population
    for ind in population:
        if ind.metrics and ind.metrics['total_score'] > max_score_agent['total_score']:
            max_score_agent['individual'] = list(ind)
            max_score_agent['seed'] = ind.seed
            max_score_agent['fitness_values'] = ind.fitness.values
            max_score_agent['metrics'] = ind.metrics
            max_score_agent['generation'] = 0
            max_score_agent['total_score'] = ind.metrics['total_score']
            max_score_agent['playstyle'] = analyse_playstyle({'metrics': ind.metrics})

    generation_history.append(gen_metrics)
    
    # Record statistics
    record = stats.compile(population)
    logbook.record(gen=0, evals=len(population), **record)
    
    # Evolution loop
    for gen in range(generations):
        # Environmental selection (includes fitness assignment)
        archive = toolbox.select(population + archive, k=archive_size)
        
        # Mating selection using tournament selection
        if len(archive) > 0:
            mating_pool = toolbox.tournament_select(archive, k=pop_size)
        else:
            # If archive is empty, select from population
            mating_pool = toolbox.tournament_select(population, k=pop_size)

        offspring = list(map(toolbox.clone, mating_pool))
        
        # Apply variation operators
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate offspring
        offspring, offspring_metrics = evaluate_population(offspring, gen + 1, num_processes)

        # Check for new max score
        for ind in offspring:
            if ind.metrics and ind.metrics['total_score'] > max_score_agent['total_score']:
                max_score_agent['individual'] = list(ind)
                max_score_agent['seed'] = ind.seed
                max_score_agent['fitness_values'] = ind.fitness.values
                max_score_agent['metrics'] = ind.metrics
                max_score_agent['generation'] = gen + 1
                max_score_agent['total_score'] = ind.metrics['total_score']
                max_score_agent['playstyle'] = analyse_playstyle({'metrics': ind.metrics})
        
        # Update population
        population = offspring
        
        # Record statistics
        combined = population + archive if archive else population
        record = stats.compile(combined)
        logbook.record(gen=gen + 1, evals=len(offspring), **record)
        
        # Store generation data
        generation_metrics = {
            'generation': gen,
            'population_size': len(population),
            'archive_size': len(archive),
            'objective_values': [ind.fitness.values for ind in population],
            'archive_values': [ind.fitness.values for ind in archive] if archive else [],
            'metrics': offspring_metrics.get('metrics', []),
            'solution_spread': offspring_metrics.get('solution_spread', 0),
            'solution_std': offspring_metrics.get('solution_std', 0),
            'objective_stats': {
                'mean': np.mean([ind.fitness.values for ind in combined], axis=0),
                'std': np.std([ind.fitness.values for ind in combined], axis=0),
                'max': np.max([ind.fitness.values for ind in combined], axis=0),
                'min': np.min([ind.fitness.values for ind in combined], axis=0)
            }
        }
        generation_history.append(generation_metrics)
        archive_history.append([ind.fitness.values for ind in archive] if archive else [])
        
        # Save checkpoint every 100 generations
        if (gen+1) % 100 == 0:
            checkpoint = {
                'generation': gen,
                'population': population,
                'archive': archive,
                'generation_metrics': generation_metrics
            }
            with open(f'pacman_spea2_checkpoint_gen_{gen+1}.pkl', 'wb') as cp_file:
                pickle.dump(checkpoint, cp_file)
    
    # Prepare final results
    final_archive = archive
    
    # Get best individuals by each objective from both population and archive
    all_individuals = population + (archive if archive else [])
    best_by_score = sorted(all_individuals, key=lambda x: x.fitness.values[0], reverse=True)[:10]
    best_by_survival = sorted(all_individuals, key=lambda x: x.fitness.values[1], reverse=True)[:10]
    best_by_efficiency = sorted(all_individuals, key=lambda x: x.fitness.values[2], reverse=True)[:10]

    # Store replay information for best individuals
    replay_info = {
        'best_by_score': [(list(ind), ind.seed, ind.fitness.values, ind.metrics) for ind in best_by_score],
        'best_by_survival': [(list(ind), ind.seed, ind.fitness.values, ind.metrics) for ind in best_by_survival],
        'best_by_efficiency': [(list(ind), ind.seed, ind.fitness.values, ind.metrics) for ind in best_by_efficiency]
        #'final_archive': [(list(ind), ind.seed, ind.fitness.values, ind.metrics) for ind in final_archive]
    }
    
    # Store comprehensive information
    performance_metrics = {
        'score_objective': {
            'best_individuals': [(list(ind), ind.seed, ind.fitness.values, ind.metrics,
                               analyse_playstyle({'metrics': ind.metrics}))
                              for ind in best_by_score],
            'max_fitness': max(best_by_score, key=lambda x: x.fitness.values[0]).fitness.values[0]
        },
        'survival_objective': {
            'best_individuals': [(list(ind), ind.seed, ind.fitness.values, ind.metrics,
                               analyse_playstyle({'metrics': ind.metrics}))
                              for ind in best_by_survival],
            'max_fitness': max(best_by_survival, key=lambda x: x.fitness.values[1]).fitness.values[1]
        },
        'efficiency_objective': {
            'best_individuals': [(list(ind), ind.seed, ind.fitness.values, ind.metrics,
                               analyse_playstyle({'metrics': ind.metrics}))
                              for ind in best_by_efficiency],
            'max_fitness': max(best_by_efficiency, key=lambda x: x.fitness.values[2]).fitness.values[2]
        },
        'max_score_agent': max_score_agent,
        'archive': [(list(ind), ind.seed, ind.fitness.values, ind.metrics,
                    analyse_playstyle({'metrics': ind.metrics}))
                   for ind in final_archive]
    }

    # Find absolute best game score across all generations
    max_score = 0
    max_score_generation = 0
    max_score_individual = None
    max_score_metrics = None

    for gen_num, gen_data in enumerate(generation_history):
        if 'population_metrics' in gen_data and 'metrics' in gen_data['population_metrics']:
            for metric in gen_data['population_metrics']['metrics']:
                if metric and metric['total_score'] > max_score:
                    max_score = metric['total_score']
                    max_score_generation = gen_num
                    # Find corresponding individual
                    for ind in population:
                        if ind.metrics and ind.metrics['total_score'] == max_score:
                            max_score_individual = ind
                            max_score_metrics = metric
                            break


    if max_score_individual is not None:
        performance_metrics['absolute_best'] = {
            'generation': max_score_generation,
            'individual': list(max_score_individual),
            'seed': max_score_individual.seed,
            'fitness_values': max_score_individual.fitness.values,
            'metrics': max_score_metrics,
            'playstyle': analyse_playstyle({'metrics': max_score_metrics}),
            'total_score': max_score
        }
    
    results = {
        'final_population': population,
        'final_archive': archive,
        'logbook': logbook,
        'generation_history': generation_history,
        'archive_history': archive_history,
        'performance_metrics': performance_metrics,
        'replay_info': replay_info,
        'config': {
            'POPULATION_SIZE': POPULATION_SIZE,
            'ARCHIVE_SIZE': archive_size,
            'GENERATIONS': GENERATIONS,
            'MUTATION_RATE': MUTATION_RATE,
            'CROSSOVER_RATE': CROSSOVER_RATE,
            'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
            'LIMITED_ACTIONS': LIMITED_ACTIONS,
            'TOURNAMENT_SIZE': toolbox.tournament_size
        }
    }
    
    # Save final results
    with open('pacman_spea2_final_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def main():
    logger.info("Starting SPEA2 optimisation for Ms. Pac-Man")
    logger.info(f"Generations: {GENERATIONS}")
    logger.info(f"Population size: {POPULATION_SIZE}")
    logger.info(f"Archive size: {POPULATION_SIZE//2}")
    logger.info(f"Number of processes: {NUM_PROCESSES}")

    toolbox = setup_spea2()
    results = run_spea2(toolbox)

    # Initialise variables to track max score info
    max_game_score = 0
    max_score_generation = 0
    max_score_seed = None
    max_score_metrics = None  # Store full metrics for the best agent
  
    # Iterate through generation history
    for gen_data in results['generation_history']:
        current_gen = gen_data['generation']
        if 'metrics' in gen_data:  
            metrics_list = gen_data['metrics']  
            for metric in metrics_list:
                if metric and metric['total_score'] > max_game_score:
                    max_game_score = metric['total_score']
                    max_score_generation = current_gen
                    max_score_metrics = metric
                    max_score_seed = metric.get('seed')  # Get seed directly from metrics 
                    
    # Get playstyle for the best agent
    if max_score_metrics:
        playstyle_info = analyse_playstyle({'metrics': max_score_metrics})
        playstyle = playstyle_info['playstyle'] if playstyle_info else "Unknown"
    else:
        playstyle = "Unknown"
    
    logger.info(f"Optimisation complete - Results saved to pacman_SPEA2_final_results.pkl")
    logger.info(f"Maximum game score achieved: {max_game_score:.0f}")
    logger.info(f"Generation with max score: {max_score_generation}")
    logger.info(f"Seed of max score: {max_score_seed}")
    logger.info(f"Playstyle of best agent: {playstyle}")

if __name__ == "__main__":
    main()
