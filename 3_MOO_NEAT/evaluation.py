import numpy as np
from functools import partial
import multiprocessing
import random
import neat
from ocatari.core import OCAtari
import os

from game_utils import (
    extract_objects_info, construct_input_vector,
    update_coverage, manhattan_distance, get_game_resolution, detect_walls_from_pills, calculate_junction_points, calculate_hybrid_reward  
)

from config import (
    LIMITED_ACTIONS, MAX_EXPECTED_FRAMES, NUM_PROCESSES, MAX_STEPS,
    GHOST_SCORE, INITIAL_LIVES, CONSECUTIVE_PILL_THRESHOLD,
    GHOST_HUNT_BONUS, EXPLORATION_BONUS, CONSECUTIVE_PILL_BONUS,
    DISTANCE_NORMALISATION, GHOST_PROXIMITY_THRESHOLD,
    logger, creator
)

from objectives import (
    calculate_progress, calculate_survival,
    calculate_efficiency
)
from classes import GameStateMetrics


def evaluate_genome(genome, config, seed=0):
    """Main genome evaluation function"""
    try:
        # Create the neural network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Initialise the environment
        env = OCAtari("MsPacman-v4", mode="ram", hud=None, render_mode=None)

        # Generate a random seed if none provided
        if not seed:
            seed = random.randint(1, 1000000)

        # Reset environment with seed
        observation, info = env.reset(seed=seed)

        steps = 0
        terminated = False
        truncated = False

        # Initialise action history tracking
        action_history = []

        # Skip intro sequence
        for _ in range(150):
            env.step(0)  # Use NOOP during intro

        # Get resolution info once at the start
        resolution_info = get_game_resolution(env)

        # # Get initial player position for debugging
        # initial_player_pos, _, _, _ = extract_objects_info(env)
        # print(f"Initial player position: {initial_player_pos}")

        # Calculate network complexity
        hidden_count = len([n for n in genome.nodes.keys() 
                           if n not in config.genome_config.input_keys and 
                           n not in config.genome_config.output_keys])
        connection_count = len([c for c in genome.connections.values() if c.enabled])
        network_complexity = hidden_count + connection_count

        # Initialise tracking
        player_pos, ghost_positions, initial_pill_positions, initial_powerpill_positions = extract_objects_info(env)
        # logger.info(f"Detected {len(initial_powerpill_positions)} powerpills in the maze")
        # logger.info(f"Detected {len(initial_pill_positions)} pills in the maze")
        
        original_pill_locations = set(initial_pill_positions)
        visited_pill_locations = set() # initialised as an empty set
        metrics = GameStateMetrics(resolution_info) # Pass resolution info 

        # Add network complexity to metrics
        metrics.network_complexity = network_complexity

        # Calculate junction points from initial pill positions
        junction_points = calculate_junction_points(initial_pill_positions)
        # logger.info(f"Detected {len(junction_points)} junction points")
        
        # save to env
        env.junction_points = junction_points 

        # save to metrics
        metrics.junction_points = junction_points

        # # During evaluation loop, verify they are being accessed
        # if hasattr(env, 'junction_points'):
        #     logger.info(f"Env has {len(env.junction_points)} junction points")
        # else:
        #     logger.warning("Env does not have junction_points attribute")

        # Store all pill positions for wall detection
        all_pill_positions = initial_pill_positions.copy()
        all_powerpill_positions = initial_powerpill_positions.copy()

        # # Debug logging for first 10 steps
        # for step in range(10):
        #     inputs = construct_input_vector(env, resolution_info)
        #     output = net.activate(inputs)
        #     action_idx = int(np.argmax(output)) % len(LIMITED_ACTIONS)
        #     action = LIMITED_ACTIONS[action_idx]
            
        #     print(f"Step {step}: Action selected: {action}")
        #     observation, reward, terminated, truncated, info = env.step(action)
            
        #     prev_x, prev_y = player_pos

        #     player_pos, ghost_positions, pill_positions, powerpill_positions = extract_objects_info(env)
        #     print(f"  New position: {player_pos}")
        
        # Additional tracking
        consecutive_pill_collection = 0
        ghost_hunt_success = 0
        exploration_score = 0
        power_pill_timing = 0
        last_score = 0
        last_pos = player_pos
        movement_diversity = set()

        # Hybrid reward tracking
        total_hybrid_reward = 0
        prev_reward = 0

        # Initialise milestone tracking attributes
        metrics.milestone_50_percent = False
        metrics.milestone_75_percent = False
        metrics.milestone_90_percent = False
        metrics.milestone_100_percent = False
        
        last_pill_count = metrics.current_pill_count
        no_pill_collection_counter = 0  # Track stagnation in pill collection
        
        while not (terminated or truncated) and steps < MAX_STEPS:
            inputs = construct_input_vector(env, resolution_info)
            output = net.activate(inputs)
            action_idx = int(np.argmax(output)) % len(LIMITED_ACTIONS)
            action = LIMITED_ACTIONS[action_idx]

            # Record actions
            action_history.append(action)
            
            observation, reward, terminated, truncated, info = env.step(action)
            # player_pos, ghost_positions, pill_positions, powerpill_positions = extract_objects_info(env)

            # Extract game state with error reporting
            try:
                player_pos, ghost_positions, pill_positions, powerpill_positions = extract_objects_info(env)
                # logger.info(f"Step {steps}: player_pos type: {type(player_pos)}")
                # logger.info(f"Step {steps}: ghost_positions type: {type(ghost_positions)}")
                # logger.info(f"Step {steps}: pill_positions type: {type(pill_positions)}")
                # logger.info(f"Step {steps}: powerpill_positions type: {type(powerpill_positions)}")
                # if player_pos is not None:
                    # logger.info(f"player_pos value: {player_pos}")
                    # logger.info(f"ghost_positions value: {ghost_positions}")
                    # logger.info(f"pill_positions value: {pill_positions}")
                    # logger.info(f"powerpill_positions value: {powerpill_positions}")
            except Exception as e:
                logger.error(f"Error extracting objects at step {steps}: {e}")
                
            # Track pill collection stagnation after metrics have been updated at least once
            if hasattr(metrics, 'current_pill_count') and metrics.current_pill_count is not None:
                if steps > 0:  # Make sure gone through at least one update
                    last_pill_count = getattr(metrics, 'last_pill_count', metrics.current_pill_count)
                    
                    if metrics.current_pill_count == last_pill_count:
                        no_pill_collection_counter += 1
                    else:
                        no_pill_collection_counter = 0
                    
                    # Store current count for next comparison
                    metrics.last_pill_count = metrics.current_pill_count
                    
                    # If agent is stuck without collecting pills for too long, add penalty
                    if no_pill_collection_counter > 100:  # Stuck for 100 frames
                        metrics.hybrid_rewards -= 0.1
                        no_pill_collection_counter = 0  # Reset counter
            
            # Add bonuses for significant pill collection milestones after metrics have been updated at least once
            if hasattr(metrics, 'current_pill_count') and hasattr(metrics, 'initial_pill_count'):
                if metrics.current_pill_count is not None and metrics.initial_pill_count is not None and metrics.initial_pill_count > 0:
                    pills_collected = metrics.initial_pill_count - metrics.current_pill_count
                    collection_percent = pills_collected / metrics.initial_pill_count
                    
                    # Check milestones
                    if not metrics.milestone_50_percent and collection_percent >= 0.5:
                        metrics.milestone_50_percent = True
                        metrics.hybrid_rewards += 0.5  # Bonus for 50% completion
                    
                    if not metrics.milestone_75_percent and collection_percent >= 0.75:
                        metrics.milestone_75_percent = True
                        metrics.hybrid_rewards += 0.6  # Bonus for 75% completion
                    
                    if not metrics.milestone_90_percent and collection_percent >= 0.9:
                        metrics.milestone_90_percent = True
                        metrics.hybrid_rewards += 0.7  # Bonus for 90% completion
                    
                    # Huge bonus for level completion
                    if not metrics.milestone_100_percent and collection_percent >= 0.99:
                        metrics.milestone_100_percent = True
                        metrics.hybrid_rewards += 1.0  # Big bonus for level completion
            
            # Calculate hybrid reward
            hybrid_reward = calculate_hybrid_reward(
                player_pos=player_pos,
                prev_pos=last_pos,
                ghost_positions=ghost_positions,
                score_change=reward,
                junction_points=junction_points,
                metrics=metrics,  
                resolution_info=resolution_info
            )
            
            total_hybrid_reward += hybrid_reward

            # Detect walls
            walls = detect_walls_from_pills(player_pos, all_pill_positions, all_powerpill_positions)

            # Store wall information in env
            env.current_walls = walls

            # Store wall information in metrics
            metrics.walls_detected = walls
            
            # Update tracking
            if player_pos and last_pos:
                dx = player_pos[0] - last_pos[0]
                dy = player_pos[1] - last_pos[1]
                if dx != 0 or dy != 0:
                    movement_diversity.add((dx, dy))
            last_pos = player_pos

            # Track junction visits
            if player_pos is not None and junction_points:
                # Convert to list if it might be a set
                junction_points_list = list(junction_points) if isinstance(junction_points, set) else junction_points
                if any(manhattan_distance(player_pos, j) < 5 for j in junction_points_list):
                    metrics.times_at_junction += 1

            # After calculating junction points
            # logger.info(f"Number of junction points detected: {len(junction_points)}")
            # if len(junction_points) > 0:
            #     logger.info(f"Sample junction points: {junction_points[:3]}")
            
            if player_pos is not None:
                update_coverage(player_pos, original_pill_locations, visited_pill_locations)
            
            # Track game events
            if reward > 0 and reward < GHOST_SCORE:
                consecutive_pill_collection += 1
            else:
                consecutive_pill_collection = 0
            
            if reward >= GHOST_SCORE:
                ghost_hunt_success += 1
                power_pill_timing += 1
                # Add to decision quality score for successful ghost hunting
                metrics.decision_quality += 0.5
            
            if len(visited_pill_locations) > exploration_score:
                exploration_score = len(visited_pill_locations)
                # Add to exploration score
                metrics.exploration_score += 0.01
            
            lives = info.get('lives', 3) if isinstance(info, dict) else 3
            frame_number = info.get('frame_number', steps) if isinstance(info, dict) else steps

            # Update ghost avoidance score
            if ghost_positions and player_pos:
                min_ghost_dist = min(manhattan_distance(player_pos, g) for g in ghost_positions)
                if min_ghost_dist > 10:  # Maintaining safe distance
                    metrics.ghost_avoidance_score += 0.01
            
            metrics.update(
                reward=reward,
                lives=lives,
                frame_number=frame_number,
                action=action_idx,
                player_pos=player_pos,
                ghost_positions=ghost_positions,
                pill_positions=pill_positions,
                powerpill_positions=powerpill_positions
            )

            # Update hybrid reward in metrics
            metrics.hybrid_rewards = total_hybrid_reward
            
            steps += 1
            prev_reward = reward
        
        env.close()

        # Calculate objectives 
        progression_objective = calculate_progress(
            metrics=metrics,
            visited_pill_count=len(visited_pill_locations),
            total_pill_count=len(original_pill_locations)
        )
        
        survival_objective = calculate_survival(
            metrics=metrics,
            player_pos=player_pos,
            ghost_positions=ghost_positions
        )
        
        efficiency_objective = calculate_efficiency(
            metrics=metrics,
            player_pos=player_pos,
            ghost_positions=ghost_positions
        )

        # Apply strategic bonuses
        if consecutive_pill_collection > CONSECUTIVE_PILL_THRESHOLD:
            progression_objective *= CONSECUTIVE_PILL_BONUS
        if ghost_hunt_success > 0:
            efficiency_objective *= GHOST_HUNT_BONUS
        if exploration_score > len(original_pill_locations) / 2:
            progression_objective *= EXPLORATION_BONUS

        # Get metrics dict
        metrics_dict = metrics.get_metrics()
        
        # Embed the seed directly in the metrics
        metrics_dict['seed'] = seed

        # Add action history to metrics
        metrics_dict['action_history'] = action_history
        
        # Also store the objective values in metrics
        metrics_dict['objective_values'] = {
            'progression': progression_objective,
            'survival': survival_objective,
            'efficiency': efficiency_objective
        }
        
        # Store the metrics dict on the genome
        genome.metrics = metrics_dict
        genome.total_score = metrics.total_score
        genome.objective_values = {
            'progression': progression_objective,
            'survival': survival_objective,
            'efficiency': efficiency_objective
        }

        # Return the fitness values and metrics (with seed embedded)
        return (
            (min(1.0, progression_objective),
             min(1.0, survival_objective),
             min(1.0, efficiency_objective)),
            metrics_dict
        )
        
    except Exception as e:
        logger.error(f"Error in evaluate_genome: {str(e)}")
        # Return minimal metrics with seed still included
        return ((0.0, 0.0, 0.0), {'total_score': 0, 'seed': seed})
      

def eval_genome_parallel(genome_config_tuple, seed=0):
    """Wrapper for parallel genome evaluation"""
    genome_id, genome, config = genome_config_tuple

    if not seed:
        seed = random.randint(1, 1000000)
    
    try:
        # Evaluate the genome
        fitness_tuple, metrics = evaluate_genome(genome, config, seed)

        return genome_id, fitness_tuple, metrics
    except Exception as e:
        logger.error(f"Error in eval_genome_parallel for genome {genome_id}: {str(e)}")
        return genome_id, (0.0, 0.0, 0.0), {'total_score': 0}, seed

def parallel_eval_genomes(genomes, config):
    """Parallel implementation of population evaluation"""
    evaluator = partial(eval_genome_parallel)
    genome_tuples = [(gid, genome, config) for gid, genome in genomes]
    
    successful_evaluations = 0
    total_evaluations = len(genome_tuples)

    # Track best genomes by objective
    best_by_objective = {
        'overall': {'genome_id': None, 'score': 0, 'seed': None},
        'progression': {'genome_id': None, 'score': 0, 'seed': None},
        'survival': {'genome_id': None, 'score': 0, 'seed': None},
        'efficiency': {'genome_id': None, 'score': 0, 'seed': None}
    }
    
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        try:
            results = pool.map(evaluator, genome_tuples)

            for genome_id, fitness_tuple, metrics in results:
                try:
                    for gid, genome in genomes:
                        if gid == genome_id:
                            genome.fitness = sum(fitness_tuple)
                            genome.multi_fitness = creator.FitnessMulti(fitness_tuple)
                            genome.metrics = metrics
                            genome.total_score = metrics.get('total_score', 0)

                             # Get seed from metrics
                            seed = metrics.get('seed')

                            # Track best by overall score
                            if genome.total_score > best_by_objective['overall']['score']:
                                best_by_objective['overall'] = {
                                    'genome_id': genome_id,
                                    'score': genome.total_score,
                                    'seed': seed
                                }
                            
                            # Track best by progression (objective 0)
                            if fitness_tuple[0] > best_by_objective['progression']['score']:
                                best_by_objective['progression'] = {
                                    'genome_id': genome_id,
                                    'score': fitness_tuple[0],
                                    'seed': seed
                                }
                            
                            # Track best by survival (objective 1)
                            if fitness_tuple[1] > best_by_objective['survival']['score']:
                                best_by_objective['survival'] = {
                                    'genome_id': genome_id,
                                    'score': fitness_tuple[1],
                                    'seed': seed
                                }
                            
                            # Track best by efficiency (objective 2)
                            if fitness_tuple[2] > best_by_objective['efficiency']['score']:
                                best_by_objective['efficiency'] = {
                                    'genome_id': genome_id,
                                    'score': fitness_tuple[2],
                                    'seed': seed
                                }
                      
                            if genome.total_score > 0:
                                successful_evaluations += 1
                            
                            break

                except Exception as e:
                    logger.error(f"Error processing genome {genome_id}: {str(e)}")
                    genome.fitness = 0.0
                    genome.multi_fitness = creator.FitnessMulti((0.0, 0.0, 0.0))
                    genome.total_score = 0

            # DIAGNOSTIC CODE: Check fitness diversity after evaluation
            fitness_values = {}
            for _, genome in genomes:
                if hasattr(genome, 'multi_fitness') and hasattr(genome.multi_fitness, 'values'):
                    fitness_values[genome.key] = genome.multi_fitness.values
            
            # Check for uniqueness
            unique_values = set(tuple(v) for v in fitness_values.values())
            logger.info(f"DIAGNOSTIC: Number of unique fitness value combinations: {len(unique_values)}")
            logger.info(f"DIAGNOSTIC: Total number of genomes with fitness: {len(fitness_values)}")
            # logger.info(f"DIAGNOSTIC: Number of genomes with tracked seeds: {seeds_tracked}")
            if len(unique_values) < 5:
                logger.warning("DIAGNOSTIC: Very low fitness diversity detected!")
                logger.warning(f"DIAGNOSTIC: Sample of values: {list(unique_values)[:5]}")

            # # Save best genomes to file for reference
            # try:
            #     with open('best_genomes_by_objective.txt', 'w') as f:
            #         for objective, data in best_by_objective.items():
            #             if 'genome_id' in data and 'seed' in data:
            #                 f.write(f"{objective}: genome_id={data['genome_id']}, seed={data['seed']}, score={data['score']}\n")
            # except Exception as e:
            #     logger.error(f"Error writing best genomes file: {e}")
            
            logger.info(f"Evaluation complete: {successful_evaluations}/{total_evaluations} genomes scored > 0")
                    
        except Exception as e:
            logger.error(f"Error in parallel evaluation: {str(e)}")
            for _, genome in genomes:
                if not hasattr(genome, 'fitness') or genome.fitness is None:
                    genome.fitness = 0.0
                if not hasattr(genome, 'multi_fitness') or genome.multi_fitness is None:
                    genome.multi_fitness = creator.FitnessMulti((0.0, 0.0, 0.0))
                if not hasattr(genome, 'total_score'):
                    genome.total_score = 0
            