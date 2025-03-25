import os
import logging
from config import NUM_PROCESSES, N_GENERATIONS, POPULATION_SIZE, logger
from evolution import run_neat_parallel
from game_utils import manhattan_distance
import traceback

# # UNCOMMENT TO TEST for player movement in OCAtari env 
# def test_actions(env):
#     """Test actions are working correctly"""
#     print("Testing basic actions...")
#     env.reset()
    
#     # Skip through intro sequence
#     print("Skipping intro sequence...")
#     for _ in range(150):
#         env.step(0)  # NOOP during intro
    
#     # Get position after intro
#     player_pos = None
#     for obj in env.objects:
#         if hasattr(obj, '_xy') and obj.__class__.__name__ == 'Player':
#             player_pos = obj._xy
#             break
    
#     print(f"Position after intro: {player_pos}")
#     starting_pos = player_pos
    
#     # Test each action
#     for action in range(5):  # 0-4 (NOOP, UP, RIGHT, LEFT, DOWN)
#         # Apply the same action multiple times to see movement
#         for _ in range(5):
#             observation, reward, terminated, truncated, info = env.step(action)
        
#         # Find player position after action
#         player_pos = None
#         for obj in env.objects:
#             if hasattr(obj, '_xy') and obj.__class__.__name__ == 'Player':
#                 player_pos = obj._xy
#                 break
        
#         print(f"Action {action} (repeated 5 times) -> Player position: {player_pos}")
#         if player_pos == starting_pos and action != 0:
#             print(f"  WARNING: No movement detected for action {action}")
    
#     # Reset environment back to starting state
#     env.reset()

def main():
    """Main function to run the NEAT-NSGA-II optimisation"""
    
    logger.info("Starting NEAT optimisation for Ms. Pac-Man with parallel processing")
    logger.info(f"Number of processes: {NUM_PROCESSES}")
    logger.info(f"Number of generations: {N_GENERATIONS}")
    logger.info(f"Warning: takes time to start with population size: {POPULATION_SIZE}")
    
    # Check for config file for NEAT configuration
    config_path = 'neat_config.txt'
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    # # Test the action space with OCAtari
    # from ocatari.core import OCAtari
    # test_env = OCAtari("MsPacman-v4", mode="ram", hud=None, render_mode=None)
    # test_actions(test_env)
    # test_env.close()
    
    try:
        # Run evolution
        winner_pop, stats, results = run_neat_parallel(config_path)
        
        # Before accessing results, ensure junction_points are lists (error handling)
        if hasattr(results, 'best_ever'):
            for objective, data in results['best_ever'].items():
                if data['genome'] and hasattr(data['genome'], 'metrics') and 'junction_points' in data['genome'].metrics:
                    if isinstance(data['genome'].metrics['junction_points'], set):
                        data['genome'].metrics['junction_points'] = list(data['genome'].metrics['junction_points'])
        
        # Get best performing genome
        best_genome = max(winner_pop.population.values(), 
                         key=lambda g: g.total_score if hasattr(g, 'total_score') else 0)
        
        # Log results
        logger.info("\nOptimisation complete")
        logger.info(f"Best genome score: {best_genome.total_score}")
        logger.info("Results saved to neat_final_results.pkl")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(traceback.format_exc())  # For full error trace
        logger.error("Optimisation failed")

if __name__ == '__main__':
    main()