import unittest
import sys
import os
import importlib
import inspect
import neat
import numpy as np
from deap import base, creator, tools

class TestVariableDefinitions(unittest.TestCase):
    """Test suite to verify all required variables are properly defined"""
    
    @classmethod
    def setUpClass(cls):
        """Import all project modules for testing"""
        cls.modules = {}
        try:
            cls.modules['config'] = importlib.import_module('config')
            cls.modules['classes'] = importlib.import_module('classes')
            cls.modules['game_utils'] = importlib.import_module('game_utils')
            cls.modules['objectives'] = importlib.import_module('objectives')
            cls.modules['evaluation'] = importlib.import_module('evaluation')
            cls.modules['evolution'] = importlib.import_module('evolution')
        except ImportError as e:
            print(f"Error importing modules: {e}")
            sys.exit(1)

    def test_config_variables(self):
        """Test that all required config variables are defined"""
        config = self.modules['config']
        required_variables = [
            'NUM_PROCESSES', 'N_GENERATIONS', 'POPULATION_SIZE', 'SEQUENCE_LENGTH',
            'LIMITED_ACTIONS', 'GHOST_SCORE', 'MAX_EXPECTED_FRAMES', 'MAX_STEPS',
            'INITIAL_LIVES', 'ELITE_SIZE', 'CROSSOVER_RATE', 'MUTATION_RATE',
            'DIVERSITY_THRESHOLD', 'NOVELTY_WEIGHT', 'ARCHIVE_SIZE',
            'K_NEAREST_NEIGHBORS', 'GHOST_PROXIMITY_THRESHOLD',
            'POWER_PILL_EFFECT_TIME', 'CONSECUTIVE_PILL_THRESHOLD',
            'EXPLORATION_REWARD_THRESHOLD', 'MAX_EXPECTED_SCORE',
            'GHOST_HUNT_BONUS', 'EXPLORATION_BONUS', 'CONSECUTIVE_PILL_BONUS',
            'DISTANCE_NORMALISATION'
        ]
        
        for var in required_variables:
            self.assertTrue(hasattr(config, var), f"Config missing variable: {var}")
            self.assertIsNotNone(getattr(config, var), f"Config variable is None: {var}")

    def test_required_imports(self):
        """Test that all required imports are available"""
        required_imports = {
            'neat': neat,
            'numpy': np,
            'deap.base': base,
            'deap.creator': creator,
            'deap.tools': tools
        }
        
        for name, module in required_imports.items():
            self.assertIsNotNone(module, f"Required import not found: {name}")

    def test_classes_variables(self):
        """Test GameStateMetrics class initialisation"""
        classes = self.modules['classes']
        metrics = classes.GameStateMetrics()
        
        required_attributes = [
            'total_score', 'initial_lives', 'current_lives', 'frames_survived',
            'ghost_points', 'pellet_points', 'total_points_from_actions',
            'scoring_actions', 'deaths', 'action_counts', 'ghost_distance_list',
            'visited_positions', 'coverage_ratio'
        ]
        
        for attr in required_attributes:
            self.assertTrue(hasattr(metrics, attr), f"GameStateMetrics missing attribute: {attr}")

    def test_evaluation_dependencies(self):
        """Test evaluation module dependencies"""
        evaluation = self.modules['evaluation']
        required_functions = [
            'evaluate_genome', 'eval_genome_parallel', 'parallel_eval_genomes'
        ]
        
        for func_name in required_functions:
            self.assertTrue(hasattr(evaluation, func_name),
                        f"Evaluation missing function: {func_name}")
            func = getattr(evaluation, func_name)
            self.assertTrue(callable(func), f"Evaluation attribute not callable: {func_name}")
            
            # Check function signature dependencies
            sig = inspect.signature(func)
            for param in sig.parameters.values():
                if param.default is not inspect.Parameter.empty:
                    # Allow 0 as a valid default value
                    self.assertIn(param.default, [0, None, ...],  # Added 0 as valid default
                            f"Function {func_name} has invalid default for {param.name}")

    def test_game_utils_functions(self):
        """Test game utility functions"""
        game_utils = self.modules['game_utils']
        required_functions = [
            'manhattan_distance', 'extract_objects_info',
            'construct_input_vector', 'update_coverage'
        ]
        
        for func_name in required_functions:
            self.assertTrue(hasattr(game_utils, func_name),
                          f"game_utils missing function: {func_name}")
            self.assertTrue(callable(getattr(game_utils, func_name)),
                          f"game_utils attribute not callable: {func_name}")

    def test_objectives_functions(self):
        """Test objective calculation functions"""
        objectives = self.modules['objectives']
        required_functions = [
            'calculate_progress', 'calculate_survival', 'calculate_efficiency'
        ]
        
        for func_name in required_functions:
            self.assertTrue(hasattr(objectives, func_name),
                          f"objectives missing function: {func_name}")
            func = getattr(objectives, func_name)
            self.assertTrue(callable(func),
                          f"objectives attribute not callable: {func_name}")

    def test_evolution_dependencies(self):
        """Test evolution module dependencies"""
        evolution = self.modules['evolution']
        required_functions = [
            'calculate_behavioral_distance', 'calculate_novelty',
            'run_neat_parallel'
        ]
        
        for func_name in required_functions:
            self.assertTrue(hasattr(evolution, func_name),
                          f"evolution missing function: {func_name}")

    def test_neat_config_file(self):
        """Test NEAT configuration file"""
        config_path = 'neat_config.txt'
        self.assertTrue(os.path.exists(config_path),
                      "NEAT config file not found")
        
        # Test loading config
        try:
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation,
                               config_path)
            self.assertIsNotNone(config)
            self.assertEqual(config.genome_config.num_inputs, 40,
                           "Incorrect number of inputs in NEAT config")
            self.assertEqual(config.genome_config.num_outputs, 4,
                           "Incorrect number of outputs in NEAT config")
        except Exception as e:
            self.fail(f"Failed to load NEAT config: {e}")

    def test_deap_setup(self):
        """Test DEAP setup"""
        self.assertTrue(hasattr(creator, "FitnessMulti"),
                      "DEAP FitnessMulti not defined")
        self.assertTrue(hasattr(creator, "Individual"),
                      "DEAP Individual not defined")
        
        # Test fitness weights
        self.assertEqual(creator.FitnessMulti.weights, (1.0, 1.0, 1.0),
                        "Incorrect fitness weights")

def additional_tests():
    """Run additional variable dependency tests"""
    ###
    pass

if __name__ == '__main__':
    unittest.main()