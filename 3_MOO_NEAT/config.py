import logging
import neat
from deap import base, creator

# Evolution Parameters
NUM_PROCESSES = 10
N_GENERATIONS = 100
POPULATION_SIZE = 500
SEQUENCE_LENGTH = 10000

# Game Parameters
LIMITED_ACTIONS = [1, 2, 3, 4]
GHOST_SCORE = 200
MAX_EXPECTED_FRAMES = 10000
MAX_STEPS = 10000
INITIAL_LIVES = 3

# Evolution Control Parameters
ELITE_SIZE = 3 # how many overall best-performing individuals are preserved in MOO implementation; not in NEAT's internal operations. Reduce to maintain fewer elites, encourage more exploration
CROSSOVER_RATE = 0.65 # reduced to favour mutation over crossover
MUTATION_RATE = 0.7 # Increase to promote more exploration
MAX_POPULATION_SIZE = 750  # Limit total population across all islands 
POPULATION_GROWTH_RATE = 1.1  # Control growth per generation (+10%)

# Diversity Parameters
DIVERSITY_THRESHOLD = 0.5 # point at which action is taken to increase pop diversity (lower for more convergence, higher for aggresive diversity)
NOVELTY_WEIGHT = 0.7 # novelty over pure performance, higher to prevent premature convergence and promote exploration of solution space
ARCHIVE_SIZE = 150 # size of history of previous behaviours for novelty calculation
K_NEAREST_NEIGHBORS = 30 # how many nearest neighbours are considered in novelty score (most similar behaviours in population archive)

# Game Thresholds
GHOST_PROXIMITY_THRESHOLD = 20
POWER_PILL_EFFECT_TIME = 100
CONSECUTIVE_PILL_THRESHOLD = 5
EXPLORATION_REWARD_THRESHOLD = 0.5

# Scoring Parameters
MAX_EXPECTED_SCORE = 5000
GHOST_HUNT_BONUS = 1.3
EXPLORATION_BONUS = 1.1
CONSECUTIVE_PILL_BONUS = 1.2

# Distance Normalisation - based on screen dimensions
SCREEN_HEIGHT = 210
SCREEN_WIDTH = 160
DISTANCE_NORMALISATION = ((SCREEN_WIDTH ** 2 + SCREEN_HEIGHT ** 2) ** 0.5) / 2

# File Paths
CHECKPOINT_DIR = 'checkpoints'
RESULTS_FILE = 'neat_final_results.pkl'
DEFAULT_CONFIG_PATH = 'neat_config.txt'


# DEAP Setup
def create_fitness_class():
    """Create DEAP fitness and individual classes"""
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

# Logging Setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NEAT Configuration
def load_neat_config(config_path=DEFAULT_CONFIG_PATH):
    """Load NEAT configuration"""
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

# Initialise DEAP creator
create_fitness_class()