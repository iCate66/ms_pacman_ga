# ms_pacman_ga: Ms. Pac-Man Evolutionary AI
This project demonstrates concepts in evolutionary computing to explore single-objective optimisation (SOO) and multi-objective optimisation (MOO) problems in the context of an AI agent learning to play a game. 

This repository contains all requisite code for the evolutionary approaches to train AI agents for playing Ms. Pac-Man, and also accompanying Jupyter notebooks to analyse and display results at every stage. Different evolutionary computing methodologies are explored across three progressive stages:

## Project Overview

This project implements and compares different evolutionary computing approaches to create effective Ms. Pac-Man playing agents:

### Stage 1: Single-objective genetic algorithm optimising for game score
   
   - Simple genetic algorithm evolving fixed length action sequences
   - Individuals are represented as sequences of actions
   - Fitness is directly measured by game score
   - Implements crossover, mutation and tournament selection

Key files: <br/><br/> 
[Prototype](1_SOO/1_Prototype/) folder containing:  <br/>  

- [PROTOTYPE.ipynb](1_SOO/1_Prototype/PROTOTYPE.ipynb) -
        Initial prototype in a self-contained Jupyter notebook <br>
- best_agent_episode-7770.mp4 -
        Video playback of best agent resulting from above notebook <br>
        
[Main](1_SOO/2_Main/) folder containing: <br/>
- [genetic_algorithm_multiprocessing_SOO.py](1_SOO/2_Main/genetic_algorithm_multiprocessing_SOO.py) - main SOO implementation featuring multiprocessing for efficiency <br>
- [run_results_from_script.ipynb](1_SOO/2_Main/run_results_from_script.ipynb) - notebook to analyse results from results pickles generated from above simulation <br>
- [compare_parameters_runs.ipynb](1_SOO/2_Main/compare_parameters_runs.ipynb) - notebook to analyse results from multiple simulation runs measuring effects of different combinations of parameters <br>
- [runs_results/](1_SOO/2_Main/runs_results/) - folder containing resulting files (pickles, videos) of simulation runs at different parameter combinations

### Stage 2: Multi-objective evolutionary algorithms 
Extends the evolution to consider multiple objectives with both NSGA-II and SPEA2.

  - NSGA-II and SPEA2 algorithm implementations
  - Three objectives for Pareto-optimal strategy discovery:
    - progression: level completion and maze exploration
    - survival: avoiding ghosts and staying alive for as long as possible
    - efficiency: strategic ghost hunting and power pill usage
  - Gameplay metrics tracking and playstyle analysis for strategic classification

Key files: <br/><br/>
[2_MOO](2_MOO) folder containing: <br>
- [NSGA2](2_MOO/NSGA2) folder containing: <br>
  - [NSGA2.py](2_MOO/NSGA2/MOO_NSGA2.py) - main implementation of NSGA-II simulation
  - 3 folders of different runs from above simulation containing pickle results files and notebook analysis <br>
- [SPEA2](2_MOO/SPEA2) folder containing: <br>
  - [SPEA2.py](2_MOO/SPEA2/MOO_SPEA2.py) - main implementation of SPEA2 simulation
  - 3 folders of different runs from above simulation containing pickle results files and notebook analysis <br>
- [Compare_NSGA2_SPEA2_Analysis.ipynb](2_MOO/Compare_NSGA2_SPEA2_Analysis.ipynb) - comparison analysis of both algorithms based on results from runs from each

### Stage 3: Neuroevolution (NEAT + NSGA-II)
Implements NEAT with NSGA-II to evolve neural network controllers.

- Dynamic network topology evolves with agent
- Island model with migration for improved diversity
- Rich input representation from game state
- Hybrid reward system and behaviour analysis

Key files: <br/><br/> 
[3_MOO_NEAT](3_MOO_NEAT) folder containing: <br>
- [config.py](3_MOO_NEAT/config.py) - configuration parameters
- [neat_config.txt](3_MOO_NEAT/neat_config.txt) - NEAT configuration
- [main.py](3_MOO_NEAT/main.py) - entry point **(run this file)**
- [classes.py](3_MOO_NEAT/classes.py) - data structures and metrics
- [evolution.py](3_MOO_NEAT/evolution.py) - core evolutionary algorithms
- [objectives.py](3_MOO_NEAT/objectives.py) - multi-objective fitness functions
- [evaluation.py](3_MOO_NEAT/evaluation.py) - agent evaluation logic
- [game_utils.py](3_MOO_NEAT/game_utils.py) - neural network vector input construction, object extraction from environment, game state processing utilities
- [test_variables.py](3_MOO_NEAT/test_variables.py) - test suite to verify required variables, imports, functions, dependencies are properly defined
- [run_tests.py](3_MOO_NEAT/run_tests.py) - implement above as unit testing module
- [3_MOO_NEAT/results](3_MOO_NEAT/results) folder containing: <br>
  - [NEAT_and_NSGA-II_Analysis.ipynb](3_MOO_NEAT/results/NEAT_and_NSGA-II_Analysis.ipynb)
  - results pickle from running simulation main.py

### Running the code

## Configuration
- Check configuration for your system and adapt the code for optimal number of core CPUs for NUM_PROCESSES in each .py simulation file as defined below
- Adjust parameters in .py simulation files (config.py for stage 3) to modify evolutionary process
- Modify neat_config.txt to change NEAT-specific parameters
- Objectives can be modified in objectives functions in .py simulation files (objectives.py for stage 3)

**Stage 1: Single-objective genetic algorithm**

Code
> python genetic_algorithm_multiprocessing_SOO.py 
<br><br>

**Stage 2: Multi-objective, NSGA-II vs SPEA2**

Run NSGA-II:

Code
> python MOO_NSGA2.py 
<br><br>

Run SPEA2:

Code:
> python MOO_SPEA2.py
<br><br>

**Stage 3: NEAT +NSGA-II**

Code:
> python main.py
<br><br>

## Acknowledgements

- **OpenAI Gymnasium and ALE-py for Ms. Pac-Man environment**

Bellemare, M. G., Naddaf, Y., Veness, J., Bowling, M. (2013) ‘The Arcade Learning Environment: An Evaluation Platform for General Agents’, Journal of Artificial Intelligence Research, Volume 47, pages 253-279, 2013
  
- **OCAtari for wrapper on Gymnasium environment** to extract object values stored in the RAM to detect the objects currently on the screen from Atari games 

Delfosse, Q., Bluml, J., Gregori, B., Sztwiertnia, S., Kersting, K. (2023) ‘OCAtari: Object-Centric Atari 2600 Reinforcement Learning Environments.’

- **DEAP library for evolutionary algorithm implementations**

Félix-Antoine Fortin, François-Michel De Rainville, Marc-André Gardner, Marc Parizeau and Christian Gagné, "DEAP: Evolutionary Algorithms Made Easy", Journal of Machine Learning Research, vol. 13, pp. 2171-2175, jul 2012

- **NEAT-Python library for neuroevolution**

McIntyre, A., Kallada, M., Miguel, C. G., Feher de Silva, C., & Netto, M. L. neat-python [Computer software]

    
    
   
      
           
     
