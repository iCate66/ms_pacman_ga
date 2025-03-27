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
   **Prototype**     
     [1_SOO/1_Prototype/PROTOTYPE.ipynb](1_SOO/1_Prototype/PROTOTYPE.ipynb) -
        Initial prototype in a self-contained Jupyter notebook <br>
     [1_SOO/1_Prototype/best_agent_episode-7770.mp4](1_SOO/1_Prototype/best_agent_episode-7770.mp4) -
        Video playback of best agent resulting from above notebook <br><br>
   **Main implementation** <br/>
     [1_SOO/2_Main/genetic_algorithm_multiprocessing_SOO.py](1_SOO/2_Main/genetic_algorithm_multiprocessing_SOO.py) - main SOO implementation featuring multiprocessing for efficiency <br>
     [1_SOO/2_Main/run_results_from_script.ipynb](1_SOO/2_Main/run_results_from_script.ipynb) - notebook to analyse results from results pickles generated from above simulation <br>
     [1_SOO/2_Main/compare_parameters_runs.ipynb](1_SOO/2_Main/compare_parameters_runs.ipynb) - notebook to analyse results from multiple simulation runs measuring effects of different combinations of parameters <br>
     [1_SOO/2_Main/runs_results/](1_SOO/2_Main/runs_results/) - folder containing resulting files (pickles, videos) of simulation runs at different parameter combinations
     
    
    
   
      
           
     
