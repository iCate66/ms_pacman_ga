import numpy as np
from game_utils import manhattan_distance
from config import (
    MAX_EXPECTED_FRAMES, GHOST_SCORE, MAX_EXPECTED_SCORE,
    GHOST_PROXIMITY_THRESHOLD, POWER_PILL_EFFECT_TIME
)

def calculate_progress(metrics, visited_pill_count, total_pill_count):
    """Progression objective with focus on level completion"""
    m = metrics.get_metrics()
    
    # Pill consumption ratio for higher completion
    if m['initial_pill_count'] is None or m['initial_pill_count'] == 0:
        regular_pill_ratio = 0
    else:
        # Calculate completion percentage
        completion_percent = (m['initial_pill_count'] - m['current_pill_count']) / m['initial_pill_count']
        
        # Apply quadratic scaling to reward higher completion more
        regular_pill_ratio = completion_percent ** 1.5
        
        # Apply bonuses for significant completion milestones
        if completion_percent > 0.2:
            regular_pill_ratio *= 1.2  # 20% bonus for 20%+ completion

        if completion_percent > 0.4:
            regular_pill_ratio *= 1.3  # 30% bonus for 40%+ completion

        if completion_percent > 0.6:
            regular_pill_ratio *= 1.4  # 40% bonus for 60%+ completion
            
        if completion_percent > 0.8:
            regular_pill_ratio *= 1.5  # 50% bonus for 80%+ completion
    
    # Movement diversity based on junction visits
    junction_visits = m.get('times_at_junction', 0)
    junction_score = min(1.0, junction_visits / 20.0)  

    # Score progression
    max_expected_score = 5000
    score_ratio = min(1.0, m['total_score'] / max_expected_score)
    
    return (0.6 * regular_pill_ratio +   
            0.2 * junction_score +       
            0.2 * score_ratio)
            # 0.1 * m.get('exploration_score',0))


def calculate_survival(metrics, player_pos, ghost_positions):
    """Survival objective with stronger ghost avoidance focus"""
    m = metrics.get_metrics()
    
    # Exponentially increase the death penalty
    survival_rate = max(0, 1.0 - (metrics.deaths ** 1.7 * 0.2))
    
    # Ghost proximity avoidance 
    if m['ghost_proximity_history']:
        close_encounters = m['frequency_low_ghost_distance']
        # Apply a stronger power transform
        ghost_avoidance = (1.0 - close_encounters) ** 2.0 
    else:
        ghost_avoidance = 0.4
    
    # Longevity bonus - reward longer survival
    time_bonus = (min(1.0, metrics.frames_survived / MAX_EXPECTED_FRAMES)) ** 0.8  
    
    # Ghost avoidance score - increased weight
    ghost_avoidance_score = min(1.0, m.get('ghost_avoidance_score', 0) * 2.5)  
    
    # Early termination penalty
    early_term_penalty = max(0.0, 1.0 - (200.0 / max(200, metrics.frames_survived)))

    # Add bonus for handling dangerous situations successfully
    dangerous_situations = sum(1 for dist, _ in m.get('ghost_proximity_history', []) if dist < 15)
    danger_handling = min(1.0, dangerous_situations / 50.0)  # Cap at 50 dangerous situations
     
    return (0.15 * survival_rate +          
            0.30 * ghost_avoidance +       
            0.15 * time_bonus +             
            0.15 * ghost_avoidance_score +  
            0.10 * early_term_penalty +    
            0.15 * danger_handling)        


def calculate_efficiency(metrics, player_pos, ghost_positions):
    """Calculate efficiency objective"""
    m = metrics.get_metrics()
    
    # Ghost hunting efficiency
    ghost_hunting_score = 0.0
    if m['total_score'] > 0:
        ghost_points_ratio = m['ghost_points'] / m['total_score']
        ghost_hunting_score = ghost_points_ratio ** 0.5
    
    # Risk taking
    ghost_proximity_score = 0.0
    if m['ghost_proximity_history']:
        close_encounters = m['frequency_low_ghost_distance']
        ghost_proximity_score = close_encounters ** 0.7
    
    # Power pill usage
    power_pill_aggression = 0.0
    powerpills_consumed = (m['initial_powerpill_count'] - m['current_powerpill_count'])
    if powerpills_consumed > 0:
        points_per_powerpill = m['ghost_points'] / (powerpills_consumed * GHOST_SCORE)
        power_pill_aggression = min(1.0, points_per_powerpill * 1.5)
    
    # Scoring speed
    points_per_frame = m['total_score'] / max(1, m['frames_survived'])
    speed_score = min(1.0, points_per_frame / 8.0)
    
    return (0.35 * ghost_hunting_score +
            0.20 * ghost_proximity_score +
            0.25 * power_pill_aggression +
            0.20 * speed_score)