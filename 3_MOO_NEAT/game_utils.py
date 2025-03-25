from config import logger
import numpy as np
import math

from config import (
    DISTANCE_NORMALISATION, GHOST_PROXIMITY_THRESHOLD,
    SCREEN_HEIGHT, SCREEN_WIDTH,
    logger
)

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_game_resolution(env):
    """Get the actual game resolution for position normalisation"""
    # OCAtari observation shape is (210, 160, 3)
    SCREEN_HEIGHT = 210
    SCREEN_WIDTH = 160
    
    # Track min/max positions of objects to determine actual play area
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    # Check current objects without resetting
    for obj in env.objects:
        if hasattr(obj, '_xy') and obj._xy:
            x, y = obj._xy
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    
    # If no bounds determined, use screen dimensions
    if min_x == float('inf'):
        min_x, max_x = 0, SCREEN_WIDTH
        min_y, max_y = 0, SCREEN_HEIGHT
    
    # Use reasonable defaults based on Ms. Pac-Man screen layout
    if max_x - min_x < 10:  # Unreasonable width
        min_x, max_x = 0, 160
    if max_y - min_y < 10:  # Unreasonable height
        min_y, max_y = 0, 170  # Adjust for score area
    
    return {
        'width': max_x - min_x,
        'height': max_y - min_y,
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'screen_width': SCREEN_WIDTH,
        'screen_height': SCREEN_HEIGHT
    }

def extract_objects_info(env):
    """Extract object positions from OCAtari environment"""
    player_pos = None
    ghost_positions = []
    pill_positions = []
    powerpill_positions = []
    
    for obj in env.objects:
        try:
            obj_name = obj.__class__.__name__

            # # DEBUG: Print object type and attributes
            # print(f"Object type: {obj.__class__.__name__}")
            # print(f"Has _xy: {hasattr(obj, '_xy')}")
            
            # Get position data
            if hasattr(obj, '_xy') and obj._xy:
                pos_x, pos_y = obj._xy
            elif hasattr(obj, 'xy') and obj.xy:
                pos_x, pos_y = obj.xy
            elif hasattr(obj, 'x') and hasattr(obj, 'y'):
                pos_x, pos_y = obj.x, obj.y
            else:
                continue  # Skip objects without position data
                
            # Add to assigned list based on object type
            lower_name = obj_name.lower()
            if "player" in lower_name:
                player_pos = (pos_x, pos_y)
            elif "ghost" in lower_name:
                ghost_positions.append((pos_x, pos_y))
            elif "powerpill" in lower_name:
                powerpill_positions.append((pos_x, pos_y))
            elif "pill" in lower_name:
                pill_positions.append((pos_x, pos_y))
                
        except Exception as e:
            print(f"Error accessing object {obj_name} attributes: {e}")
            continue
    
    return player_pos, ghost_positions, pill_positions, powerpill_positions

def normalise_position(pos, resolution_info):
    """Normalise a position based on game resolution"""
    if not pos:
        return (0.0, 0.0)
    
    x, y = pos
    norm_x = (x - resolution_info['min_x']) / (resolution_info['width'])
    norm_y = (y - resolution_info['min_y']) / (resolution_info['height'])
    return (
        max(0.0, min(1.0, norm_x)),
        max(0.0, min(1.0, norm_y))
    )

def calculate_normalised_velocity(current_pos, prev_pos, resolution_info):
    """Calculate normalised velocity vector"""
    if not (current_pos and prev_pos) or prev_pos == (0, 0):
        return (0.0, 0.0)
        
    dx = (current_pos[0] - prev_pos[0]) / resolution_info['width']
    dy = (current_pos[1] - prev_pos[1]) / resolution_info['height']
    return (dx, dy)

def construct_input_vector(env, resolution_info):
    """Construct a normalised input vector (40 inputs) for neural network"""
    player_info = None
    ghosts_info = []
    pills_info = []
    powerpills_info = []
    inputs = []
    
    # Process all objects
    for obj in env.objects:
        obj_type = obj.__class__.__name__
        
        if obj_type == "NoObject":
            continue

        # Ensure default values if objects are not found
        if player_info is None:
            player_info = {'position': (0.5, 0.5), 'velocity': (0.0, 0.0)}
            
        pos = obj._xy if hasattr(obj, '_xy') else None
        prev_pos = obj._prev_xy if hasattr(obj, '_prev_xy') else None
        
        if not pos:
            continue
            
        norm_pos = normalise_position(pos, resolution_info)
        velocity = calculate_normalised_velocity(pos, prev_pos, resolution_info)
        
        obj_data = {
            'position': norm_pos,
            'velocity': velocity
        }
        
        if "Player" in obj_type:
            player_info = obj_data
        elif "Ghost" in obj_type:
            ghosts_info.append(obj_data)
        elif "PowerPill" in obj_type:
            powerpills_info.append(obj_data)
        elif "Pill" in obj_type:
            pills_info.append(obj_data)
    
    # 1. PLAYER INFORMATION (2 values) - position
    if player_info:
        inputs.extend([
            player_info['position'][0],  # X position
            player_info['position'][1],  # Y position
        ])
    else:
        inputs.extend([0.5, 0.5])  # Default centre position


    # Reference to player position for later use
    player_pos = player_info['position'] if player_info else (0.5, 0.5)
    
    # 2. GHOST INFORMATION (12 values: 3 per ghost)
    # For each ghost: x, y, distance to player
    for i in range(4):  # For each ghost
        if i < len(ghosts_info):
            ghost = ghosts_info[i]
            ghost_pos = ghost['position']
            distance = ((ghost_pos[0] - player_pos[0])**2 + 
                       (ghost_pos[1] - player_pos[1])**2)**0.5
            
            inputs.extend([
                ghost_pos[0],                # X position
                ghost_pos[1],                # Y position
                min(1.0, distance)           # Distance to player
            ])
        else:
            inputs.extend([0.0, 0.0, 1.0])  # Default values
    
    # 3. NEAREST GHOST DIRECTION (4 values)
    # Directional information about the nearest ghost
    nearest_ghost_dir = [0.0, 0.0, 0.0, 0.0]  # Default: no direction (up, right, down, left)
    if ghosts_info:
        # Find nearest ghost
        distances = [(((g['position'][0] - player_pos[0])**2 + 
                     (g['position'][1] - player_pos[1])**2)**0.5, i) 
                    for i, g in enumerate(ghosts_info)]
        nearest_idx = min(distances, key=lambda x: x[0])[1]
        nearest_ghost = ghosts_info[nearest_idx]
        
        # Calculate direction vector
        dx = nearest_ghost['position'][0] - player_pos[0]
        dy = nearest_ghost['position'][1] - player_pos[1]
        
        # Set directional flags (stronger in the dominant direction)
        if abs(dx) > abs(dy):  # Horizontal dominant
            if dx > 0:
                nearest_ghost_dir[1] = 1.0  # Right
            else:
                nearest_ghost_dir[3] = 1.0  # Left
        else:  # Vertical dominant
            if dy > 0:
                nearest_ghost_dir[2] = 1.0  # Down
            else:
                nearest_ghost_dir[0] = 1.0  # Up
    
    inputs.extend(nearest_ghost_dir)
    
    # 4. POWER PILL INFORMATION (8 values)
    # For each power pill: x, y (max 4 power pills)
    for i in range(4):
        if i < len(powerpills_info):
            pp = powerpills_info[i]
            pp_pos = pp['position']
            inputs.extend([
                pp_pos[0],  # X position
                pp_pos[1],  # Y position
            ])
        else:
            inputs.extend([0.0, 0.0])
    
    # 5. NEAREST PILL DIRECTION (4 values)
    # Direction to the nearest regular pill
    nearest_pill_dir = [0.0, 0.0, 0.0, 0.0]  # Default: no direction (up, right, down, left)
    if pills_info:
        # Sort pills by distance to player
        sorted_pills = sorted(
            pills_info,
            key=lambda p: ((p['position'][0] - player_pos[0])**2 + 
                          (p['position'][1] - player_pos[1])**2)
        )
        
        if sorted_pills:
            nearest_pill = sorted_pills[0]
            # Calculate direction vector
            dx = nearest_pill['position'][0] - player_pos[0]
            dy = nearest_pill['position'][1] - player_pos[1]
            
            # Set directional flags (stronger in the dominant direction)
            if abs(dx) > abs(dy):  # Horizontal dominant
                if dx > 0:
                    nearest_pill_dir[1] = 1.0  # Right
                else:
                    nearest_pill_dir[3] = 1.0  # Left
            else:  # Vertical dominant
                if dy > 0:
                    nearest_pill_dir[2] = 1.0  # Down
                else:
                    nearest_pill_dir[0] = 1.0  # Up
    
    inputs.extend(nearest_pill_dir)
    
    # 6. WALL DETECTION (4 values)
    # Use detect_walls_from_pills derived means for wall detection as OCAtari does not provide wall/corridor detection
    all_pill_positions = [p['position'] for p in pills_info]
    all_powerpill_positions = [p['position'] for p in powerpills_info]

    # Get original coordinates (not normalised) for detect_walls_from_pills
    player_orig_pos = None
    for obj in env.objects:
        if hasattr(obj, '_xy') and obj.__class__.__name__ == 'Player':
            player_orig_pos = obj._xy
            break

    if player_orig_pos:
        walls = detect_walls_from_pills(player_orig_pos, all_pill_positions, all_powerpill_positions, threshold=10)
        inputs.extend([
            1.0 if walls.get('up', True) else 0.0,
            1.0 if walls.get('right', True) else 0.0, 
            1.0 if walls.get('down', True) else 0.0,
            1.0 if walls.get('left', True) else 0.0
        ])
    else:
        # Default wall detection if player position unknown
        inputs.extend([1.0, 1.0, 1.0, 1.0])
    
    # 7. GAME STATE (6 values)
    # Power state and remaining pills ratio
    power_active = 0.0
    if hasattr(env, 'metrics') and hasattr(env.metrics, 'get_metrics'):
        m = env.metrics.get_metrics()
        power_frames = m.get('powerpill_consumption_frames', [])
        if power_frames and m.get('frames_survived'):
            # Consider powered if used power pill in last 100 frames
            power_active = 1.0 if (m['frames_survived'] - max(power_frames)) < 100 else 0.0
        
        # Remaining pill ratio
        if m.get('initial_pill_count') is not None and m.get('initial_pill_count') > 0:
            remaining_ratio = m.get('current_pill_count', 0) / m.get('initial_pill_count', 1)
            inputs.append(remaining_ratio)
        else:
            inputs.append(1.0)
    else:
        inputs.extend([0.0, 1.0])  # Default values
    # Final input: Power active state
    inputs.append(power_active) # Should be 3 total for this section (power state + pill ratio + duplicate power state)

    # Power pill count
    power_pill_count = min(1.0, len(powerpills_info) / 4.0)  # Normalise to 0-1
    inputs.append(power_pill_count)
    
    # Remaining lives (normalised)
    if hasattr(env, 'metrics') and hasattr(env.metrics, 'get_metrics'):
        m = env.metrics.get_metrics()
        lives = m.get('current_lives', 3)
        inputs.append(lives / 3.0)  # Normalise to 0-1
    else:
        inputs.append(1.0)  # Default full lives
    
    # Simple time progress indicator
    if hasattr(env, 'metrics') and hasattr(env.metrics, 'get_metrics'):
        m = env.metrics.get_metrics()
        frames = m.get('frames_survived', 0)
        inputs.append(min(1.0, frames / 10000.0))  # Normalise to 0-1 up to 10000 frames
    else:
        inputs.append(0.0)  # Default zero progress
        
    # Verify 40 inputs
    assert len(inputs) == 40, f"Expected 40 inputs, got {len(inputs)}"
    
    return np.array(inputs)

def update_coverage(player_pos, original_pill_locations, visited_pill_locations, threshold=10):
    """Update coverage tracking for visited pills"""
    if player_pos is None:
        return
        
    for pill_loc in original_pill_locations:
        if pill_loc not in visited_pill_locations:
            if manhattan_distance(player_pos, pill_loc) < threshold:
                visited_pill_locations.add(pill_loc)


# implement a wall detection mechanism by observing player positions relative to initial pill positions
def detect_walls_from_pills(player_pos, pill_positions, powerpill_positions, threshold=10):
    """Detect walls by analysing pill placement around the player position"""
    walls = {
        'up': True,     # Assume walls by default
        'right': True,
        'down': True,
        'left': True
    }
    
    if not player_pos:
        return walls
    
    x, y = player_pos
    
    # Look for pills/paths in each direction
    for pill_pos in pill_positions + powerpill_positions:
        px, py = pill_pos
        
        # Check if pill is aligned horizontally or vertically with player
        if abs(px - x) < threshold:  # Vertically aligned
            if py < y and abs(py - y) < threshold:
                walls['up'] = False    # No wall above
            elif py > y and abs(py - y) < threshold:
                walls['down'] = False  # No wall below
                
        if abs(py - y) < threshold:  # Horizontally aligned
            if px > x and abs(px - x) < threshold:
                walls['right'] = False  # No wall to right
            elif px < x and abs(px - x) < threshold:
                walls['left'] = False   # No wall to left
    
    return walls

def calculate_junction_points(pill_positions, threshold=10):
    """Enhanced junction point detection for Ms. Pac-Man maze"""
    junctions = []
    
    # Convert pill positions to a set for faster lookups
    pill_set = set(pill_positions)
    
    # For each pill, check potential junction
    for x, y in pill_positions:
        # Count available directions (up, right, down, left)
        directions = 0
        direction_vectors = [(0, -threshold), (threshold, 0), (0, threshold), (-threshold, 0)]
        
        for dx, dy in direction_vectors:
            # Check if there's a pill in this direction
            nearby_pills = [
                (x+dx, y+dy),
                (x+dx//2, y+dy//2),  # Check halfway to handle irregular spacing
                (x+dx*2//3, y+dy*2//3)
            ]
            
            for nearby in nearby_pills:
                if any(manhattan_distance(nearby, p) < threshold for p in pill_positions):
                    directions += 1
                    break
        
        # If this pill has paths in 3 or more directions, deemed a junction
        if directions >= 3:
            junctions.append((x, y))
    
    # Filter out junctions that are too close to each other
    filtered_junctions = []
    for j in junctions:
        if not any(manhattan_distance(j, other) < threshold for other in filtered_junctions):
            filtered_junctions.append(j)
    
    # logger.info(f"Found {len(filtered_junctions)} junction points in the maze")
    return filtered_junctions

def calculate_hybrid_reward(player_pos, prev_pos, ghost_positions, score_change, junction_points, metrics=None, resolution_info=None):
    """Hybrid reward with clear incentives"""
    reward = 0
    
    # Import the GHOST_SCORE constant
    from config import GHOST_SCORE
    
    # Skip if player position is invalid
    if not player_pos or not prev_pos:
        return 0
    
    # Base movement reward
    if player_pos != prev_pos:
        reward += 0.02  # Small constant reward for any movement
    
    # Ghost avoidance
    if ghost_positions:
        min_ghost_distance = min(manhattan_distance(player_pos, g) for g in ghost_positions)
        
        # Simple tiered rewards
        if min_ghost_distance < 5:  # Very close
            reward -= 0.2
        elif min_ghost_distance < 15:  # Medium distance
            reward -= 0.05
        else:  # Safe distance
            reward += 0.05
    
    # Pill collection reward
    if score_change > 0 and score_change < GHOST_SCORE:  # Pills
        reward += 0.3
    
    # Ghost eating reward
    if score_change >= GHOST_SCORE:  # Ate ghost
        reward += 0.3
    
    # Junction reward 
    if junction_points and any(manhattan_distance(player_pos, j) < 5 for j in junction_points):
        reward += 0.05
    
    # Level completion incentive
    if metrics is not None:
        m = metrics.get_metrics() if hasattr(metrics, 'get_metrics') else {}
        if m.get('initial_pill_count') and m.get('current_pill_count'):
            # Add increasing bonuses when approaching level completion
            collected_percent = (m['initial_pill_count'] - m['current_pill_count']) / m['initial_pill_count']
            if collected_percent > 0.3:  # Over 30% completed
                reward += 0.3
            if collected_percent > 0.5:  # Over 50% completed
                reward += 0.5
            if collected_percent > 0.7:  # Over 70% completed
                reward += 0.7
            if collected_percent > 0.8:  # Over 80% completed
                reward += 0.8
            if collected_percent > 0.9:  # Over 90% completed
                reward += 0.9
            if collected_percent > 0.95:  # Nearly complete
                reward += 1.0  # Very large bonus for near-completion
            
            # Add consistent reward based on completion percentage
            reward += collected_percent * 0.2

    return max(-0.5, min(0.5, reward))

