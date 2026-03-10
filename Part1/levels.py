"""
Level definitions for the gridworld environment.
Each level is a dictionary containing grid layout and element positions.

Legend:
    '.' = empty floor
    '#' = rock (blocks movement)
    'A' = apple (+1 reward)
    'K' = key (enables chest opening)
    'C' = chest (+2 reward when opened with key)
    'F' = fire (instant death)
    'M' = monster (instant death, moves probabilistically)
    'S' = start position
"""

# Level 0: Simple level with apples on the right side (Q-Learning demonstration)
LEVEL_0 = {
    "name": "Level 0 - Apples Only",
    "description": "Basic level to demonstrate Q-Learning shortest path",
    "grid": [
        ". . . . . . . . . .",
        ". . . . . . . . . .",
        ". . . . . . . . . .",
        ". . . . . . . . A .",
        ". . . . . . . . A .",
        "S . . . . . . . A .",
        ". . . . . . . . A .",
        ". . . . . . . . . .",
        ". . . . . . . . . .",
        ". . . . . . . . . .",
    ],
}

# Level 1: Level with fire hazards to demonstrate SARSA's conservative behavior
LEVEL_1 = {
    "name": "Level 1 - Fire Hazards",
    "description": "Level with fire to show SARSA conservative policy",
    "grid": [
        ". . . . . . . . . .",
        ". . . . . . . . . .",
        ". . . . F F F . . .",
        ". . . . F . F . A .",
        ". . . . F . F . A .",
        "S . . . . . . . A .",
        ". . . . F . F . A .",
        ". . . . F . F . . .",
        ". . . . F F F . . .",
        ". . . . . . . . . .",
    ],
}

# Level 2: Multiple apples with key and chest
LEVEL_2 = {
    "name": "Level 2 - Keys and Chests",
    "description": "Level with apples, key, and chest",
    "grid": [
        ". . . . . . . . . .",
        ". # # # . . . . . .",
        ". # A # . . . K . .",
        ". # . # . . . . . .",
        ". . . . . # # # . .",
        "S . . . . # C # . .",
        ". . . . . # . # . .",
        ". . . . . . . . . .",
        ". . A . . . . . A .",
        ". . . . . . . . . .",
    ],
}

# Level 3: More complex layout with multiple objectives
LEVEL_3 = {
    "name": "Level 3 - Complex Layout",
    "description": "Complex level with multiple apples, keys, and chests",
    "grid": [
        ". . . . # . . . . .",
        ". A . . # . . A . .",
        ". . . . # . . . . .",
        ". . . . . . . . . .",
        "# # # . . . . # # #",
        ". . . . . . . . . .",
        ". . K . . . . . C .",
        ". . . . # . . . . .",
        ". A . . # . . . A .",
        "S . . . # . . . . .",
    ],
}

# Level 4: Monsters with simple layout
LEVEL_4 = {
    "name": "Level 4 - Monster Encounter",
    "description": "Level with monsters that move probabilistically",
    "grid": [
        ". . . . . . . . . .",
        ". . . . . . . . . .",
        ". . . . . . . . A .",
        ". . . M . . . . A .",
        ". . . . . . . . A .",
        "S . . . . . M . A .",
        ". . . . . . . . . .",
        ". . . . . . . . . .",
        ". . . . . . . . . .",
        ". . . . . . . . . .",
    ],
}

# Level 5: Complex level with monsters, keys, and chests
LEVEL_5 = {
    "name": "Level 5 - Monster Gauntlet",
    "description": "Complex level with monsters, keys, chests, and hazards",
    "grid": [
        ". . . . . . . . . .",
        ". A . . # . . . A .",
        ". . . . # . . . . .",
        ". . M . . . . M . .",
        ". . . . K . . . . .",
        "S . . . . . . . . .",
        ". . . . . . . . . .",
        ". . M . # . . M . .",
        ". . . . # . . . . .",
        ". . . . # . . C . .",
    ],
}

# Level 6: Large exploration level for intrinsic reward testing
LEVEL_6 = {
    "name": "Level 6 - Exploration Challenge",
    "description": "Sparse rewards requiring exploration with intrinsic motivation",
    "grid": [
        ". . . . . . . . . . . .",
        ". . . . # # # # . . . .",
        ". . . . # . . # . . A .",
        ". . . . # . . # . . . .",
        ". . . . . . . . . . . .",
        "S . . . . . . . . . . .",
        ". . . . . . . . . . . .",
        ". . . . # . . # . . . .",
        ". . K . # . . # . . . .",
        ". . . . # # # # . . . .",
        ". . . . . . . . . . C .",
        ". . . . . . . . . . . .",
    ],
}

# List of all levels
ALL_LEVELS = [LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4, LEVEL_5, LEVEL_6]


def parse_grid(level_data):
    """
    Parse grid string representation into structured data.
    
    Returns:
        Dictionary containing:
            - width: grid width
            - height: grid height
            - start_position: tuple (row, col)
            - rocks: set of (row, col)
            - apples: set of (row, col)
            - keys: set of (row, col)
            - chests: set of (row, col)
            - fire: set of (row, col)
            - monsters: set of (row, col)
    """
    grid_lines = level_data["grid"]
    height = len(grid_lines)
    width = len(grid_lines[0].split())
    
    parsed_data = {
        "name": level_data["name"],
        "description": level_data["description"],
        "width": width,
        "height": height,
        "start_position": None,
        "rocks": set(),
        "apples": set(),
        "keys": set(),
        "chests": set(),
        "fire": set(),
        "monsters": set(),
    }
    
    for row_index, row_string in enumerate(grid_lines):
        cells = row_string.split()
        for column_index, cell in enumerate(cells):
            position = (row_index, column_index)
            
            if cell == "#":
                parsed_data["rocks"].add(position)
            elif cell == "A":
                parsed_data["apples"].add(position)
            elif cell == "K":
                parsed_data["keys"].add(position)
            elif cell == "C":
                parsed_data["chests"].add(position)
            elif cell == "F":
                parsed_data["fire"].add(position)
            elif cell == "M":
                parsed_data["monsters"].add(position)
            elif cell == "S":
                parsed_data["start_position"] = position
    
    return parsed_data

