def get_state(observation):
    state = np.zeros((7, 7, 19))

    # Encode the game board
    for i, row in enumerate(observation['board']):
        for j, cell in enumerate(row):
            if cell == observation['player']:
                state[i, j, 0] = 1
            elif cell == observation['players'][observation['player'] - 1]:
                state[i, j, 1] = 1
            elif cell > 0:
                state[i, j, 2] = 1

    # Encode the resources
    state[:, :, 3] = observation['resources'] / 100.

    # Encode the units
    for unit in observation['units']:
        if unit['team'] == observation['player']:
            if unit['type'] == 'worker':
                state[unit['y'], unit['x'], 4] = 1
            else:
                state[unit['y'], unit['x'], 5] = 1
                state[unit['y'], unit['x'], 6] = unit['cargo'] / 100.
        else:
            if unit['type'] == 'worker':
                state[unit['y'], unit['x'], 7] = 1
            else:
                state[unit['y'], unit['x'], 8] = 1
                state[unit['y'], unit['x'], 9] = unit['cargo'] / 100.

    # Encode the player's research points and research levels
    state[:, :, 10] = observation['player_research_points'] / 200.
    state[:, :, 11] = observation['player_research_level'] / 4.

    # Encode the opponent's research points and research levels
    state[:, :, 12] = observation['opponent_research_points'] / 200.
    state[:, :, 13] = observation['opponent_research_level'] / 4.

    # Encode the time
    state[:, :, 14] = observation['step'] / 360.

    # Encode the player's last actions
    for i, action in enumerate(observation['last_actions']):
        if action.startswith('c'):
            state[:, :, 15 + i] = 1
        elif action.startswith('m'):
            state[:, :, 15 + i] = 2

    # Encode the day/night cycle
    state[:, :, 17] = observation['night']

    # Encode the player's ID
    state[:, :, 18] = observation['player'] / 2.

    return state.flatten()
