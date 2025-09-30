def get_new_reward(obs):
    cart_position, cart_velocity, pole_angle, pole_velocity = obs
    
    # Penalize for being close to the termination boundaries
    # Cart position penalty
    cart_pos_penalty = 0
    if abs(cart_position) > 2.0:  # Closer to the edge
        cart_pos_penalty = (abs(cart_position) - 2.0) * 0.5 
        
    # Pole angle penalty
    pole_angle_penalty = 0
    if abs(pole_angle) > 0.15: # Closer to the termination angle
        pole_angle_penalty = (abs(pole_angle) - 0.15) * 1.0

    # Penalize for high velocities (less stable)
    velocity_penalty = abs(cart_velocity) * 0.1 + abs(pole_velocity) * 0.2

    # Base reward for staying alive
    reward = 1.0 - cart_pos_penalty - pole_angle_penalty - velocity_penalty
    
    # Ensure reward is not negative, though the penalties are designed to be small
    reward = max(0, reward) 
    
    return reward