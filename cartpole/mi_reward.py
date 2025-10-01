def get_new_reward(obs):
    """
    Calculates a new reward based on the observation of the CartPole environment.

    The reward is shaped to encourage the agent to keep the pole balanced and the cart within bounds.
    A higher reward is given for states closer to the upright position and centered cart.
    Penalties are applied for states that are close to termination conditions.

    Args:
        obs (np.ndarray): The observation from the environment, containing:
            - obs[0]: Cart Position
            - obs[1]: Cart Velocity
            - obs[2]: Pole Angle
            - obs[3]: Pole Angular Velocity

    Returns:
        float: The shaped reward.
    """
    import numpy as np
    
    cart_pos, cart_vel, pole_angle, pole_vel = obs

    # Define termination boundaries for clarity
    max_cart_pos = 2.4
    max_pole_angle = np.deg2rad(12)  # 12 degrees in radians

    # Reward for keeping the pole upright
    # Closer to 0 angle gives a higher reward
    pole_reward = 1.0 - abs(pole_angle) / max_pole_angle

    # Reward for keeping the cart within bounds
    # Closer to 0 position gives a higher reward
    cart_reward = 1.0 - abs(cart_pos) / max_cart_pos

    # Combine rewards, with pole angle being more critical
    # You can adjust these weights based on empirical results
    shaped_reward = 0.5 * pole_reward + 0.5 * cart_reward

    # Add a small penalty for velocity to encourage stability
    # This is a subtle shaping and might not be strictly necessary
    velocity_penalty = -0.01 * (abs(cart_vel) + abs(pole_vel))

    # Ensure the reward is not excessively large or small, and avoid large negative rewards
    # that could destabilize learning.
    final_reward = shaped_reward + velocity_penalty

    # Clip the reward to prevent extreme values, especially near termination
    # A small positive reward for staying alive is generally good.
    # If the episode is about to terminate, the reward should reflect that.
    if abs(pole_angle) > max_pole_angle * 0.8 or abs(cart_pos) > max_cart_pos * 0.8:
        final_reward -= 0.5  # Stronger penalty when close to termination

    # Ensure a minimum positive reward for simply surviving a step
    # This helps prevent the agent from getting stuck in states with zero or negative rewards
    # if the shaping is too aggressive.
    return max(final_reward, 0.1)
