"""
Thompson Sampling for Bayesian Reinforcement Learning

This module implements Thompson sampling for action selection in uncertain environments.
It samples possible world models from the current belief distribution and evaluates actions
based on optimality for each sampled model.
"""

import numpy as np
from scipy.stats import beta
import math
import random

class ThompsonSampler:
    """
    Implements Thompson sampling for action selection under uncertainty
    """
    def __init__(self, n_samples=10):
        self.n_samples = n_samples
    
    def sample_models(self, beliefs):
        """
        Sample models from belief distributions
        
        Args:
            beliefs: Dictionary mapping parameter IDs to belief distributions
                    Each distribution is represented as {'alpha': α, 'beta': β}
        
        Returns:
            List of sampled models, where each model is a dictionary of parameter values
        """
        models = []
        
        for _ in range(self.n_samples):
            model = {}
            
            for param_id, belief in beliefs.items():
                # Sample from Beta distribution
                if 'alpha' in belief and 'beta' in belief:
                    alpha = belief['alpha']
                    beta_val = belief['beta']
                    
                    # Ensure alpha, beta are valid
                    if alpha <= 0: alpha = 0.1
                    if beta_val <= 0: beta_val = 0.1
                    
                    # Sample from Beta distribution
                    sample = np.random.beta(alpha, beta_val)
                    model[param_id] = sample
                else:
                    # Default to uniform sampling if distribution is unspecified
                    model[param_id] = random.random()
            
            models.append(model)
        
        return models
    
    def evaluate_actions(self, actions, models, utility_function):
        """
        Evaluate actions across sampled models
        
        Args:
            actions: List of possible actions
            models: List of sampled models
            utility_function: Function (action, model) -> utility value
        
        Returns:
            Dictionary mapping actions to expected utilities
        """
        action_utilities = {action: 0.0 for action in actions}
        
        # Evaluate each action across all sampled models
        for action in actions:
            for model in models:
                utility = utility_function(action, model)
                action_utilities[action] += utility / len(models)
        
        return action_utilities
    
    def select_action(self, actions, beliefs, utility_function):
        """
        Select an action using Thompson sampling
        
        Args:
            actions: List of possible actions
            beliefs: Dictionary of current belief distributions
            utility_function: Function (action, model) -> utility value
        
        Returns:
            Selected action and its expected utility
        """
        # Sample models from beliefs
        models = self.sample_models(beliefs)
        
        # Evaluate actions across sampled models
        action_utilities = self.evaluate_actions(actions, models, utility_function)
        
        # Select action with highest expected utility
        best_action = max(action_utilities, key=action_utilities.get)
        best_utility = action_utilities[best_action]
        
        return best_action, best_utility

# Example application for robot navigation

def example_use_in_robot():
    """
    Example of how to use Thompson sampling for robot navigation
    """
    # Define belief distributions about obstacle velocities
    # Each obstacle has a belief about its x and y velocities
    obstacle_beliefs = {
        'obstacle_1': {
            'vx': {'alpha': 3.0, 'beta': 1.5},  # Belief about x velocity 
            'vy': {'alpha': 2.0, 'beta': 2.0}   # Belief about y velocity
        },
        'obstacle_2': {
            'vx': {'alpha': 1.5, 'beta': 3.0},
            'vy': {'alpha': 2.5, 'beta': 1.5}
        }
    }
    
    # Define possible actions (e.g., directions in degrees)
    possible_directions = [0, 45, 90, 135, 180, 225, 270, 315]
    
    # Initialize Thompson sampler
    sampler = ThompsonSampler(n_samples=20)
    
    # Utility function that evaluates a direction based on collision risk
    def evaluate_direction_utility(direction, model):
        # In a real implementation, this would:
        # 1. Predict future obstacle positions using sampled velocities
        # 2. Check if moving in this direction risks collision
        # 3. Return high utility for safe directions, low for risky ones
        
        # Simplified example:
        # Generate a safety score based on direction and obstacle predicted positions
        safety_score = 0.0
        
        # For each obstacle, calculate safety
        for obstacle_id in ['obstacle_1', 'obstacle_2']:
            # Get sampled velocities from model
            vx = model.get(f"{obstacle_id}_vx", 0)
            vy = model.get(f"{obstacle_id}_vy", 0)
            
            # Predict future position (simplified)
            # In real implementation, you would use actual obstacle positions & robot position
            future_x = vx * 2.0  # Look ahead 2 seconds
            future_y = vy * 2.0
            
            # Calculate angle to future obstacle position (simplified)
            # In real implementation, this would be relative to robot position
            obstacle_angle = math.degrees(math.atan2(future_y, future_x)) % 360
            
            # Calculate angular distance between direction and obstacle
            angle_diff = min(abs(direction - obstacle_angle), 360 - abs(direction - obstacle_angle))
            
            # High safety if direction is far from obstacle direction
            direction_safety = min(1.0, angle_diff / 180.0)
            safety_score += direction_safety
        
        # Average safety across obstacles
        safety_score /= 2.0
        
        # Add goal alignment component (simplified)
        # In real implementation, this would consider the goal direction
        goal_direction = 45  # Example goal direction (NE)
        goal_diff = min(abs(direction - goal_direction), 360 - abs(direction - goal_direction))
        goal_alignment = 1.0 - min(1.0, goal_diff / 180.0)
        
        # Combined utility: safety is primary, goal alignment secondary
        utility = safety_score * 0.7 + goal_alignment * 0.3
        
        return utility
    
    # Use Thompson sampling to select direction
    best_direction, expected_utility = sampler.select_action(
        possible_directions, obstacle_beliefs, evaluate_direction_utility)
    
    print(f"Selected direction: {best_direction}° with expected utility: {expected_utility:.2f}")
    
    # In a real robot implementation:
    # 1. This would be called at each decision step
    # 2. Beliefs would be updated based on new obstacle observations
    # 3. Robot would move in the selected direction
    # 4. Process repeats

def velocity_belief_update_example():
    """
    Example showing how to update velocity beliefs based on observations
    """
    # Initial belief about obstacle velocity
    # Relatively uninformative prior
    obstacle_belief = {
        'vx': {'alpha': 2.0, 'beta': 2.0},
        'vy': {'alpha': 2.0, 'beta': 2.0}
    }
    
    # Simulated obstacle tracking
    previous_pos = [100, 150]
    previous_time = 10.0
    
    # New observation
    current_pos = [105, 148]
    current_time = 10.5
    
    # Calculate observed velocity
    dt = current_time - previous_time
    observed_vx = (current_pos[0] - previous_pos[0]) / dt
    observed_vy = (current_pos[1] - previous_pos[1]) / dt
    
    # Update beliefs using Bayesian update
    # For x-velocity (positive reinforces alpha, negative reinforces beta)
    if observed_vx > 0:
        obstacle_belief['vx']['alpha'] += observed_vx
    else:
        obstacle_belief['vx']['beta'] += abs(observed_vx)
    
    # For y-velocity
    if observed_vy > 0:
        obstacle_belief['vy']['alpha'] += observed_vy
    else:
        obstacle_belief['vy']['beta'] += abs(observed_vy)
    
    # Calculate MAP velocity estimate
    vx_map = (obstacle_belief['vx']['alpha'] - 1) / (obstacle_belief['vx']['alpha'] + obstacle_belief['vx']['beta'] - 2)
    vy_map = (obstacle_belief['vy']['alpha'] - 1) / (obstacle_belief['vy']['alpha'] + obstacle_belief['vy']['beta'] - 2)
    
    # Scale to actual velocity range (e.g., -3 to 3 pixels/step)
    max_velocity = 3.0
    vx_estimate = vx_map * 2 * max_velocity - max_velocity
    vy_estimate = vy_map * 2 * max_velocity - max_velocity
    
    print(f"Observed velocity: ({observed_vx:.2f}, {observed_vy:.2f})")
    print(f"Updated belief: vx_alpha={obstacle_belief['vx']['alpha']:.2f}, vx_beta={obstacle_belief['vx']['beta']:.2f}")
    print(f"Updated belief: vy_alpha={obstacle_belief['vy']['alpha']:.2f}, vy_beta={obstacle_belief['vy']['beta']:.2f}")
    print(f"MAP velocity estimate: ({vx_estimate:.2f}, {vy_estimate:.2f})")

# Examples to demonstrate use
if __name__ == "__main__":
    print("Example of Thompson sampling for direction selection:")
    example_use_in_robot()
    
    print("\nExample of velocity belief updating:")
    velocity_belief_update_example()