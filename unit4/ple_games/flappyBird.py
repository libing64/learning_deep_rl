from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import time


class SimpleAgent:
    """
    Simple agent for FlappyBird that randomly chooses actions
    """
    def __init__(self, allowed_actions):
        self.allowed_actions = allowed_actions
    
    def pickAction(self, reward, observation):
        """
        Pick an action based on observation
        
        Args:
            reward: Current reward
            observation: Current game observation (screen RGB)
        
        Returns:
            action: Selected action from allowed_actions
        """
        # Simple random agent - randomly chooses between actions
        return random.choice(self.allowed_actions)


# Create game and PLE environment
game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)

# Initialize PLE before getting action set
p.init()

# Create agent
agent = SimpleAgent(allowed_actions=p.getActionSet())

# Training/playing parameters
nb_frames = 1000  # Number of frames to run
reward = 0.0
total_reward = 0.0

# Main loop
for i in range(nb_frames):
    if p.game_over():
        p.reset_game()
        print(f"Game over at frame {i}, Total reward: {total_reward:.2f}")
        total_reward = 0.0
    
    observation = p.getScreenRGB()
    action = agent.pickAction(reward, observation)
    reward = p.act(action)
    total_reward += reward

    time.sleep(0.033)
print(f"Finished {nb_frames} frames")