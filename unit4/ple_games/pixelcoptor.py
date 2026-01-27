from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import random
import time


# Create game and PLE environment
game = Pixelcopter()
p = PLE(game, fps=30, display_screen=True)

# Initialize PLE before getting action set
p.init()

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
    action = p.act(random.choice(p.getActionSet()))
    reward = p.act(action)  # this will return the reward for the action
    total_reward += reward

    time.sleep(0.033)
print(f"Finished {nb_frames} frames")

