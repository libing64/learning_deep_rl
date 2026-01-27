from pyvirtualdisplay import Display
import numpy as np
import gymnasium as gym
import random
import pickle5 as pickle
from tqdm.notebook import tqdm


virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# create env
env = gym.make('SpaceInvaders-v4', render_mode='rgb_array')


# train command
python -m rl_zoo3.train --algo dqn  --env SpaceInvadersNoFrameskip-v4 -f logs/ -c dqn.yml


# evaluate command 
!python -m rl_zoo3.enjoy  --algo dqn  --env SpaceInvadersNoFrameskip-v4  --no-render  --n-timesteps 5000  --folder logs/