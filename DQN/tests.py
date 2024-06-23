import gymnasium as gym
import torch
from dqn_model import Qnet
from collections import deque
from utils import frames_to_input
import numpy as np

action_space = [[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,0,1]]
frame_stack = deque(maxlen=5)
action_duration = 25
env = gym.make('CarRacing-v2',render_mode='human')
model = Qnet.load("model_6000.pth")



env.reset()



for i in range(20):
        action = [0,0,0]
        observation, reward, done, truncated, info = env.step(action)

        frame_stack.append(observation)

for j in range(50):
    with torch.no_grad():
        nn_output = model(frames_to_input(frame_stack))
    index = np.argmax(nn_output.detach().numpy())
    print(index)
    action = action_space[index]
    for _ in range(action_duration):
            observation, reward, done, truncated, info = env.step(action)
            frame_stack.append(observation)