from dqn_model import DQN
from collections import deque
from utils import frames_to_input
import random
import gymnasium as gym
import numpy as np
import torch
import copy


#env = gym.make('CarRacing-v2',render_mode='human')
env = gym.make('CarRacing-v2')
dqn = DQN()
frame_stack_size = 5
frame_stack = deque(maxlen=frame_stack_size)
num_of_episodes = 10001
action_duration = 25
epsilon = 1
epsilon_min  = 0.01
epsilon_decay = 0.9997
max_steps = 50
negative_reward_steps_threshold = 3
batch_size = 128
update_frequency = 5

for n in range(num_of_episodes):
    
    env.reset()
    
    ## Wait for map to normalize
    for i in range(20):
        action = dqn.action_space[0]
        observation, reward, done, truncated, info = env.step(action)

        frame_stack.append(observation)

    steps = 0
    negative_reward_steps = 0

    while( not done and not truncated):
        
        if random.random()>epsilon:
            with torch.no_grad():
                nn_output = dqn.agent(frames_to_input(frame_stack))
            index = np.argmax(nn_output.detach().numpy())
            action = dqn.action_space[index]
        else:
            index = random.randint(0,4)
            action = dqn.action_space[index]

        total_reward = 0
        next_state_frames = deque(maxlen=frame_stack_size)
        for _ in range(action_duration):
            observation, reward, done, truncated, info = env.step(action)
            
            total_reward+=reward
            next_state_frames.append(observation)
        #print(action)
        #print(total_reward)
        dqn.remember(frame_stack,index,total_reward,next_state_frames)
    
        frame_stack = copy.deepcopy(next_state_frames)
        
    
        if total_reward<0:
            negative_reward_steps+=1
        else:
            negative_reward_steps=0
        
        if negative_reward_steps>negative_reward_steps_threshold:
            break
        
        
        steps+=1
        if steps>max_steps:
            break

    if n%1000 == 0:
        dqn.agent.save(f'model_{n}.pth')

    if n%10 == 0:
        print(f'Epsiode {n} total reward {total_reward} epsilon {epsilon}')

    if len(dqn.memory)>batch_size:
        states,actions,rewards,next_states = dqn.memory_sample(batch_size)
        dqn.optimize(states,actions,rewards,next_states)

    if n % update_frequency == 0:
        dqn.target.load_state_dict(dqn.agent.state_dict())
    
    if epsilon>epsilon_min:
        epsilon *= epsilon_decay


dqn.agent.save(f'model.pth')






    







