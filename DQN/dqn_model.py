import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random
import copy
from utils import frames_to_input

frame_buffer = 5

memory_size = 500
sample_size = 16
learning_rate_a = 0.01
gamma = 0.95

class Qnet(nn.Module):

    def __init__(self,output_size):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=frame_buffer, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def save(self,path):
        torch.save(self.state_dict(), path)

    def load(path):
        model = Qnet(5)
        model.load_state_dict(torch.load(path))
        return model
    
class DQN():

    def __init__(self):

        self.action_space = [[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,0,1]]

        self.agent = Qnet(len(self.action_space))
        self.target = Qnet(len(self.action_space))

        self.memory = deque(maxlen=memory_size)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate_a)
    
    def remember(self,state_frames,action_index,reward,next_state_frames):
        self.memory.append([copy.deepcopy(state_frames),action_index,reward,copy.deepcopy(next_state_frames)])

    def memory_sample(self,sample_size):
        batch = random.sample(self.memory, sample_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        
        for i in range(sample_size):
            states.append(frames_to_input(batch[i][0]))
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            next_states.append(frames_to_input(batch[i][3]))

        states = torch.cat(states,dim=0)
        next_states = torch.cat(next_states,dim=0)
        return states,actions,rewards,next_states
    
    def optimize(self,states,actions,rewards,next_states):

        target_q_values = []
        for i in range(len(states)):
            with torch.no_grad():
                target = self.agent.forward(states[i].unsqueeze(0))
                new_q = self.target.forward(next_states[i].unsqueeze(0))
                target[0][actions[i]] = rewards[i] + gamma* max(new_q[0])
                target_q_values.append(target)

        target_q_values = torch.cat(target_q_values,dim=0)
        q_values = self.agent.forward(states)

        loss = F.mse_loss(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    

