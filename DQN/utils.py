import torch
import numpy as np

def frames_to_input(frame_stack):
    
    input_batch = []
    l = len(frame_stack)
    for i in range(l):
        input_tensor = rgb_to_grayscale(frame_stack[i])
        input_tensor = np.array(input_tensor/255)
        input_tensor = input_tensor.astype(np.float32)
        input_tensor = torch.from_numpy(input_tensor)
        input_batch.append(input_tensor)

    input_batch = torch.from_numpy(np.array(input_batch).reshape(1,l,96,96))
    return input_batch


def rgb_to_grayscale(rgb_array):

    grayscale_array = np.dot(rgb_array[...,:3], [0.2989, 0.5870, 0.1140])
    return grayscale_array