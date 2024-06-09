import gym
import cv2
import numpy as np
from deap import base, creator, tools
import random

# Define the fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize the toolbox
toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", np.random.uniform, -1, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the evaluation function
def evalOneMax(individual):
    env = gym.make('CarRacing-v2', render_mode='rgb_array')
    observation = env.reset()
    total_reward = 0
    for _ in range(1000):
        action = individual
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward,

# Register the evaluation, crossover, mutation and selection operators
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create the population and run the evolution process
pop = toolbox.population(n=300)
CXPB, MUTPB, NGEN = 0.5, 0.2, 40

# Video recording
width, height = 600, 400
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("Car Racing Evo.mp4", fourcc, 30.0, (width, height))

for g in range(NGEN):
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

    # Test the model
    for episode in range(1, 10+1):
        # Reset the environment
        env = gym.make('CarRacing-v2')
        observation = env.reset()
        done, score = False, 0
        while not done:
            # Predict the action
            action = pop[0]  # Use the best individual
            observation, reward, done, info = env.step(action)
            score += reward
            # Recording environment
            frame = env.render(mode='rgb_array')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (width, height))
            # Write frame to video
            video_writer.write(frame)
        print(f"Episode {episode} score: {score}")

# Release the video writer and close the environment
video_writer.release()
env.close()
cv2.destroyAllWindows()