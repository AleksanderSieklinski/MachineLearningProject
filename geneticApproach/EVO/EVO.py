import gym
import cv2
import numpy as np
from deap import base, creator, tools
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class EvoCarRacing:
    def __init__(self):
        self.width, self.height = 600, 400
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter("Car Racing Evo.mp4", self.fourcc, 30.0, (self.width, self.height))
        self.env = gym.make('CarRacing-v2', render_mode='rgb_array')
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", np.random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, 3)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evalOneMax)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evalOneMax(self, individual):
        observation = self.env.reset()
        total_reward = 0
        for _ in range(1000):
            action = individual
            step = self.env.step(action)
            # Unpacking the step
            observation, reward, done, info = step[0], step[1], step[2], step[3]
            total_reward += reward
            if done:
                break
        return total_reward,

    def run(self):
        pop = self.toolbox.population(n=300)
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        for g in range(NGEN):
            print(f"Generation {g}")
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
            best_ind = tools.selBest(pop, 1)[0]
            print(f"Best individual: {best_ind}")
            print(f"Best fitness: {best_ind.fitness.values[0]}")
        return pop

    def test(self, pop):
        for episode in range(1, 10+1):
            observation = self.env.reset()
            done, score = False, 0
            while not done:
                action = pop[0]
                step = self.env.step(action)
                # Unpacking the step
                observation, reward, done, info = step[0], step[1], step[2], step[3]
                score += reward
                frame = self.env.render()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (self.width, self.height))
                self.video_writer.write(frame)
            print(f"Episode {episode} score: {score}")

    def runTestAndRecord(self):
        pop = self.run()
        self.test(pop)
        self.video_writer.release()
        self.env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    evo_car_racing = EvoCarRacing()
    evo_car_racing.runTestAndRecord()