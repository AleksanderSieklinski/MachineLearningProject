import gymnasium as gym
import cv2
import numpy as np
import random
from deap import base, creator, tools
import warnings
import logging
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)


class EvoCarRacing:
    def __init__(self):
        run_number = self.__get_next_run_number()
        self.__setup_logger(run_number)
        self.__setup_video_writer()
        self.__setup_environment()
        self.__setup_genetic_algorithm()

    @staticmethod
    def __get_next_run_number():
        log_dir = 'training/logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        files = os.listdir(log_dir)
        log_files = [f for f in files if f.endswith('.log')]
        return len(log_files) + 1

    def __setup_logger(self, run_number):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not os.path.exists('training/logs'):
            os.makedirs('training/logs')
        file_handler = logging.FileHandler(f'training/logs/EVO_{run_number}.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def __setup_video_writer(self):
        self.width, self.height = 600, 400
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter("Car Racing Evo.mp4", self.fourcc, 30.0, (self.width, self.height))

    def __setup_environment(self):
        self.env = gym.make('CarRacing-v2', render_mode='rgb_array')

    def __setup_genetic_algorithm(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", np.random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, 3)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.__evalOneMax)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def __evalOneMax(self, individual):
        self.env.reset()
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

    def __run(self):
        self.logger.info('Starting the genetic algorithm')
        pop = self.toolbox.population(n=20)
        # CXPB - The probability with which two individuals are crossed
        # MUTPB - The probability for mutating an individual
        # NGEN - The number of generations
        CXPB, MUTPB, NGEN = 0.7, 0.3, 150
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
            self.logger.info(f"Generation {g}, Best individual: {best_ind}, Best fitness: {best_ind.fitness.values[0]}")
        return pop

    def __record_best(self, pop):
        self.logger.info('Recording the best individual')
        for episode in range(1, 10 + 1):
            self.env.reset()
            done, score = False, 0
            while not done:
                action = pop[0]
                step = self.env.step(action)
                observation, reward, done, info = step[0], step[1], step[2], step[3]
                score += reward
                frame = self.env.render()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (self.width, self.height))
                self.video_writer.write(frame)
            print(f"Episode {episode} score: {score}")
            self.logger.info(f"Episode {episode}, Score: {score}")

    def runTestAndRecord(self):
        self.logger.info('Running test and recording')
        pop = self.__run()
        self.__record_best(pop)
        self.video_writer.release()
        self.env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    evo_car_racing = EvoCarRacing()
    evo_car_racing.runTestAndRecord()
