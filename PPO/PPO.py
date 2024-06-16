import gymnasium as gym
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def main():

    env = gym.make('CarRacing-v2', render_mode = 'rgb_array')
    env = DummyVecEnv([lambda : env])
    model = PPO("CnnPolicy", 'CarRacing-v2',
                tensorboard_log = 'training/logs',
                batch_size = 128,
                clip_range = 0.2,
                ent_coef = 0.0,
                gae_lambda = 0.95,
                gamma = 0.99,
                learning_rate = 0.0005,
                max_grad_norm = 0.5,
                n_epochs = 10)
    
    w, h = 400, 400

    videoWriterFr = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter("Car Racing run2 PPO.mp4", videoWriterFr, 30.0, (w, h))
    
    #Run for 2^19 = 524288 episodes
    model.learn(total_timesteps = 524288 , log_interval = 5, progress_bar = True)

    testRuns = 15
    for testRun in range(1, testRuns+1):
        numberOfNegativeSteps = 0
        observation = env.reset()
        done = False
        totalReward = 0
        while not done:
   
            action, _ = model.predict(observation)
            step = env.step(action)

            observation, reward, done, info = step[0], step[1], step[2], step[3]
            totalReward += reward

            if reward < 0:
                numberOfNegativeSteps+=1
            if numberOfNegativeSteps>300:
                break

            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (w, h))

            videoWriter.write(frame)
        print(f'Test run {testRun} total reward {totalReward}')

    videoWriter.release()
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()