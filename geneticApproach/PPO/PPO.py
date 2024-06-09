import gymnasium as gym
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def main():
    # Create the environment
    env = gym.make('CarRacing-v2', render_mode = 'rgb_array')
    env = Monitor(DummyVecEnv([lambda : env]))
    env = DummyVecEnv([lambda : env])
    model = PPO("CnnPolicy", 'CarRacing-v2',
                tensorboard_log = 'training/logs',
                batch_size = 128,
                clip_range = 0.2,
                ent_coef = 0.0,
                gae_lambda = 0.95,
                gamma = 0.99,
                learning_rate = 0.0003,
                max_grad_norm = 0.5,
                n_epochs = 10)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    # Video recording
    width, height = 600, 400
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("Car Racing PPO.mp4", fourcc, 30.0, (width, height))
    episodes = 10
    # Train the model
    model.learn(total_timesteps = 400_000, log_interval = 5, progress_bar = True)
    evaluate_policy(model, env, n_eval_episodes = episodes)
    # Test the model
    for episode in range(1, episodes+1):
        # Reset the environment
        observation = env.reset()
        done, score = False, 0
        while not done:
            # Predict the action
            action, _ = model.predict(observation)
            step = env.step(action)
            # Unpacking the step
            observation, reward, done, info = step[0], step[1], step[2], step[3]
            score += reward
            # Recording envorment
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (width, height))
            # Write frame to video
            video_writer.write(frame)
        print(f"Episode {episode} score: {score}")
    # Release the video writer and close the environment
    video_writer.release()
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()