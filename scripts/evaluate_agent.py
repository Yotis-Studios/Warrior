import torch
from envs.raifu_env import RaifuWarsEnv
from agents.agent import WarriorAgent

def load_trained_agent(model_path):
    agent = WarriorAgent()
    agent.model.load_state_dict(torch.load(model_path))
    return agent

def evaluate_agent(agent, env, num_episodes):
    total_rewards = 0
    
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_rewards += episode_reward
        print(f'Episode {episode + 1}: Reward = {episode_reward}')
    
    average_reward = total_rewards / num_episodes
    print(f'Average Reward over {num_episodes} episodes = {average_reward}')

if __name__ == "__main__":
    model_path = 'models/trained_model.pth'  # Path to the saved model file
    agent = load_trained_agent(model_path)
    
    env = RaifuWarsEnv()
    
    num_evaluation_episodes = 100
    evaluate_agent(agent, env, num_evaluation_episodes)
