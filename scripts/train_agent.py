from envs.raifu_env import RaifuWarsEnv
from agents.agent import WarriorAgent

def train_agent():
    env = RaifuWarsEnv()
    agent = WarriorAgent()
    
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = agent.select_action(observation)
            observation, reward, done, info = env.step(action)
            # ... training logic ...

if __name__ == "__main__":
    train_agent()
