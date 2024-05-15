import numpy as np
from pettingzoo.mpe import simple_tag_v3
from RandomAgent import get_random_action
from GreedyAgent import get_greedy_action
import matplotlib.pyplot as plt

NUM_GOOD = 1
NUM_ADVERSARIES = 1
NUM_OBSTACLES = 0
MAX_CYCLES = 50

simple_tag = simple_tag_v3.parallel_env(num_good=NUM_GOOD, num_adversaries=NUM_ADVERSARIES, num_obstacles=NUM_OBSTACLES, max_cycles=MAX_CYCLES, continuous_actions=False, render_mode=None)
 
def run_agents(env, action_func, n_episodes):
    observations, infos = env.reset()

    results = {}
    for agent in env.agents:
        results[agent] = 0

    for episode in range(n_episodes):
        observations, infos = env.reset()

        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent.startswith("adversary"):
                    actions[agent] = action_func(agent, observations) #insert get_action function here
                else:
                    actions[agent] = 0
                    continue
                
            observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent, reward in rewards.items():
                results[agent] += reward

        print("episode: ", episode, " completed")
        
    env.close()

    return results #total reward for all agents over all episodes

if __name__ == "__main__":
    print("hello world!")
    num_episodes = 100
    results = run_agents(env=simple_tag, action_func=get_greedy_action, n_episodes=num_episodes)

    #print results
    for agent, total_reward in results.items():
        print(agent, " : ", total_reward/num_episodes)
    
    agents = list(results.keys())
    rewards = [reward/num_episodes for reward in list(results.values())]
    plt.bar(agents, rewards, color='red', edgecolor='black')

    # Add title and labels
    plt.title('Greedy Predator, N=100 episodes')
    plt.xlabel('Agents')
    plt.ylabel('Avg reward pr episode')

    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    #draw a block line to indocate the x-axis
    plt.axhline(0, color='black', linewidth=0.8)

    # Show the plot
    plt.show()



