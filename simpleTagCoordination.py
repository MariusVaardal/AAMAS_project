import numpy as np
import math
from utils.utils import get_timestep_reward
from typing import Type, TypeVar
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from pettingzoo.mpe import simple_tag_v3

from agent_types.CoordinatingAgent import CoordinatingAgent
from agent_types.GreedyAgent import GreedyAgent
from agent_types.RandomAgent import RandomAgent
from agent_types.AvoidingAgent import AvoidingAgent
from agent_types.AvoidingNearestAdversaryAgent import AvoidingNearestAdversaryAgent
from agent_types.ImmobileAgent import ImmobileAgent

NUM_GOOD = 1
NUM_LANDMARKS = 0
MAX_CYCLES = 200
NUM_EPISODES = 50

RENDER_MODE = "human"

T = TypeVar('T')

def run_simple_tag_and_get_results(
        num_adversaries: int, 
        max_cycles: int, 
        render_mode,
        good_agent_type: Type[T],
        adversary_type: Type[T],
        num_episodes: int
        ):
    env = simple_tag_v3.parallel_env(num_good=NUM_GOOD, num_adversaries=num_adversaries, num_obstacles=NUM_LANDMARKS, max_cycles=max_cycles, continuous_actions=False, render_mode=render_mode)
    env.reset()
    coord_agents = [adversary_type(name, num_adversaries, NUM_LANDMARKS) for name in env.agents if name != "agent_0"]

    coord_agents.append(good_agent_type("agent_0", num_adversaries, NUM_LANDMARKS))

    episode_rewards = []
    for episode in range(1, num_episodes + 1):
        observations, infos = env.reset()
        for agent in coord_agents:
            agent.observations = observations[agent.name]
            agent.update_observed_agent_positions()
            
        episode_reward = 0
        while env.agents:
            actions = {}
            sum_of_distances_from_target = 0
            for agent in coord_agents:
                action = agent.get_action()
                actions[agent.name] = action if action != None else 0
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent in coord_agents:
                # agent.see(observations[agent.name])
                agent.observations = observations[agent.name]
                agent.update_observed_agent_positions()
            episode_reward += get_timestep_reward(rewards)
        episode_rewards.append(episode_reward)
        # print(f"Episode: {episode}. Reward: {sum(episode_rewards)}")
    env.close()
    avg_rew = sum(episode_rewards) / len(episode_rewards)
    # print(f"Avg : {avg_rew}. For {len(episode_rewards)} number of episodes")
    return avg_rew

def run_simple_tag_and_plot_results(
        num_adversaries_list: list[int], 
        good_agent_types: list[Type[T]],
        adversary_types: list[Type[T]],
        num_episodes: int
        ):
    # Define fonts
    font_normal = FontProperties(style='normal')
    font_italic = FontProperties(style='italic')
    font_oblique = FontProperties(style='oblique')

    fig_list = []
    for i, num_adv in tqdm(enumerate(num_adversaries_list), total=len(num_adversaries_list)):
        fig, ax = plt.subplots(1, len(good_agent_types), figsize=(10, 8))
        fig.suptitle(f'{num_adv} ADVERSARIES', 
                    fontsize=18, 
                    fontweight='bold', 
                    bbox=dict(facecolor='#abdbe3', 
                               edgecolor='black', 
                               boxstyle='Square, pad=0.5'),
                    fontproperties=font_oblique)

        fig.subplots_adjust(wspace=0.4)
        # Set y-label
        ax[0].set_ylabel(f'Avg. reward over {num_episodes} episodes', 
                         fontsize=12, 
                         fontweight=500, 
                         fontproperties=font_italic)
        for i in range(len(good_agent_types)):
            x = [adv.__name__ for adv in adversary_types]
            y = [run_simple_tag_and_get_results(num_adv, MAX_CYCLES, None, good_agent_types[i], adv_type, num_episodes) for adv_type in adversary_types]
            bars = ax[i].bar(x, 
                      y, 
                      color=['blue', 'red', 'green'], 
                      alpha=0.7,
                      edgecolor='black',
                      linewidth=1.5,  
                      )
            ax[i].grid(True, axis='y')
            ax[i].set_title(f'{good_agent_types[i].__name__}', fontsize=12, fontweight='roman')
            ax[i].tick_params(axis='x', 
                              labelrotation=25, 
                              grid_color='grey', 
                              grid_alpha=0.2, 
                              labelsize=9)
            
            # Adjust the height a little bit
            pos = ax[i].get_position()
            ax[i].set_position([pos.x0, pos.y0, pos.width, pos.height - 0.1])

            # Add text over bars to indicate the bar's height
            for bar in bars:
                height = bar.get_height()
                ax[i].text(bar.get_x() + bar.get_width() / 2, 
                           height + 0.1, 
                           f'{int(height)}', 
                           ha='center', 
                           va='bottom',
                           fontsize=10)

        fig.text(0.5, 0.86, 'Good agent type:', 
                 ha='center', 
                 fontsize=14, 
                 fontproperties=font_oblique, 
                 fontweight=700)
        fig_list.append(fig)
    
    for fig in fig_list:
        fig.show()
        fig.savefig(f'./plots/{fig._suptitle.get_text()}. {num_episodes} episodes.png', bbox_inches='tight')
    plt.show()

run_simple_tag_and_plot_results([2, 3, 4], [ImmobileAgent, RandomAgent, AvoidingAgent, AvoidingNearestAdversaryAgent], [RandomAgent, GreedyAgent, CoordinatingAgent], NUM_EPISODES)








