import numpy as np
import math
from utils.utils import get_timestep_reward
from typing import Type, TypeVar
from tqdm import tqdm
import tkinter as tk
from tkinter import messagebox
import pygame.freetype

import matplotlib
matplotlib.use('Qt5Agg')  # Use an interactive backend
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from pettingzoo.mpe import simple_tag_v3

from agent_types.AvoidingAgent import AvoidingAgent
from agent_types.ImmobileAgent import ImmobileAgent
from agent_types.RandomAgent import RandomAgent
from agent_types.GreedyAgent import GreedyAgent
from agent_types.CoordinatingAgent import CoordinatingAgent
from agent_types.CoordinatingAgentDemo import CoordinatingAgentDemo
from agent_types.AvoidingNearestAdversaryAgent import AvoidingNearestAdversaryAgent
from agent_types.RLAgent_2_adv import RLAgent2
from agent_types.RLAgent_3_adv import RLAgent3_1k, RLAgent3_100k, RLAgent3_100M
from agent_types.RLAgent_4_adv import RLAgent4_1k, RLAgent4_50k, RLAgent4_4M, RLAgent4_50M

NUM_GOOD = 1
NUM_LANDMARKS = 0
MAX_CYCLES = 200
NUM_EPISODES = 5
SAVE_PLOTS = False

RENDER_MODE = None


T = TypeVar('T')
def run_simple_tag_and_get_results(
        num_adversaries: int, 
        max_cycles: int, 
        render_mode,
        good_agent_type: Type[T],
        adversary_type: Type[T],
        num_episodes: int,
        RLAgent_model_name=None
        ):
    env = simple_tag_v3.parallel_env(num_good=NUM_GOOD, num_adversaries=num_adversaries, num_obstacles=NUM_LANDMARKS, max_cycles=max_cycles, continuous_actions=False, render_mode=render_mode)
    env.reset()
    if RLAgent_model_name != None:
        coord_agents = [adversary_type(name, num_adversaries, NUM_LANDMARKS, RLAgent_model_name) for name in env.agents if name != "agent_0"]
    else:
        coord_agents = [adversary_type(name, num_adversaries, NUM_LANDMARKS) for name in env.agents if name != "agent_0"]

    coord_agents.append(good_agent_type("agent_0", num_adversaries, NUM_LANDMARKS))

    episode_rewards = []
    for episode in range(1, num_episodes + 1):
        observations, infos = env.reset()
            
        episode_reward = 0
        while env.agents:
            # Update the observations of all the agents
            for agent in coord_agents:
                agent.see(observations[agent.name])

            # get the actions from all the agents according to their policies
            actions = {}
            for agent in coord_agents:
                action = agent.get_action()
                actions[agent.name] = action if action != None else 0
            
            # make a step in the environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # update the episode reward
            episode_reward += get_timestep_reward(rewards)

        episode_rewards.append(episode_reward)

    env.close()
    avg_rew = sum(episode_rewards) / len(episode_rewards)
    return avg_rew

def run_simple_tag_and_plot_results(
        num_adversaries_list: list[int], 
        good_agent_types: list[Type[T]],
        adversary_types: list[Type[T]],
        num_episodes: int,
        colors: list
        ):
    # Define fonts
    font_normal = FontProperties(style='normal')
    font_italic = FontProperties(style='italic')
    font_oblique = FontProperties(style='oblique')

    fig_list = []
    for i, num_adv in tqdm(enumerate(num_adversaries_list), total=len(num_adversaries_list), desc="Different num adv"):
        fig, ax = plt.subplots(1, len(good_agent_types), figsize=(10, 8))
        fig.suptitle(f'{num_adv} ADVERSARIES', 
                    fontsize=18, 
                    fontweight='bold', 
                    bbox=dict(facecolor='#abdbe3', 
                               edgecolor='black', 
                               boxstyle='Square, pad=0.5'),
                    fontproperties=font_oblique)

        fig.subplots_adjust(wspace=0.4)

        ax[0].set_ylabel(f'Avg. reward over {num_episodes} episodes', 
                         fontsize=12, 
                         fontweight=500, 
                         fontproperties=font_italic)
        for i in tqdm(range(len(good_agent_types)), desc="Good agent types"):
            x = [adv.__name__ for adv in adversary_types]
            y = [run_simple_tag_and_get_results(num_adv, MAX_CYCLES, RENDER_MODE, good_agent_types[i], adv_type, num_episodes) for adv_type in tqdm(adversary_types, desc="Adversary types", total=len(adversary_types))]
            bars = ax[i].bar(x, 
                      y, 
                      color=colors, 
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
        if SAVE_PLOTS:
            fig.savefig(f'./plots/{fig._suptitle.get_text()}. {num_episodes} episodes.png', bbox_inches='tight')
    plt.show()

# #run for graphical demo:
def run_demo(max_cycles=60, num_episodes=1):
    root = tk.Tk()
    root.withdraw()
    
    messagebox.showinfo("Demo", "Avoiding prey vs Greedy predators")
    run_simple_tag_and_get_results(3, max_cycles, "human", AvoidingAgent, GreedyAgent, num_episodes)

    messagebox.showinfo("Demo", "Random prey vs Coordinating(demo) predators")
    run_simple_tag_and_get_results(3, max_cycles, "human", RandomAgent, CoordinatingAgentDemo, num_episodes)
    
    messagebox.showinfo("Demo", "AvoidingNearestAdversary prey vs Coordinating(demo) predators")
    run_simple_tag_and_get_results(3, max_cycles, "human", AvoidingNearestAdversaryAgent, CoordinatingAgentDemo, num_episodes)

    messagebox.showinfo("Demo", "Avoiding prey vs RL predators after training for 1000 steps")
    run_simple_tag_and_get_results(3, max_cycles, "human", AvoidingAgent, RLAgent3_1k, num_episodes)

    messagebox.showinfo("Demo", "Avoiding prey vs RL predators after training for 100 000 steps")
    run_simple_tag_and_get_results(3, max_cycles, "human", AvoidingAgent, RLAgent3_100k, num_episodes)

    messagebox.showinfo("Demo", "Avoiding prey vs RL predators after training for 100 000 000 steps")
    run_simple_tag_and_get_results(3, max_cycles, "human", AvoidingAgent, RLAgent3_100M, num_episodes)

    root.destroy()

run_demo()
run_simple_tag_and_plot_results([2, 3, 4], [ImmobileAgent, RandomAgent, AvoidingAgent, AvoidingNearestAdversaryAgent], [RandomAgent, GreedyAgent, CoordinatingAgent], NUM_EPISODES, ['blue', 'red', 'green'])
run_simple_tag_and_plot_results([2], [AvoidingAgent, AvoidingNearestAdversaryAgent], [CoordinatingAgent, RLAgent2], NUM_EPISODES, ['green', 'purple'])
run_simple_tag_and_plot_results([3], [AvoidingAgent, AvoidingNearestAdversaryAgent], [CoordinatingAgent, RLAgent3_100M], NUM_EPISODES, ['green', 'purple'])
run_simple_tag_and_plot_results([4], [AvoidingAgent, AvoidingNearestAdversaryAgent], [CoordinatingAgent, RLAgent4_50M], NUM_EPISODES, ['green', 'purple'])
run_simple_tag_and_plot_results([4], [AvoidingAgent, AvoidingNearestAdversaryAgent], [RLAgent4_1k, RLAgent4_50k, RLAgent4_4M, RLAgent4_50M], NUM_EPISODES, ['purple', 'purple', 'purple', 'purple'])

#if we want to run the script from the command line with arguments:
"""
def main(run_graphical_demo, reproduce_plots):
    if run_graphical_demo:
        run_demo()

    if reproduce_plots:
        run_simple_tag_and_plot_results([2, 3, 4], [ImmobileAgent, RandomAgent, AvoidingAgent, AvoidingNearestAdversaryAgent], [RandomAgent, GreedyAgent, CoordinatingAgent], NUM_EPISODES)
        run_simple_tag_and_plot_results([3], [AvoidingAgent, AvoidingNearestAdversaryAgent], [GreedyAgent, CoordinatingAgent, RLAgent], NUM_EPISODES)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AASMA Project Script')
    parser.add_argument('--run_graphical_demo', action='store_true', help='runs the graphical demo')
    parser.add_argument('--reproduce_plots', action='store_true', help='reproduces the plots from the report')
    args = parser.parse_args()
    
    main(args.run_graphical_demo, args.reproduce_plots)
"""



