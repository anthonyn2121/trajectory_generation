import setup
import numpy as np
import json
import matplotlib.pyplot as plt
from environment_toolkit.environment import Environment
from environment_toolkit.occupancy_map import OccupancyMap
from a_star.a_star import graph_search
from trajectory_generator import PolyTraj


def plot_trajectory(trajectory, waypoints, ax):
    trajectory = np.asarray(trajectory).reshape((200, 3))
    start = waypoints[0]
    goal = waypoints[-1]

    xs, ys, zs = waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]
    ax.scatter(xs, ys, zs, color="magenta", label="Waypoints")
    ax.scatter(start[0], start[1], start[2], color='green', marker='o')
    ax.text(start[0], start[1], start[2], "START", color='green')
    ax.scatter(goal[0], goal[1], goal[2], color='green', marker='o')
    ax.text(goal[0], goal[1], goal[2], "GOAL", color='red')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig("mytrajectory.png")
    plt.show(block=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", '-e', default="environment_toolkit/worlds/test_simple.json", help="JSON file that describes environment")
    parser.add_argument("--order", "-o", type=int, default=4)
    args = parser.parse_args()

    world_filepath = args.environment
    with open(world_filepath, 'r') as file:
        world = json.load(file)

    world_data = {key:world[key] for key in ["bounds", "blocks"]}
    env = Environment(world_data)
    start = tuple(world['start'])
    goal = tuple(world['goal'])
    margin = world['margin']
    resolution = world['resolution']
    max_simulation_time = 100

    waypoints = graph_search(env, start, goal, margin, resolution)
    waypoints = np.asarray([start] + waypoints + [goal]).reshape((-1, 3))
    waypoints = np.unique(waypoints, axis=0)

    traj = PolyTraj(args.order, waypoints)
    time_span = np.linspace(0, max_simulation_time, 200)
    trajectory = []
    for t in time_span:
        pos = traj.update(t)
        if (np.any(pos) == None):
            pos = trajectory[-1]
        trajectory.append(pos)

    ax = env.get_plot()
    
    plot_trajectory(trajectory, waypoints, ax)