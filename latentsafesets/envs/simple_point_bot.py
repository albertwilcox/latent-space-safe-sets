"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise. State
representation is (x, y). Action representation is (dx, dy).
"""

import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image, ImageDraw
from skimage.transform import resize

"""
Constants associated with the PointBot env.
"""

WINDOW_WIDTH = 180
WINDOW_HEIGHT = 150

MAX_FORCE = 3


class SimplePointBot(Env, utils.EzPickle):

    def __init__(self, from_pixels=True,
                 walls=None,
                 start_pos=(30, 75),
                 end_pos=(150, 75),
                 horizon=100,
                 constr_penalty=-100,
                 goal_thresh=3,
                 noise_scale=0.125):
        utils.EzPickle.__init__(self)
        self.done = self.state = None
        self.horizon = horizon
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.goal_thresh = goal_thresh
        self.noise_scale = noise_scale
        self.constr_penalty = constr_penalty
        self.action_space = Box(-np.ones(2) * MAX_FORCE,
                                np.ones(2) * MAX_FORCE)
        if from_pixels:
            self.observation_space = Box(-1, 1, (3, 64, 64))
        else:
            self.observation_space = Box(-np.ones(2) * np.float('inf'),
                                         np.ones(2) * np.float('inf'))
        self._episode_steps = 0
        # self.obstacle = self._complex_obstacle(OBSTACLE_COORDS)
        if walls is None:
            walls = [((75, 55), (100, 95))]
        self.walls = [self._complex_obstacle(wall) for wall in walls]
        self.wall_coords = walls
        self._from_pixels = from_pixels
        self._image_cache = {}

    def step(self, a):
        a = self._process_action(a)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a)
        cur_reward = self.step_reward(self.state, a)
        self.state = next_state
        self._episode_steps += 1
        constr = self.obstacle(next_state)
        self.done = self._episode_steps >= self.horizon

        if self._from_pixels:
            obs = self._state_to_image(self.state)
        else:
            obs = self.state
        return obs, cur_reward, self.done, {
            "constraint": constr,
            "reward": cur_reward,
            "state": old_state,
            "next_state": next_state,
            "action": a
        }

    def reset(self, random_start=False):
        if random_start:
            self.state = np.random.random(2) * (WINDOW_WIDTH, WINDOW_HEIGHT)
            if self.obstacle(self.state):
                self.reset(True)
        else:
            self.state = self.start_pos + np.random.randn(2)
        self.done = False
        self._episode_steps = 0
        if self._from_pixels:
            obs = self._state_to_image(self.state)
        else:
            obs = self.state
        return obs

    def render(self, mode='human'):
        return self._draw_state(self.state)

    def _draw_state(self, state):
        BCKGRND_COLOR = (0, 0, 0)
        ACTOR_COLOR = (255, 0, 0)
        OBSTACLE_COLOR = (0, 0, 255)

        def draw_circle(draw, center, radius, color):
            lower_bound = tuple(np.subtract(center, radius))
            upper_bound = tuple(np.add(center, radius))
            draw.ellipse([lower_bound, upper_bound], fill=color)

        im = Image.new("RGB", (WINDOW_WIDTH, WINDOW_HEIGHT), BCKGRND_COLOR)
        draw = ImageDraw.Draw(im)

        draw_circle(draw, state, 10, ACTOR_COLOR)
        for wall in self.wall_coords:
            draw.rectangle(wall, fill=OBSTACLE_COLOR, outline=(0, 0, 0), width=1)

        return np.array(im)

    def _next_state(self, s, a, override=False):
        if self.obstacle(s):
            return s

        next_state = s + a + self.noise_scale * np.random.randn(len(s))
        next_state = np.clip(next_state, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT))
        return next_state

    def step_reward(self, s, a):
        """
        Returns -1 if not in goal otherwise 0
        """
        return int(np.linalg.norm(np.subtract(self.end_pos, s)) < self.goal_thresh) - 1

    def obstacle(self, s):
        return any([wall(s) for wall in self.walls])

    @staticmethod
    def _complex_obstacle(bounds):
        """
        Returns a function that returns true if a given state is within the
        bounds and false otherwise
        :param bounds: bounds in form [[X_min, Y_min], [X_max, Y_max]]
        :return: function described above
        """
        min_x, min_y = bounds[0]
        max_x, max_y = bounds[1]

        def obstacle(state):
            if type(state) == np.ndarray:
                lower = (min_x, min_y)
                upper = (max_x, max_y)
                state = np.array(state)
                component_viol = (state > lower) * (state < upper)
                return np.product(component_viol, axis=-1)
            if type(state) == torch.Tensor:
                lower = torch.from_numpy(np.array((min_x, min_y)))
                upper = torch.from_numpy(np.array((max_x, max_y)))
                component_viol = (state > lower) * (state < upper)
                return torch.prod(component_viol, dim=-1)

        return obstacle

    @staticmethod
    def _process_action(a):
        return np.clip(a, -MAX_FORCE, MAX_FORCE)

    def _state_to_image(self, state):
        def state_to_int(state):
            return int(state[0]), int(state[1])

        state = state_to_int(state)
        image = self._image_cache.get(state)
        if image is None:
            image = self._draw_state(state)
            image = image.transpose((2, 0, 1))
            image = (resize(image, (3, 64, 64)) * 255).astype(np.uint8)
            self._image_cache[state] = image
        return image

    def draw(self, trajectories=None, heatmap=None, plot_starts=False, board=True, file=None,
             show=False):
        """
        Draws the desired trajectories and heatmaps (probably would be a safe set) to pyplot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if heatmap is not None:
            assert heatmap.shape == (WINDOW_HEIGHT, WINDOW_WIDTH)
            heatmap = np.flip(heatmap, axis=0)
            im = plt.imshow(heatmap, cmap='hot')
            plt.colorbar(im)

        if board:
            self.draw_board(ax)

        if trajectories is not None and type(trajectories) == list:
            if type(trajectories[0]) == list:
                self.plot_trajectories(ax, trajectories, plot_starts)
            if type(trajectories[0]) == dict:
                self.plot_trajectory(ax, trajectories, plot_starts)

        ax.set_aspect('equal')
        ax.autoscale_view()

        if file is not None:
            plt.savefig(file)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_trajectory(self, ax, trajectory, plot_start=False):
        self.plot_trajectories(ax, [trajectory], plot_start)

    def plot_trajectories(self, ax, trajectories, plot_start=False):
        """
        Renders a trajectory to pyplot. Assumes you already have a plot going
        :param ax:
        :param trajectories: Trajectories to impose upon the graph
        :param plot_start: whether or not to draw a circle at the start of the trajectory
        :return:
        """

        for trajectory in trajectories:
            states = np.array([frame['obs'] for frame in trajectory])
            plt.plot(states[:, 0], WINDOW_HEIGHT - states[:, 1])
            if plot_start:
                start = states[0]
                start_circle = plt.Circle((start[0], WINDOW_HEIGHT - start[1]),
                                          radius=2, color='lime')
                ax.add_patch(start_circle)

    def draw_board(self, ax):
        plt.xlim(0, WINDOW_WIDTH)
        plt.ylim(0, WINDOW_HEIGHT)

        for wall in self.wall_coords:
            width, height = np.subtract(wall[1], wall[0])
            ax.add_patch(
                patches.Rectangle(
                    xy=wall[0],  # point of origin.
                    width=width,
                    height=height,
                    linewidth=1,
                    color='red',
                    fill=True
                )
            )

        circle = plt.Circle(self.start_pos, radius=3, color='k')
        ax.add_patch(circle)
        circle = plt.Circle(self.end_pos, radius=3, color='k')
        ax.add_patch(circle)
        ax.annotate("start", xy=(self.start_pos[0], self.start_pos[1] - 8), fontsize=10,
                    ha="center")
        ax.annotate("goal", xy=(self.end_pos[0], self.end_pos[1] - 8), fontsize=10, ha="center")


class SimplePointBotLong(SimplePointBot):
    def __init__(self, from_pixels=True):
        super().__init__(from_pixels,
                         start_pos=(15, 20),
                         end_pos=(165, 20),
                         walls=[((80, 55), (100, 150)),
                                ((30, 0), (45, 100)),
                                ((30, 120), (45, 150)),
                                ((135, 0), (150, 120))],
                         horizon=500)
