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
                 start_pos=(30, 75),#if not random start, then you start (around) here
                 end_pos=(150, 75),#a single point
                 horizon=100,#by default
                 constr_penalty=-100,
                 goal_thresh=3,
                 noise_scale=0.125):#0.125
        utils.EzPickle.__init__(self)
        self.done = self.state = None
        self.horizon = horizon
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.goal_thresh = goal_thresh
        self.noise_scale = noise_scale#0.125 by default
        self.constr_penalty = constr_penalty
        self.action_space = Box(-np.ones(2) * MAX_FORCE,
                                np.ones(2) * MAX_FORCE)
        if from_pixels:
            self.observation_space = Box(-1, 1, (3, 64, 64))
        else:
            self.observation_space = Box(-np.ones(2) * np.float('inf'),
                                         np.ones(2) * np.float('inf'))
        self._episode_steps = 0#start with 0!
        # self.obstacle = self._complex_obstacle(OBSTACLE_COORDS)
        if walls is None:
            walls = [((75, 55), (100, 95))]#the position and dimension of the wall
        self.walls = [self._complex_obstacle(wall) for wall in walls]#140, the bound of the wall
        #it is a list of functions that depend on states
        self.wall_coords = walls
        self._from_pixels = from_pixels
        self._image_cache = {}#it is a dictionary

    def step(self, a):#it returns#if using cbf dot estimator, then see the sepsafety function below
        a = self._process_action(a)#line 166, an action satisfying the constraint
        old_state = self.state.copy()#2d state
        next_state = self._next_state(self.state, a)#122, go to the next 2d state with noise
        cur_reward = self.step_reward(self.state, a)#130, the reward of the current state
        self.state = next_state#move on to the next step
        self._episode_steps += 1
        constr = self.obstacle(next_state)#line 136 to check if next_state is obstacle
        self.done = self._episode_steps >= self.horizon#just over time limit!

        if self._from_pixels:
            obs = self._state_to_image(self.state)#line 169#it is a 3-channel image
        else:
            obs = self.state#it is a 2-d state
        return obs, cur_reward, self.done, {
            "constraint": constr,#it is also a dictionary!
            "reward": cur_reward,
            "state": old_state,
            "next_state": next_state,
            "action": a#the current action!
        }

    def stepsafety(self, a):#it returns
        a = self._process_action(a)#line 166, an action satisfying the constraint
        old_state = self.state.copy()#2d state
        next_state = self._next_state(self.state, a)#122, go to the next 2d state with noise
        cur_reward = self.step_reward(self.state, a)#130, the reward of the current state
        self.state = next_state#move on to the next step
        self._episode_steps += 1
        constr = self.obstacle(next_state)#line 136 to check if next_state is obstacle
        self.done = self._episode_steps >= self.horizon#just over time limit!

        if self._from_pixels:
            obs = self._state_to_image(self.state)#line 169#it is a 3-channel image
        else:
            obs = self.state#it is a 2-d state
        #find the nearest distance to the obstacle according to different regions
        #I set 8 regions versue the central obstacle: upper left, upper middle, upper right, right middle, lower right, lower middle, lower left, left middle
        if (old_state<=self.wall_coords[0][0]).all():#left upper#old_state#check it!
            reldistold=old_state-self.wall_coords[0][0]#relative distance old#np.linalg.norm()
        elif self.wall_coords[0][0][0]<=old_state[0]<=self.wall_coords[0][1][0] and old_state[1]<=self.wall_coords[0][0][1]:
            reldistold = np.array([0,old_state[1] - self.wall_coords[0][0][1]])#middle up
        elif old_state[0]>=self.wall_coords[0][1][0] and old_state[1]<=self.wall_coords[0][0][1]:
            reldistold = old_state - (self.wall_coords[0][1][0],self.wall_coords[0][0][1])#upper right
        elif old_state[0]>=self.wall_coords[0][1][0] and self.wall_coords[0][0][1]<=old_state[1]<=self.wall_coords[0][1][1]:
            reldistold = np.array([old_state[0] - self.wall_coords[0][1][0],0])#right middle
        elif (old_state>=self.wall_coords[0][1]).all():#old_state#lower right
            reldistold = old_state - self.wall_coords[0][1]
        elif self.wall_coords[0][0][0]<=old_state[0]<=self.wall_coords[0][1][0] and old_state[1]>=self.wall_coords[0][1][1]:
            reldistold = np.array([0,old_state[1] - self.wall_coords[0][1][1]])#middle down/lower middle
        elif old_state[0]<=self.wall_coords[0][0][0] and old_state[1]>=self.wall_coords[0][1][1]:
            reldistold = (old_state - (self.wall_coords[0][0][0],self.wall_coords[0][1][1]))#lower left
        elif old_state[0]<=self.wall_coords[0][0][0] and self.wall_coords[0][0][1]<=old_state[1]<=self.wall_coords[0][1][1]:
            reldistold = np.array([old_state[0] - self.wall_coords[0][0][0],0])#middle left
        else:
            #print(old_state)#it can be [98.01472841 92.11425524]
            reldistold=np.array([0,0])#9.9#
        hvalueold = np.linalg.norm(reldistold) ** 2 - 15 ** 2#get the value of the h function
        if self._from_pixels:
            obs = self._state_to_image(self.state)#line 169#it is a 3-channel image
        else:
            obs = self.state#it is a 2-d state
        if (next_state <= self.wall_coords[0][0]).all():  # old_state#check it!
            reldistnew = next_state - self.wall_coords[0][0]#relative distance new # np.linalg.norm()
        elif self.wall_coords[0][0][0] <= next_state[0] <= self.wall_coords[0][1][0] and next_state[1] <= \
                self.wall_coords[0][0][1]:
            reldistnew = np.array([0, next_state[1] - self.wall_coords[0][0][1]])
        elif next_state[0] >= self.wall_coords[0][1][0] and next_state[1] <= self.wall_coords[0][0][1]:
            reldistnew = next_state - (self.wall_coords[0][1][0], self.wall_coords[0][0][1])
        elif next_state[0] >= self.wall_coords[0][1][0] and self.wall_coords[0][0][1] <= next_state[1] <= \
                self.wall_coords[0][1][1]:
            reldistnew = np.array([next_state[0] - self.wall_coords[0][1][0], 0])
        elif (next_state >= self.wall_coords[0][1]).all():  # old_state
            reldistnew = next_state - self.wall_coords[0][1]
        elif self.wall_coords[0][0][0] <= next_state[0] <= self.wall_coords[0][1][0] and next_state[1] >= \
                self.wall_coords[0][1][1]:
            reldistnew = np.array([0, next_state[1] - self.wall_coords[0][1][1]])
        elif next_state[0] <= self.wall_coords[0][0][0] and next_state[1] >= self.wall_coords[0][1][1]:
            reldistnew = (next_state - (self.wall_coords[0][0][0], self.wall_coords[0][1][1]))
        elif next_state[0] <= self.wall_coords[0][0][0] and self.wall_coords[0][0][1] <= next_state[1] <= \
                self.wall_coords[0][1][1]:
            reldistnew = np.array([next_state[0] - self.wall_coords[0][0][0], 0])
        else:
            # print(old_state)#it can be [98.01472841 92.11425524]
            reldistnew = np.array([0, 0])  # 9.9#
        hvaluenew = np.linalg.norm(reldistnew) ** 2 - 15 ** 2
        hvd=hvaluenew-hvalueold#hvd for h value difference
        return obs, cur_reward, self.done, {
            "constraint": constr,#it is also a dictionary!
            "reward": cur_reward,
            "state": old_state,
            "next_state": next_state,
            "action": a,#the current action!
            "rdo":reldistold,#rdo for relative distance old#array now!
            "rdn": reldistnew,#rdn for relative distance new#array now!
            "hvo": hvalueold,#hvo for h value old
            "hvn":hvaluenew,#hvn for h value new
            "hvd":hvd#hvd for h value difference

        }

    def reset(self, random_start=False):
        if random_start:
            self.state = np.random.random(2) * (WINDOW_WIDTH, WINDOW_HEIGHT)
            if self.obstacle(self.state):
                self.reset(True)
        else:
            self.state = self.start_pos + np.random.randn(2)#respawn around the start_pos
        self.done = False#unless the starting point is within 3 meters of the goal point!
        self._episode_steps = 0
        if self._from_pixels:#then move to this case as Nik points out
            obs = self._state_to_image(self.state)#line 169
        else:#I start from this condition
            obs = self.state
        return obs

    def render(self, mode='human'):
        return self._draw_state(self.state)#see line 103

    def _draw_state(self, state):#it returns the image in the form of 3-channel numpy array
        BCKGRND_COLOR = (0, 0, 0)#black#why not green?
        ACTOR_COLOR = (255, 0, 0)#red
        OBSTACLE_COLOR = (0, 0, 255)#blue

        def draw_circle(draw, center, radius, color):#draw is a few lines later!
            lower_bound = tuple(np.subtract(center, radius))
            upper_bound = tuple(np.add(center, radius))
            draw.ellipse([lower_bound, upper_bound], fill=color)

        im = Image.new("RGB", (WINDOW_WIDTH, WINDOW_HEIGHT), BCKGRND_COLOR)#draw the background
        draw = ImageDraw.Draw(im)#on this blank cloth?

        draw_circle(draw, state, 10, ACTOR_COLOR)#draw a circle at state with radius=10 in red!!!
        for wall in self.wall_coords:#draw an obstacle with blue with black outline width 1
            draw.rectangle(wall, fill=OBSTACLE_COLOR, outline=(0, 0, 0), width=1)

        return np.array(im)#you have got the image in the form of numpy array!

    def _next_state(self, s, a, override=False):
        if self.obstacle(s):
            return s#you cannot go further as you will run into obstacles!
        #is it just single integrator dynamics?
        next_state=s+a+self.noise_scale*np.random.randn(len(s))#dim (len(s))#what dynamics?
        next_state = np.clip(next_state, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT))#must within the map
        return next_state

    def step_reward(self, s, a):
        """
        Returns -1 if not in goal otherwise 0
        """#then what is the point of the goal region classifier?#It will lead to mis-classification
        return int(np.linalg.norm(np.subtract(self.end_pos, s)) < self.goal_thresh) - 1

    def obstacle(self, s):#as long as there is one state that is dangerous, it is dangerous
        return any([wall(s) for wall in self.walls])#1 or 0, right?

    @staticmethod
    def _complex_obstacle(bounds):
        """#it returns a function, OK?
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
                return np.product(component_viol, axis=-1)#when both are 1, it means obstacle!
            if type(state) == torch.Tensor:
                lower = torch.from_numpy(np.array((min_x, min_y)))
                upper = torch.from_numpy(np.array((max_x, max_y)))
                component_viol = (state > lower) * (state < upper)
                return torch.prod(component_viol, dim=-1)

        return obstacle

    @staticmethod
    def _process_action(a):#seems just to avoid saturation/to satisfy constraints
        return np.clip(a, -MAX_FORCE, MAX_FORCE)

    def _state_to_image(self, state):#you get the observation of the state
        def state_to_int(state):
            return int(state[0]), int(state[1])#where you are currently

        state = state_to_int(state)#get the coordinate of this state
        image = self._image_cache.get(state)#state is the key#seems none at this point
        if image is None:
            image = self._draw_state(state)#see line 103#3 channel images
            image = image.transpose((2, 0, 1))#put the channels first!
            image = (resize(image, (3, 64, 64)) * 255).astype(np.uint8)
            self._image_cache[state] = image#key: state, value: image
        return image#mainly to get the image

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
            self.draw_board(ax)#see line 237

        if trajectories is not None and type(trajectories) == list:
            if type(trajectories[0]) == list:
                self.plot_trajectories(ax, trajectories, plot_starts)#line 219
            if type(trajectories[0]) == dict:
                self.plot_trajectory(ax, trajectories, plot_starts)#line 216

        ax.set_aspect('equal')
        ax.autoscale_view()

        if file is not None:
            plt.savefig(file)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_trajectory(self, ax, trajectory, plot_start=False):#line 219
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
            plt.plot(states[:, 0], WINDOW_HEIGHT - states[:, 1])#horizontal and vertical axis
            if plot_start:
                start = states[0]#the horizontal and vertical axis of the starting point
                start_circle = plt.Circle((start[0], WINDOW_HEIGHT - start[1]),
                                          radius=2, color='lime')
                ax.add_patch(start_circle)

    def draw_board(self, ax):
        plt.xlim(0, WINDOW_WIDTH)
        plt.ylim(0, WINDOW_HEIGHT)

        for wall in self.wall_coords:#in simple env, there is only one wall
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
                    ha="center")#just annotating the texts!
        ax.annotate("goal", xy=(self.end_pos[0], self.end_pos[1] - 8), fontsize=10, ha="center")


class SimplePointBotLong(SimplePointBot):
    def __init__(self, from_pixels=True):
        super().__init__(from_pixels,
                         start_pos=(15, 20),
                         end_pos=(165, 20),
                         walls=[((80, 55), (100, 150)),#multiple obstacles
                                ((30, 0), (45, 100)),
                                ((30, 120), (45, 150)),
                                ((135, 0), (150, 120))],
                         horizon=500)
