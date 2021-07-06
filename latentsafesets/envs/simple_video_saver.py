import gym
from gym.wrappers import LazyFrames
import os

import moviepy.editor as mpy


class SimpleVideoSaver(gym.Wrapper):
    def __init__(self, env: gym.Env, video_dir):
        super().__init__(env)
        self.env = env
        self.dir = video_dir

        os.mkdir(self.dir)
        self.video_buffer = []
        self.count = 0

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if type(next_obs) == LazyFrames:
            next_obs_in = next_obs[0]
        else:
            next_obs_in = next_obs
        self.video_buffer.append(next_obs_in.transpose((1, 2, 0)))
        return next_obs, reward, done, info

    def reset(self, **kwargs):
        if len(self.video_buffer) > 0:
            self._make_movie()
        self.video_buffer = []
        self.count += 1

        obs = self.env.reset(**kwargs)
        if type(obs) == LazyFrames:
            obs_in = obs[0]
        else:
            obs_in = obs
        self.video_buffer.append(obs_in.transpose((1, 2, 0)))

        return obs

    def _make_movie(self):
        file = os.path.join(self.dir, '%d.gif' % self.count)
        clip = mpy.ImageSequenceClip(self.video_buffer, fps=10)
        clip.write_gif(file)
