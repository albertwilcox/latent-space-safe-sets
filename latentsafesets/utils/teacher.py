from abc import ABC

from latentsafesets.envs import simple_point_bot as spb

import numpy as np


class AbstractTeacher(ABC):

    def __init__(self, env, noisy=False, on_policy=True, horizon=None):
        self.env = env
        self.noisy = noisy
        self.on_policy = on_policy

        self.ac_high = env.action_space.high
        self.ac_low = env.action_space.low
        self.noise_std = (self.ac_high - self.ac_low) / 5
        self.random_start = False
        if horizon is None:
            self.horizon = env.horizon

    def generate_demonstrations(self, num_demos, store_noisy=True, noise_param=None):
        demonstrations = []
        for i in range(num_demos):
            demo = self.generate_trajectory(noise_param, store_noisy=store_noisy)
            reward = sum([frame['reward'] for frame in demo])
            print('Trajectory %d, Reward %d' % (i, reward))
            demonstrations.append(demo)
        return demonstrations

    def generate_trajectory(self, noise_param=None, store_noisy=True):
        """
        The teacher initially tries to go northeast before going to the origin
        """
        self.reset()
        transitions = []
        obs = self.env.reset()
        # state = np.zeros((0, 0))
        state = None
        done = False
        for i in range(self.horizon):
            if state is None:
                action = self.env.action_space.sample().astype(np.float64)
            else:
                action = self._expert_control(state, i).astype(np.float64)
            if self.noisy:
                action_input = np.random.normal(action, self.noise_std)
                action_input = np.clip(action_input, self.ac_low, self.ac_high)
            else:
                action_input = action

            if store_noisy:
                action = action_input

            next_obs, reward, done, info = self.env.step(action_input)
            transition = {'obs': obs, 'action': tuple(action), 'reward': float(reward),
                          'next_obs': next_obs, 'done': int(done),
                          'constraint': 0, 'safe_set': 0,
                          'on_policy': int(self.on_policy)}
            # print({k: v.dtype for k, v in transition.items() if 'obs' in k})
            transitions.append(transition)
            state = None # info['next_state']
            obs = next_obs

            if done:
                break

        transitions[-1]['done'] = 1

        rtg = 0
        ss = 0
        for frame in reversed(transitions):
            if frame['reward'] >= 0:
                ss = 1

            frame['safe_set'] = ss
            frame['rtg'] = rtg

            rtg = rtg + frame['reward']

        # assert done, "Did not reach the goal set on task completion."
        # V = self.env.values()
        # for i, t in enumerate(transitions):
        #     t['values'] = V[i]
        return transitions

    def _expert_control(self, state, i):
        raise NotImplementedError("Override in subclass")

    def reset(self):
        pass


class SimplePointBotTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super().__init__(env, noisy)
        self.goal = (150, 75)

    def _expert_control(self, s, t):
        if t < 20:
            goal = np.array((30, 15))
        elif t < 60:
            goal = np.array((150, 15))
        else:
            goal = self.goal

        act = np.subtract(goal, s)
        act = np.clip(act, -3, 3)
        return act


class ConstraintTeacher(AbstractTeacher):
    def __init__(self, env, noisy=True):
        super().__init__(env, noisy, on_policy=False)
        self.d = (np.random.random(2) * 2 - 1) * spb.MAX_FORCE
        self.goal = (88, 75)
        self.random_start = True

    def _expert_control(self, state, i):
        if i < 15:
            return self.d
        else:
            to_obstactle = np.subtract(self.goal, state)
            to_obstacle_normalized = to_obstactle / np.linalg.norm(to_obstactle)
            to_obstactle_scaled = to_obstacle_normalized * spb.MAX_FORCE / 2
            return to_obstactle_scaled

    def reset(self):
        self.d = (np.random.random(2) * 2 - 1) * spb.MAX_FORCE


class ReacherTeacher(AbstractTeacher):
    def __init__(self, env, noisy=True):
        super().__init__(env, noisy=noisy, horizon=100)

    def _expert_control(self, state, i):
        if i < 40:
            goal = np.array((np.pi, 0))
        else:
            goal = np.array((np.pi * .75, 0))

        angle = state[:2]
        act = goal - angle
        act = np.clip(act, -1, 1)
        return act

class RopeTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super().__init__(env, horizon=15)
        self.horizon = 15

class ClothTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super().__init__(env, horizon=15)
        self.horizon = 15

class GymClothTeacher(AbstractTeacher):
    def __init__(self,env,noisy=False):
        super().__init__(env,noisy=noisy,horizon=20)
        self.horizon = 20 


class ReacherConstraintTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super(ReacherConstraintTeacher, self).__init__(env, noisy, on_policy=False)
        self.direction = 1
        self.random_start = True

    def _expert_control(self, state, i):
        angle = state[:2]
        goal1 = np.array((np.pi * .53, 0.7 * np.pi))
        goal2 = np.array((np.pi, -0.7 * np.pi))
        goal = min(goal1, goal2, key=lambda x: np.linalg.norm(angle - x))
        act = goal - angle
        # act = np.random.normal((self.direction, 0), 1)
        act = np.clip(act, -1, 1)
        return act

    def reset(self):
        self.direction = self.direction * -1


class PushTeacher(AbstractTeacher):

    def __init__(self, env, noisy):
        super(PushTeacher, self).__init__(env, False)
        self.demonstrations = []
        self.default_noise = 0.2
        self.block_id = 0
        self.horizon = 150

    def _expert_control(self, state, i):
        action, block_done = self.env.expert_action(block=self.block_id, noise_std=0.004)
        if block_done:
            self.block_id += 1
            self.block_id = min(self.block_id, 2)

        return action

    def reset(self):
        self.block_id = 0


class StrangeTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super(StrangeTeacher, self).__init__(env, noisy, on_policy=False)
        self.d_act = env.action_space.shape
        self.high = env.action_space.high
        self.low = env.action_space.low
        self.std = (self.high - self.low) / 10
        self.last_action = env.action_space.sample()
        self.random_start = True
        self.horizon = 20

    def _expert_control(self, state, i):
        action = np.random.normal(self.last_action, self.std)
        action = np.clip(action, self.low, self.high)
        self.last_action = action
        return action

    def reset(self):
        self.last_action = self.env.action_space.sample()


class OutburstPushTeacher(AbstractTeacher):
    def __init__(self, env, noisy):
        super(OutburstPushTeacher, self).__init__(env, False, False)
        # self.block_id = 0
        self.horizon = 150
        self.outburst = False

    def _expert_control(self, state, i):
        if np.random.random() > .8:
            self.outburst = True

        if np.random.random() > .9:
            self.outburst = False

        if self.outburst:
            return self.env.action_space.sample().astype(np.float64)

        return np.array((0, -0.02))

    def reset(self):
        self.block_id = 0
        self.outburst = False



