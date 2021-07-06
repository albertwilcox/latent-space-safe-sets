import latentsafesets.envs.simple_point_bot as spb

import numpy as np
from tqdm import tqdm


def evaluate_safe_set(s_set,
                      env,
                      file=None,
                      plot=True,
                      show=False,
                      skip=2):
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            state = env._state_to_image((x, y)) / 255
            row_states.append(state)
        vals = s_set.safe_set_probability_np(np.array(row_states)).squeeze()
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y+1, ::2], data[y+1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show)

    return data


def evaluate_value_func(value_func,
                        env,
                        file=None,
                        plot=True,
                        show=False,
                        skip=2):
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            state = env._state_to_image((x, y)) / 255
            row_states.append(state)
        vals = value_func.get_value_np(np.array(row_states)).squeeze()
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show)

    return data


def evaluate_constraint_func(constraint,
                             env,
                             file=None,
                             plot=True,
                             show=False,
                             skip=2):
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            state = env._state_to_image((x, y)) / 255
            row_states.append(state)
        vals = constraint.prob(np.array(row_states)).squeeze()
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show, board=False)

    return data

