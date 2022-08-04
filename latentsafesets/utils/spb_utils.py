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
        vals = constraint.prob(np.array(row_states)).squeeze()#it is like calling forward of const_estimator!
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

def evaluate_cbfdot_func(cbfdot,
                             env,
                             file=None,
                             plot=True,
                             show=False,
                             skip=1,
                             action=(0,0)):#(1,1)):#2):#
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    #if walls is None:
    walls = [((75, 55), (100, 95))]  # the position and dimension of the wall
    #self.walls = [self._complex_obstacle(wall) for wall in walls]  # 140, the bound of the wall
    # it is a list of functions that depend on states
    selfwall_coords = np.array(walls)
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            old_state = np.array((x,y))#env._state_to_image((x, y)) / 255
            #print('old_state',old_state)#tuple
            #print('selfwall_coords[0][0]',selfwall_coords[0][0])#tuple
            #print('(old_state <= selfwall_coords[0][0])',(old_state <= selfwall_coords[0][0]))
            if (old_state <= selfwall_coords[0][0]).all():  # old_state#check it!
                reldistold = old_state - selfwall_coords[0][0]  # np.linalg.norm()
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] <= \
                    selfwall_coords[0][0][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][0][1]])
            elif old_state[0] >= selfwall_coords[0][1][0] and old_state[1] <= selfwall_coords[0][0][1]:
                reldistold = old_state - (selfwall_coords[0][1][0], selfwall_coords[0][0][1])
            elif old_state[0] >= selfwall_coords[0][1][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][1][0], 0])
            elif (old_state >= selfwall_coords[0][1]).all():  # old_state
                reldistold = old_state - selfwall_coords[0][1]
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] >= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][1][1]])
            elif old_state[0] <= selfwall_coords[0][0][0] and old_state[1] >= selfwall_coords[0][1][1]:
                reldistold = (old_state - (selfwall_coords[0][0][0], selfwall_coords[0][1][1]))
            elif old_state[0] <= selfwall_coords[0][0][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][0][0], 0])
            else:
                # print(old_state)#it can be [98.01472841 92.11425524]
                reldistold = np.array([0, 0])  # 9.9#
            rda=np.concatenate((reldistold,action))#thanks it is one-by-one
            row_states.append(rda)
        vals = cbfdot.cbfdots(np.array(row_states)).squeeze()#it is like calling forward of const_estimator!
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