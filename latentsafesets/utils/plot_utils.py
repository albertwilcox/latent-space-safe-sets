import latentsafesets.utils.pytorch_utils as ptu
import latentsafesets.utils.spb_utils as spbu
from latentsafesets.envs import SimplePointBot

import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import LazyFrames

from moviepy.editor import VideoClip


def loss_plot(losses, file, title=None):
    if title is None:
        title = 'Loss'
    log_scale = np.min(losses) > 0
    simple_plot(losses, title=title, show=False, file=file, ylabel='Loss', xlabel='Iters', log=log_scale)


def simple_plot(data, std=None, title=None, show=False, file=None, ylabel=None, xlabel=None, log=False):
    plt.figure()
    if log:
        plt.semilogy(data)
    else:
        plt.plot(data)

    if std is not None:
        assert not log, 'not sure how to implement this with log'
        upper = np.add(data, std)
        lower = np.subtract(data, std)
        xs = np.arange(len(lower))
        plt.fill_between(xs, lower, upper, alpha=0.3)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if file is not None:
        plt.savefig(file)

    if show:
        plt.show()
    else:
        plt.close()


def make_movie(trajectory, file):
    def float_to_int(im):
        if np.max(im) <= 1:
            im = im * 255
            im = im.astype(int)
        im = np.nan_to_num(im)
        return im
    ims = []
    for frame in trajectory:
        if type(frame) == dict:
            frame = frame['obs']
        if type(frame) == LazyFrames:
            frame = frame[0]
        if type(frame) == np.ndarray:
            ims.append(float_to_int(frame.transpose((1, 2, 0))))
            # print(float_to_int(frame.transpose((1, 2, 0))).shape)
        else:
            raise ValueError

    def make_frame(t):
        """Returns an image of the frame for time t."""
        # ... create the frame with any library here ...
        return ims[int(round(t*10))]

    if 'gif' in file:
        codec = 'gif'
    elif 'mp4' in file:
        codec = 'mpeg4'
    else:
        codec = None

    duration = int(np.ceil(len(ims) / 10))
    while len(ims) < duration * 10 + 1:
        ims.append(ims[-1])
    clip = VideoClip(make_frame, duration=duration)
    clip.set_fps(10)
    clip.write_videofile(file, fps=10, codec=codec)


def visualize_value(obs, value_func, file, env=None):
    """
    Sorts the observations to show which ones have high value, which ones low
    """
    if issubclass(type(env), SimplePointBot):
        spbu.evaluate_value_func(value_func, env, file=file, skip=2)
        return
    values = value_func.forward_np(obs, already_embedded=True).squeeze()
    obs = ptu.to_numpy(value_func.encoder.decode(ptu.torchify(obs)))
    sort_ind = np.argsort(values)
    low_ind = sort_ind[:5]
    high_ind = sort_ind[-5:]
    low_vals = values[low_ind]

    if len(obs.shape) == 5:
        obs = obs[:, 0]

    low_obs = obs[low_ind]
    high_vals = values[high_ind]
    high_obs = obs[high_ind]

    fig, axs = plt.subplots(2, 5)

    for i in range(5):
        axs[0][i].imshow(high_obs[i].squeeze().transpose((1, 2, 0)))
        axs[0][i].set_title('%3.3f' % high_vals[i])
        axs[0][i].set_axis_off()

        axs[1][i].imshow(low_obs[i].squeeze().transpose((1, 2, 0)))
        axs[1][i].set_title('%3.3f' % low_vals[i])
        axs[1][i].set_axis_off()

    plt.savefig(file)
    plt.close()


def visualize_safe_set(obs, safe_set, file, env=None):
    if issubclass(type(env), SimplePointBot):
        spbu.evaluate_safe_set(safe_set, env, file=file, skip=2)
        return
    ss = safe_set.safe_set_probability_np(obs, already_embedded=True).squeeze()

    obs = ptu.to_numpy(safe_set.encoder.decode(ptu.torchify(obs)))
    ss = ss > 0.8
    nonzeros = np.nonzero(ss)[0]
    zeros = np.nonzero(np.logical_not(ss))[0]

    if len(nonzeros) > 0:
        nonzero_inds = nonzeros[np.random.randint(len(nonzeros), size=5)]
    if len(zeros) > 0:
        zero_inds = zeros[np.random.randint(len(zeros), size=5)]

    fig, axs = plt.subplots(2, 5)

    if len(obs.shape) == 5:
        obs = obs[:, 0]

    for i in range(5):
        if len(nonzeros) > 0:
            im = obs[nonzero_inds[i]].squeeze().transpose((1, 2, 0))
        else:
            im = np.zeros((64, 64, 3))
        axs[0][i].imshow(im)
        axs[0][i].set_title('In SS')
        axs[0][i].set_axis_off()

        if len(zeros) > 0:
            im = obs[zero_inds[i]].squeeze().transpose((1, 2, 0))
        else:
            im = np.zeros((64, 64, 3))
        axs[1][i].imshow(im)
        axs[1][i].set_title('Out SS')
        axs[1][i].set_axis_off()

    plt.savefig(file)
    plt.close()


def visualize_onezero(obs, onezero, file, env=None):
    if issubclass(type(env), SimplePointBot):
        spbu.evaluate_constraint_func(onezero, env, file=file, skip=2)
        return
    ss = onezero.prob(obs, already_embedded=True).squeeze()
    obs = ptu.to_numpy(onezero.encoder.decode(ptu.torchify(obs)))
    # print(ss)
    ss = ss > 0.8
    nonzeros = np.nonzero(ss)[0]
    zeros = np.nonzero(np.logical_not(ss))[0]
    # print(nonzeros, zeros)

    if len(nonzeros) > 0:
        nonzero_inds = nonzeros[np.random.randint(len(nonzeros), size=5)]
    if len(zeros) > 0:
        zero_inds = zeros[np.random.randint(len(zeros), size=5)]

    if len(obs.shape) == 5:
        obs = obs[:, 0]

    fig, axs = plt.subplots(2, 5)

    for i in range(5):
        if len(nonzeros) > 0:
            im = obs[nonzero_inds[i]].squeeze().transpose((1, 2, 0))
        else:
            im = np.zeros((64, 64, 3))
        axs[0][i].imshow(im)
        axs[0][i].set_title('In')
        axs[0][i].set_axis_off()

        if len(zeros) > 0:
            im = obs[zero_inds[i]].squeeze().transpose((1, 2, 0))
        else:
            im = np.zeros((64, 64, 3))
        axs[1][i].imshow(im)
        axs[1][i].set_title('Out')
        axs[1][i].set_axis_off()

    plt.savefig(file)
    plt.close()


def visualize_dynamics(obs_seqs, act_seqs, dynamics_func, encoder, file):
    """

    :param obs_seqs: Sequence of observations, (n, time, *d_obs)
    :param act_seqs: Sequence of actions, (n, time, d_act)
    :param dynamics_func:
    :param file:
    """
    ims = []

    for obs_seq, act_seq in list(zip(obs_seqs, act_seqs)):
        act_seq = act_seq[None]
        obs_0 = ptu.torchify(obs_seq.squeeze()[0])

        act_seq = ptu.torchify(act_seq)
        predictions = dynamics_func.predict(obs_0, act_seq, already_embedded=True)
        predictions_decoded = encoder.decode(predictions).detach().cpu().numpy()
        predictions_decoded = predictions_decoded[0].squeeze()

        predictions_model_mean = predictions.mean(dim=0)
        predictions_mm_decoded = encoder.decode(predictions_model_mean).detach().cpu().numpy().squeeze()

        obs_seq_recoded = encoder.decode(ptu.torchify(obs_seq)).detach().cpu().numpy()

        obs_seq = obs_seq[1:]
        obs_seq_recoded = obs_seq_recoded[1:]
        predictions_decoded = predictions_decoded[:-1]
        predictions_mm_decoded = predictions_mm_decoded[:-1]

        for rec, pred1, pred2 in list(zip(obs_seq_recoded.squeeze(),
                                          predictions_decoded, predictions_mm_decoded)):
            if len(rec.shape) == 4: # frame stacking
                rec = rec[0]
                pred1 = pred1[0]
                pred2 = pred2[0]
            ims.append(np.concatenate((rec, pred1, pred2), axis=2))

    make_movie(ims, file)


def visualize_plan(obs, act_seq, dynamics, file):
    encoder = dynamics.encoder

    act_seq = act_seq[None]
    obs = ptu.torchify(obs.squeeze())

    act_seq = ptu.torchify(act_seq)
    predictions = dynamics.predict(obs, act_seq, already_embedded=True)
    predictions_model_mean = predictions.mean(dim=0)
    predictions_decoded = encoder.decode(predictions_model_mean).detach().cpu().numpy().squeeze()

    ims = []
    for pred in list(predictions_decoded):
        if len(pred.shape) == 4:  # frame stacking
            pred = pred[0]
        ims.append(pred)
    make_movie(ims, file)

