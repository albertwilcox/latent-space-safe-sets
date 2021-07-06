from latentsafesets.rl_trainers import VAETrainer, SafeSetTrainer, Trainer, \
    ValueTrainer, ConstraintTrainer, GoalIndicatorTrainer, PETSDynamicsTrainer

from latentsafesets.utils import LossPlotter, EncoderDataLoader

import os


class MPCTrainer(Trainer):

    def __init__(self, env, params, modules):

        self.params = params
        self.env = env

        self.logdir = params['logdir']

        loss_plotter = LossPlotter(os.path.join(params['logdir'], 'loss_plots'))
        self.encoder_data_loader = EncoderDataLoader(params)

        self.trainers = []

        self.trainers.append(VAETrainer(params, modules['enc'], loss_plotter))
        self.trainers.append(PETSDynamicsTrainer(params, modules['dyn'], loss_plotter))
        self.trainers.append(ValueTrainer(env, params, modules['val'], loss_plotter))
        self.trainers.append(SafeSetTrainer(env, params, modules['ss'], loss_plotter))
        self.trainers.append(ConstraintTrainer(env, params, modules['constr'], loss_plotter))
        self.trainers.append(GoalIndicatorTrainer(env, params, modules['gi'], loss_plotter))

    def initial_train(self, replay_buffer):
        update_dir = os.path.join(self.logdir, 'initial_train')
        os.makedirs(update_dir, exist_ok=True)
        for trainer in self.trainers:
            if type(trainer) == VAETrainer:
                trainer.initial_train(self.encoder_data_loader, update_dir)
            else:
                trainer.initial_train(replay_buffer, update_dir)

    def update(self, replay_buffer, update_num):
        update_dir = os.path.join(self.logdir, 'update_%d' % update_num)
        os.makedirs(update_dir, exist_ok=True)
        for trainer in self.trainers:
            trainer.update(replay_buffer, update_dir)
