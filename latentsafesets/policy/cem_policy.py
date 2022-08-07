"""
Code inspired by https://github.com/kchua/handful-of-trials/blob/master/dmbrl/controllers/MPC.py
                 https://github.com/quanvuong/handful-of-trials-pytorch/blob/master/MPC.py
                 https://github.com/kchua/handful-of-trials/blob/master/dmbrl/misc/optimizers/cem.py
"""

from .policy import Policy

import latentsafesets.utils.pytorch_utils as ptu
import latentsafesets.utils.spb_utils as spbu
from latentsafesets.modules import VanillaVAE, PETSDynamics, ValueFunction, ConstraintEstimator, \
    GoalIndicator, CBFdotEstimator#they are all there

import torch
import numpy as np
import gym

import logging

log = logging.getLogger('cem')


class CEMSafeSetPolicy(Policy):
    def __init__(self, env: gym.Env,
                 encoder: VanillaVAE,
                 safe_set,
                 value_function: ValueFunction,
                 dynamics_model: PETSDynamics,
                 constraint_function: ConstraintEstimator,
                 goal_indicator: GoalIndicator,
                 cbfdot_function: CBFdotEstimator,
                 params):
        log.info("setting up safe set and dynamics model")

        self.env = env
        self.encoder = encoder
        self.safe_set = safe_set#safe set estimator
        self.dynamics_model = dynamics_model
        self.value_function = value_function
        self.constraint_function = constraint_function
        self.goal_indicator = goal_indicator
        self.cbfdot_function = cbfdot_function
        self.logdir = params['logdir']

        self.d_act = params['d_act']#2
        self.d_obs = params['d_obs']#dimension of observation (3,64,64)
        self.d_latent = params['d_latent']#32
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.plan_hor = params['plan_hor']#H=5
        self.random_percent = params['random_percent']#1 in spb
        self.popsize = params['num_candidates']#1000
        self.num_elites = params['num_elites']#100
        self.max_iters = params['max_iters']#5
        self.safe_set_thresh = params['safe_set_thresh']#0.8
        self.safe_set_thresh_mult = params['safe_set_thresh_mult']#0.8
        self.safe_set_thresh_mult_iters = params['safe_set_thresh_mult_iters']#5
        self.constraint_thresh = params['constr_thresh']#0.2
        self.goal_thresh = params['gi_thresh']#0.5
        self.ignore_safe_set = params['safe_set_ignore']#False, for ablation study!#changed to true after using cbf dot
        self.ignore_constraints = params['constr_ignore']#false
        self.ignore_cbfdots = params['cbfd_ignore']  # false

        self.mean = torch.zeros(self.d_act)#the dimension of action
        self.std = torch.ones(self.d_act)
        self.ac_buf = np.array([]).reshape(0, self.d_act)#action buffer?
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

    @torch.no_grad()
    def act(self, obs):#if using cbf, see the function actcbfd later on
        """
        Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation. Cannot be a batch

        Returns: An action (and possibly the predicted cost)
        """

        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!

        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initally 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where num_constraint_satisfying=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        return self.env.action_space.sample()#really random action!

                    itr = 0#let's start over with itr=0 in this case!#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue

                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]#get those elite trajectories
                #print('elites.shape',elites.shape)#once it is torch.Size([1, 5, 2]), it's gone!
                #print('elites',elites)

                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)
                # print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!
                #print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:#this means that there is only one trajectory
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)#
                    #print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.0 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?

                action_samples = self._sample_actions_normal(self.mean, self.std)
                #print('action_samples', action_samples)#it becomes nan!

            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                #print('emb.shape',emb.shape)# torch.Size([1, 32])
                #print('action_samples.shape',action_samples.shape)#torch.Size([1000, 5, 2])
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials

                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20

                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#call forward#each in the model
                    #print(constraints_all.shape)#torch.Size([20, 1000, 5, 1])
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] > self.constraint_thresh, dim=1)#those that violate the constraints
                    #tempe=torch.max(constraints_all, dim=0)#a tuple of (value, indices)
                    #print('tempe',tempe.shape)  # val's shape is torch.Size([1000, 5, 1])
                    #print((torch.max(constraints_all, dim=0)[0]).shape)#torch.Size([1000, 5, 1])
                    #print((torch.max(constraints_all, dim=0)[0] > self.constraint_thresh).shape)#torch.Size([1000, 5, 1])
                    #print('constraint_viols.shape',constraint_viols.shape)#sum will consume the 1st dim (5 in this case)#1000 0,1,2,3,4,5s
                else:
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!

                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction
                    safe_set_viols = torch.mean(safe_set_all#not max this time
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#f_G in the paper(1000,1)
                #maybe the self.goal_thresh is a bug source?
                values = values + (constraint_viols + safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()

            itr += 1#CEM Evolution method

        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy()

    @torch.no_grad()
    def actcbfd(self, obs,state):#some intermediate step that the cbf dot part still requires states rather than latent states
        """
        Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation. Cannot be a batch

        Returns: An action (and possibly the predicted cost)
        """

        # encode observation:
        obs = ptu.torchify(obs).reshape(1, *self.d_obs)#just some data processing
        emb = self.encoder.encode(obs)#in latent space now!

        itr = 0#
        reset_count = 0#
        act_ss_thresh = self.safe_set_thresh#initially 0.8
        act_cbfd_thresh=self.safe_set_thresh#initially 0.8
        while itr < self.max_iters:#5
            if itr == 0:
                # Action samples dim (num_candidates, planning_hor, d_act)
                if self.mean is None:#right after reset
                    action_samples = self._sample_actions_random()#1000*5 2d array
                else:
                    num_random = int(self.random_percent * self.popsize)#sample 1000 trajectories
                    num_dist = self.popsize - num_random#=0 when random_percent=1
                    action_samples_dist = self._sample_actions_normal(self.mean, self.std, n=num_dist)
                    action_samples_random = self._sample_actions_random(num_random)#uniformly random from last iter ation
                    action_samples = torch.cat((action_samples_dist, action_samples_random), dim=0)
            else:
                # Chop off the numer of elites so we don't use constraint violating trajectories
                num_constraint_satisfying = sum(values > -1e5)#no any constraints violation
                #print(num_constraint_satisfying)
                iter_num_elites = min(num_constraint_satisfying, self.num_elites)#max(2,min(num_constraint_satisfying, self.num_elites))#what about doing max(2) to it?
                #what if I change this into num_constraint_satisfying+2?
                if num_constraint_satisfying == 0:#it is definitely a bug not to include the case where ncs=1!
                    reset_count += 1
                    act_ss_thresh *= self.safe_set_thresh_mult#*0.8 by default
                    act_cbfd_thresh *= self.safe_set_thresh_mult  # *0.8 by default
                    if reset_count > self.safe_set_thresh_mult_iters:
                        self.mean = None
                        return self.env.action_space.sample()#really random action!

                    itr = 0#that is why it always stops at iteration 0 when error occurs!
                    self.mean, self.std = None, None
                    continue

                # Sort
                sortid = values.argsort()#if it goes to this step, the num_constraint_satisfying should >=1
                actions_sorted = action_samples[sortid]
                elites = actions_sorted[-iter_num_elites:]
                #print('elites.shape',elites.shape)#once it is torch.Size([1, 5, 2]), it's gone!
                #print('elites',elites)

                # Refitting to Best Trajs
                self.mean, self.std = elites.mean(0), elites.std(0)
                # print('self.mean',self.mean,'self.std',self.std)#it's self.std that got nan!
                #print(self.std[0,0])
                #import ipdb#it seems that they are lucky to run into the following case
                if torch.isnan(self.std[0,0]):#self.std[0,0]==torch.nan:
                    #ipdb.set_trace()
                    print('elites.shape',elites.shape)#
                    #print('nan',self.std[0,0])
                    #self.std=0.5*torch.rand_like(self.mean)+0.1#1e-2#is it just a work around?
                    self.std = 0.8 * torch.ones_like(self.mean)#0.0 * torch.ones_like(self.mean)##1.0 * torch.ones_like(self.mean)# 1e-2#is it just a work around?
                    #0.8 is the hyperparameter I choose which I think may have good performance
                action_samples = self._sample_actions_normal(self.mean, self.std)#(1000,5,2)
                #print('action_samples', action_samples)#it becomes nan!

            if itr < self.max_iters - 1:#why the ensemble param in dynamics is 5! For MPC!
                # dimension (num_models, num_candidates, planning_hor, d_latent)
                #print('emb.shape',emb.shape)# torch.Size([1, 32])
                #print('action_samples.shape',action_samples.shape)#torch.Size([1000, 5, 2])
                predictions = self.dynamics_model.predict(emb, action_samples, already_embedded=True)
                num_models, num_candidates, planning_hor, d_latent = predictions.shape#the possible H sequence of all candidates' all trials

                last_states = predictions[:, :, -1, :].reshape(
                    (num_models * num_candidates, d_latent))#the last state under the action sequence#the 20000*32 comes out!
                all_values = self.value_function.get_value(last_states, already_embedded=True)#all values from 1000 candidates*20 particles
                nans = torch.isnan(all_values)
                all_values[nans] = -1e5
                values = torch.mean(all_values.reshape((num_models, num_candidates, 1)), dim=0)#reduce to (1000,1), take the mean of 20

                #print(state.shape)#(2,)
                #print(state.dtype)#float64
                storch=ptu.torchify(state)#state torch
                #print(action_samples.shape)#torch.Size([1000, 5, 2])
                #print(action_samples.dtype)#torch.float32
                se=storch+action_samples#se means state estimated#shape(1000,5,2)
                #se1=stateevolve

                walls=[((75,55),(100,95))]#
                #I devide the map into 8 regions clockwise: left up, middle up, right up, right middle, right down, middle down, left down, left middle
                rd1h = torch.where((se[:, :, 0]<=walls[0][0][0])*(se[:, :, 1]<=walls[0][0][1]), se[:, :, 0]-walls[0][0][0], se[:, :, 0])
                #Thus, rd1h means relative distance region 1 horizontal, where region 1 means left up of the centeral obstacle
                rd1v = torch.where((se[:, :, 0] <= walls[0][0][0]) * (se[:, :, 1] <= walls[0][0][1]),se[:, :, 1] - walls[0][0][1], se[:, :, 1])
                #and consequently, rd1v means relative distance region 1 vertical, which gets the relative distance in the vertical directions
                rd1=torch.concat((rd1h.reshape(rd1h.shape[0],rd1h.shape[1],1),rd1v.reshape(rd1v.shape[0],rd1v.shape[1],1)),dim=2)
                #we concatenate them to recover the 2-dimensional coordinates
                rd2h = torch.where((rd1[:, :, 0] > walls[0][0][0])*(rd1[:, :, 0] <= walls[0][1][0]) * (rd1[:, :, 1] <= walls[0][0][1]),
                                   0*rd1[:, :, 0] , rd1[:, :, 0])#region 2 is the middle up of the centeral obstacle
                rd2v = torch.where((rd1[:, :, 0] > walls[0][0][0])*(rd1[:, :, 0] <= walls[0][1][0]) * (rd1[:, :, 1] <= walls[0][0][1]),
                                   rd1[:, :, 1] - walls[0][0][1], rd1[:, :, 1])
                rd2 = torch.concat(
                    (rd2h.reshape(rd2h.shape[0], rd2h.shape[1], 1), rd2v.reshape(rd2v.shape[0], rd2v.shape[1], 1)),
                    dim=2)
                rd3condition=(rd2[:, :, 0]>walls[0][1][0])*(rd2[:, :, 1]<=walls[0][0][1])#this condition is to see if it is in region 3
                rd3h=torch.where(rd3condition,rd2[:, :, 0]-walls[0][1][0], rd2[:, :, 0])#h means horizontal
                rd3v = torch.where(rd3condition, rd2[:, :, 1] - walls[0][0][1], rd2[:, :, 1])#v means vertical
                rd3 = torch.concat(
                    (rd3h.reshape(rd3h.shape[0], rd3h.shape[1], 1), rd3v.reshape(rd3v.shape[0], rd3v.shape[1], 1)),
                    dim=2)
                rd4condition = (rd3[:, :, 0] > walls[0][1][0]) * (rd3[:, :, 1] > walls[0][0][1])*(rd3[:, :, 1] <= walls[0][1][1])
                rd4h = torch.where(rd4condition, rd3[:, :, 0] - walls[0][1][0], rd3[:, :, 0])  # h means horizontal
                rd4v = torch.where(rd4condition, 0*rd3[:, :, 1], rd3[:, :, 1])  # v means vertical
                rd4 = torch.concat(
                    (rd4h.reshape(rd4h.shape[0], rd4h.shape[1], 1), rd4v.reshape(rd4v.shape[0], rd4v.shape[1], 1)),
                    dim=2)
                rd5condition = (rd4[:, :, 0] > walls[0][1][0]) * (rd4[:, :, 1] > walls[0][1][1])
                rd5h = torch.where(rd5condition, rd4[:, :, 0] - walls[0][1][0], rd4[:, :, 0])  # h means horizontal
                rd5v = torch.where(rd5condition, rd4[:, :, 1]- walls[0][1][1], rd4[:, :, 1])  # v means vertical
                rd5 = torch.concat(
                    (rd5h.reshape(rd5h.shape[0], rd5h.shape[1], 1), rd5v.reshape(rd5v.shape[0], rd5v.shape[1], 1)),
                    dim=2)
                rd6condition = (rd5[:, :, 0] <= walls[0][1][0]) *(rd5[:, :, 0] > walls[0][0][0]) * (rd5[:, :, 1] > walls[0][1][1])
                rd6h = torch.where(rd6condition, 0*rd5[:, :, 0], rd5[:, :, 0])  # h means horizontal
                rd6v = torch.where(rd6condition, rd5[:, :, 1] - walls[0][1][1], rd5[:, :, 1])  # v means vertical
                rd6 = torch.concat(
                    (rd6h.reshape(rd6h.shape[0], rd6h.shape[1], 1), rd6v.reshape(rd6v.shape[0], rd6v.shape[1], 1)),
                    dim=2)
                rd7condition = (rd6[:, :, 0] <= walls[0][0][0]) * (rd6[:, :, 1] > walls[0][1][1])
                rd7h = torch.where(rd7condition, rd6[:, :, 0]- walls[0][0][0], rd6[:, :, 0])  # h means horizontal
                rd7v = torch.where(rd7condition, rd6[:, :, 1] - walls[0][1][1], rd6[:, :, 1])  # v means vertical
                rd7 = torch.concat(
                    (rd7h.reshape(rd7h.shape[0], rd7h.shape[1], 1), rd7v.reshape(rd7v.shape[0], rd7v.shape[1], 1)),
                    dim=2)
                rd8condition = (rd7[:, :, 0] <= walls[0][0][0]) *(rd7[:, :, 1] <= walls[0][1][1])* (rd7[:, :, 1] > walls[0][0][1])
                rd8h = torch.where(rd8condition, rd7[:, :, 0] - walls[0][0][0], rd7[:, :, 0])  # h means horizontal
                rd8v = torch.where(rd8condition, 0*rd7[:, :, 1], rd7[:, :, 1])  # v means vertical
                rd8 = torch.concat(
                    (rd8h.reshape(rd8h.shape[0], rd8h.shape[1], 1), rd8v.reshape(rd8v.shape[0], rd8v.shape[1], 1)),
                    dim=2)#dim: (1000,5,2)
                rdn=torch.norm(rd8,dim=2)#rdn for relative distance norm
                #print(rdn.shape)#torch.Size([1000, 5])
                cbf=rdn**2-15**2#13**2#20:30#don't forget the square!# Note that this is also used in the online training afterwards
                acbf=-cbf*act_ss_thresh#acbf means alpha cbf, the minus class k function#0.8 will be replaced later#don't forget the negative sign!
                asrv1=action_samples[:,:,0]#asrv1 means action sample reversed in the 1st dimension (horizontal dimension)!
                asrv2=action_samples[:,:,1]#-action_samples[:,:,1]#asrv2 means action sample reversed in the 2st dimension (vertical dimension)!
                asrv = torch.concat((asrv1.reshape(asrv1.shape[0], asrv1.shape[1], 1), asrv2.reshape(asrv2.shape[0], asrv2.shape[1], 1)),dim=2)  # dim: (1000,5,2)
                rda=torch.concat((rd8,asrv),dim=2)#check if it is correct!#rda: relative distance+action will be thrown later into the cbf dot network


                # Blow up cost for trajectories that are not constraint satisfying and/or don't end up
                #   in the safe set
                if not self.ignore_constraints:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    constraints_all = torch.sigmoid(self.constraint_function(predictions, already_embedded=True))#all the candidates#each in the model
                    constraint_viols = torch.sum(torch.max(constraints_all, dim=0)[0] >= self.constraint_thresh, dim=1)#those that violate the constraints#if constraint_viols>=1, then game over!
                else:#ignore the constraints
                    constraint_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!

                #self.ignore_cbfdots=True#just for 10:57 at Aug 4th
                if not self.ignore_cbfdots:#Do I add the CBF term here?#to see the constraint condition of 1000 trajs
                    cbfdots_all = self.cbfdot_function(rda, already_embedded=True)#all the candidates#torch.sigmoid()#each in the model#(20,1000,5)
                    #print(cbfdots_all.shape)#torch.Size([1000, 5, 1])#
                    cbfdots_all=cbfdots_all.reshape(cbfdots_all.shape[0],cbfdots_all.shape[1])#
                    #print('cbfdots_all', cbfdots_all)
                    cbfdots_viols = torch.sum(cbfdots_all<acbf, dim=1)#those that violate the constraints#1000 0,1,2,3,4,5s#
                    #print('acbf',acbf)#bigger than or equal to is the right thing to do! The violations are <!
                    #print('cbfdots_viols',cbfdots_viols)
                    cbfdots_viols=cbfdots_viols.reshape(cbfdots_viols.shape[0],1)#the threshold now should be predictions dependent
                    #print('cbfdots_viols.shape',cbfdots_viols.shape)
                    #print('cbfdots_viols',cbfdots_viols)
                    #print(cbfdots.shape)
                else:#if ignoring the cbf dot constraints
                    cbfdots_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)#no constraint violators!
                self.ignore_safe_set=True#Including 18:47 Aug 4th as well as 15:14 Aug 5th
                if not self.ignore_safe_set:
                    safe_set_all = self.safe_set.safe_set_probability(last_states, already_embedded=True)#get the prediction for the safety of the last state
                    safe_set_viols = torch.mean(safe_set_all#not max this time, but the mean of the 20 candidates
                                                .reshape((num_models, num_candidates, 1)),#(20,1000,1)
                                                dim=0) < act_ss_thresh#(1000,1)
                else:#ignore safe set constraints
                    safe_set_viols = torch.zeros((num_candidates, 1), device=ptu.TORCH_DEVICE)
                goal_preds = self.goal_indicator(predictions, already_embedded=True)#the prob of being goal at those states#Do I add the CBF term here?(20,1000,5)
                goal_states = torch.sum(torch.mean(goal_preds, dim=0) > self.goal_thresh, dim=1)#sum over planning horizon#f_G in the paper(1000,1)
                #maybe the self.goal_thresh is a bug source?
                values = values + (constraint_viols +cbfdots_viols+ safe_set_viols) * -1e5 + goal_states#equation 2 in paper!
                values = values.squeeze()#all those violators, assign them with big cost of -1e5

            itr += 1#CEM Evolution method

        # Return the best action
        action = actions_sorted[-1][0]#the best one
        return action.detach().cpu().numpy()

    def reset(self):#where it is used?
        # It's important to call this after each episode
        self.mean, self.std = None, None

    def _sample_actions_random(self, n=None):
        if n is None:
            n = self.popsize#1000
        rand = torch.rand((n, self.plan_hor, self.d_act))#(1000,5,2)
        scaled = rand * (self.ac_ub - self.ac_lb)
        action_samples = scaled + self.ac_lb#something random between ac_lb and ac_ub
        return action_samples.to(ptu.TORCH_DEVICE)#size of (1000,5,2)

    def _sample_actions_normal(self, mean, std, n=None):#sample from a normal distribution with mean=mean and std=std
        if n is None:
            n = self.popsize

        smp = torch.empty(n, self.plan_hor, self.d_act).normal_(
            mean=0, std=1).to(ptu.TORCH_DEVICE)
        mean = mean.unsqueeze(0).repeat(n, 1, 1).to(ptu.TORCH_DEVICE)
        std = std.unsqueeze(0).repeat(n, 1, 1).to(ptu.TORCH_DEVICE)

        # Sample new actions
        action_samples = smp * std + mean
        # TODO: Assuming action space is symmetric, true for maze and shelf for now
        action_samples = torch.clamp(
            action_samples,
            min=self.env.action_space.low[0],
            max=self.env.action_space.high[0])

        return action_samples
