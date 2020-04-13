""" This file defines the base algorithm class. """

import abc
import copy
import logging

import random
import numpy as np

from gps.algorithm.config import ALG
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo
from gps.utility.general_utils import extract_condition
from gps.proto.gps_pb2 import  JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES
import rllab.misc.logger as logger
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
import pdb

LOGGER = logging.getLogger(__name__)


class Algorithm(object):
    """ Algorithm superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG)
        config.update(hyperparams)
        self._hyperparams = config

        if 'train_conditions' in hyperparams:
            self._cond_idx = hyperparams['train_conditions']
            self.M = len(self._cond_idx)
        else:
            self.M = hyperparams['conditions']
            self._cond_idx = range(self.M)
            self._hyperparams['train_conditions'] = self._cond_idx
            self._hyperparams['test_conditions'] = self._cond_idx
        self.iteration_count = 0

        # Grab a few values from the agent.
        agent = self._hyperparams['agent']
        self.T = self._hyperparams['T'] = agent.T
        self.dU = self._hyperparams['dU'] = agent.dU
        self.dX = self._hyperparams['dX'] = agent.dX
        self.dO = self._hyperparams['dO'] = agent.dO

        init_traj_distr = config['init_traj_distr']
        init_traj_distr['x0'] = agent.x0
        init_traj_distr['dX'] = agent.dX
        init_traj_distr['dU'] = agent.dU
        del self._hyperparams['agent']  # Don't want to pickle this.

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        if self._hyperparams['fit_dynamics']:
            dynamics = self._hyperparams['dynamics']

        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            if self._hyperparams['fit_dynamics']:
                self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._cond_idx[m]
            )
            # pdb.set_trace()
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        self.traj_opt = hyperparams['traj_opt']['type'](
            hyperparams['traj_opt']
        )
        if type(hyperparams['cost']) == list:
            self.cost = [
                hyperparams['cost'][i]['type'](hyperparams['cost'][i])
                for i in range(self.M)
            ]
        else:
            self.cost = [
                hyperparams['cost']['type'](hyperparams['cost'])
                for _ in range(self.M)
            ]
        self.base_kl_step = self._hyperparams['kl_step']

    @abc.abstractmethod
    def iteration(self, sample_list):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")

    def _update_dynamics(self):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to
        current samples.
        """
        for m in range(self.M):
            cur_data = self.cur[m].sample_list
            X = cur_data.get_X()
            U = cur_data.get_U()

            # Update prior and fit dynamics.
            self.cur[m].traj_info.dynamics.update_prior_delta(cur_data)
            self.cur[m].traj_info.dynamics.fit_delta(X, U)

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = np.diag(
                np.maximum(np.var(x0, axis=0),
                           self._hyperparams['initial_state_var'])
            )

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += \
                        Phi + (N*priorm) / (N+priorm) * \
                        np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
        for cond in range(self.M):
            self.new_traj_distr[cond], self.cur[cond].eta = \
                    self.traj_opt.update(cond, self)

    def _eval_cost(self, cond, verbose=True):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[cond].sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        cvel = np.zeros((N, T))
        for n in range(N):
            sample = self.cur[cond].sample_list[n]
            sample_u = sample.get_U()
            # jp = sample.get(END_EFFECTOR_POINTS) #8 dims
            eepv = sample.get(END_EFFECTOR_POINT_VELOCITIES) #velocity on the x-axis
            forward_reward = eepv[:, 0]
            ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(sample_u), axis = 1)
            l = -forward_reward + ctrl_cost    
            # Get costs.

            cc[n, :] = np.zeros(T)
            cs[n, :] = l
            cvel[n, :] = forward_reward
            
        for n in range(N):
            for t in range(T):
                # for i in range(29, 113):
                #    Cm[n][t][i][i] = 0.5 * 1e-3
                for i in range(23, 29):
                    Cm[n][t][i][i] = 1e-1
                    
                # Cm[n][t][20][22] = -0.5
                # Cm[n][t][22][20] = -0.5
                cv[n][t][20] = -1.0
                
            '''
            l, lx, lu, lxx, luu, lux = self.cost[cond].eval(sample)
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * \
                    np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update
            '''
        

        # Fill in cost estimate.
        self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond].cs = cs  # True value of cost.
        if verbose:
            prefix=''
            undiscounted_returns = [-sum(path) for path in cs]
            logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
            logger.record_tabular('NumTrajs', len(cs))
            logger.record_tabular('StdReturn', np.std(undiscounted_returns))
            logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular('MinReturn', np.min(undiscounted_returns))

            ave_vel = np.array([np.mean(path) for path in cvel])
            min_vel = np.array([np.min(path) for path in cvel])
            max_vel = np.array([np.max(path) for path in cvel])
            std_vel = np.array([np.std(path) for path in cvel])
            logger.record_tabular(prefix+'AverageAverageVelocity', np.mean(ave_vel))
            logger.record_tabular(prefix+'AverageMinVelocity', np.mean(min_vel))
            logger.record_tabular(prefix+'AverageMaxVelocity', np.mean(max_vel))
            logger.record_tabular(prefix+'AverageStdVelocity', np.mean(std_vel))
            

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration
        counter.
        """
        self.iteration_count += 1
        self.prev = copy.deepcopy(self.cur)
        # TODO: change IterationData to reflect new stuff better
        for m in range(self.M):
            self.prev[m].new_traj_distr = self.new_traj_distr[m]
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
        delattr(self, 'new_traj_distr')

    def _set_new_mult(self, predicted_impr, actual_impr, m):
        """
        Adjust step size multiplier according to the predicted versus
        actual improvement.
        """
        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4,
                                               predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(
            min(new_mult * self.cur[m].step_mult,
                self._hyperparams['max_step_mult']),
            self._hyperparams['min_step_mult']
        )
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('Increasing step size multiplier to %f', new_step)
        else:
            LOGGER.debug('Decreasing step size multiplier to %f', new_step)

    def _measure_ent(self, m):
        """ Measure the entropy of the current trajectory. """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :]))
            )
        return ent

    # For pickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_random_state'] = random.getstate()
        state['_np_random_state'] = np.random.get_state()
        return state

    # For unpickling.
    def __setstate__(self, state):
        self.__dict__ = state
        random.setstate(state.pop('_random_state'))
        np.random.set_state(state.pop('_np_random_state'))
