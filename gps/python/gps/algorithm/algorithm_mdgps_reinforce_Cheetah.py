""" This file defines the MD-based GPS algorithm. """
import copy
import logging

import numpy as np
import scipy as sp

from gps.algorithm.algorithm_Cheetah import Algorithm
from gps.algorithm.algorithm_mdgps_Cheetah import AlgorithmMDGPS
from gps.algorithm.algorithm_utils import PolicyInfo
from gps.algorithm.config import ALG_MDGPS
from gps.sample.sample_list import SampleList
from gps.algorithm.baseline.LinearFeatureBaseline import LinearFeatureBaseline

LOGGER = logging.getLogger(__name__)


class AlgorithmMDGPS_Reinforce(AlgorithmMDGPS, Algorithm):
    """
    Sample-based joint policy learning and trajectory optimization with
    (approximate) mirror descent guided policy search algorithm, then Reinforce
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_MDGPS)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        policy_prior = self._hyperparams['policy_prior']
        for m in range(self.M):
            self.cur[m].pol_info = PolicyInfo(self._hyperparams)
            self.cur[m].pol_info.policy_prior = \
                    policy_prior['type'](policy_prior)

        self.policy_opt = self._hyperparams['policy_opt']['type'](
            self._hyperparams['policy_opt'], self.dO, self.dU
        )

        self.baseline = LinearFeatureBaseline()

    def iteration(self, sample_lists):
        """
        Run iteration of MDGPS-based guided policy search.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples and evaluate the costs.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)

        # Update dynamics linearizations.
        self._update_dynamics()

        # On the first iteration, need to catch policy up to init_traj_distr.
        if self.iteration_count == 0:
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
            self._update_policy()

        # Update policy linearizations.
        for m in range(self.M):
            self._update_policy_fit(m)

        # C-step
        if self.iteration_count > 0:
            self._stepadjust()
        self._update_trajectories()

        # S-step
        self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()

    def _mf_iteration(self, sample_lists, verbose):
        # Store the samples and evaluate the costs.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m, verbose=verbose)

        # MF-step
        self._update_mf_policy()

    def _update_mf_policy(self):
        """ Compute the new policy. """
        dU, dO, T = self.dU, self.dO, self.T
        gamma  = 1.0
        traj_X = np.zeros((0, T, dO)) 
        traj_U = np.zeros((0, T, dU))
        traj_C = np.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            states = samples.get_X()
            actions = samples.get_U()
            costs = self.cur[m].cs
            accumulate_costs = np.zeros_like(costs)
            accumulate_costs[:,-1] = costs[:,-1]
            for t in range(T-2, -1, -1):
                accumulate_costs[:, t] = gamma*accumulate_costs[:, t+1] + costs[:, t]
            traj_X = np.concatenate((traj_X, states))
            traj_U = np.concatenate((traj_U, actions))
            traj_C = np.concatenate((traj_C, accumulate_costs))
        
        #traj_X = np.reshape(traj_X, (-1, dO))
        traj_U = np.reshape(traj_U, (-1, dU))
        #traj_C = np.reshape(traj_C, (-1, ))
        baselines = np.concatenate([self.baseline.predict(state) for state in traj_X])
        traj_A =  np.reshape(traj_C, (-1, )) - baselines
        self.baseline.fit(traj_X, traj_C)
        self.policy_opt.update_mf(np.reshape(traj_X, (-1, dO)), traj_U, traj_A)

    def _update_policy(self):
        """ Compute the new policy. """
        dU, dO, T = self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        obs_data, tgt_mu = np.zeros((0, T, dO)), np.zeros((0, T, dU))
        tgt_prc, tgt_wt = np.zeros((0, T, dU, dU)), np.zeros((0, T))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            X = samples.get_X()
            N = len(samples)
            traj, pol_info = self.new_traj_distr[m], self.cur[m].pol_info
            mu = np.zeros((N, T, dU))
            prc = np.zeros((N, T, dU, dU))
            wt = np.zeros((N, T))
            # Get time-indexed actions.
            for t in range(T):
                # Compute actions along this trajectory.
                prc[:, t, :, :] = np.tile(traj.inv_pol_covar[t, :, :],
                                          [N, 1, 1])
                for i in range(N):
                    mu[i, t, :] = (traj.K[t, :, :].dot(X[i, t, :]) + traj.k[t, :])
                wt[:, t].fill(pol_info.pol_wt[t])
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.get_obs()))
        self.policy_opt.update_imit(obs_data, tgt_mu, tgt_prc, tgt_wt)
