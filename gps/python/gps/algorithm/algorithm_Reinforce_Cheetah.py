""" This file defines the MD-based GPS algorithm. """
import copy
import logging

import numpy as np
import scipy as sp

from gps.algorithm.algorithm_Cheetah import Algorithm
from gps.algorithm.algorithm_utils import PolicyInfo
from gps.algorithm.config import ALG_MF
from gps.sample.sample_list import SampleList
from gps.algorithm.baseline.LinearFeatureBaseline import LinearFeatureBaseline

LOGGER = logging.getLogger(__name__)


class AlgorithmReinforce(Algorithm):
    """
    Sample-based joint policy learning and trajectory optimization with
    (approximate) mirror descent guided policy search algorithm.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG_MF)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        self.policy_opt = self._hyperparams['policy_opt']['type'](
            self._hyperparams['policy_opt'], self.dO, self.dU
        )

        self.baseline = LinearFeatureBaseline()

    def iteration(self, sample_lists):
        """
        Run iteration of REINFORCE.

        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Store the samples and evaluate the costs.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)

        # S-step
        self._update_policy()

    def _update_policy(self):
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
        self.policy_opt.update(np.reshape(traj_X, (-1, dO)), traj_U, traj_A)
