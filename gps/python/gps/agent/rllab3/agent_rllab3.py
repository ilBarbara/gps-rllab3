from __future__ import print_function

import sys
print (sys.path)
#sys.path.remove('/home2/wsdm/gyy/sjh_project/rllab')
sys.path.append('../../env/rllab')
sys.path.append('../../dnc')
print (sys.path)

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
import theano
import theano.tensor as TT
from lasagne.updates import adam

import time
import pygame
from rllab.misc.resolve import load_class

# Environment Imports
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
import dnc.envs as dnc_envs


""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

#import mjcpy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_MUJOCO
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, IMAGE_FEAT, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, NOISE

from gps.sample.sample import Sample


class AgentRllab3Ant(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_MUJOCO)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._setup_conditions()
        self._setup_world(hyperparams['filename'])

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        self._world = []
        self._model = []

        # Initialize Mujoco worlds. If there's only one xml file, create a single world object,
        # otherwise create a different world for each condition.
        for i in range(self._hyperparams['conditions']):
            self._world.append(TfEnv(normalize(dnc_envs.create_deterministic('ant'))))
        # Initialize x0.
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            self.x0.append(self._world[i].reset())
        
    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        # Create new sample, populate first time step.
        feature_fn = None
        if 'get_features' in dir(policy):
            feature_fn = policy.get_features
        new_sample = self._init_sample(condition, feature_fn=feature_fn)
        mj_X = self._world[condition].reset()     #initial state in mj_world, condition-specific
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                self._model[condition]['body_pos'][idx, :] += \
                        var * np.random.randn(1, 3)

        timestep = 0.05
        speedup = 1
        # Take the sample.
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)       #get state from _data in sample class
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = mj_U
            
            if (t + 1) < self.T:
                mj_X, reward, terminal, _ = self._world[condition].step(mj_U)

                if verbose:
                    self._world[condition].render()
                    time.sleep(timestep / speedup)

                # import time as ttime      
                #self._data = self._world[condition].get_data()     #get data from mj_world
                self._set_sample(new_sample, mj_X, reward, t, condition, feature_fn=feature_fn)
        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init(self, condition):
        """
        Set the world to a given model, and run kinematics.
        Args:
            condition: Which condition to initialize.
        """

        # Initialize world/run kinematics
        self._world[condition].reset()
        x0 = self._hyperparams['x0'][condition]
        idx = len(x0) // 2
        data = {'qpos': x0[:idx], 'qvel': x0[idx:]}
        #self._world[condition].set_data(data)
        #self._world[condition].kinematics()

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
            feature_fn: funciton to comptue image features from the observation.
        """
        sample = Sample(self)

        # Initialize world/run kinematics
        #self._init(condition)

        # Initialize sample with stuff from _data
        data = self._world[condition].reset()          #get data from mj_world, condition-specific
        sample.set(JOINT_ANGLES, data[0:8], t=0)    #Set _data in sample class
        sample.set(JOINT_VELOCITIES, data[8:16], t=0)
        sample.set(END_EFFECTOR_POINTS, data[16:82], t=0)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, data[82:148], t=0)
        #sample.set(END_EFFECTOR_POINT_JACOBIANS, np.array(0.0), t=0)

        return sample

    def _set_sample(self, sample, mj_X, reward, t, condition, feature_fn=None):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
            feature_fn: function to compute image features from the observation.
        """
        sample.set(JOINT_ANGLES, mj_X[0:8], t=t+1)   #Set _data in sample class
        sample.set(JOINT_VELOCITIES, mj_X[8:16], t=t+1)
        sample.set(END_EFFECTOR_POINTS, mj_X[16:82], t=t+1)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, mj_X[82:148], t=t+1)
        #sample.set(END_EFFECTOR_POINT_JACOBIANS, np.array(reward), t=t+1)

    def _get_image_from_obs(self, obs):
        imstart = 0
        imend = 0
        image_channels = self._hyperparams['image_channels']
        image_width = self._hyperparams['image_width']
        image_height = self._hyperparams['image_height']
        for sensor in self._hyperparams['obs_include']:
            # Assumes only one of RGB_IMAGE or CONTEXT_IMAGE is present
            if sensor == RGB_IMAGE or sensor == CONTEXT_IMAGE:
                imend = imstart + self._hyperparams['sensor_dims'][sensor]
                break
            else:
                imstart += self._hyperparams['sensor_dims'][sensor]
        img = obs[imstart:imend]
        img = img.reshape((image_width, image_height, image_channels))
        return img




class AgentRllab3(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_MUJOCO)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._setup_conditions()
        self._setup_world(hyperparams['filename'])

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        self._world = []
        self._model = []

        # Initialize Mujoco worlds. If there's only one xml file, create a single world object,
        # otherwise create a different world for each condition.
        for i in range(self._hyperparams['conditions']):
            self._world.append(TfEnv(normalize(dnc_envs.create_stochastic('pick'))))
        # Initialize x0.
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            self.x0.append(self._world[i].reset())
        
    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        # Create new sample, populate first time step.
        feature_fn = None
        if 'get_features' in dir(policy):
            feature_fn = policy.get_features
        new_sample = self._init_sample(condition, feature_fn=feature_fn)
        mj_X = self._world[condition].reset()     #initial state in mj_world, condition-specific
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                self._model[condition]['body_pos'][idx, :] += \
                        var * np.random.randn(1, 3)
        # Take the sample.
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)       #get state from _data in sample class
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = mj_U
            
            if (t + 1) < self.T:
                mj_X, reward, terminal, _ = self._world[condition].step(mj_U)
                
                #self._data = self._world[condition].get_data()     #get data from mj_world
                self._set_sample(new_sample, mj_X, reward, t, condition, feature_fn=feature_fn)
        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init(self, condition):
        """
        Set the world to a given model, and run kinematics.
        Args:
            condition: Which condition to initialize.
        """

        # Initialize world/run kinematics
        self._world[condition].reset()
        x0 = self._hyperparams['x0'][condition]
        idx = len(x0) // 2
        data = {'qpos': x0[:idx], 'qvel': x0[idx:]}
        #self._world[condition].set_data(data)
        #self._world[condition].kinematics()

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
            feature_fn: funciton to comptue image features from the observation.
        """
        sample = Sample(self)

        # Initialize world/run kinematics
        #self._init(condition)

        # Initialize sample with stuff from _data
        data = self._world[condition].reset()          #get data from mj_world, condition-specific
        sample.set(JOINT_ANGLES, data[0:7], t=0)    #Set _data in sample class
        sample.set(JOINT_VELOCITIES, data[7:14], t=0)
        sample.set(END_EFFECTOR_POINTS, data[14:24], t=0)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, data[24:34], t=0)
        #sample.set(END_EFFECTOR_POINT_JACOBIANS, np.array(0.0), t=0)

        return sample

    def _set_sample(self, sample, mj_X, reward, t, condition, feature_fn=None):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
            feature_fn: function to compute image features from the observation.
        """
        sample.set(JOINT_ANGLES, mj_X[0:7], t=t+1)   #Set _data in sample class
        sample.set(JOINT_VELOCITIES, mj_X[7:14], t=t+1)
        sample.set(END_EFFECTOR_POINTS, mj_X[14:24], t=t+1)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, mj_X[24:34], t=t+1)
        #sample.set(END_EFFECTOR_POINT_JACOBIANS, np.array(reward), t=t+1)

    def _get_image_from_obs(self, obs):
        imstart = 0
        imend = 0
        image_channels = self._hyperparams['image_channels']
        image_width = self._hyperparams['image_width']
        image_height = self._hyperparams['image_height']
        for sensor in self._hyperparams['obs_include']:
            # Assumes only one of RGB_IMAGE or CONTEXT_IMAGE is present
            if sensor == RGB_IMAGE or sensor == CONTEXT_IMAGE:
                imend = imstart + self._hyperparams['sensor_dims'][sensor]
                break
            else:
                imstart += self._hyperparams['sensor_dims'][sensor]
        img = obs[imstart:imend]
        img = img.reshape((image_width, image_height, image_channels))
        return img
