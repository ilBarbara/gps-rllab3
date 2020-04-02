""" Hyperparameters for MJC peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
# from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.agent.rllab3.agent_rllab3Hopper import AgentRllab3Hopper
from gps.algorithm.algorithm_traj_opt_Hopper import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum_Hopper import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    END_EFFECTOR_POINTS: 5,
    JOINT_VELOCITIES: 6,
    JOINT_ANGLES: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 3,
}

PR2_GAINS = np.array([1.0, 1.0, 1.0])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/experiments/rllab3_hopper_traj_opt/'


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentRllab3Hopper,
    'filename': './mjc_models/hopper.xml',
    'x0': np.concatenate([np.array([0.1, 0.1, -1.54]),
                          np.zeros(3)]),
    'dt': 0.001,
    'substeps': 1,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [np.array([0, 0, 0])],
    #'pos_body_offset': [[np.array([0, 0.2, 0])], [np.array([0, 0.1, 0])],
    #                    [np.array([0, -0.1, 0])], [np.array([0, -0.2, 0])]],
    'T': 200,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 1000,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS,
}

fk_cost = {
    'type': CostFK,
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'wp': np.array([1, 1, 1, 1, 1, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost],
    'weights': [1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 20,
    'verbose_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
