from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
# from gps.agent.box2d.agent_box2d import AgentBox2D
# from gps.agent.box2d.arm_world import ArmWorld
from gps.agent.rllab3.agent_rllab3Swimmer import AgentRllab3Swimmer
from gps.algorithm.algorithm_badmm_Swimmer import AlgorithmBADMM
# from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum_Swimmer import CostSum
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
# from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.config import generate_experiment_info
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION

SENSOR_DIMS = {
    END_EFFECTOR_POINTS: 5,
    JOINT_VELOCITIES: 5,
    JOINT_ANGLES: 3,
    END_EFFECTOR_POINT_VELOCITIES: 4,
    ACTION: 2,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/experiments/rllab3_swimmer_badmm/'


common = {
    'experiment_name': 'my_badmm_example' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentRllab3Swimmer,
    'filename': './mjc_models/half_cheetah.xml',
    'x0': np.concatenate([np.array([0.1, 0.1]),
                          np.zeros(2)]),
    'dt': 0.01,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [np.array([0, 0, 0])],
    #[[np.array([-0.08, -0.08, 0])], [np.array([-0.08, 0.08, 0])],
    #[np.array([0.08, 0.08, 0])], [np.array([0.08, -0.08, 0])]],
    'T': 200,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
}

algorithm = {
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
    'iterations': 100,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.2,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 0.1,
    'stiffness': 0.01,
    'dt': agent['dt'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.array([1, 1])
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost],
    'weights': [1e-5],
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

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'sensor_dims': SENSOR_DIMS,
    },
    'weights_file_prefix': EXP_DIR + 'policy',
    'iterations': 3000,
    'network_model': tf_network
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 20,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
