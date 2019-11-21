""" Default configuration for policy optimization. """
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
try:
    #from gps.algorithm.policy_opt.policy_opt_utils import construct_fc_network
    from gps.algorithm.policy_opt.tf_model_example import tf_network, multi_modal_network
except ImportError:
    #construct_fc_network = None
    tf_network = None

import os

# config options shared by both caffe and tf.
GENERIC_CONFIG = {
    # Initialization.
    'init_var': 0.1,  # Initial policy variance.
    'ent_reg': 0.0,  # Entropy regularizer.
    # Solver hyperparameters.
    'iterations': 5000,  # Number of iterations per inner iteration.
    'batch_size': 25,
    'lr': 0.001,  # Base learning rate (by default it's fixed).
    'lr_policy': 'fixed',  # Learning rate policy.
    'momentum': 0.9,  # Momentum.
    'weight_decay': 0.005,  # Weight decay.
    'solver_type': 'Adam',  # Solver type (e.g. 'SGD', 'Adam', etc.).
    # set gpu usage.
    'use_gpu': 1,  # Whether or not to use the GPU for caffe training.
    'gpu_id': 0,
    'random_seed': 1,
}

'''
POLICY_OPT_CAFFE = {
    # Other hyperparameters.
    'network_model': construct_fc_network,  # Either a filename string
                                            # or a function to call to
                                            # create NetParameter.
    'network_arch_params': {},  # Arguments to pass to method above.
    'weights_file_prefix': '',
}

POLICY_OPT_CAFFE.update(GENERIC_CONFIG)
'''

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 7,
}

POLICY_OPT_TF = {
    # Other hyperparameters.
    'network_model': tf_network,
    'network_params': {
        'sensor_dims': SENSOR_DIMS,
        'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        
    },  
    
    'copy_param_scope': 'conv_params',
    'fc_only_iterations': 0,
}

POLICY_OPT_TF.update(GENERIC_CONFIG)
