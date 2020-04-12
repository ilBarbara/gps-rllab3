""" This file provides an example tensorflow network used to define a policy. """
import tensorflow as tf
from gps.algorithm.policy_opt.tf_utils import TfMap_mfmb
from gps.algorithm.policy_opt.tf_model_example import get_loss_layer
import numpy as np

def init_weights(shape, name=None):
    return tf.get_variable(name, initializer=tf.random_normal(shape, stddev=0.01))

def init_bias(shape, name=None):
    return tf.get_variable(name, initializer=tf.zeros(shape, dtype='float'))

def get_input_layer(dim_input, dim_output):
    """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, action that the policy takes at this state.
        cost: accumulate cost in the future."""
    net_input = tf.placeholder("float", [None, dim_input], name='nn_input')
    action = tf.placeholder('float', [None, dim_output], name='action')
    cost = tf.placeholder('float', [None, ], name='cost')
    precision = tf.placeholder('float', [None, dim_output, dim_output], name='precision')
    return net_input, action, cost, precision

def get_param_layers(param_input, dim_output, name=""):
    '''
    input: None*dim_input
    output: None*dim_output
    '''
    cur_bias = init_bias([1, dim_output], name= name + 'std_const')
    tile_shape = tf.concat([tf.shape(param_input)[:-1], [1,]], 0)
    cur_top = tf.tile(cur_bias, tile_shape)
    return cur_top, [], [cur_bias], cur_bias

def get_mlp_layers(mlp_input, number_layers, dimension_hidden, name=""):
    """compute MLP with specified number of layers.
        math: sigma(Wx + b)
        for each layer, where sigma is by default relu"""
    cur_top = mlp_input
    weights = []
    biases = []
    for layer_step in range(0, number_layers):
        in_shape = cur_top.get_shape().dims[1].value
        cur_weight = init_weights([in_shape, dimension_hidden[layer_step]], name= name + 'w_' + str(layer_step))
        cur_bias = init_bias([dimension_hidden[layer_step]], name= name + 'b_' + str(layer_step))
        weights.append(cur_weight)
        biases.append(cur_bias)
        if layer_step != number_layers-1:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
        else:
            cur_top = tf.matmul(cur_top, cur_weight) + cur_bias

    return cur_top, weights, biases

def get_mf_loss(mean, log_std, action, cost, dim):
    """
    The loss layer used for the MLP network is obtained through this class.
    cost * log pi(a|s) = cost * (-log(sigma) - (a-mu)^2/2sigma^2)
    """
    std = tf.math.exp(log_std)
    log_sigma = tf.reduce_sum(log_std, 1)#[batch,]
    diff = tf.square(tf.divide(action-mean, std))
    diff = -0.5*tf.reduce_sum(diff, 1)#[batch,]
    return tf.reduce_mean(cost*(diff-log_sigma-0.5*np.log(2*np.pi)*dim))

def mfmb_network(dim_input=27, dim_output=7, batch_size=25, network_config=None):
    """
    Specifying a fully-connected network in TensorFlow.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
        network_config: dictionary of network structure parameters
    Returns:
        a TfMap object used to serialize, inputs, outputs, and loss.
    """
    n_layers = 3 if 'n_layers' not in network_config else network_config['n_layers'] + 1
    dim_hidden = (n_layers - 1) * [40] if 'dim_hidden' not in network_config else network_config['dim_hidden']
    dim_hidden.append(dim_output)#last layer outputs mean and (diag) precision

    nn_input, action, cost, precision = get_input_layer(dim_input, dim_output)
    mean, weights_FC1, biases_FC1 = get_mlp_layers(nn_input, n_layers, dim_hidden, name="mean")
    # if adaptive_std:
    #     log_std, weights_FC2, biases_FC2 = get_mlp_layers(nn_input, n_layers, dim_hidden, name="std")
    #     log_std = tf.nn.leaky_relu(log_std + 5., 0.2) - 5.  
    # else:
    #     log_std, weights_FC2, biases_FC2, log_std_weight = get_param_layers(nn_input, dim_output, name="param_std")      
    #     log_std = tf.nn.leaky_relu(log_std + 5., 0.2) - 5. 
    log_std, weights_FC2, biases_FC2, log_std_weight = get_param_layers(nn_input, dim_output, name="param_std")      
    log_std = tf.nn.leaky_relu(log_std + 5., 0.2) - 5.       
    fc_vars = weights_FC1 + weights_FC2 + biases_FC1 + biases_FC2
    loss_out_mf = get_mf_loss(mean=mean, log_std=log_std, action=action, cost=cost, dim=dim_output)
    loss_out_imit = get_loss_layer(mlp_out=mean, action=action, precision=precision, batch_size=batch_size)

    return TfMap_mfmb(nn_input, action, cost, precision, \
        mean, log_std, loss_out_mf, loss_out_imit, log_std_weight,\
        policy_type="mfmb"), fc_vars, []

