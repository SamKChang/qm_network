import tensorflow as tf
#from qm_layer import *
import qmnet_layer as qnl
import qmnet_tools as qnt

def Fock_error_test(graph, n_basis):
    """
    Directly optimize Fock diagonalization error does not work
    probably related to variational principle on energy
    and probably related to structure of error surface
    """
    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I, N_layer=2)
#        for _ in range(5):
#            C_prd = qnl.SCF_fc_layer(I, C_prd, keep_prob=keep_prob)
#            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=5)
        for _ in range(2):
            C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=1)
        y_prd = qnl.Fock_matrix_occ_error(I, Er, occ, nn, C_prd)
            #y_prd = tf.nn.l2_loss(qnl.Fock_matrix(I, Er, occ, nn, C_prd))
            #y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)

    return y_prd, C_prd, param

def Fock_test(graph, n_basis):

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I, N_layer=2)
        C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=0)
        y_prd = qnl.Fock_matrix_occ_error(I, Er, occ, nn, C_prd)

    return y_prd, C_prd, param

def I2_R5h2b_d_R3h1b_d_R2h1b_O5(graph, n_basis):
    """good for H2 but seems to be bad for A2
       A2 err: 0.0004
       However, for sto-6g with p basis function, 
       strongly overfit is observed. 
       The prediction error is not improving
       even for test data point (with molecule type in the training set)
    """

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I)
#        with tf.variable_scope("rnn1"):
#            C_prd = qnl.SCF_rnn_layer(I, C_prd, n_basis*n_basis*2, 1, 10, 1)
#        C_prd = tf.nn.dropout(C_prd, keep_prob=keep_prob)
        with tf.variable_scope("rnn2"):
            C_prd = qnl.SCF_rnn_layer(I, C_prd, n_basis*n_basis, 1, 10)
#        C_prd = tf.nn.dropout(C_prd, keep_prob=keep_prob)
#        with tf.variable_scope("rnn3"):
#            C_prd = qnl.SCF_rnn_layer(I, C_prd, n_basis*n_basis, 1, 2)
        C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=5)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)

    return y_prd, C_prd, param

def I2_R5h2b_R3h1b_O2(graph, n_basis):
    """good for H2 but seems to be bad for A2
       A2 err: 0.0004
    """

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I, N_layer=2)
        with tf.variable_scope("rnn1"):
            C_prd = qnl.SCF_rnn_layer(I, C_prd, n_basis*n_basis*2, 1, 5, 1)
        with tf.variable_scope("rnn2"):
            C_prd = qnl.SCF_rnn_layer(I, C_prd, n_basis*n_basis, 1, 3, 0)
        C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=2)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)

    return y_prd, C_prd, param

def I0_R8h1b_O3(graph, n_basis):
    """A2 err: 0.004"""

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I)
        C_prd = qnl.SCF_rnn_layer(I, C_prd, n_basis*n_basis, 1, 8, 1)
        C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=3)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)

    return y_prd, C_prd, param

def Ix3_R5h2b_O2(graph, n_basis):
    """A2 err: 0.004"""

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I, N_layer=3)
        C_prd = qnl.SCF_rnn_layer(I, C_prd, n_basis*n_basis*2, 1, 5, 1)
        C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=2)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)

    return y_prd, C_prd, param

def Ix3_R5_O2(graph, n_basis):
    """good for H2 but seems to be bad for A2"""

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I, N_layer=3)
        C_prd = qnl.SCF_rnn_layer(I, C_prd, n_basis*n_basis, 1, 5)
        C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=2)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)

    return y_prd, C_prd, param

def Ix3_R5h1b_O0(graph, n_basis):

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I, N_layer=3)
        C_prd = qnl.SCF_rnn_layer(I, C_prd, n_basis*n_basis, 1, 5)
        C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=0)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)

    return y_prd, C_prd, param

def Ix0_dS5x5_O3x5(graph, n_basis):
    """
    initialize-(dropout-SCFx5)x5-(orthogoalx3)x5
    1. initialize without hidden layer
    2. 5 times of dropoutlayer and SCF layer with 5 hidden layer
    3. 5 times of orthogonal layer with 3 hidden layer
    """

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I)
        for _ in range(5):
            C_prd = qnl.SCF_fc_layer(I, C_prd, keep_prob=keep_prob)
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=5)
        for _ in range(5):
            C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=3)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)
    
    return y_prd, C_prd, param

def I3_S3O3x5d_S3O3x5_O3x3(graph, n_basis):
    """
    """

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I, N_layer=3)
        for _ in range(5):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=3)
            C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=3)
        C_prd = qnl.SCF_fc_layer(I, C_prd, keep_prob=keep_prob)
        for _ in range(5):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=3)
            C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=3)
        for _ in range(3):
            C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=3)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)
    
    return y_prd, C_prd, param

def I3_S5x10_d_S5x5_dS2_O3x5(graph, n_basis):
    """
    """

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I, N_layer=3)
        for _ in range(10):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=5)
        C_prd = qnl.SCF_fc_layer(I, C_prd, keep_prob=keep_prob)
        for _ in range(5):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=5)
        C_prd = qnl.SCF_fc_layer(I, C_prd, keep_prob=keep_prob)
        C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=2)
        for _ in range(5):
            C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=3)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)
    
    return y_prd, C_prd, param

def I0_S2x5_O1x3(graph, n_basis):
    """
    """

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I)
        for _ in range(5):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=2)
        for _ in range(3):
            C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=1)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)
    
    return y_prd, C_prd, param

def I0_S1x2_d_S1x2_O1x3(graph, n_basis):
    """
    Works well for H2
    """

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I)
        for _ in range(2):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=1)
        C_prd = tf.nn.dropout(C_prd, keep_prob)
        for _ in range(2):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=1)
        for _ in range(3):
            C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=1)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)
    
    return y_prd, C_prd, param

def I0_S1x2_d_S1x2_O1x3_SCF(graph, n_basis):
    """
    Works well for H2
    """

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I)
        for _ in range(2):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=1)
        C_prd = tf.nn.dropout(C_prd, keep_prob)
        for _ in range(2):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=1)
        for _ in range(3):
            C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=1)
        C_prd = qnl.SCF_iteration_layer(I, Er, occ, nn, C_prd)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)
    
    return y_prd, C_prd, param

def I0_S1x4_O1_SCFx2(graph, n_basis):

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I)
        for _ in range(4):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=1)
        C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=1)
        for _ in range(2):
            C_prd = qnl.SCF_iteration_layer(I, Er, occ, nn, C_prd)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)
    
    return y_prd, C_prd, param

def I0_S1x4_SCFx1(graph, n_basis):

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I)
        for _ in range(4):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=1)
        C_prd = qnl.SCF_iteration_layer(I, Er, occ, nn, C_prd)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)
    
    return y_prd, C_prd, param

def I0_S1x4_O1_SCFx1(graph, n_basis):

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I)
        for _ in range(4):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=1)
        C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=1)
        C_prd = qnl.SCF_iteration_layer(I, Er, occ, nn, C_prd)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)
    
    return y_prd, C_prd, param

def I0_S1x4_O2_SCFx1(graph, n_basis):

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I)
        for _ in range(4):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=1)
        for _ in range(2):
            C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=1)
        for _ in range(1):
            C_prd = qnl.SCF_iteration_layer(I, Er, occ, nn, C_prd)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)
    
    return y_prd, C_prd, param

def I0_S1x4_O3_SCFx1(graph, n_basis):

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        # model
        C_prd = qnl.SCF_fc_layer(I)
        for _ in range(4):
            C_prd = qnl.SCF_fc_layer(I, C_prd, N_layer=1)
        for _ in range(3):
            C_prd = qnl.orthogonal_fc_layer(I, C_prd, N_layer=1)
        for _ in range(1):
            C_prd = qnl.SCF_iteration_layer(I, Er, occ, nn, C_prd)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)
    
    return y_prd, C_prd, param

def HF_iter_20(graph, n_basis):

    with graph.as_default():
        # input tensors
        I, Er, occ, nn, keep_prob, param = qnt.get_input(graph, n_basis)

        C_prd = None
        for _ in range(20):
            C_prd = qnl.SCF_iteration_layer(I, Er, occ, nn, C_prd)
        y_prd = qnl.HF_energy_layer(I, Er, occ, nn, C_prd)

    return y_prd, C_prd, param
