import tensorflow as tf
import numpy as np

tf.reset_default_graph()

def orth(A):
    """??"""
    AA =tf.matmul(tf.transpose(A),A)
    L = tf.cholesky(AA)
    return tf.transpose(tf.matrix_solve(L,tf.transpose(A)))

def lrelu(x, alpha):
    """Leaky ReLU activation function"""
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def flatten(x_tensor):
    """reshape stacked channels in to flatten nodes for fully connected layer"""
    
    if type(x_tensor) is np.ndarray:
        shape = x_tensor.shape
    else:
        shape = x_tensor.get_shape().as_list()
        
    flatten_shape = np.prod(shape[1:])
    
    return tf.reshape(x_tensor, [-1, flatten_shape])

def SCF_fc_layer(I, C_prd=None, keep_prob=1.0, sess=None, N_layer=0, reg_scale=None):
    """fully connected layers to map S, K, V, Er_j, Er_x, C_prd_old to C_prd_new"""
    
    # use initial guess if C_prd is not provided
    if C_prd is None:
        C_prd = I[:,-2]
        shape = C_prd.shape[-2:]
    else:    
        shape = C_prd.get_shape()[-2:]
        
    if reg_scale is not None:
        regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
    else:
        regularizer = None
    # features extracted from molecular Hamiltonian
    S, K, V, Er_j, Er_x = (I[:,i] for i in range(5))
    
    shape_product = int(S.shape[1] * S.shape[2])
    input_flatten = flatten(tf.stack((S, K, V, Er_j, Er_x, C_prd), axis=1))
    for _ in range(N_layer):
        input_flatten = tf.contrib.layers.fully_connected(
            input_flatten, 
            input_flatten.get_shape().as_list()[-1],
            weights_regularizer = regularizer
        )
    C_added = tf.contrib.layers.fully_connected(
        input_flatten, 
        shape_product,
        weights_regularizer = regularizer
    )
    # leaky relu is used to ensure no zero eigen values
    C_out = lrelu(tf.reshape(C_added , [-1, tf.shape(C_prd)[1], tf.shape(C_prd)[2]]), 0.1)
    C_out = tf.nn.dropout(C_out, keep_prob=keep_prob)
    C_out.set_shape(S.get_shape())
    
    return C_out

def _init_LSTM(batch_size, rnn_size, keep_prob=1.0, N_layer=1):

    lstm_list = []

    for _ in range(N_layer):
        cell = tf.contrib.rnn.LSTMCell(rnn_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        lstm_list.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(lstm_list)

    #initial_state = tf.identity(cell.zero_state(batch_size, tf.float64), name='initial_state')
    initial_state = cell.zero_state(batch_size, tf.float64)

    return cell, initial_state

def SCF_rnn_layer(I, C_prd, rnn_size=None, keep_prob=1.0, N_layer=1, fc_layer=0):

    shape_product = int(I[0].shape[1] * I[0].shape[2])
    cell, initial_state = _init_LSTM(tf.shape(C_prd)[0], rnn_size, keep_prob, N_layer)

    inputs = flatten(C_prd)
    #outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float64)
    outputs, state = tf.nn.static_rnn(cell, [inputs], dtype=tf.float64)
    outputs = outputs[0]
    state = tf.identity(state, name = 'final_state')

    for _ in range(fc_layer):
        Wr = tf.Variable(tf.truncated_normal(
            (outputs.get_shape().as_list()[-1], inputs.get_shape().as_list()[-1]),
            dtype=tf.float64
        ), dtype=tf.float64, name='weights_rnn')
        br = tf.Variable(tf.zeros(
            inputs.get_shape().as_list()[-1], dtype=tf.float64
        ), dtype=tf.float64, name='biases_rnn')

        outputs = tf.add(tf.matmul(outputs, Wr), br)

    C_out = tf.reshape(outputs, [-1, tf.shape(C_prd)[1], tf.shape(C_prd)[2]])
    C_out.set_shape(C_prd.get_shape())

    return C_out

def orthogonal_qr_fc_layer(I, C_prd, keep_prob=1.0, N_layer=1):
    """
    orthogonalize predicted MO coefficients, C_prd
    some flexibility seems to be necessary to overfit to H2 system
    """
    C_prd.set_shape(I[:,0].get_shape())
    input_flatten = flatten(C_prd)
 
    itr = 0
    for _ in range(N_layer):
        Wn = tf.Variable(tf.truncated_normal(
            (input_flatten.get_shape().as_list()[-1], input_flatten.get_shape().as_list()[-1]),
            dtype=tf.float64
        ), dtype=tf.float64, name='weights_qr')
        bn = tf.Variable(tf.zeros(
            input_flatten.get_shape().as_list()[-1], dtype=tf.float64
        ), dtype=tf.float64, name='biases_qr')

        input_flatten = tf.add(tf.matmul(input_flatten, Wn), bn)

        if itr < N_layer - 1:
            input_flatten = tf.nn.dropout(input_flatten, keep_prob=keep_prob)

        itr += 1

    C_new = tf.reshape(
        input_flatten,
        [-1, tf.shape(C_prd)[1], tf.shape(C_prd)[2]]
    )
    C_new.set_shape(C_prd.get_shape())

    C_out = tf.qr(C_new)[0]

    return tf.matmul(I[:,-1], C_out)

def orthogonal_fc_layer(I, C_prd, keep_prob=1.0, N_layer=1):
    """
    orthogonalize predicted MO coefficients, C_prd
    some flexibility seems to be necessary to overfit to H2 system
    """
    input_flatten = flatten(C_prd)
 
    itr = 0
    for _ in range(N_layer):
        Wn = tf.Variable(tf.truncated_normal(
            (input_flatten.get_shape().as_list()[-1], input_flatten.get_shape().as_list()[-1]),
            dtype=tf.float64
        ), dtype=tf.float64, name='weights_n')
        bn = tf.Variable(tf.zeros(
            input_flatten.get_shape().as_list()[-1], dtype=tf.float64
        ), dtype=tf.float64, name='biases_n')

        input_flatten = tf.add(tf.matmul(input_flatten, Wn), bn)

        if itr < N_layer - 1:
            input_flatten = tf.nn.dropout(input_flatten, keep_prob=keep_prob)

        itr += 1

    C_new = tf.reshape(
        input_flatten,
        [-1, tf.shape(C_prd)[1], tf.shape(C_prd)[2]]
    )
    
    C_sym = tf.matmul(tf.transpose(C_new, perm=[0,2,1]), C_new)
    C_sym.set_shape(I[:,0].get_shape())


    _, C_out = tf.self_adjoint_eig(C_sym)
    C_out.set_shape(I[:,0].get_shape())
    
    return tf.matmul(I[:,-1], C_out)

def HF_energy_layer(I, Er, occ, nn, C_prd):
    """return predicted energy from predicted MO coefficients C_prd"""
    K, Na = I[:,1], I[:,2]
    
    batch = tf.shape(I)[0]
    C_occ = tf.multiply(tf.expand_dims(occ, 1), C_prd)
    dm = tf.matmul(C_occ, tf.transpose(C_prd, perm=[0,2,1]))
    
    ind = tf.transpose(tf.stack([tf.range(batch), tf.range(batch)]))
    J_kernel = tf.gather_nd(tf.tensordot(dm, Er, axes=([1,2], [1,3])), ind)
    X_kernel = tf.gather_nd(tf.tensordot(dm, Er, axes=([1,2], [1,2])), ind)
    
    Ek = tf.trace(tf.matmul(dm, K)) * 2
    Ev = tf.trace(tf.matmul(dm, Na)) * 2
    Ej = tf.trace(tf.matmul(dm, J_kernel)) * 2
    Ex = -tf.trace(tf.matmul(dm, X_kernel))
    
    E_total = tf.add(Ek, Ev)
    E_total = tf.add(E_total, Ej)
    E_total = tf.add(E_total, Ex)
    E_total = tf.add(E_total, nn)
    
    return E_total

def Fock_matrix(I, Er, occ, nn, C_prd):
    """return predicted energy from predicted MO coefficients C_prd"""
    K, V = I[:,1], I[:,2]
    
    batch = tf.shape(I)[0]
    C_occ = tf.multiply(tf.expand_dims(occ, 1), C_prd)
    dm = tf.matmul(C_occ, tf.transpose(C_prd, perm=[0,2,1]))
    
    ind = tf.transpose(tf.stack([tf.range(batch), tf.range(batch)]))
    J_kernel = tf.gather_nd(tf.tensordot(dm, Er, axes=([1,2], [1,3])), ind)
    X_kernel = tf.gather_nd(tf.tensordot(dm, Er, axes=([1,2], [1,2])), ind)

    Fock = tf.add(tf.add(tf.add(K, V), 2*J_kernel), -1*X_kernel)
    
    return Fock

def SCF_iteration_layer(I, Er, occ, nn, C_prd=None):

    if C_prd is None:
        C_prd = I[:,-2]
    F = Fock_matrix(I, Er, occ, nn, C_prd)
    X = I[:, -1]
    XT = tf.transpose(X, perm=[0,2,1])
    
    XFX = tf.matmul(XT, tf.matmul(F, X))

    _, C_out = tf.self_adjoint_eig(XFX)
    C_out.set_shape(I[:,0].get_shape())
    
    return tf.matmul(X, C_out)

def Fock_matrix_error(I, Er, occ, nn, C_prd):
    """return predicted energy from predicted MO coefficients C_prd"""

    S = I[:, 0]

    F = Fock_matrix(I, Er, occ, nn, C_prd)

    Fc_occ = tf.matmul(F, C_prd)
    Sc_occ = tf.matmul(S, C_prd)
    Fc_occT = tf.transpose(Fc_occ, perm=[0,2,1])
    Sc_occT = tf.transpose(Sc_occ, perm=[0,2,1])
    FS = tf.matrix_diag_part(tf.matmul(Fc_occT, Sc_occ))
    SS = tf.matrix_diag_part(tf.matmul(Sc_occT, Sc_occ))
    SS_inv = tf.where(tf.less(SS, 1e-7), SS, 1./SS)
    epsilon = tf.multiply(FS, SS_inv)
    eSc = tf.multiply(tf.expand_dims(epsilon, 1), Sc_occ)
    residue = tf.add(Fc_occ, -eSc)

    return tf.reduce_sum(tf.multiply(residue, residue))

def Fock_matrix_occ_error(I, Er, occ, nn, C_prd):
    """return predicted energy from predicted MO coefficients C_prd"""

    S = I[:, 0]

    F = Fock_matrix(I, Er, occ, nn, C_prd)

    Fc_occ = tf.matmul(F, C_prd)
    Sc_occ = tf.matmul(S, C_prd)
    Fc_occT = tf.transpose(Fc_occ, perm=[0,2,1])
    Sc_occT = tf.transpose(Sc_occ, perm=[0,2,1])
    FS = tf.matrix_diag_part(tf.matmul(Fc_occT, Sc_occ))
    SS = tf.matrix_diag_part(tf.matmul(Sc_occT, Sc_occ))
    SS_inv = tf.where(tf.less(SS, 1e-7), SS, 1./SS)
    epsilon = tf.multiply(FS, SS_inv)
    eSc = tf.multiply(tf.expand_dims(epsilon, 1), Sc_occ)
    residue = tf.multiply(tf.expand_dims(occ, 1), tf.add(Fc_occ, -eSc))

    return tf.reduce_sum(tf.multiply(residue, residue))
