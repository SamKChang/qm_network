import numpy as np
from sklearn.model_selection import train_test_split
import qctoolkit as qtk
import tensorflow as tf

def get_shape(tensor):
    """return tensor shape for batched input"""
    shape = list(tensor.shape)
    shape.insert(0, None)
    return shape

def get_input(graph, n_basis):

    with graph.as_default():
        # input tensors
        I = tf.placeholder(tf.float64, shape=(None, 7, n_basis, n_basis), name='I')
        Er = tf.placeholder(tf.float64, shape=(None, n_basis, n_basis, n_basis, n_basis), name='Er')
        occ = tf.placeholder(tf.float64, shape=(None, n_basis), name='occ')
        nn = tf.placeholder(tf.float64, shape=(None,), name='nn')
        keep_prob = tf.placeholder(tf.float64, name='keep_prob')
        param = [I, Er, occ, nn, keep_prob]

    return I, Er, occ, nn, keep_prob, param

def get_optimizer(graph, err, learn_rate_steps, learn_rate_start, learn_rate_base, learn_rate_min, optimizer=tf.train.AdamOptimizer):

    with graph.as_default():
        n_epoch = tf.Variable(0, trainable=False)
        learn_rate = tf.maximum(tf.train.exponential_decay(
                                    learn_rate_start, n_epoch, learn_rate_steps, learn_rate_base),
                                learn_rate_min
                               )

        #optimizer = tf.train.AdamOptimizer(learn_rate).minimize(err)
    return optimizer(learn_rate).minimize(err), learn_rate, n_epoch

def get_input_tensor(mol):
    """Extract matrices and pack as tensor"""
    S, K, Na, C0, X, Er = mol.olp, mol.kin, mol.na, mol.initial_mov, mol.X, mol.er
    Er_j = np.diagonal(np.diagonal(Er))

    Er_x = np.diagonal(np.diagonal(Er, axis1=0, axis2=2), axis1=1, axis2=0)
    I = np.stack([S, K, Na, Er_j, Er_x, C0, X])
    nn = mol.ht_external['nn']
    occ = mol.ht_exp_alpha.occupations
    return I, Er, occ, nn
    
def get_output(I, Er, occ, nn, C):
    """take input tensor and approximated MO coefficients, C, to compute total energy"""
    K = I[1]
    Na = I[2]
    dm = (C * occ).dot(C.T)
    
    Ek = np.trace(dm.dot(K)) * 2
    Ev = np.trace(dm.dot(Na)) * 2
    Ej = np.trace(dm.dot(np.tensordot(dm, Er, axes=([0,1], [0,2])))) * 2
    Ex = -np.trace(dm.dot(np.tensordot(dm, Er, axes=([0,1], [0,1]))))
    E_tot = np.sum([Ek, Ev, Ej, Ex, nn])
    return E_tot

def train_test_valid(inp_grp, valid_ind, train_inds=None, random_seed=0):
    if valid_ind >= len(inp_grp):
        valid_ind = len(inp_grp) - 1
    inp_valid = inp_grp[valid_ind]
    if valid_ind > 0:
        if train_inds is None: train_inds = range(len(inp_grp))
        inps_all = list(qtk.flatten([inp_grp[i] for i in train_inds if i != valid_ind]))
    else:
        inps_all = list(qtk.flatten([inp_grp[i] for i in train_inds]))
    inp_train, inp_test = train_test_split(inps_all, random_state=random_seed, test_size=0.1)
    return inp_train, inp_test, inp_valid

def validation_inds(inp_valid):
    E = np.array([inp.Et for inp in inp_valid])
    ind_min = np.argmin(E)
    ind_max = len(E) - 1
    ind_mid = ind_min + (ind_max - ind_min) / 2
    ind_rep = ind_min / 2
    return ind_rep, ind_min, ind_mid, ind_max

def tfIO(mol_list, inds=None):
    I_list, Er_list, occ_list, nn_list, C_list, labels = [[] for _ in range(6)]
    
    if inds is None:
        inds = range(len(mol_list))
        
    for i in inds:
        mol = mol_list[i]
        C = mol.ht_exp_alpha.coeffs.__array__()
        C_list.append(C)
        I, Er, occ, nn = get_input_tensor(mol)
        I_list.append(I)
        Er_list.append(Er)
        occ_list.append(occ)
        nn_list.append(nn)
        labels.append(get_output(I, Er, occ, nn, C))
        
    return np.stack(I_list).astype(np.float64), np.stack(Er_list).astype(np.float64), \
           np.stack(occ_list).astype(np.float64), np.stack(nn_list).astype(np.float64), \
           np.stack(C_list).astype(np.float64), np.stack(labels).astype(np.float64)

def tfIO_batch(mol_list, batch_size, inds=None):
    
    def batches(mol_list):
        for i in range(0, len(mol_list), batch_size):
            yield mol_list[i:i + batch_size]

    I, Er, occ, nn, C, y = [], [], [], [], [], []
    if inds is None:
        for mol_batch in batches(mol_list):
            I_b, Er_b, occ_b, nn_b, C_b, y_b = tfIO(mol_batch)
            I.append(I_b)
            Er.append(Er_b)
            occ.append(occ_b)
            nn.append(nn_b)
            C.append(C_b)
            y.append(y_b)
    else:
        I_b, Er_b, occ_b, nn_b, C_b, y_b = tfIO(mol_list, inds)
        I.append(I_b)
        Er.append(Er_b)
        occ.append(occ_b)
        nn.append(nn_b)
        C.append(C_b)
        y.append(y_b)
    return I, Er, occ, nn, C, y
