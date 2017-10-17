import qmnet_tools as qnt
from datetime import datetime
import tensorflow as tf
import numpy as np
import qctoolkit as qtk
import qmnet_layer as qnl

def train(qmnet_model, inp_grp, root_name='model', learn_rate_min=0.0001, valid_ind=2, 
          train_list=None, print_step=5, save_step=1000, hist_step=1, molstr_list=None,
          threshold=1E-3, maxiter=100000, from_scratch=False, kp=1.0, loss='energy',
          learn_rate_steps=50, learn_rate_start=0.1, learn_rate_base=0.5,
          batch_size=32, random_seed=0, max_mean_ratio=50., reg_scale = 0.,
          optimizer=tf.train.AdamOptimizer, fock_scale=0.001,
          save_full_valid_Cprd=False,
         ):
    
    def log_msg(msg, output=True):
        log.write(msg)
        log.flush()
        if output:
            print msg

    if train_list is None:
        train_list = range(len(inp_grp))
    
    #####################################
    # TRAINING/TESTING/VALIDATION SETUP #
    #####################################
    inp_train, inp_test, inp_valid = qnt.train_test_valid(inp_grp, valid_ind, train_list, random_seed)
    I_train, Er_train, occ_train, nn_train, C_train, y_train = qnt.tfIO_batch(inp_train, batch_size)
    I_test, Er_test, occ_test, nn_test, C_test, y_test = qnt.tfIO_batch(inp_test, batch_size)
    I_valid, Er_valid, occ_valid, nn_valid, C_valid, y_valid = qnt.tfIO_batch(inp_valid, batch_size)
    i_rep, i_min, i_mid, i_max = qnt.validation_inds(inp_valid)
    v_inds = [i_rep, i_min, i_mid, i_max]
    I_valid2, Er_valid2, occ_valid2, nn_valid2, C_valid2, y_valid2 = qnt.tfIO(inp_valid, inds=v_inds)
    
    #################
    # STATUS REPORT #
    #################
    status = "root name: " + root_name + "\n"
    status += "model: " + str(qmnet_model) + "\n"
    status += "train/test/validation size: %d/%d/%d \n" % (len(inp_train), len(inp_test), len(inp_valid))
    status += "optimizer: " + str(optimizer) + "\n"
    status += "loss: " + str(loss) + "\n"
    status += "reg_scale: " + str(reg_scale) + "\n"
    status += "keep_prob: " + str(kp) + "\n"
    status += "batch size: " + str(I_train[0].shape[0]) + "\n"
    status += "number of batches: " + str(len(I_train)) + "\n"
    if molstr_list is not None:
        status += "training set: " + str([molstr_list[i] for i in train_list]) + "\n"
        status += "validation: " + molstr_list[valid_ind] + "\n"
    status += "learning rate min: " + str(learn_rate_min) + "\n"
    status += "learning rate steps: " + str(learn_rate_steps) + "\n"
    status += "learning rate start: " + str(learn_rate_start) + "\n"
    status += "learning rate base: " + str(learn_rate_base) + "\n"
    status += "print step: " + str(print_step) + "\n"
    status += "history step: " + str(hist_step) + "\n"
    status += "threshold: " + str(threshold) + "\n"
    status += "maxiter: " + str(maxiter) + "\n"
    status += "execuate time: " + datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "\n\n"
    
    log = open("%s.log" % root_name, 'w')
    log_msg(status)

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        
        # setup model variables
        y = tf.placeholder(tf.float64, name='y')
        y_prd, C_prd, param = qmnet_model(g, inp_valid[0].olp.shape[0])
        I, Er, occ, nn, keep_prob = param
        if loss == 'energy':
            err = tf.reduce_mean(tf.abs(y - y_prd))
            energy_err = err
        elif loss == 'fock':
            err = qnl.Fock_matrix_error(I, Er, occ, nn, C_prd)
            energy_err = tf.reduce_mean(tf.abs(y - y_prd))
        elif loss == 'energy_minimize':
            err = tf.reduce_mean(y_prd)
            energy_err = tf.reduce_mean(tf.abs(y - y_prd))

        # construct weights/biases regularizer
        reg_loss = tf.zeros([1], tf.float64)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if reg_scale > 0:
            for v in variables:
                v2 = tf.multiply(variables[0], variables[0])
                reg_loss = reg_scale * tf.add(reg_loss, tf.reduce_sum(v2))

        optimizer, learn_rate, n_epoch = qnt.get_optimizer(
            g, tf.add(err, reg_loss), learn_rate_steps, learn_rate_start, learn_rate_base, learn_rate_min, optimizer
        )
        C_out = C_prd

        saver = tf.train.Saver()
        save_path = "%s.ckpt" % root_name

        sess = tf.Session()
        try:
            train_err_hist = []
            test_err_hist = []
            valid_err_hist = []
            C_valid_hist = []
            C_test_hist = []
            learn_rate_hist = []

            if not from_scratch:
                # load history and continue 
                try:
                    saver.restore(sess, save_path)
                    try:
                        train_err_hist, test_err_hist, valid_err_hist, C_valid_hist, C_test_hist, learn_rate_hist = \
                        qtk.load('%s_hist.pkl' % root_name)
                        msg = 'model and history loaded, continue optimizing'
                    except:
                        msg = 'model loaded but history is corrupted, continue optimizing'
                except:
                    sess.run(tf.global_variables_initializer())
                    msg = 'no model found, or reload failed, start from scratch'
            else:
                msg = 'start from scratch...'
                sess.run(tf.global_variables_initializer())
                
            log_msg(msg + "\n")
            epoch = len(test_err_hist) * hist_step
            
            test_err, train_err = 1, 1
            while (test_err > threshold or train_err > threshold) and epoch < maxiter:

                ############
                # TRAINING #
                ############
                # parse training set to optimizer and update model #
                train_err_list = []
                train_max_list = []
                train_mean_list = []
                for I_b, Er_b, occ_b, nn_b, C_b, y_b in \
                zip(I_train, Er_train, occ_train, nn_train, C_train, y_train):
                    train_dict = {I:I_b, Er:Er_b, occ:occ_b, nn:nn_b, y:y_b, keep_prob:kp, n_epoch:epoch}
                    sess.run(optimizer, feed_dict=train_dict)
                    train_err_list.append(sess.run(energy_err, feed_dict=train_dict))
                train_err = np.average(train_err_list, weights=[len(_) for _ in I_train])
                current_learn_rate = sess.run(learn_rate, feed_dict=train_dict)

                
                ###########
                # TESTING #
                ###########
                # parse test set to model to check test set error
                test_err_list = []
                for I_t, Er_t, occ_t, nn_t, C_t, y_t in zip(I_test, Er_test, occ_test, nn_test, C_test, y_test):
                    test_dict = {I:I_t, Er:Er_t, occ:occ_t, nn:nn_t, y:y_t, keep_prob:1.0}
                    test_err_list.append(sess.run(energy_err, feed_dict=test_dict))
                test_err = np.average(test_err_list, weights=[len(_) for _ in I_test])


                test2_dict = {
                    I:I_test[0][:3], 
                    Er:Er_test[0][:3], 
                    occ:occ_test[0][:3], 
                    nn:nn_test[0][:3], 
                    y:y_test[0][:3],
                    keep_prob:1.0
                }


                ##############
                # VALIDATION #
                ##############
                # check validation performance
                valid_err_list = []
                C_valid_hist_list = []
                for I_v, Er_v, occ_v, nn_v, C_v, y_v in \
                zip(I_valid, Er_valid, occ_valid, nn_valid, C_valid, y_valid):
                    valid_dict = {I:I_v, Er:Er_v, occ:occ_v, nn:nn_v, y:y_v, keep_prob:1.0}
                    if not save_full_valid_Cprd:
                        valid_err_list.append(sess.run(energy_err, feed_dict=valid_dict))
                    else:
                        ve, Cv = sess.run([energy_err, C_out], feed_dict=valid_dict)
                        valid_err_list.append(ve)
                        C_valid_hist_list.append(np.array(Cv))
                valid_err = np.average(valid_err_list, weights=[len(_) for _ in I_valid])
                if save_full_valid_Cprd:
                    C_valid_hist.append(np.concatenate(C_valid_hist_list))

                ################
                # SAVE HISTORY #
                ################
                valid2_dict = {I:I_valid2, Er:Er_valid2, occ:occ_valid2, nn:nn_valid2, y:y_valid2, keep_prob:1.0}
                train_err_hist.append(train_err)
                learn_rate_hist.append(current_learn_rate)
                test_err_hist.append(test_err)
                C_test_hist.append(np.array(sess.run(C_out, feed_dict=test2_dict)))
                valid_err_hist.append(valid_err)
                if not save_full_valid_Cprd:
                    C_valid_hist.append(np.array(sess.run(C_out, feed_dict=valid2_dict)))
                #################
                # STATUS REPORT #
                #################
                epoch = len(test_err_hist)
                msg = "epoch: %5d train/test/validatoin error: %f/%f/%f at rate %f" % \
                (epoch, train_err, test_err, valid_err, current_learn_rate)

                log_msg(msg + "\n", False)

                if epoch % print_step == 0 or epoch % save_step == 0:
                    print msg
                    if epoch % save_step == 0:
                        hs = hist_step
                        qtk.save([train_err_hist[::hs], test_err_hist[::hs], valid_err_hist[::hs], 
                                  C_valid_hist[::hs], C_test_hist[::hs], learn_rate_hist[::hs]], 
                                 "%s_hist.pkl" % root_name)
                        save_path_out = saver.save(sess, save_path)
                        msg = "Model and history backup"
                        log_msg(msg + "\n")

            ######################
            # TERMINATE AND SAVE #
            ######################
            save_path_out = saver.save(sess, save_path)
            msg = "Model saved in file: %s" % save_path_out
            log_msg(msg + "\n")
            hs = hist_step
            qtk.save([train_err_hist[::hs], test_err_hist[::hs], valid_err_hist[::hs], 
                      C_valid_hist[::hs], C_test_hist[::hs], learn_rate_hist[::hs]], 
                     "%s_hist.pkl" % root_name)
            print "optimization history saved"
            log.close()
            sess.close()
            print "session and log file closed"

        except KeyboardInterrupt:
            ######################
            # TERMINATE AND SAVE #
            ######################
            new_msg = "keyboard interrupt, current stage: %s" % msg
            log_msg(new_msg + "\n")
            save_path_out = saver.save(sess, save_path)
            msg = "Model saved in file: %s" % save_path_out
            log_msg(msg + "\n")
            qtk.save([train_err_hist, test_err_hist, valid_err_hist, C_valid_hist, C_test_hist, learn_rate_hist], 
                     "%s_hist.pkl" % root_name)
            print "optimization history saved"
            log.close()
            sess.close()
            print "session and log file closed"
