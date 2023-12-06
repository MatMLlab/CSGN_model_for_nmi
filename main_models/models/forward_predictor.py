"""
Feed-forward prediction process;
Including Capsule encoder, inference block, output block
"""
import os
import json
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras.layers import Input, Embedding
import matplotlib as mpl
import matplotlib.pyplot as plt

from Loss import Pre_loss
from utils import CryMat_Gen
from utils import data_pro
from .submodels import GaussianExpansion
from Loss import mse_metric, pred_loss_1

mpl.use('TkAgg')

class EpochCounter(keras.callbacks.Callback):
    """
    Count the number of epochs and save it.

    Parameters
    ----------
    filepath: str
      the path of the json file that saves the current number of epochs and
      number of training stage
    train_stage: int
      current training stage, starting with 0
    """
    def __init__(self, filepath, train_stage=0):
        super(EpochCounter, self).__init__()
        self.filepath = filepath
        self.train_stage = train_stage

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filepath, 'w') as f:
            json.dump({'epoch': epoch,
                       'stage': self.train_stage}, f)

class SaveOptimizerState(keras.callbacks.Callback):
    """
    Save the state of optimizer at the end of each epoch.

    Parameters
    ----------
    filepath: str
      the path of the pickle (.pkl) file that saves the optimizer state
    """
    def __init__(self, filepath):
        super(SaveOptimizerState, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = keras.backend.batch_get_value(symbolic_weights)
        with open(self.filepath, 'wb') as f:
            pickle.dump(weight_values, f)

DTYPES = {
    "float32": {"numpy": np.float32, "tf": tf.float32},
    "float16": {"numpy": np.float16, "tf": tf.float16},
    "int32": {"numpy": np.int32, "tf": tf.int32},
    "int16": {"numpy": np.int16, "tf": tf.int16},
}


class DataType:
    """
    Data types for tensorflow. This enables users to choose
    from 32-bit float and int, and 16-bit float and int
    """

    np_float = np.float32
    np_int = np.int32
    tf_float = tf.float32
    tf_int = tf.int32

    @classmethod
    def set_dtype(cls, data_type: str) -> None:
        """
        Class method to set the data types
        Args:
            data_type (str): '16' or '32'
        """
        if data_type.endswith("32"):
            float_key = "float32"
            int_key = "int32"
        elif data_type.endswith("16"):
            float_key = "float16"
            int_key = "int16"
        else:
            raise ValueError("Data type not known, choose '16' or '32'")

        cls.np_float = DTYPES[float_key]["numpy"]
        cls.tf_float = DTYPES[float_key]["tf"]
        cls.np_int = DTYPES[int_key]["numpy"]
        cls.tf_int = DTYPES[int_key]["tf"]


class Forward_predictor(object):
    """
    Construct a graph network model with or without explicit atom features
    if n_feature is specified then a general graph model is assumed,
    otherwise a crystal graph model with z number as atom feature is assumed.
    """
    def __init__(self, sess, MatRep_model, Cap_model, cry_graph,
                 str_data, bg_data, mode = 'hse',
                 learning_rate=0.0005,batch_size = 16,
                 k_eig=0, job_dir='./Results/0705', n_epoch=10):

        """
        Args:
            MatRep_model: (str) material representation model based on crystal structure,
                                atom information, bond and so on.
            crystal_graph: (str) data process for crystal material
            Cap_model: (str) capsule autoencoder model
            mode: (str) the calculation mode of DFT
            learning_rate: (float) learning rate
            k_eig: (float)
            job_dir: (str) weight path
            n_epoch: (int) number of training steps
        """

        self.sess = sess
        self.MatRep_model = MatRep_model
        self.Cap_model = Cap_model
        self.cry_graph = cry_graph
        self.mode = mode
        self.k_eig = k_eig
        self.pred_loss = Pre_loss()
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.str_data = str_data
        self.bg_data = bg_data

        print("[*] Initialize model successfully...")

    def check_train_state(self):
        try:
            with open(os.path.join(self.job_dir, 'train_state.json')) as f:
                data = json.load(f)
                init_epoch, init_stage = data['epoch'], data['stage']
            if init_epoch == self.n_epoch - 1:
                init_epoch = 0
                init_stage += 1
            else:
                init_epoch += 1
            weights_file = os.path.join(self.job_dir,
                                        'inos_last_model.hdf5'.format(
                                            init_stage))
            self.model.load_weights(weights_file)
            is_continue = True
        except IOError:
            init_epoch, init_stage, is_continue = 0, 0, False
        return init_epoch, init_stage, is_continue

    def load_datagen(self, data_mode = None):
        # extract pbe data
        train_data, train_tar,test_data, test_tar = data_pro(self.bg_data, self.str_data,
                                                             self.cry_graph)
        self.test_data = test_data
        self.test_tar = test_tar
        self.train_tar = train_tar

        self.train_gen, self.steps_per_train = CryMat_Gen(train_graphs = train_data,
                                                          train_targets = train_tar,
                                                          graph_converter = self.cry_graph,
                                                          batch_size = self.batch_size,
                                                          mode = 'train',).load_train_data()
        #self.train_generator, self.steps_per_train = self.train_gen.load_train_data()

        self.val_gen, self.steps_per_val = CryMat_Gen(val_graphs = test_data,
                                                      val_targets = test_tar,
                                                      graph_converter = self.cry_graph,
                                                      batch_size = self.batch_size,
                                                      mode = 'val').load_val_data()
        #self.val_generator, self.steps_per_val = self.val_gen.load_val_data()


    def build_predictor(self,
                        nfeat_edge: int = None,
                        nfeat_global: int = None,
                        nfeat_node: int = None,
                        nvocal: int = 108,
                        embedding_dim: int = 80,
                        nbvocal: int = None,
                        bond_embedding_dim: int = None,
                        ngvocal: int = 108,
                        global_embedding_dim: int = 80,
                        **kwargs):

        """build prediction model with capsule graphs constructed using the `direct` backend"""



        # atom inputs
        x_com_w = Input(shape=(None,), dtype=DataType.tf_float, name="composition_weights")
        x_ele_idx = Input(shape=(None,), dtype=DataType.tf_int, name="elements_index")
        x_atom_fea = Input(shape=(None, 103), dtype=DataType.tf_float, name="elements_fea")
        x_fea = Input(shape=(None,), dtype=DataType.tf_int, name="fea_list")
        x_nbr = Input(shape=(None,), dtype=DataType.tf_int, name="nvr_list")
        x_ele = Input(shape=(None,), dtype=DataType.tf_int, name="atom_ele_index")

        if nfeat_node is None:
            # only z as feature
            x1 = Input(shape=(None, ), dtype=DataType.tf_int, name="atom_int_input")
            x1_ = Embedding(nvocal, embedding_dim, name="atom_embedding")(x1)
        else:
            x1 = Input(shape=(None, nfeat_node), name="atom_feature_input")
            x1_ = x1

        # bond inputs
        if nfeat_edge is None:
            if bond_embedding_dim is not None:
                # bond attributes are integers for embedding
                x2 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_int_input")
                x2_ = Embedding(nbvocal, bond_embedding_dim, name="bond_embedding")(x2)
            else:
                # the bond attributes are float distance
                x2 = Input(shape=(None, ), dtype=DataType.tf_float, name="bond_float_input")
                #centers = kwargs.get("centers", None)
                centers = np.linspace(0, 6, 100)
                #width = kwargs.get("width", None)
                width = 0.5
                if centers is None and width is None:
                    raise ValueError(
                        "If the bond attributes are single float values, "
                        "we expect the value to be expanded before passing "
                        "to the models. Therefore, `centers` and `width` for "
                        "Gaussian basis expansion are needed"
                    )
                x2_ = GaussianExpansion(centers = np.linspace(0, 6, 100),
                                        width = 0.5)(x2)  # type: ignore
        else:
            x2 = Input(shape=(None, nfeat_edge), name="bond_feature_input")
            x2_ = x2

        # state inputs
        if nfeat_global is None:
            if global_embedding_dim is not None:
                # global state inputs are embedding integers
                x3 = Input(shape=(None,), dtype=DataType.tf_int, name="state_int_input")
                x3_ = Embedding(ngvocal, global_embedding_dim, name="state_embedding")(x3)
            else:
                # take default vector of two zeros
                x3 = Input(shape=(None, 2), dtype=DataType.tf_float, name="state_default_input")
                x3_ = x3
        else:
            x3 = Input(shape=(None, nfeat_global), name="state_feature_input")
            x3_ = x3
        x4 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_index_1_input")
        x5 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_index_2_input")
        x6 = Input(shape=(None, ), dtype=DataType.tf_int, name="atom_graph_index_input")
        x7 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_graph_index_input")

        x21 = Input(shape=(None, nfeat_edge), name="bond_feature_input_1")
        x2_1 = x21
        x4_1 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_index_1_input_1")
        x5_1 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_index_2_input_1")
        x7_1 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_graph_index_input_1")

        x22 = Input(shape=(None, nfeat_edge), name="bond_feature_input_2")
        x2_2 = x22
        x4_2 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_index_1_input_2")
        x5_2 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_index_2_input_2")
        x7_2 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_graph_index_input_2")

        x23 = Input(shape=(None, nfeat_edge), name="bond_feature_input_3")
        x2_3 = x23
        x4_3 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_index_1_input_3")
        x5_3 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_index_2_input_3")
        x7_3 = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_graph_index_input_3")


        real_bg = Input(shape=(None,51), dtype=DataType.tf_float, name="real_pdos")

        # crystal representation and feature process
        branch_1 = self.MatRep_model(x_com_w, x_ele_idx, x_atom_fea, x_fea, x_nbr, x_ele,
                                     x1_, x2_, x3_, x4, x5, x6, x7,
                                     x2_1, x4_1, x5_1, x7_1, x2_2, x4_2, x5_2, x7_2, x2_3, x4_3, x5_3, x7_3) # clusters

        # capsule auto-ecoder and prediction
        atom, bond, atom_1, bond_1, atom_2, bond_2,atom_3, bond_3, element_weight, cry_vec, u1_, u1_1, u1_2, u1_3 = branch_1
        mse_loss, pace_pre = self.Cap_model(atom, bond, atom_1, bond_1, atom_2, bond_2, atom_3, bond_3,# multi clusters
                                  element_weight, cry_vec, u1_, u1_1, u1_2, u1_3, x4, x6, real_bg)

        self.inputs = [x_com_w, x_ele_idx, x_atom_fea, x_fea, x_nbr, x_ele,
                  x1, x2, x3, x4, x5, x6, x7,
                       x2_1, x4_1, x5_1,x7_1, x2_2, x4_2, x5_2, x7_2,
                       x2_3, x4_3, x5_3, x7_3, real_bg]

        outputs = [mse_loss, pace_pre]

        #with tf.Session() as sess:
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
            #session.run(tf.tables_initializer())
        self.model = tf.keras.Model(inputs = self.inputs, outputs = outputs)
        #self.model.summary(line_length = 150, positions = [0.45, 0.6, 0.7, 1])

        #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = self.learning_rate, rho = 0.8)
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1 = 0.9, beta_2 = 0.999)

        self.losses = [{'tf_op_layer_cluster_capsule/mse_loss':self.Cap_model._mse_loss}]


    def train_model(self):
        FIDELITIES = ['hse']
        # load data and build model

        #self.load_datagen(data_mode=self.mode)
        self.build_predictor(nfeat_edge = 100,
                             nfeat_global = None,
                             nfeat_node = None,
                             ngvocal = len(FIDELITIES),
                             global_embedding_dim = 64,
                             bond_embedding_dim = None,
                             nblocks = 3,
                             nvocal = 108,
                             npass = 2)

        self.load_datagen(data_mode=self.mode)

        init_epoch, init_stage, is_continue = self.check_train_state()

        #with tf.Session() as sess:
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
        loss = []
        for epoch in range(self.n_epoch):
            print("\nStart of epoch %d" % (epoch,))

            #lr = 1.5*self.learning_rate/(epoch + 1)
            #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9)
            lr = 3.17*self.learning_rate/(epoch/10 + 1)
            #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
            print("******Learning rate is:******", lr)

            for l, loss_function in enumerate(self.losses):
                    # skip this round if previous training records are found
                if l < init_stage:
                        continue
                    ### compile model ###
                self.model.compile(optimizer =  self.optimizer, loss = loss_function)


                self.model.fit_generator(generator = self.train_gen,
                                         steps_per_epoch = self.steps_per_train,
                                         validation_data = self.val_gen,
                                         validation_steps = self.steps_per_val,
                                         use_multiprocessing = False,
                                         epochs = 96,
                                         #callbacks = callbacks,
                                         shuffle=False,
                                         initial_epoch = init_epoch,)

                # Results analysis
                pred_target = self.model.predict_generator(self.train_gen)

                a = np.array(pred_target[-1]).reshape(-1)
                b = np.array(self.train_tar).reshape(-1)

                np.savetxt('./Results/pre_data/pre_data_%s-%d.csv' % (epoch, l), a, delimiter=',')
                np.savetxt('./Results/pre_data/real_data_%s-%d.csv' % (epoch, l), b, delimiter=',')
                l1_loss = np.mean(np.abs(a - b))

                test_pred_target = self.model.predict_generator(self.val_gen)

                c = np.array(test_pred_target[-1]).reshape(-1)
                d = np.array(self.test_tar).reshape(-1)

                np.savetxt('./Results/pre_data/test_pre_data_%s-%d.csv' % (epoch, l), c, delimiter=',')
                np.savetxt('./Results/pre_data/test_real_data_%s-%d.csv' % (epoch, l), d, delimiter=',')
                test_l1_loss = np.mean(np.abs(c - d))

                print("Random 1 prediction L1 loss - %d: " % epoch, l1_loss)
                print("Random 1 prediction test L1 loss - %d: " % epoch, test_l1_loss)

                fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
                style = dict(size=32, color='gray')
                axs.plot(a[:102], 'b')
                axs.plot(b[:102], 'r')
                fig.suptitle('Prediction & Ground truth data - %s-%d' % (epoch, l))
                # plt.yticks([0, 0.5, 1])
                # plt.xticks([0, 0.5, 1])
                plt.savefig('./Results/plot/plot_gen & real_%d_%s' % (epoch, l))
                plt.close()
                #plt.show()
                loss.append((l1_loss))

            if epoch % 6 == 0 and epoch != 0:
                # N atom clusters
                atom_che2 = self.model.get_layer('block_add_atom_2').output[0]  # N = 2
                atom_che3 = self.model.get_layer('block_add_atom_5').output[0]  # N = 3
                atom_che5 = self.model.get_layer('block_add_atom_8').output[0]  # N = 5
                atom_che7 = self.model.get_layer('block_add_atom_11').output[0]  # N = 7

                node_vec2 = self.model.get_layer('set2set_node').output[0]  # N = 2
                node_vec3 = self.model.get_layer('set2set_node_1').output[0]  # N = 2
                node_vec5 = self.model.get_layer('set2set_node_2').output[0]  # N = 2
                node_vec7 = self.model.get_layer('set2set_node_3').output[0]  # N = 2

                # bond vectors
                bond_che2 = self.model.get_layer('block_add_bond_1').output[0]  # N = 2
                bond_che3 = self.model.get_layer('block_add_bond_4').output[0]  # N = 3
                bond_che5 = self.model.get_layer('block_add_bond_7').output[0]  # N = 5
                bond_che7 = self.model.get_layer('block_add_bond_10').output[0]  # N = 7

                bond_vec2 = self.model.get_layer('set2set_bond').output[0]  # N = 2
                bond_vec3 = self.model.get_layer('set2set_bond_1').output[0]  # N = 2
                bond_vec5 = self.model.get_layer('set2set_bond_2').output[0]  # N = 2
                bond_vec7 = self.model.get_layer('set2set_bond_3').output[0]  # N = 2

                # mass vectors
                mass_che2 = self.model.get_layer('block_add_mass_2').output[0]  # N = 2
                mass_che3 = self.model.get_layer('block_add_mass_5').output[0]  # N = 3
                mass_che5 = self.model.get_layer('block_add_mass_8').output[0]  # N = 5
                mass_che7 = self.model.get_layer('block_add_mass_11').output[0]  # N = 7

                # total material chemical environments
                cry_che = self.model.get_layer('concatenate').output[0]

                #bond_che = self.model.get_layer('tf_op_layer_SegmentMean').output

                # cluster capsules
                #cap_prob2 = self.model.get_layer('tf_op_layer_cluster_capsule/cap_prob_1').output
                #cap_prob3 = self.model.get_layer('tf_op_layer_cluster_capsule/cap_prob_2').output
                #cap_prob5 = self.model.get_layer('tf_op_layer_cluster_capsule/cap_prob_3').output
                #cap_prob7 = self.model.get_layer('tf_op_layer_cluster_capsule/cap_prob_4').output

                pose_pres_fea2 = self.model.get_layer('tf_op_layer_cluster_capsule/pose_pres_fea').output
                pose_pres_fea3 = self.model.get_layer('tf_op_layer_cluster_capsule/pose_pres_fea_1').output
                pose_pres_fea5 = self.model.get_layer('tf_op_layer_cluster_capsule/pose_pres_fea_2').output
                pose_pres_fea7 = self.model.get_layer('tf_op_layer_cluster_capsule/pose_pres_fea_3').output

                """
                clu_sigma_1 =self.model.get_layer('tf_op_layer_cluster_capsule/out_sigma').output
                clu_sigma_2 =self.model.get_layer('tf_op_layer_cluster_capsule/out_sigma_1').output
                clu_sigma_3 =self.model.get_layer('tf_op_layer_cluster_capsule/out_sigma_2').output
                clu_sigma_4 =self.model.get_layer('tf_op_layer_cluster_capsule/out_sigma_3').output

                clu_mu_1 =self.model.get_layer('tf_op_layer_cluster_capsule/out_mu').output
                clu_mu_2 =self.model.get_layer('tf_op_layer_cluster_capsule/out_mu_1').output
                clu_mu_3 =self.model.get_layer('tf_op_layer_cluster_capsule/out_mu_2').output
                clu_mu_4 =self.model.get_layer('tf_op_layer_cluster_capsule/out_mu_3').output

                clu_alpha_1 =self.model.get_layer('tf_op_layer_cluster_capsule/out_alpha').output
                clu_alpha_2 =self.model.get_layer('tf_op_layer_cluster_capsule/out_alpha_1').output
                clu_alpha_3 =self.model.get_layer('tf_op_layer_cluster_capsule/out_alpha_2').output
                clu_alpha_4 =self.model.get_layer('tf_op_layer_cluster_capsule/out_alpha_3').output
                """

                gau_sample_1 = self.model.get_layer('tf_op_layer_cluster_capsule/gau_sample_1').output
                gau_sample_2 = self.model.get_layer('tf_op_layer_cluster_capsule/gau_sample_2').output
                gau_sample_3 = self.model.get_layer('tf_op_layer_cluster_capsule/gau_sample_3').output
                gau_sample_4 = self.model.get_layer('tf_op_layer_cluster_capsule/gau_sample_4').output

                mol_fea2 = self.model.get_layer('tf_op_layer_cluster_capsule/mol_clus').output
                mol_fea3 = self.model.get_layer('tf_op_layer_cluster_capsule/mol_clus_1').output
                mol_fea5 = self.model.get_layer('tf_op_layer_cluster_capsule/mol_clus_2').output
                mol_fea7 = self.model.get_layer('tf_op_layer_cluster_capsule/mol_clus_3').output

                mol_cry_caps = self.model.get_layer('tf_op_layer_cluster_capsule/mol_cry_caps').output
                #gau_alpha = self.model.get_layer('tf_op_layer_cluster_capsule/gau_alpha').output
                #cry_gau_alpha = self.model.get_layer('tf_op_layer_cluster_capsule/mol_alpha_cry').output
                residual_mol = self.model.get_layer('tf_op_layer_cluster_capsule/add_7').output
                #resample_pool = self.model.get_layer('tf_op_layer_cluster_capsule/resample_pool').output

                #gaussian_z = self.model.get_layer('tf_op_layer_cluster_capsule/mul_7').output
                #gaussian_fea = self.model.get_layer('tf_op_layer_cluster_capsule/linear_2/add').output


                # layer 2
                atom_che_layer2 = tf.keras.Model(inputs=self.model.input, outputs=atom_che2)
                atom_che_layer3 = tf.keras.Model(inputs=self.model.input, outputs=atom_che3)
                atom_che_layer5 = tf.keras.Model(inputs=self.model.input, outputs=atom_che5)
                atom_che_layer7 = tf.keras.Model(inputs=self.model.input, outputs=atom_che7)

                node_vec_layer2 = tf.keras.Model(inputs=self.model.input, outputs=node_vec2)
                node_vec_layer3 = tf.keras.Model(inputs=self.model.input, outputs=node_vec3)
                node_vec_layer5 = tf.keras.Model(inputs=self.model.input, outputs=node_vec5)
                node_vec_layer7 = tf.keras.Model(inputs=self.model.input, outputs=node_vec7)

                bond_che_layer2 = tf.keras.Model(inputs=self.model.input, outputs=bond_che2)
                bond_che_layer3 = tf.keras.Model(inputs=self.model.input, outputs=bond_che3)
                bond_che_layer5 = tf.keras.Model(inputs=self.model.input, outputs=bond_che5)
                bond_che_layer7 = tf.keras.Model(inputs=self.model.input, outputs=bond_che7)

                bond_vec_layer2 = tf.keras.Model(inputs=self.model.input, outputs=bond_vec2)
                bond_vec_layer3 = tf.keras.Model(inputs=self.model.input, outputs=bond_vec3)
                bond_vec_layer5 = tf.keras.Model(inputs=self.model.input, outputs=bond_vec5)
                bond_vec_layer7 = tf.keras.Model(inputs=self.model.input, outputs=bond_vec7)

                mass_che_layer2 = tf.keras.Model(inputs=self.model.input, outputs=mass_che2)
                mass_che_layer3 = tf.keras.Model(inputs=self.model.input, outputs=mass_che3)
                mass_che_layer5 = tf.keras.Model(inputs=self.model.input, outputs=mass_che5)
                mass_che_layer7 = tf.keras.Model(inputs=self.model.input, outputs=mass_che7)

                cry_che_layer = tf.keras.Model(inputs=self.model.input, outputs=cry_che)

                #cap_prob_layer2 = tf.keras.Model(inputs=self.model.input, outputs=cap_prob2)
                #cap_prob_layer3 = tf.keras.Model(inputs=self.model.input, outputs=cap_prob3)
                #cap_prob_layer5 = tf.keras.Model(inputs=self.model.input, outputs=cap_prob5)
                #cap_prob_layer7 = tf.keras.Model(inputs=self.model.input, outputs=cap_prob7)


                pose_pres_fea_layer2 = tf.keras.Model(inputs=self.model.input, outputs=pose_pres_fea2)
                pose_pres_fea_layer3 = tf.keras.Model(inputs=self.model.input, outputs=pose_pres_fea3)
                pose_pres_fea_layer5 = tf.keras.Model(inputs=self.model.input, outputs=pose_pres_fea5)
                pose_pres_fea_layer7 = tf.keras.Model(inputs=self.model.input, outputs=pose_pres_fea7)
                """
                clu_sigma_layer1 =tf.keras.Model(inputs=self.model.input, outputs=clu_sigma_1)
                clu_sigma_layer2 =tf.keras.Model(inputs=self.model.input, outputs=clu_sigma_2)
                clu_sigma_layer3 =tf.keras.Model(inputs=self.model.input, outputs=clu_sigma_3)
                clu_sigma_layer4 =tf.keras.Model(inputs=self.model.input, outputs=clu_sigma_4)

                clu_mu_layer1 =tf.keras.Model(inputs=self.model.input, outputs=clu_mu_1)
                clu_mu_layer2 =tf.keras.Model(inputs=self.model.input, outputs=clu_mu_2)
                clu_mu_layer3 =tf.keras.Model(inputs=self.model.input, outputs=clu_mu_3)
                clu_mu_layer4 =tf.keras.Model(inputs=self.model.input, outputs=clu_mu_4)

                clu_alpha_layer1 =tf.keras.Model(inputs=self.model.input, outputs=clu_alpha_1)
                clu_alpha_layer2 =tf.keras.Model(inputs=self.model.input, outputs=clu_alpha_2)
                clu_alpha_layer3 =tf.keras.Model(inputs=self.model.input, outputs=clu_alpha_3)
                clu_alpha_layer4 =tf.keras.Model(inputs=self.model.input, outputs=clu_alpha_4)
                """

                gau_sample_layer1 = tf.keras.Model(inputs=self.model.input, outputs=gau_sample_1)
                gau_sample_layer2 = tf.keras.Model(inputs=self.model.input, outputs=gau_sample_2)
                gau_sample_layer3 = tf.keras.Model(inputs=self.model.input, outputs=gau_sample_3)
                gau_sample_layer4 = tf.keras.Model(inputs=self.model.input, outputs=gau_sample_4)

                mol_fea_layer2 = tf.keras.Model(inputs=self.model.input, outputs=mol_fea2)
                mol_fea_layer3 = tf.keras.Model(inputs=self.model.input, outputs=mol_fea3)
                mol_fea_layer5 = tf.keras.Model(inputs=self.model.input, outputs=mol_fea5)
                mol_fea_layer7 = tf.keras.Model(inputs=self.model.input, outputs=mol_fea7)

                mol_cry_caps_layer = tf.keras.Model(inputs=self.model.input, outputs=mol_cry_caps)
                #gau_alpha_layer = tf.keras.Model(inputs=self.model.input, outputs=gau_alpha)
                residual_mol_layer = tf.keras.Model(inputs=self.model.input, outputs=residual_mol)
                #cry_gau_alpha_layer = tf.keras.Model(inputs=self.model.input, outputs=cry_gau_alpha)
                #gaussian_z_layer = tf.keras.Model(inputs=self.model.input, outputs=gaussian_z)
                #gaussian_fea_layer = tf.keras.Model(inputs=self.model.input, outputs=gaussian_fea)
                #bond_che_layer = tf.keras.Model(inputs=self.model.input, outputs=bond_che)

                # predictive layer
                atom_che_out2 = atom_che_layer2.predict(self.train_gen)
                atom_che_out3 = atom_che_layer3.predict(self.train_gen)
                atom_che_out5 = atom_che_layer5.predict(self.train_gen)
                atom_che_out7 = atom_che_layer7.predict(self.train_gen)

                node_vec_out2 = node_vec_layer2.predict(self.train_gen)
                node_vec_out3 = node_vec_layer3.predict(self.train_gen)
                node_vec_out5 = node_vec_layer5.predict(self.train_gen)
                node_vec_out7 = node_vec_layer7.predict(self.train_gen)

                bond_che_out2 = bond_che_layer2.predict(self.train_gen)
                bond_che_out3 = bond_che_layer3.predict(self.train_gen)
                bond_che_out5 = bond_che_layer5.predict(self.train_gen)
                bond_che_out7 = bond_che_layer7.predict(self.train_gen)

                bond_vec_out2 = bond_vec_layer2.predict(self.train_gen)
                bond_vec_out3 = bond_vec_layer3.predict(self.train_gen)
                bond_vec_out5 = bond_vec_layer5.predict(self.train_gen)
                bond_vec_out7 = bond_vec_layer7.predict(self.train_gen)

                mass_che_out2 = mass_che_layer2.predict(self.train_gen)
                mass_che_out3 = mass_che_layer3.predict(self.train_gen)
                mass_che_out5 = mass_che_layer5.predict(self.train_gen)
                mass_che_out7 = mass_che_layer7.predict(self.train_gen)

                cry_che_out = cry_che_layer.predict(self.train_gen)

                #cap_prob_out2 = cap_prob_layer2.predict(self.train_gen)
                #cap_prob_out3 = cap_prob_layer3.predict(self.train_gen)
                #cap_prob_out5 = cap_prob_layer5.predict(self.train_gen)
                #cap_prob_out7 = cap_prob_layer7.predict(self.train_gen)

                pose_pres_fea_out2 = pose_pres_fea_layer2.predict(self.train_gen)
                pose_pres_fea_out3 = pose_pres_fea_layer3.predict(self.train_gen)
                pose_pres_fea_out5 = pose_pres_fea_layer5.predict(self.train_gen)
                pose_pres_fea_out7 = pose_pres_fea_layer7.predict(self.train_gen)
                """
                clu_sigma_out1 =clu_sigma_layer1.predict(self.train_gen)
                clu_sigma_out2 =clu_sigma_layer2.predict(self.train_gen)
                clu_sigma_out3 =clu_sigma_layer3.predict(self.train_gen)
                clu_sigma_out4 =clu_sigma_layer4.predict(self.train_gen)

                clu_mu_out1 =clu_mu_layer1.predict(self.train_gen)
                clu_mu_out2 =clu_mu_layer2.predict(self.train_gen)
                clu_mu_out3 =clu_mu_layer3.predict(self.train_gen)
                clu_mu_out4 =clu_mu_layer4.predict(self.train_gen)

                clu_alpha_out1 =clu_alpha_layer1.predict(self.train_gen)
                clu_alpha_out2 =clu_alpha_layer2.predict(self.train_gen)
                clu_alpha_out3 =clu_alpha_layer3.predict(self.train_gen)
                clu_alpha_out4 =clu_alpha_layer4.predict(self.train_gen)
                """

                gau_sample_out1 = gau_sample_layer1.predict(self.train_gen)
                gau_sample_out2 =gau_sample_layer2.predict(self.train_gen)
                gau_sample_out3 =gau_sample_layer3.predict(self.train_gen)
                gau_sample_out4 =gau_sample_layer4.predict(self.train_gen)
                fig = plt.figure(figsize=(12, 20))
                for i in range(12):
                    n = i+1
                    plt.subplot(12, 1, n)
                    plt.plot(gau_sample_out1[i], "r")
                plt.savefig('./Results/plot/2_atom_ked_%s_%d' % (epoch, l))
                plt.close()

                fig = plt.figure(figsize=(12, 20))
                for i in range(12):
                    n = i+1
                    plt.subplot(12, 1, n)
                    plt.plot(gau_sample_out4[i], "r")
                plt.savefig('./Results/plot/7_atom_ked_%s_%d' % (epoch, l))
                plt.close()

                mol_fea_out2 = mol_fea_layer2.predict(self.train_gen)
                mol_fea_out3 = mol_fea_layer3.predict(self.train_gen)
                mol_fea_out5 = mol_fea_layer5.predict(self.train_gen)
                mol_fea_out7 = mol_fea_layer7.predict(self.train_gen)

                mol_cry_caps_out = mol_cry_caps_layer.predict(self.train_gen)
                #gau_alpha_out = gau_alpha_layer.predict(self.train_gen)
                residual_mol_out = residual_mol_layer.predict(self.train_gen)
                #cry_gau_alpha_out = cry_gau_alpha_layer.predict(self.train_gen)

                #gaussian_z_out = gaussian_z_layer.predict(self.train_gen)
                #gaussian_fea_out = gaussian_fea_layer.predict(self.train_gen)

                # output layer
                np.savetxt('./Results/mid_data/1_atom_che_out2%s_%d.csv' % (epoch, l), atom_che_out2, delimiter=',')
                np.savetxt('./Results/mid_data/1_atom_che_out3%s_%d.csv' % (epoch, l), atom_che_out3, delimiter=',')
                np.savetxt('./Results/mid_data/1_atom_che_out5%s_%d.csv' % (epoch, l), atom_che_out5, delimiter=',')
                np.savetxt('./Results/mid_data/1_atom_che_out7%s_%d.csv' % (epoch, l), atom_che_out7, delimiter=',')

                np.savetxt('./Results/mid_data/2_node_vec_out2%s_%d.csv' % (epoch, l), node_vec_out2, delimiter=',')
                np.savetxt('./Results/mid_data/2_node_vec_out3%s_%d.csv' % (epoch, l), node_vec_out3, delimiter=',')
                np.savetxt('./Results/mid_data/2_node_vec_out5%s_%d.csv' % (epoch, l), node_vec_out5, delimiter=',')
                np.savetxt('./Results/mid_data/2_node_vec_out7%s_%d.csv' % (epoch, l), node_vec_out7, delimiter=',')

                np.savetxt('./Results/mid_data/3_bond_che_out2%s_%d.csv' % (epoch, l), bond_che_out2, delimiter=',')
                np.savetxt('./Results/mid_data/3_bond_che_out3%s_%d.csv' % (epoch, l), bond_che_out3, delimiter=',')
                np.savetxt('./Results/mid_data/3_bond_che_out5%s_%d.csv' % (epoch, l), bond_che_out5, delimiter=',')
                np.savetxt('./Results/mid_data/3_bond_che_out7%s_%d.csv' % (epoch, l), bond_che_out7, delimiter=',')

                np.savetxt('./Results/mid_data/4_bond_vec_out2%s_%d.csv' % (epoch, l), bond_vec_out2, delimiter=',')
                np.savetxt('./Results/mid_data/4_bond_vec_out3%s_%d.csv' % (epoch, l), bond_vec_out3, delimiter=',')
                np.savetxt('./Results/mid_data/4_bond_vec_out5%s_%d.csv' % (epoch, l), bond_vec_out5, delimiter=',')
                np.savetxt('./Results/mid_data/4_bond_vec_out7%s_%d.csv' % (epoch, l), bond_vec_out7, delimiter=',')

                np.savetxt('./Results/mid_data/5_mass_che_out2%s_%d.csv' % (epoch, l), mass_che_out2, delimiter=',')
                np.savetxt('./Results/mid_data/5_mass_che_out3%s_%d.csv' % (epoch, l), mass_che_out3, delimiter=',')
                np.savetxt('./Results/mid_data/5_mass_che_out5%s_%d.csv' % (epoch, l), mass_che_out5, delimiter=',')
                np.savetxt('./Results/mid_data/5_mass_che_out7%s_%d.csv' % (epoch, l), mass_che_out7, delimiter=',')

                np.savetxt('./Results/mid_data/6_cry_che_out%s_%d.csv' % (epoch, l), cry_che_out, delimiter=',')

                #np.savetxt('./Results/mid_data/7_cap_prob_out2%s_%d.csv' % (epoch, l), cap_prob_out2, delimiter=',')
                #np.savetxt('./Results/mid_data/7_cap_prob_out3%s_%d.csv' % (epoch, l), cap_prob_out3, delimiter=',')
                #np.savetxt('./Results/mid_data/7_cap_prob_out5%s_%d.csv' % (epoch, l), cap_prob_out5, delimiter=',')
                #np.savetxt('./Results/mid_data/7_cap_probout7%s_%d.csv' % (epoch, l), cap_prob_out7, delimiter=',')

                np.savetxt('./Results/mid_data/7_pose_pres_fea_out2%s_%d.csv' % (epoch, l), pose_pres_fea_out2, delimiter=',')
                np.savetxt('./Results/mid_data/7_pose_pres_fea_out3%s_%d.csv' % (epoch, l), pose_pres_fea_out3, delimiter=',')
                np.savetxt('./Results/mid_data/7_pose_pres_fea_out5%s_%d.csv' % (epoch, l), pose_pres_fea_out5, delimiter=',')
                np.savetxt('./Results/mid_data/7_pose_pres_fea_out7%s_%d.csv' % (epoch, l), pose_pres_fea_out7, delimiter=',')
                """
                np.savetxt('./Results/mid_data/7_sigma_out2%s_%d.csv' % (epoch, l), clu_sigma_out1, delimiter=',')
                np.savetxt('./Results/mid_data/7_sigma_out3%s_%d.csv' % (epoch, l), clu_sigma_out2, delimiter=',')
                np.savetxt('./Results/mid_data/7_sigma_out5%s_%d.csv' % (epoch, l), clu_sigma_out3, delimiter=',')
                np.savetxt('./Results/mid_data/7_sigma_out7%s_%d.csv' % (epoch, l), clu_sigma_out4, delimiter=',')

                np.savetxt('./Results/mid_data/7_mu_out2%s_%d.csv' % (epoch, l), clu_mu_out1, delimiter=',')
                np.savetxt('./Results/mid_data/7_mu_out3%s_%d.csv' % (epoch, l), clu_mu_out2, delimiter=',')
                np.savetxt('./Results/mid_data/7_mu_out5%s_%d.csv' % (epoch, l), clu_mu_out3, delimiter=',')
                np.savetxt('./Results/mid_data/7_mu_out7%s_%d.csv' % (epoch, l), clu_mu_out4, delimiter=',')

                np.savetxt('./Results/mid_data/7_alpha_out2%s_%d.csv' % (epoch, l), clu_alpha_out1, delimiter=',')
                np.savetxt('./Results/mid_data/7_alpha_out3%s_%d.csv' % (epoch, l), clu_alpha_out2, delimiter=',')
                np.savetxt('./Results/mid_data/7_alpha_out5%s_%d.csv' % (epoch, l), clu_alpha_out3, delimiter=',')
                np.savetxt('./Results/mid_data/7_alpha_out7%s_%d.csv' % (epoch, l), clu_alpha_out4, delimiter=',')
                """

                np.savetxt('./Results/mid_data/7_sample_out2%s_%d.csv' % (epoch, l), gau_sample_out1, delimiter=',')
                np.savetxt('./Results/mid_data/7_sample_out3%s_%d.csv' % (epoch, l), gau_sample_out2, delimiter=',')
                np.savetxt('./Results/mid_data/7_sample_out5%s_%d.csv' % (epoch, l), gau_sample_out3, delimiter=',')
                np.savetxt('./Results/mid_data/7_sample_out7%s_%d.csv' % (epoch, l), gau_sample_out4, delimiter=',')

                np.savetxt('./Results/mid_data/8_mol_fea_out2%s_%d.csv' % (epoch, l), mol_fea_out2, delimiter=',')
                np.savetxt('./Results/mid_data/8_mol_fea_out3%s_%d.csv' % (epoch, l), mol_fea_out3, delimiter=',')
                np.savetxt('./Results/mid_data/8_mol_fea_out5%s_%d.csv' % (epoch, l), mol_fea_out5, delimiter=',')
                np.savetxt('./Results/mid_data/8_mol_fea_out7%s_%d.csv' % (epoch, l), mol_fea_out7, delimiter=',')

                np.savetxt('./Results/mid_data/9_mol_cry_caps_out%s_%d.csv' % (epoch, l), mol_cry_caps_out, delimiter=',')
                #np.savetxt('./Results/mid_data/9_gau_alpha_out%s_%d.csv' % (epoch, l), gau_alpha_out, delimiter=',')

                np.savetxt('./Results/mid_data/9_residual_mol_out%s_%d.csv' % (epoch, l), residual_mol_out, delimiter=',')
                #np.savetxt('./Results/mid_data/9_cry_gau_alpha_out%s_%d.csv' % (epoch, l), cry_gau_alpha_out, delimiter=',')

                #np.savetxt('./Results/mid_data/10_gaussian_z_out%s_%d.csv' % (epoch, l), gaussian_z_out, delimiter=',')
                #np.savetxt('./Results/mid_data/10_gaussian_fea_out%s_%d.csv' % (epoch, l), gaussian_fea_out, delimiter=',')


        # delete training and validation data to save memory

        loss = np.array(loss)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        style = dict(size=32, color='gray')
        axs.plot(loss, 'b')
        fig.suptitle('Prediction & Ground truth data')
        # plt.yticks([0, 0.5, 1])
        # plt.xticks([0, 0.5, 1])
        #plt.savefig('./Results/loss_plot' % (epoch, l))
        plt.show()
        np.savetxt('./Results/pre_data/loss_curve.csv', loss, delimiter=',')

            # delete training and validation data to save memory
        del self.train_gen
        del self.val_gen
        del self.model
            #self.predict_model(self.job_dir)

    def predict_model(self, result_dir):

        self.test_generator = CryMat_Gen(self.test_data,
                                         self.test_tar,
                                         batch_size = self.batch_size,
                                         mode = 'test')
        self.build_predictor()

        # load weights
        weights_file = os.path.join(self.job_dir, 'last_model.hdf5')
        self.model.load_weights(weights_file)
        raw_preds = self.model.predict_generator(generator=self.test_generator,
                                                 use_multiprocessing=False)
        preds = reorder_predictions(raw_preds, len(self.test_data),tau = 1)
        np.save(os.path.join(result_dir, 'test_pred.npy'), preds)
        preds_placeholder = tf.placeholder(raw_preds.dtype,
                                           shape=raw_preds.shape)
        metric_vamp = self.vamp.metric_VAMP(None, preds_placeholder)
        metric_vamp2 = self.vamp.metric_VAMP2(None, preds_placeholder)
        with tf.Session() as sess:
            results = sess.run([metric_vamp, metric_vamp2],
                               feed_dict={preds_placeholder: raw_preds})
        np.savetxt(os.path.join(result_dir, 'test_eval.csv'),np.array(results), delimiter=',')

def reorder_predictions(raw_predictions, num_data, tau):
    """
    Reorder raw prediction array

    Parameters
    ----------
    raw_predictions: shape (num_data * (F - tau), num_atom, 2 * n_classes)
    predictions: shape (num_data, F, num_atom, n_classes)
    """
    if (raw_predictions.shape[0] % num_data != 0 or
            len(raw_predictions.shape) != 3 or
            raw_predictions.shape[2] % 2 != 0):
        raise ValueError('Bad format!')
    n_classes = raw_predictions.shape[2] // 2
    num_atom = raw_predictions.shape[1]
    raw_predictions = raw_predictions.reshape(num_data, -1, num_atom,
                                              n_classes * 2)
    assert np.allclose(raw_predictions[:, tau:, :, :n_classes],
                       raw_predictions[:, :-tau, :, n_classes:])
    predictions = np.concatenate([raw_predictions[:, :, :, :n_classes],
                                  raw_predictions[:, -tau:, :, n_classes:]],
                                 axis=1)
    return predictions


def load_keras_optimizer(model, filepath):
    """
    Load the state of optimizer.

    Parameters
    ----------
    optimizer: keras optimizer
      the optimizer for loading the state
    filepath: str
      the path of the pickle (.pkl) file that saves the optimizer state
    """
    with open(filepath, 'rb') as f:
        weight_values = pickle.load(f)
    model._make_train_function()
    model.optimizer.set_weights(weight_values)








