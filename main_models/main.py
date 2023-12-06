"""
This is the capsule spectral model for PDOS property prediction and material generation.
The cs-model consist of two part:
            1. Predictor: Predict property based on material representation
               and multi-dimensional descriptors;

            2. Designer: Inverse generate material samples with fixed representation
               based on learned params by predictor

Starting time: 2022.05.03
"""

from __future__ import print_function
from __future__ import division
from absl import flags
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
import numpy as np
import json
import gzip

from cluster_graph_process.crystal import CrystalGraph
from models.forward_predictor import Forward_predictor
from models.multi_atom_cluster_representation import MatRep
from models import cap_enc
from cluster_graph_process.graph import GaussianDistance
from Configs import *

# GPU SETTING
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict Tensorflow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=46666)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


crystal_graph = CrystalGraph(
    bond_converter = GaussianDistance(centers=np.linspace(0, 6, 100),
                                      width=0.5), cutoff=6.0)
# DATA LOADING
with open('/data1/1-PDOS_Case/1-datas/4-DATA/phdos_e3nn_len51max1000_fwin101ord3.pkl', 'rb') as f:
    data_dict = pickle.load(f)


material_id = data_dict['material_id']
crystal_cif = data_dict['cif']
phfre = data_dict['phfre']
phdos = data_dict['phdos']
phfre_gt = data_dict['phfre_gt']
phdos_gt = data_dict['phdos_gt']

print('---NUMBER OF DATAs ARE:---', len(phdos))
print('---ALL DATA ARE SUCCESSFULLY PREPARED---')


def main(_):
    FLAGS = tf.compat.v1.app.flags.FLAGS  # pylint: disable=invalid-name,redefined-outer-name
    config = FLAGS
    FLAGS.__dict__['config'] = config

    model_dict = cap_enc.get(FLAGS)
    cap_model = model_dict.model

    if FLAGS.use_gpu:
        # added to control the gpu memory
        with tf.compat.v1.Session() as sess:
            pre_model = Forward_predictor(sess, MatRep_model=MatRep, Cap_model=cap_model,
                                          cry_graph=crystal_graph, str_data=crystal_cif,
                                          bg_data=phdos, batch_size=FLAGS.batch_size,
                                          learning_rate=FLAGS.lr, n_epoch=FLAGS.training_epoch)
            if FLAGS.phase == 'train':
                pre_model.train_model()
            elif FLAGS.phase == 'test':
                pre_model.predict_model(FLAGS.job_dir)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.compat.v1.Session() as sess:
            pre_model = Forward_predictor(sess, MatRep_model=MatRep, Cap_model=cap_model,
                                          cry_graph=crystal_graph, str_data=crystal_cif,
                                          bg_data=phdos, batch_size=FLAGS.batch_size,
                                          learning_rate=FLAGS.lr, n_epoch=FLAGS.training_epoch)
            if FLAGS.phase == 'train':
                pre_model.train_model()
            elif FLAGS.phase == 'test':
                pre_model.predict_model(FLAGS.job_dir)
            else:
                print('[!]Unknown phase')
                exit(0)



if __name__ == '__main__':
    FLAGS = flags.FLAGS
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    tf.compat.v1.app.run()

