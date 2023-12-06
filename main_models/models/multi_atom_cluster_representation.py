"""
Representation of atom, bond, and features for material .
"""

from typing import Callable
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Dense, Add, Dropout, Lambda, \
    BatchNormalization, Permute, Activation, Concatenate, Reshape, LSTM, Multiply
from tensorflow.compat.v1.keras.regularizers import l2
import sonnet as snt
from models.layers.set2set import Set2Set
from .Repnet import MatRepLayer
from .inferencer_1 import QKVAttention


def MatRep(x_com_w, x_ele_idx, atom_fea, x_fea, x_nbr,ele_index,
           x1_, x2_, x3_, x4, x5, x6, x7, # x2_: bond, x4: bond_index_1, x5: bond_index_2
           x2_1, x4_1, x5_1, x7_1, # cluster_2
           x2_2, x4_2, x5_2, x7_2,# cluster_3
           x2_3, x4_3, x5_3, x7_3,# cluster_4
           nblocks: int = 3,
           n1: int = 128,
           n2: int = 64,
           n3: int = 128,
           n4: int = 80,
           npass: int = 3,
           act: Callable = tf.nn.softplus,
           l2_coef: float = None,
           dropout: float = None,
           dropout_on_predict: bool = True,
           use_bn: bool = True,
           n_conv: int = 3,
           **kwargs):
    """

    Args:
        x_com_w: element weight in crystals
        x_ele_idx: element index of crystals
        atom_fea: atom embedding vectors
        x_fea: atom index of crystals
        x_nbr: neighbor atom index correspond to atoms
        x1_: elements of crystal
        x2_: bond information
        x3_: state information
        x4: atom index related by bonds
        x5: neighbor atom index related by bonds
        x6: atom source of crystals (x1_)
        x7: bond source of crystals (x2_)

        x81:
        x91:
        x82:
        x92:

        x21:
        x41:
        x51:
        x71:
        real_pdos: ground truth data
        nblocks: iter numbers
        n1, n2, n3, n4: units
        npass:
        act: activation funtion
        l2_coef:
        dropout:
        dropout_on_predict:
        use_bn:
        n_conv:
        **kwargs:

    Returns: N-Multi-body Clusters of Crystals

    """

    # Get the setting for the training kwarg of Dropout
    dropout_training = True if dropout_on_predict else None

    if l2_coef is not None:
        reg = l2(l2_coef)
    else:
        reg = None

    # two feedforward layers
    def nn(x, n_hiddens=None, name_prefix=None):
        if name_prefix is None:
            name_prefix = "FF"
        out = x
        for k, i in enumerate(n_hiddens):
            out = Dense(i, activation = act,
                        name = "%s_%d" % (name_prefix, k))(out)
        return out

    def one_block(x_com_w, x_ele_idx, atom_fea, x_fea, x_nbr,ele_index,
                  atom, bond, u, index1, index2, gatom, gbond,
                  has_ff=True, block_index=0, n=0):
        if has_ff:
            atom = nn(atom, n_hiddens=[n1, n2], name_prefix="block_%d_atom_ff" % block_index)
            bond = nn(bond, n_hiddens=[n1, n2], name_prefix="block_%d_bond_ff" % block_index)
            u = nn(u, n_hiddens=[n1, n2], name_prefix="block_%d_state_ff" % block_index)
        else:
            atom = atom
            bond = bond
            u = u
        out = MatRepLayer([n1, n1, n2],
                          [n1, n1, n2],
                          [n1, n1, n2],
                          pool_method="mean",
                          activation=tf.nn.relu,
                          name="MatRep_%d" % (block_index + n))([x_com_w, x_ele_idx, atom_fea, x_fea, x_nbr,ele_index,
                                                                   atom, bond, u, index1, index2, gatom, gbond])

        x1_temp = out[0]
        x2_temp = out[1]
        x3_temp = out[2]
        comp_temp = out[3]
        bond_2 = out[4]

        if dropout:
            x1_temp = Dropout(dropout, name="dropout_atom_%d" % block_index)(x1_temp, training=dropout_training)
            x2_temp = Dropout(dropout, name="dropout_bond_%d" % block_index)(x2_temp, training=dropout_training)
            x3_temp = Dropout(dropout, name="dropout_state_%d" % block_index)(x3_temp, training=dropout_training)
        return x1_temp, x2_temp, x3_temp, comp_temp, bond_2


    #atom chemical environment
    def gdy_encoder(final_vec):

        inps = final_vec
        # data process and reshape the data
        nbr_att_model = QKVAttention()
        atom_nbr_fea = nbr_att_model(inps, inps, inps)

        full_fea = tf.concat([atom_nbr_fea, inps], -1)
        nbr_vec = snt.BatchApply(snt.Linear(256, name='che_out'))(full_fea)


        return nbr_vec

    x1_ = nn(x1_, n_hiddens=[n2], name_prefix="preblock_atom") #(1, ?, 64)  --> (1, ?, 64)
    x2_ = nn(x2_, n_hiddens=[n2], name_prefix="preblock_bond") #(1, ?, 100) --> (1, ?, 64)
    x3_ = nn(x3_, n_hiddens=[n2], name_prefix="preblock_mass") #(1, ?, 64) --> (1, ?, 64)
    x2_1 = nn(x2_1, n_hiddens=[n2], name_prefix="preblock_bond_1") #(1, ?, 100) --> (1, ?, 64)
    x2_2 = nn(x2_2, n_hiddens=[n2], name_prefix="preblock_bond_2") #(1, ?, 100) --> (1, ?, 64)
    x2_3 = nn(x2_3, n_hiddens=[n2], name_prefix="preblock_bond_3") #(1, ?, 100) --> (1, ?, 64)

    #comps = 0
    def atom_che(x_com_w, x_ele_idx, atom_fea, x_fea, x_nbr,ele_index,
                 node, bond, state, index1, index2, gatom, gbond, n=1):


        for i in range(3):
            if i == 0:
                has_ff = False
            else:
                has_ff = True
            node_1 = node
            bond_1 = bond
            state_1 = state
            node_1, bond_1, state_1, x_comp, bond_2 = one_block(x_com_w, x_ele_idx, atom_fea, x_fea, x_nbr,ele_index,
                                               node_1, bond_1,state_1, index1, index2, gatom, gbond, has_ff,
                                               block_index=int(i + n), n=n)


            node_1 = Dense(64)(node_1)
            # skip connection
            node = Add(name="block_add_atom_%d" % int(i + n))([node, 1 * node_1])
            bond = Add(name="block_add_bond_%d" % int(i + n))([bond, 1 * bond_1])
            state = Add(name="block_add_mass_%d" % int(i + n))([state, 1 * state_1])
            x_comp += x_comp

        return node, bond_1, state, x_comp

    def atom2mol(inp, ind):

        ind = tf.reshape(ind, (-1,))
        # maxes = tf.math.segment_max(inp[0], ind)

        inp = tf.squeeze(inp, 0)
        out = tf.math.segment_mean(inp, ind)

        return out


    # Multi-body clusters (MBC) index
    # MBC construction
    # 2-body clusters
    atom1_, bond1_, u1_, x_comp = atom_che(x_com_w, x_ele_idx, atom_fea, x_fea, x_nbr, ele_index,
                                           x1_, x2_, x3_, x4, x5, x6, x7, n=0)
    # 3-body clusters
    atom1_1, bond1_1, u1_1, x_comp_1 = atom_che(x_com_w, x_ele_idx, atom_fea, x_fea, x_nbr, ele_index,
                                           x1_, x2_1, x3_, x4_1, x5_1, x6, x7_1, n=3)
    # 5-body clusters
    atom1_2, bond1_2, u1_2, x_comp_2 = atom_che(x_com_w, x_ele_idx, atom_fea, x_fea, x_nbr, ele_index,
                                           x1_, x2_2, x3_, x4_2, x5_2, x6, x7_2, n=6)
    # 7-body clusters
    atom1_3, bond1_3, u1_3, x_comp_3 = atom_che(x_com_w, x_ele_idx, atom_fea, x_fea, x_nbr, ele_index,
                                           x1_, x2_3, x3_, x4_3, x5_3, x6, x7_3, n=9)

    node_vec = Set2Set(T = 3, n_hidden=128, kernel_constraint=None, name = "set2set_node")([atom1_,x6])
    bond_vec = Set2Set(T=3, n_hidden=128, kernel_constraint=None, name="set2set_bond")([bond1_, x7])
    u_vec = Set2Set(T=3, n_hidden=128, kernel_constraint=None, name="set2set_mass")([u1_, x6])

    node_vec_1 = Set2Set(T=3, n_hidden=128, kernel_constraint=None, name="set2set_node_1")([atom1_1, x6])
    bond_vec_1 = Set2Set(T=3, n_hidden=128, kernel_constraint=None, name="set2set_bond_1")([bond1_1, x7_1])
    u1_vec = Set2Set(T=3, n_hidden=128, kernel_constraint=None, name="set2set_mass_1")([u1_1, x6])

    node_vec_2 = Set2Set(T=3, n_hidden=128, kernel_constraint=None, name="set2set_node_2")([atom1_2, x6])
    bond_vec_2 = Set2Set(T=3, n_hidden=128, kernel_constraint=None, name="set2set_bond_2")([bond1_2, x7_2])
    u2_vec = Set2Set(T=3, n_hidden=128, kernel_constraint=None, name="set2set_mass_2")([u1_2, x6])

    node_vec_3 = Set2Set(T=3, n_hidden=128, kernel_constraint=None, name="set2set_node_3")([atom1_3, x6])
    bond_vec_3 = Set2Set(T=3, n_hidden=128, kernel_constraint=None, name="set2set_bond_3")([bond1_3, x7_3])
    u3_vec = Set2Set(T=3, n_hidden=128, kernel_constraint=None, name="set2set_mass_3")([u1_3, x6])

    cry_vec = Concatenate(axis=-1)([node_vec, bond_vec,u_vec,
                                    node_vec_1, bond_vec_1,u1_vec,
                                    node_vec_2,bond_vec_2,u2_vec,
                                    node_vec_3,bond_vec_3,u3_vec])
    cry_vec = tf.nn.dropout(tf.nn.softplus(cry_vec), 0.9)

    bond1_ = atom2mol(bond1_, x4)
    bond1_1 = atom2mol(bond1_1, x4_1)
    bond1_2 = atom2mol(bond1_2, x4_2)
    bond1_3 = atom2mol(bond1_3, x4_3)

    x_comp = BatchNormalization()(x_comp, training=True)
    x_comp = tf.nn.dropout(tf.nn.softplus(x_comp), 0.9)
    x_comp = tf.expand_dims(x_comp, 0)


    return atom1_, bond1_, atom1_1, bond1_1, atom1_2, bond1_2, atom1_3, bond1_3, x_comp, cry_vec, u1_, u1_1, u1_2, u1_3 # atom, bond, state, element_weight

def _concat_nbrs(inps):
    """
    Concatenate neighbor features based on graph structure into a full
    feature vector

    B: Batch size
    N: Number of atoms in each crystal
    M: Max number of neighbors
    atom_fea_len: the length of atom features
    bond_fea_len: the length of bond features

    Parameters
    ----------
    atom_fea: (B, N, atom_fea_len).
    bond_fea: (B, N, M, bond_fea_len)
    nbr_list: (B, N, M)

    Returns
    -------
    total_fea: (B, N, M, 2 * atom_fea_len + bond_fea_len)
    """


    # extraction of atom correlation by nbr_attention
    nbr_att_model = QKVAttention()
    atom_nbr_fea = nbr_att_model(inps, inps, inps)

    #atom_self_fea = tf.tile(tf.expand_dims(atom_fea, 2), [1, 1, 1, 1])
    full_fea = tf.concat([atom_nbr_fea, inps], -1)

    return full_fea

def _concat_nbrs_output_shape(input_shapes):
    """
    Needed since the inference of output shape fail in Keras for non-float tensors.
    """
    B = input_shapes[0]
    N = input_shapes[1]
    M = input_shapes[2]

    return (B, N, 2*M)