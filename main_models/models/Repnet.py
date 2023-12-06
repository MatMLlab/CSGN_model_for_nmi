"""
Megnet graph layer implementation
"""
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as kb
from tensorflow.compat.v1.keras.layers import Dense

from models.layers.base import GraphNetworkLayer
from .submodels import repeat_with_index
from models.layers.stoi_tansformer import Stoi_Rep

class MatRepLayer(GraphNetworkLayer):


    def __init__(
        self,
        units_v,
        units_e,
        units_u,
        pool_method="mean",
        activation = tf.nn.selu,
        use_bias=True,
        kernel_initializer = "glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):

        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.units_v = units_v
        self.units_e = units_e
        self.units_u = units_u
        self.pool_method = pool_method

        model_params = {
            "n_target": 1,
            "elem_emb_len": 200, #
            "elem_fea_len": 256,
            "n_graph": 3,
            "elem_heads": 3,
            "elem_gate": [256],
            "elem_msg": [256],
            "cry_heads": 3,
            "cry_gate": [256],
            "cry_msg": [256],
            "out_hidden": [1024, 512, 256, 128, 64],
        }

        self.Stoi_Rep = Stoi_Rep(**model_params)

        if pool_method == "mean":
            self.reduce_method = tf.reduce_mean
            self.seg_method = tf.math.segment_mean
        elif pool_method == "sum":
            self.reduce_method = tf.reduce_sum
            self.seg_method = tf.math.segment_sum
        else:
            raise ValueError("Pool method: " + pool_method + " not understood!")

    def build(self, input_shapes):

        vdim = int(input_shapes[6][2])
        edim = int(input_shapes[7][2])
        udim = int(input_shapes[8][2])


        with kb.name_scope(self.name):



            with kb.name_scope("phi_e"):
                #e_shapes = [2 * vdim + edim + udim] + self.units_e
                e_shapes = [2 * vdim + 2*edim] + self.units_e
                e_shapes = list(zip(e_shapes[:-1], e_shapes[1:]))
                self.phi_e_weights = [
                    self.add_weight(
                        shape = i,
                        name = "weight_e_%d" % j,
                        regularizer = self.kernel_regularizer,
                        constraint = self.kernel_constraint,
                    )
                    for j, i in enumerate(e_shapes)
                ]
                if self.use_bias:
                    self.phi_e_biases = [
                        self.add_weight(
                            shape=(i[-1],),
                            initializer=self.bias_initializer,
                            name="bias_e_%d" % j,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                        )
                        for j, i in enumerate(e_shapes)
                    ]
                else:
                    self.phi_e_biases = None

            with kb.name_scope("phi_v"):
                #v_shapes = [edim + vdim + udim] + self.units_v
                v_shapes = [edim + vdim] + self.units_v
                v_shapes = list(zip(v_shapes[:-1], v_shapes[1:]))

                self.phi_v_weights = [
                    self.add_weight(
                        shape=i,
                        name="weight_v_%d" % j,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint)
                    for j, i in enumerate(v_shapes)]

                if self.use_bias:
                    self.phi_v_biases = [
                        self.add_weight(
                            shape=(i[-1],),
                            name="bias_v_%d" % j,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                        )
                        for j, i in enumerate(v_shapes)
                    ]
                else:
                    self.phi_v_biases = None


            with kb.name_scope("phi_u"):
                u_shapes = [udim] + self.units_u
                u_shapes = list(zip(u_shapes[:-1], u_shapes[1:]))
                self.phi_u_weights = [
                    self.add_weight(
                        shape=i,
                        name="weight_u_%d" % j,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                    )
                    for j, i in enumerate(u_shapes)
                ]
                if self.use_bias:
                    self.phi_u_biases = [
                        self.add_weight(
                            shape=(i[-1],),
                            name="bias_u_%d" % j,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                        )
                        for j, i in enumerate(u_shapes)
                    ]
                else:
                    self.phi_u_biases = None



        self.built = True

    def compute_output_shape(self, input_shape):

        node_feature_shape = input_shape[6]
        edge_feature_shape = input_shape[7]
        #state_feature_shape = input_shape[8]
        output_shape = [
            (node_feature_shape[0], node_feature_shape[1], self.units_v[-1]),
            (edge_feature_shape[0], edge_feature_shape[1], self.units_e[-1]),
            #(state_feature_shape[0], state_feature_shape[1], self.units_u[-1]),
        ]
        return output_shape

    def phi_u_h(self, inputs):

        com_w, ele_idx, atom_fea, x_fea, x_nbr, ele_index, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs


        crys_fea = self.Stoi_Rep(
            com_w, atom_fea, x_fea, x_nbr, ele_idx, ele_index
        )

        return crys_fea

    def phi_e(self, inputs):
        # bond transformer

        com_w, ele_idx, atom_fea, x_fea, x_nbr,  ele_index, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs

        index1 = tf.reshape(index1, (-1,))
        index2 = tf.reshape(index2, (-1,))
        fs = tf.gather(nodes, index1, axis=1)
        fr = tf.gather(nodes, index2, axis=1)
        concate_node = tf.concat([fs, fr], -1)
        u_expand = repeat_with_index(u, gbond, axis=1)
        concated = tf.concat([concate_node, u_expand, edges], -1) # (1, ?, 16 * 16)
        #concated_1 = tf.squeeze(concated, 0)


        out = self._mlp(concated, self.phi_e_weights, self.phi_e_biases)
        out = tf.nn.softplus(out)

        return out

    def phi_v(self, b_ei_p, inputs):
        # atom transformer

        com_w, ele_idx, atom_fea, x_fea, x_nbr,  ele_index, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs

        #u_expand = repeat_with_index(u, gbond, axis=1)
        concated = tf.concat([b_ei_p, nodes], -1) # (1, ?, 16 * 13)
        out = self._mlp(concated, self.phi_v_weights, self.phi_v_biases)
        out = tf.nn.softplus(out)

        return out


    def phi_u(self, inputs):
        # state transformer

        #concated = tf.concat([b_e_p, b_v_p, inputs[8]], -1) # (1, ?, 16 * 18)


        out = self._mlp(inputs[8], self.phi_u_weights, self.phi_u_biases)
        return out


    def rho_e_v(self, e_p, inputs):

        com_w, ele_idx, atom_fea, x_fea, x_nbr,  ele_index, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs

        index1 = tf.reshape(index1, (-1,))
        out = tf.expand_dims(self.seg_method(tf.squeeze(e_p), index1), axis=0)

        return out

    def rho_e_u(self, e_p, inputs):

        com_w, ele_idx, atom_fea, x_fea, x_nbr,  ele_index, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs

        gbond = tf.reshape(gbond, (-1,))
        out = tf.expand_dims(self.seg_method(tf.squeeze(e_p), gbond), axis=0)
        return out

    def rho_v_u(self, v_p, inputs):

        com_w, ele_idx, atom_fea, x_fea, x_nbr,  ele_index, \
        nodes, edges, u, index1, index2, gnode, gbond = inputs

        gbond = tf.reshape(gbond, (-1,))
        #index1 = tf.reshape(index1, (-1,))
        v_p = Dense(64)(v_p)
        out =tf.expand_dims(self.seg_method(tf.squeeze(v_p, axis=0), gbond), axis=0)
        #out = tf.expand_dims(self.seg_method(tf.squeeze(out), index1), axis=0)
        return out

    def _mlp(self, input_, weights, biases):
        if biases is None:
            biases = [0] * len(weights)
        act = input_
        for w, b in zip(weights, biases):
            output = kb.dot(act, w) + b
            act = self.activation(output)
        return output

    def get_config(self):

        config = {
            "units_e": self.units_e,
            "units_v": self.units_v,
            "units_u": self.units_u,
            "pool_method": self.pool_method,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
