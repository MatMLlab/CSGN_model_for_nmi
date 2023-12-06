"""
Capsule Encoder for material representation
"""
from absl import flags
import sonnet as snt
from models.layers import primary
from .inferencer_1 import SetTransformer
from .spectrum_generator import ClusterCapsule
from .spectrum_generator import MaterialCapsule
import tensorflow.compat.v1 as tf
from monty.collections import AttrDict


def get(config):
  """Builds the model."""
  model = make_capsule(config)


  lr = config.lr
  if config.use_lr_schedule:
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(
        global_step=global_step,
        learning_rate=lr,
        decay_steps=1e4,
        decay_rate=.96)

  eps = 1e-2 / float(config.batch_size)  ** 2
  opt = tf.train.RMSPropOptimizer(config.lr, momentum=.9, epsilon=eps)

  return AttrDict(model=model, opt=opt, lr=config.lr)




def make_capsule(config):
    """Build Material Capsules"""

    img_size = [51,1]
    #img_size = config.material_size
    template_size = [config.template_size] * 2

    cnn_encoder = snt.nets.ConvNet2D(
        output_channels = [64] * 4,
        kernel_shapes = [3],
        strides = [1, 1, 1, 1],
        paddings = [snt.SAME],
        activate_final = True,
        activation = tf.nn.softplus)

    part_encoder = primary.CapsuleMaterialEncoder(
        cnn_encoder,
        config.n_part_caps,
        config.n_part_caps_dims,
        n_features = config.n_part_special_features,
        similarity_transform = False,
        encoder_type ='linear')

    part_decoder = primary.TemplateBasedMaterialDecoder(
        output_size = img_size,
        template_size = template_size,
        n_channels = config.n_channels,
        learn_output_scale = False,
        colorize_templates = config.colorize_templates,
        use_alpha_channel = config.use_alpha_channel,
        template_nonlin = config.template_nonlin,
        color_nonlin = config.color_nonlin,
    )

    obj_encoder = SetTransformer(
        n_layers = 3,
        n_heads = 1,
        n_dims = 512,
        n_output_dims = 256,
        n_outputs = config.n_obj_caps,
        layer_norm = True,
        dropout_rate = 0.15)

    obj_decoder = MaterialCapsule(
        config.n_obj_caps,
        2,
        config.n_part_caps,
        n_caps_params = config.n_obj_caps_params,
        n_hiddens = 128,
        learn_vote_scale = True,
        deformations = True,
        noise_type = 'uniform',
        noise_scale = 4.,
        similarity_transform = False)

    model = ClusterCapsule(
        primary_encoder = part_encoder,
        primary_decoder = part_decoder,
        encoder = obj_encoder,
        decoder = obj_decoder,
        stop_grad_caps_inpt = True,
        stop_grad_caps_target = True,
        input_key = 'image',
        label_key = 'label',
        n_classes = 10)

    return model

