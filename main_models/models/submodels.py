"""
Functional modules
Operation utilities on lists and arrays
"""
from typing import Callable, Any
from monty.json import MSONable
#import tensorflow.compat.v1.keras.backend as kb
from tensorflow.compat.v1.keras.activations import deserialize, serialize  # noqa
from tensorflow.compat.v1.keras.activations import get as keras_get
from typing import Tuple

from tensorflow.compat.v1.keras.layers import Layer
import numpy as np
from pymatgen.core import Structure, Molecule
from pymatgen.optimization.neighbors import find_points_in_spheres

from models.layers.config import DataType
from cluster_graph_process.typing import StructureOrMolecule
from typing import Union, List
from cluster_graph_process.typing import OptStrOrCallable
from collections import Iterable
import tensorflow.compat.v1 as tf






def get(identifier: OptStrOrCallable = None) -> Callable[..., Any]:
    """
    Get activations by identifier

    Args:
        identifier (str or callable): the identifier of activations

    Returns:
        callable activation

    """
    try:
        return keras_get(identifier)
    except ValueError:
        if isinstance(identifier, str):
            return deserialize(identifier, custom_objects=globals())
    raise ValueError("Could not interpret:", identifier)



class Converter(MSONable):
    """
    Base class for atom or bond converter
    """

    def convert(self, d: Any) -> Any:
        """
        Convert the object d
        Args:
            d (Any): Any object d

        Returns: returned object
        """
        raise NotImplementedError



def get_graphs_within_cutoff(
    structure: StructureOrMolecule, cutoff: float = 5.0, numerical_tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get graph representations from structure within cutoff
    Args:
        structure (pymatgen Structure or molecule)
        cutoff (float): cutoff radius
        numerical_tol (float): numerical tolerance

    Returns:
        center_indices, neighbor_indices, images, distances
    """
    if isinstance(structure, Structure):
        lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
        pbc = np.array([1, 1, 1], dtype=int)
    elif isinstance(structure, Molecule):
        lattice_matrix = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]], dtype=float)
        pbc = np.array([0, 0, 0], dtype=int)
    else:
        raise ValueError("structure type not supported")
    r = float(cutoff)
    cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords, cart_coords, r=r, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
    )
    center_indices = center_indices.astype(DataType.np_int)
    neighbor_indices = neighbor_indices.astype(DataType.np_int)
    images = images.astype(DataType.np_int)
    distances = distances.astype(DataType.np_float)
    exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
    return center_indices[exclude_self], neighbor_indices[exclude_self], images[exclude_self], distances[exclude_self]


def expand_1st(x: np.ndarray) -> np.ndarray:
    """
    Adding an extra first dimension

    Args:
        x: (np.array)
    Returns:
         (np.array)
    """
    return np.expand_dims(x, axis=0)

def to_list(x: Union[Iterable, np.ndarray]) -> List:
    """
    If x is not a list, convert it to list
    """
    if isinstance(x, Iterable):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()  # noqa
    return [x]


class GaussianExpansion(Layer):
    """
    Simple Gaussian expansion.
    A vector of distance [d1, d2, d3, ..., dn] is expanded to a
    matrix of shape [n, m], where m is the number of Gaussian basis centers

    """

    def __init__(self, centers, width, **kwargs):
        """
        Args:
            centers (np.ndarray): Gaussian basis centers
            width (float): width of the Gaussian basis
            **kwargs:
        """
        self.centers = np.array(centers).ravel()
        self.width = width
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        build the layer
        Args:
            input_shape (tuple): tuple of int for the input shape
        """
        self.built = True

    def call(self, inputs, masks=None):
        """
        The core logic function

        Args:
            inputs (tf.Tensor): input distance tensor, with shape [None, n]
            masks (tf.Tensor): bool tensor, not used here
        """
        return tf.math.exp(-((inputs[:, :, None] - self.centers[None, None, :]) ** 2) / self.width ** 2)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape, used in older keras API
        """
        return input_shape[0], input_shape[1], len(self.centers)

    def get_config(self):
        """
        Get layer configurations
        """
        base_config = super().get_config()
        config = {"centers": self.centers.tolist(), "width": self.width}
        return dict(list(base_config.items()) + list(config.items()))


def _repeat(x: tf.Tensor, n: tf.Tensor, axis: int = 1) -> tf.Tensor:
    """
    Given an tensor x (N*M*K), repeat the middle axis (axis=1)
    according to repetition indicator n (M, )
    for example, if M = 3, axis=1, and n = Tensor([3, 1, 2]),
    and the final tensor would have the shape (N*6*3) with the
    first one in M repeated 3 times,
    second 1 time and third 2 times.

     Args:
        x: (3d Tensor) tensor to be augmented
        n: (1d Tensor) number of repetition for each row
        axis: (int) axis for repetition

    Returns:
        (3d Tensor) tensor after repetition
    """
    # get maximum repeat length in x
    assert len(n.shape) == 1
    maxlen = tf.reduce_max(input_tensor=n)
    x_shape = tf.shape(input=x)
    x_dim = len(x.shape)
    # create a range with the length of x
    shape = [1] * (x_dim + 1)
    shape[axis + 1] = maxlen
    # tile it to the maximum repeat length, it should be of shape
    # [xlen, maxlen] now
    x_tiled = tf.tile(tf.expand_dims(x, axis + 1), tf.stack(shape))

    new_shape = tf.unstack(x_shape)
    new_shape[axis] = -1
    new_shape[-1] = x.shape[-1]
    x_tiled = tf.reshape(x_tiled, new_shape)
    # create a sequence mask using x
    # this will create a boolean matrix of shape [xlen, maxlen]
    # where result[i,j] is true if j < x[i].
    mask = tf.sequence_mask(n, maxlen)
    mask = tf.reshape(mask, (-1,))
    # mask the elements based on the sequence mask
    return tf.boolean_mask(tensor=x_tiled, mask=mask, axis=axis)

def repeat_with_index(x: tf.Tensor, index: tf.Tensor, axis: int = 1):
    """
    Given an tensor x (N*M*K), repeat the middle axis (axis=1)
    according to the index tensor index (G, )
    for example, if axis=1 and n = Tensor([0, 0, 0, 1, 2, 2])
    then M = 3 (3 unique values),
    and the final tensor would have the shape (N*6*3) with the
    first one in M repeated 3 times,
    second 1 time and third 2 times.

     Args:
        x: (3d Tensor) tensor to be augmented
        index: (1d Tensor) repetition tensor
        axis: (int) axis for repetition
    Returns:
        (3d Tensor) tensor after repetition
    """
    index = tf.reshape(index, (-1,))
    _, _, n = tf.unique_with_counts(index)
    return _repeat(x, n, axis)





