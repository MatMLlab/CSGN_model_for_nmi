"""
This is the data process for material data and all data solution.
"""

from pymatgen.core import Structure
from copy import deepcopy
from typing import Dict, List, Union
import numpy as np
from cluster_graph_process.graph import GraphBatchDistanceConvert, GraphBatchGenerator, StructureGraph
from models.layers.preprocessing import DummyScaler, Scaler





def atom_fillna(data, max):
    # Get lengths of each row of data
    lens = len(data)
    #data = np.array(data)

    # Mask of valid places in each row
    pad = [0] * (max - lens)

    out = data + pad
    return out

def bond_fillna(data, max):
    # Get lengths of each row of data
    lens = data.shape[0]

    # Mask of valid places in each row
    pad = np.zeros((max - lens), dtype=float)

    out = np.hstack((data, pad))
    return out

def data_pro(bandgap_data,  structure_data, crystal_graph):

    targets = np.array(bandgap_data)

    graphs_valid = []
    targets_valid = []
    mat_ids = []
    n = 0

    for i, (s, t) in enumerate(zip(structure_data, targets)):
        # print(n)
        structures = Structure.from_str("\n".join(s), "CIF")
        graph = crystal_graph.convert(structures)
        graphs_valid.append(graph)
        targets_valid.append(t)
        mat_ids.append(i)
        n += 1

    #targets_valid = np.array(targets_valid)
    #max_tar = np.max(targets_valid)
    #min_tar = np.min(targets_valid)
    #targets_valid = (targets_valid - min_tar) / (max_tar - min_tar)

    final_graphs = {i: j for i, j in zip(mat_ids, graphs_valid)}
    final_targets = {i: j for i, j in zip(mat_ids, targets_valid)}

    from sklearn.model_selection import train_test_split

    train_ids, test_ids = train_test_split(mat_ids, train_size=0.9, test_size=0.1,
                                           random_state=66,
                                           shuffle=True)

    train_ids1 = np.array(train_ids)
    np.savetxt('./train_ids1.csv', train_ids1, delimiter=',')
    test_ids = np.array(test_ids)
    np.savetxt('./test_ids1.csv', test_ids, delimiter=',')

    print("Train, val and test data sizes are ", len(train_ids), len(test_ids))


    def get_graphs_targets(ids):
        """
        Get graphs and targets list from the ids

        Args:
            ids (List): list of ids

        Returns:
            list of graphs and list of target values
        """
        ids = [i for i in ids if i in final_graphs]
        return [final_graphs[i] for i in ids], [final_targets[i] for i in ids]

    train_graphs, train_targets = get_graphs_targets(train_ids)

    test_graphs, test_targets = get_graphs_targets(test_ids)

    return train_graphs, train_targets, test_graphs, test_targets



class CryMat_Gen(object):
    """
    Make the generator for keras.fit_generator
    """
    def __init__(self,
                 train_graphs: List[Dict] = None,
                 train_targets: List[float] = None, batch_size = 16,
                 sample_weights: List[float] = None, mode = 'train',
                 val_graphs: List[Dict] = None,
                 val_targets: List[float] = None,
                 target_scaler: Scaler = DummyScaler(),
                 graph_converter = StructureGraph,
                 scrub_failed_structures: bool = False,
                 ):
        self.train_graphs = train_graphs
        self.train_targets = train_targets
        self.batch_size = batch_size
        self.sample_weights = sample_weights
        self.mode = mode
        self.val_graphs = val_graphs
        self.val_targets = val_targets
        self.target_scaler = target_scaler
        self.graph_converter = graph_converter
        self.scrub_failed_structures = scrub_failed_structures


        #try:
        #    if self.mode == 'train':
        #        self.load_train_data()
        #    else:
        #        self.load_val_data()
        #except KeyError:
        #    raise KeyError('Data loader failed: choose `other mode` to load' ' data preprocessed.')


    def load_train_data(self):
        train_nb_atoms = [len(i["atom"]) for i in self.train_graphs]

        train_targets = [self.target_scaler.transform(i, j) for i, j in zip(self.train_targets, train_nb_atoms)]

        train_inputs = self.graph_converter.get_flat_data(self.train_graphs, train_targets)

        train_generator = self._create_generator(*train_inputs,
                                                 sample_weights=self.sample_weights,
                                                 batch_size=self.batch_size)
        steps_per_train = int(np.ceil(len(self.train_graphs) / self.batch_size))


        return train_generator, steps_per_train

    def load_val_data(self):
        val_nb_atoms = [len(i["atom"]) for i in self.val_graphs]

        val_targets = [self.target_scaler.transform(i, j) for i, j in zip(self.val_targets, val_nb_atoms)]

        val_inputs = self.graph_converter.get_flat_data(self.val_graphs, val_targets)

        val_generator = self._create_generator(*val_inputs,
                                                 sample_weights=self.sample_weights,
                                                 batch_size=self.batch_size)
        steps_per_val = int(np.ceil(len(self.val_graphs) / self.batch_size))

        return val_generator, steps_per_val

    def _create_generator(self, *args, **kwargs) -> Union[GraphBatchDistanceConvert, GraphBatchGenerator]:
        if hasattr(self.graph_converter, "bond_converter"):
            kwargs.update({"distance_converter": self.graph_converter.bond_converter})
            return GraphBatchDistanceConvert(*args, **kwargs)
        return GraphBatchGenerator(*args, **kwargs)









