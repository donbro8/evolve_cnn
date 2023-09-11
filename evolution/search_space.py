import itertools
import numpy as np
import uuid
from math import fsum

class SearchSpace:

    sampled_instances = set()

    def __init__(self,
            layer_config: dict = {
                'Conv2D':{
                    'filters':[8,16,32], 
                    'kernel_size':[(1,1),(3,3)], 
                    'activation':['relu'], 
                    'padding':['same']
                    },
                'AveragePooling2D':{
                    'pool_size':[(3,3)],
                    'strides':[(1,1)],
                    'padding':['same']
                    }
            }
        ) -> None:
        search_space = []
        for layer in layer_config.keys():
            combinations = list(itertools.product(*layer_config[layer].values()))
            search_space += [(layer, dict(zip(layer_config[layer].keys(), combination))) for combination in combinations]
        self.search_space = search_space
        self.layer_types = list(layer_config.keys())
        self.distributed_search_space = {layer_type:[layer for layer in self.search_space if layer[0] == layer_type] for layer_type in self.layer_types}
        print(f"Search space contains {len(self.search_space)} possible instances.")


    def sample_from_search_space(self, n_samples: int = 1, sample_probabilities = None) -> list[tuple[str, dict]]:
        
        samples = []

        if sample_probabilities is None:
            print("Warning: No sample probabilities provided. Sampling uniformly from search layers...")
            sample_probabilities = np.array([1]*len(self.layer_types))/len(self.layer_types)

        else:
            if len(sample_probabilities) != len(self.layer_types):
                raise ValueError("Sample probabilities must be the same length as the number of layer types in the search space.")

            elif fsum(sample_probabilities) != 1.0:
                print("Warning: Sample probabilities must sum to 1.0. Normalising sample probabilities...")
                sample_probabilities = np.array(sample_probabilities)/np.sum(sample_probabilities)
        
        for _ in range(n_samples):
            sample_type = np.random.choice(self.layer_types, p = sample_probabilities)
            sub_search_space = self.distributed_search_space[sample_type]
            sample = sub_search_space[np.random.randint(len(sub_search_space))]
            sample = (sample[0] + '_' + str(len(self.__class__.sampled_instances) + 1) + '_' + str(uuid.uuid4())[:4], sample[1])
            self.__class__.sampled_instances.add(sample[0])
            samples.append(sample)
        return samples
    

    def get_base_node(self, node: tuple[str, dict]) -> tuple[str, dict]:
        return node[0].split('_')[0], node[1]

    def get_node_display_name(self, node: tuple[str, dict]) -> str:
        node_type = node[0].split('_')[0]
        return '_'.join([node_type, str(node[1])])