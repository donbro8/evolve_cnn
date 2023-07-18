from itertools import product
import numpy as np
from graphs import Node
import yaml

class SearchSpace(Node):

    layer_space = []

    def __init__(self, layers: list) -> None:
        self.layers = {}
        self.local_layers = []
        self.global_layers = []
        layers = layers.copy()
        for layer in layers:
            layer_type = layer.pop("layer_type")

            layer_location = layer.pop("location")
            if layer_location == "local":
                self.local_layers.append(layer_type)
            elif layer_location == "global":
                self.global_layers.append(layer_type)
            else:
                raise ValueError(f"Invalid layer location. Must be one of ['local', 'global']")
            
            id_count = 1
            
            if "parameter_set_type" in layer.keys():
                parameter_set_type = layer.pop("parameter_set_type")
                rounding = layer.pop("rounding")
                for key, value in parameter_set_type.items():
                    if value == "continuous":
                        min_value = np.min(layer[key])
                        max_value = np.max(layer[key])
                        n_values = int((max_value - max_value)*10**rounding + 1)
                        layer[key] = np.linspace(min_value, max_value, n_values)

                include_keys = parameter_set_type.keys()
                    
                layer_permutations = [dict(zip(layer.keys(), values)) for values in product(*[layer[key] if key in include_keys else [value] for key, value in layer.items()])]

                for layer_permutation in layer_permutations:

                    if layer_type in self.layers:
                        self.layers[layer_type].append(layer_permutation)
                    else:
                        self.layers[layer_type] = [layer_permutation]

                    id_count += 1
            else:
                self.layers[layer_type] = [layer]

        self.convert_to_tuple(['pool_size'])

    def __repr__(self) -> str:
        return str(self.layers)

    def get_random_layer_type(self, layer_types: list) -> str:
        return np.random.choice(layer_types)
    
    def get_random_layer_attributes(self, layer_type: str) -> dict:
        if layer_type not in self.layers.keys():
            raise ValueError(f"Invalid layer type. Must be one of {self.layers.keys()}")
        return np.random.choice(self.layers[layer_type])
    
    def assign_node_layer(self, node_type: str, attributes: dict) -> Node:
        existing_nodes = [(node.node_type, node.attributes) for node in SearchSpace.layer_space]
        if (node_type, attributes) not in existing_nodes:
            node = Node(node_type, attributes)
            SearchSpace.layer_space.append(node)
        else:
            index = existing_nodes.index((node_type, attributes))
            node = SearchSpace.layer_space[index].copy()
        return node
    
    def sample_nodes(self, n_nodes: int, layer_types: list) -> list[Node]:
        nodes = []
        for _ in range(n_nodes):
            layer_type = self.get_random_layer_type(layer_types)
            attributes = self.get_random_layer_attributes(layer_type)
            nodes.append(self.assign_node_layer(layer_type, attributes))
        return nodes
    
    def convert_to_tuple(self, tuple_values: list[str]) -> None:
        for layer_type in self.layers.keys():
            for layer in self.layers[layer_type]:
                for key, value in layer.items():
                    if key in tuple_values:
                        layer[key] = tuple([value, value])


    
class LayersLoader:
    @staticmethod
    def load_layers_from_yaml(file_path: str) -> SearchSpace:
        with open(file_path, "r") as f:
            layers = yaml.safe_load(f)

        return SearchSpace(layers)