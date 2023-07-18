from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, MaxPooling2D, AveragePooling2D, SpatialDropout2D, GlobalAveragePooling2D, Flatten, Lambda, Concatenate
from tensorflow.keras.models import Model
from graphs import Graph
import numpy as np

class BuildLayer(Layer):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.graph = graph
        self.params = 0
        for node in self.graph.nodes:
            index = self.graph.nodes.index(node)
            if node.node_type == 'Conv2D':
                self.graph.layers[index] = Conv2D(**node.attributes)
            elif node.node_type == 'BatchNormalization':
                self.graph.layers[index] = BatchNormalization()
            elif node.node_type == 'Dense':
                self.graph.layers[index] = Dense(**node.attributes)
            elif node.node_type == 'MaxPooling2D':
                self.graph.layers[index] = MaxPooling2D(**node.attributes)
            elif node.node_type == 'AveragePooling2D':
                self.graph.layers[index] = AveragePooling2D(**node.attributes)
            elif node.node_type == 'SpatialDropout2D':
                self.graph.layers[index] = SpatialDropout2D(**node.attributes)
            elif node.node_type == 'GlobalAveragePooling2D':
                self.graph.layers[index] = GlobalAveragePooling2D()
            elif node.node_type == 'Flatten':
                self.graph.layers[index] = Flatten()
            elif node.node_type == 'Identity':
                self.graph.layers[index] = Lambda(lambda x: x)
            else:
                raise ValueError(f"Invalid node type {node.node_type} - Node type must be one of 'Conv2D', 'BatchNormalization', 'Dense', 'MaxPooling2D', 'AveragePooling2D', 'SpatialDropout2D', 'GlobalAveragePooling2D', 'Flatten', 'Identity'")
            
    def get_layers(self):
        layers = []
        for node in self.graph.order_nodes:
            layers.append(self.graph.layers[self.graph.nodes.index(node)])
        return layers
    
    def count_params(self):
        params = 0
        for layer in self.get_layers():
            params += layer.count_params()
        self.params = params
    
    def call(self, inputs):
        x = inputs
        self.defined_nodes = [None  for _ in range(len(self.graph.order_nodes))]
        for node in self.graph.order_nodes:
            node_index = self.graph.nodes.index(node)
            layer = self.graph.layers[node_index]
            node_inputs = self.graph.node_inputs[node_index]
            if len(node_inputs) == 0:
                self.defined_nodes[self.graph.order_nodes.index(node)] = layer(x)
            elif len(node_inputs) == 1:
                self.defined_nodes[self.graph.order_nodes.index(node)] = layer(self.defined_nodes[self.graph.order_nodes.index(node_inputs[0])])
            else:
                concat = Concatenate()([self.defined_nodes[self.graph.order_nodes.index(node_input)] for node_input in node_inputs])
                self.defined_nodes[self.graph.order_nodes.index(node)] = layer(concat)
        return self.defined_nodes[-1]


class BuildModel(Model):
    def __init__(self, graphs: list[Graph]) -> None:
        super().__init__()
        self.graphs = graphs
        for graph in graphs:
            setattr(self, graph.id, BuildLayer(graph))


    def count_params(self):
        params = 0
        for graph in self.graphs:
            getattr(self, graph.id).count_params()
            params += getattr(self, graph.id).params
        self.params = params
        return params
    
    
    def call(self, inputs):
        x = inputs
        for graph in self.graphs:
            x = getattr(self, graph.id)(x)
        return x


class TrainModel:
    def __init__(self,
                 graphs: list[Graph], 
                 train_data: tuple[np.ndarray], 
                 test_data: tuple[np.ndarray], 
                 validation_data: tuple[np.ndarray] = None,
                 batch_size: int = 32, 
                 epochs: int = 2, 
                 verbose: int = 1,
                 optimizer: str = 'adam',
                 loss: str = 'categorical_crossentropy',
                 metrics: list[str] = ['accuracy']
                 ) -> None:
        self.model = BuildModel(graphs)
        self.train_data = train_data
        self.test_data = test_data
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model.build(input_shape = (None,) + self.train_data[0].shape[1:])
        self.model_params = self.model.count_params()
        self.model.compile(
            optimizer = optimizer,
            loss = loss,
            metrics = metrics
        )
        self.history = self.model.fit(
            x = self.train_data[0],
            y = self.train_data[1],
            batch_size = self.batch_size,
            epochs = self.epochs,
            verbose = self.verbose,
            validation_data = validation_data
        )

    def evaluate_model(self):
        return self.history.history['val_accuracy'][-1]