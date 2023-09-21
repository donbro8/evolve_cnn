from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, MaxPooling2D, AveragePooling2D, SpatialDropout2D, GlobalAveragePooling2D, Flatten, Lambda, Concatenate, Input, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback 
import time
import networkx as nx
import numpy as np

class BuildLayer(Layer):

    def __init__(self, graph, **kwargs) -> None:
        super(BuildLayer, self).__init__(**kwargs)
        self.graph = graph
        self.params = 0
        self.layers = []
        self.nodes = list(nx.topological_sort(self.graph))
        for node in self.nodes:
            node_type = node.split('_')[0]
            node_attributes = self.graph.nodes[node]
            if node_type == 'Conv2D':
                self.layers.append(Conv2D(**node_attributes))
            elif node_type == 'BatchNormalization':
                self.layers.append(BatchNormalization())
            elif node_type == 'Dense':
                self.layers.append(Dense(**node_attributes))
            elif node_type == 'MaxPooling2D':
                self.layers.append(MaxPooling2D(**node_attributes))
            elif node_type == 'AveragePooling2D':
                self.layers.append(AveragePooling2D(**node_attributes))
            elif node_type == 'SpatialDropout2D':
                self.layers.append(SpatialDropout2D(**node_attributes))
            elif node_type == 'GlobalAveragePooling2D':
                self.layers.append(GlobalAveragePooling2D())
            elif node_type == 'Flatten':
                self.layers.append(Flatten())
            elif node_type == 'ReLU':
                self.layers.append(ReLU())
            elif node_type in ['Identity', 'input', 'output']:
                self.layers.append(Lambda(lambda x: x))
            else:
                print(f"Warning: Node type {node_type} not recognised.")
    

    def count_params(self):
        params = 0
        for layer in self.layers:
            params += layer.count_params()
        self.params = params

    def get_config(self):
        config = super().get_config()
        config.params = self.count_params()
        return config
    
    def call(self, inputs):
        x = inputs
        self.defined_nodes = [None  for _ in range(len(self.nodes))]
        for i in range(len(self.nodes)):

            node_inputs = list(self.graph.predecessors(self.nodes[i]))

            if len(node_inputs) == 0:

                self.defined_nodes[i] = self.layers[i](x)

            elif len(node_inputs) == 1:
                self.defined_nodes[i] = self.layers[i](self.defined_nodes[self.nodes.index(node_inputs[0])])

            else:
                concat = Concatenate()([self.defined_nodes[self.nodes.index(node_input)] for node_input in node_inputs])
                self.defined_nodes[self.nodes.index(self.nodes[i])] = self.layers[i](concat)
        return self.defined_nodes[-1]


class TimeOutCallback(Callback):

    def __init__(self, timeout: int = 600) -> None:
        self.timeout = timeout
        self.timeout_reached = False

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > self.timeout:
            self.model.stop_training = True
            self.timeout_reached = True

    def on_train_batch_begin(self, batch, logs=None):
        if time.time() - self.start_time > self.timeout:
            self.model.stop_training = True
            self.timeout_reached = True

    def on_train_batch_end(self, batch, logs=None):
        if time.time() - self.start_time > self.timeout:
            self.model.stop_training = True
            self.timeout_reached = True



class ModelCompiler():

    def __init__(
        self,
        input_graph: nx.DiGraph,
        output_graph: nx.DiGraph,
        normal_cell_graph: nx.DiGraph = None,
        reduction_cell_graph: nx.DiGraph = None,
        normal_cell_repeats: int = 3,
        substructure_repeats: int = 3

    ) -> None:
        self.input_graph = input_graph
        self.normal_cell_graph = normal_cell_graph
        self.output_graph = output_graph
        self.reduction_cell_graph = reduction_cell_graph
        self.normal_cell_repeats = normal_cell_repeats
        self.substructure_repeats = substructure_repeats


    def build_model(self, input_shape: tuple[int]):
        input_layer = Input(shape = input_shape, name = 'Input Layer')
        x = BuildLayer(self.input_graph, name = 'IC')(input_layer)
        for M in range(self.substructure_repeats):
            if self.normal_cell_graph is not None:
                for N in range(self.normal_cell_repeats):
                    x = BuildLayer(self.normal_cell_graph, name = 'NC_' + str(M + 1) + '_' + str(N + 1))(x)
            if self.reduction_cell_graph is not None:
                x = BuildLayer(self.reduction_cell_graph, name = 'RC_' + str(M + 1))(x)
        x = BuildLayer(self.output_graph, name = 'OC')(x)
        return Model(inputs = input_layer, outputs = x)


    def train_model(
        self,
        training_data: tuple[np.ndarray, np.ndarray],
        validation_data: tuple[np.ndarray, np.ndarray],
        model: Model,
        batch_size: int = 32, 
        epochs: int = 2, 
        verbose: int = 1,
        optimizer: str = 'adam',
        loss: str = 'categorical_crossentropy',
        metrics: list[str] = ['accuracy','mse'],
        measure_time: bool = True,
        train_timeout: int = 600
    ):
        model.compile(
            optimizer = optimizer,
            loss = loss,
            metrics = metrics
        )
        if measure_time:
            start_time = time.time()

        callback_timeout = TimeOutCallback(train_timeout)

        history = model.fit(
            x = training_data[0],
            y = training_data[1],
            batch_size = batch_size,
            epochs = epochs,
            steps_per_epoch = int(np.ceil(training_data[0].shape[0] / batch_size)),
            verbose = verbose,
            validation_data = validation_data,
            callbacks = [callback_timeout]
        )
        if measure_time:
            end_time = time.time()
            history.history['training_time'] = end_time - start_time
            history.history['timeout_reached'] = callback.timeout_reached
        return history