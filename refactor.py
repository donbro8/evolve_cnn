import numpy as np
from itertools import product, combinations
import yaml
from keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, AveragePooling2D, Flatten, Dense, BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D, Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from keras.utils import plot_model, to_categorical
from tensorflow import Tensor
import uuid

np.random.seed(0)
    

class Node:

    node_instances = []

    def __init__(self, node_type: str, attributes: dict = {}) -> None:
        # existing_nodes = [(node.node_type, node.attributes) for node in Node.node_instances]
        # if (node_type, attributes) not in existing_nodes:
        self.node_type = node_type
        self.attributes = attributes
        # Think about moving these out of the class and allow for editable node to be created for an individual
        # Node space should not be editable, however the attributes of a node with respect to an individual should be
        # Likewise, the connection space should not be editable. If a connection in the connection space defines the innovation number
        # Within the context of the individual and connection can be enabled or disabled
        # self.input_layer = None
        # self.layer = None
        # self.graph = None
        self.uuid = uuid.uuid4().hex
        Node.node_instances.append(self)
        self.id = '_'.join(['node', self.node_type.lower(), str(len(Node.node_instances)), self.uuid[:4]])
        # else:
        #     print(f"Node with node type {node_type} and attributes {attributes} already exists. Returning existing node.")
        #     index = existing_nodes.index((node_type, attributes))
        #     self = Node.node_instances[index].copy()

    def __repr__(self) -> str:
        return f"(ID: {self.id} | Node type: {self.node_type} | Node attributes: {self.attributes})"
    
    def copy(self) -> None:
        return Node(self.node_type, self.attributes)
    


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
                self.local_layers.append(layer)
            elif layer_location == "global":
                self.global_layers.append(layer)
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



class Connection:

    connection_instances = []

    def __init__(self, node_in: Node, node_out: Node) -> None: # , enabled: bool = True
        existing_connections = [(connection.node_in, connection.node_out) for connection in Connection.connection_instances]
        if (node_in, node_out) not in existing_connections:
            self.node_in = node_in
            self.node_out = node_out
            self.uuid = uuid.uuid4().hex
            Connection.connection_instances.append(self)
            self.id = '_'.join(['connection', self.node_in.id, self.node_out.id, str(len(Connection.connection_instances)), self.uuid[:4]])
        else:
            print(f"Connection with node in {node_in} and node out {node_out} already exists. Returning existing connection.")
            index = existing_connections.index((node_in, node_out))
            self = Connection.connection_instances[index]


    def __repr__(self) -> str:
        return f"(ID: {self.id} | Node in: {self.node_in} | Node out: {self.node_out})" # , {self.enabled}




class Graph:

    graph_instances = []

    def __init__(self, connections: list[Connection], all_enabled: bool = True, enabled_connections: list[bool] = []) -> None:
        self.connections = connections
        if all_enabled:
            self.enabled_connections = [True for _ in range(len(self.connections))]
        elif not all_enabled and len(enabled_connections) == len(self.connections):
            self.enabled_connections = enabled_connections
        else:
            raise ValueError(f"Invalid enabled connections list. Must be a list of booleans with length equal to the number of connections in the graph")
        self.uuid = uuid.uuid4().hex
        Graph.graph_instances.append(self)
        self.id = '_'.join(['graph', str(len(Graph.graph_instances)), self.uuid[:4]])
        self.update_graph_info()

    def __repr__(self) -> str:
        return f"ID: {self.id}"
    
    def update_graph_info(self) -> None:
        self.nodes_in = [connection.node_in for connection in self.connections]
        self.nodes_out = [connection.node_out for connection in self.connections]
        self.nodes = list(set(self.nodes_in + self.nodes_out))
        self.node_inputs = [[] for _ in range(len(self.nodes))]
        self.layers = [None for _ in range(len(self.nodes))]
        self.start_nodes = [connection.node_in for connection in self.connections if connection.node_in not in self.nodes_out]
        self.end_nodes = [connection.node_out for connection in self.connections if connection.node_out not in self.nodes_in]
        self.valid_nodes = self.update_valid_nodes(self.start_nodes, self.end_nodes)
        self.node_clusters = []
        for node in self.start_nodes:
            self.node_clusters += self.get_node_inputs(node, visited=[])
        self.order_nodes = self.order_input_nodes()

    def order_input_nodes(self) -> None:
        ordered_nodes = list(set([node for node in self.start_nodes if node in self.valid_nodes]))
        remaining_nodes = [node for node in set(self.node_clusters) if node not in ordered_nodes]
        while len(remaining_nodes) > 0:
            node = remaining_nodes.pop(0)
            index = self.nodes.index(node)
            if set(self.node_inputs[index]).issubset(set(ordered_nodes)):
                ordered_nodes.append(node)
            else:
                remaining_nodes.append(node)
        return ordered_nodes
        
    def add_connection(self, connection: Connection, enabled: bool = True) -> None:
        if connection not in self.connections:
            self.connections.append(connection)
            self.enabled_connections.append(enabled)
            self.update_graph_info()
        else:
            raise Warning(f"Connection {connection} already exists in graph {self.id} - cannot add duplicate connection")

    def delete_connection(self, connection: Connection) -> None:
        if connection in self.connections:
            index = self.connections.index(connection)
            self.enabled_connections.pop(index)
            self.connections.pop(index)
            self.update_graph_info()
        else:
            raise Warning(f"Connection {connection} does not exist in graph {self.id} - cannot delete connection")

    def switch_connection(self, connection: Connection) -> None:
        if connection in self.connections:
            index = self.connections.index(connection)
            self.enabled_connections[index] = not self.enabled_connections[index]
            self.update_graph_info()
        else:
            raise Warning(f"Connection {connection} does not exist in graph {self.id} - cannot switch connection")
    
    
    def depth_first_search(self, node: Node, visited: set = set(), connected_nodes: set = set(), enabled_only: bool = True) -> list:
        if node not in visited:
            visited.add(node)
            for i in range(len(self.connections)):
                if self.connections[i].node_in == node and (self.enabled_connections[i] or not enabled_only):
                    connected_nodes.add(self.connections[i].node_out)
                    self.depth_first_search(self.connections[i].node_out, visited, connected_nodes)
        return connected_nodes
    
    
    def breadth_first_search(self, node: Node, enabled_only: bool = True) -> list:
        queue = [node]
        visited = [node]
        while len(queue) > 0:
            node = queue.pop(0)
            for i in range(len(self.connections)):
                if self.connections[i].node_in == node and (self.enabled_connections[i] or not enabled_only):
                    visited.append(self.connections[i].node_out)
                    queue.append(self.connections[i].node_out)
        return visited
    
    
    def get_node_neighbours_in(self, node: Node, enabled_only: bool = True) -> list:
        return [self.connections[i].node_in for i in range(len(self.connections)) if self.connections[i].node_out == node and (self.enabled_connections[i] or not enabled_only)]


    def get_node_neighbours_out(self, node: Node, enabled_only: bool = True) -> list:
        return [self.connections[i].node_out for i in range(len(self.connections)) if self.connections[i].node_in == node and (self.enabled_connections[i] or not enabled_only)]

    def get_node_inputs(self, node: Node, visited: list = [], enabled_only: bool = True) -> list:
        neighbours_out = self.get_node_neighbours_out(node = node, enabled_only = enabled_only)
        if node not in visited and node in self.valid_nodes:
            for neighbour in neighbours_out:
                if neighbour not in self.valid_nodes:
                    continue
                neighbours_in = self.get_node_neighbours_in(node = neighbour, enabled_only = enabled_only)
                index = self.nodes.index(neighbour)
                self.node_inputs[index] = [node_in for node_in in neighbours_in if node_in in self.valid_nodes]
                self.get_node_inputs(neighbour, visited)
            visited.append(node)
        return visited[::-1]
    

    def check_continuity(self, node_start: Node, node_end: Node, enabled_only: bool = True) -> bool:
        connected_nodes = self.depth_first_search(node_start, visited = set(), connected_nodes = set(), enabled_only = enabled_only)
        return node_end in connected_nodes
    
    
    def check_recursion(self, node: Node, enabled_only = True) -> bool:
        connected_nodes = self.depth_first_search(node, visited = set(), connected_nodes = set(), enabled_only = enabled_only)
        return node in connected_nodes
    

    def update_valid_nodes(self, start_nodes: list[Node], end_nodes: list[Node]) -> list:
        valid_nodes = set()
        node_combinations = list(product(start_nodes, end_nodes))
        for node_start, node_end in node_combinations:
            if self.check_continuity(node_start, node_end):
                valid_nodes.add(node_start)
                valid_nodes.add(node_end)
                for node in self.nodes:
                    if self.check_continuity(node_start, node) and self.check_continuity(node, node_end):
                        valid_nodes.add(node)
        return valid_nodes
    
    def get_random_connection(self, enabled_only: bool = True) -> Connection:
        connections = [self.connections[i] for i in range(len(self.connections)) if self.enabled_connections[i] or not enabled_only]
        return np.random.choice(connections)
    
    def get_random_node(self, add_node_point: bool = False, node_point: str = 'start') -> Node:
        if add_node_point:
            possible_nodes = self.nodes + [node_point]
        else:
            possible_nodes = self.nodes
        return np.random.choice(possible_nodes)
    


class RandomGraph(Graph):
    def __init__(self, possible_nodes: list[Node], connection_density: float = 0.5) -> None:
        if len(possible_nodes) > 1:
            self.possible_nodes = possible_nodes
            self.connections = []
            possible_connections = list(combinations(self.nodes, 2))
            max_possible_connections = len(possible_connections)
            np.random.shuffle(possible_connections)
            while len(self.connections) < np.ceil(connection_density*max_possible_connections):
                node_in, node_out = possible_connections.pop(0)
                connection = Connection(node_in.copy(), node_out.copy()) # Separation between search space nodes and graph nodes
                self.connections.append(connection)
            return super().__init__(self.connections)
        else:
            raise ValueError(f"Cannot create random graph with {len(possible_nodes)} nodes - must have at least 2 nodes")


# The idea here is to have hierarchical graphs, where each node is a graph in itself
# The network can be structured as a graph of graphs, where each node is a graph
# An unpacked graph is a graph where all of the nodes are no longer graphs
# The depth of a node/layer is the maximum depth of any node in the graph, i.e. the number of subgraphs for a node
# e.g Network -> Nodes has a depth of 0
# e.g Network -> Subgraph Level 1 -> Node has a depth of 1
# e.g Network -> Subgraph Level 1 -> Subgraph Level 2 -> Node has a depth of 2
# e.g Network -> Subgraph Level 1 -> Subgraph Level 2 -> Subgraph Level 3 -> Node has a depth of 3
# Example:
#   Network Level 0: Network = Single Node with no connections that is a graph
#   Graph Level 1: Node 1 = Input -> Node 2 = Hidden -> Node 3 = Output
#       Node 1 Node Level 2: [(C1, BN1)], => max depth = 2
#       Node 2 Graph Level 2: [(B1, B2), (B1, B3)]
#           B1 Node Level 3: [(C1, C2), (C1, C3), (C1, P1)] => max depth = 3
#           B2 Node Level 3: [(C2, C3), (C2, P1)] => max depth = 3
#           B3 Node Level 3: [(C3, P1)] => max depth = 3
#       Node 3 Graph Level 2: [(D1, D2)] => max depth = 2
class UnpackGraph(Graph):
    def __init__(self, graph: Graph) -> None:
        for node in graph.nodes:
            if node.graph == None:
                pass
            else:
                pass
        self.nodes = graph.nodes
        self.connections = graph.connections
        return super().__init__(self.connections, name = 'unpacked_graph')


# Currently setup for a linear connection of graphs, ordered from left to right
class CombinedGraph(Graph):
    def __init__(self, graphs: list[Graph]) -> None:
        self.nodes = [node for graph in graphs for node in graph.input_nodes]
        self.connections = [graph.connections for graph in graphs]
        i = 0
        max_len = 2*len(self.connections) - 1
        while len(self.connections)  < max_len:
            new_connection = [Connection(self.connections[i][-1].node_out, self.connections[i+1][0].node_in, True)]
            self.connections.insert(i+1, new_connection)
            i += 2
        self.connections = [connection for connections in self.connections for connection in connections]
        return super().__init__(self.connections)
        


class Individual(Graph):

    individual_instances = []

    def __init__(self, connections: list[Connection]) -> None:
        self.connections = connections
        self.individual = super().__init__(self.connections)
        self.fitness = 0.5
        self.uuid = uuid.uuid4().hex
        Individual.individual_instances.append(self)
        self.id = '_'.join(['individual', str(len(Individual.individual_instances)), self.uuid[:4]])

    def __repr__(self) -> str:
        return str(self.individual)
    
    def copy(self) -> None:
        return Individual(self.connections)


class Species:

    species_instances = []

    def __init__(self, members: list[Individual]) -> None:
        self.members = members
        self.n_offspring = 0
        self.update_species_info()
        self.get_new_representative()
        self.uuid = uuid.uuid4().hex
        Species.species_instances.append(self)
        self.id = '_'.join(['species', len(Species.species_instances) - 1, self.uuid[:4]])


    def __repr__(self) -> str:
        return f"Species ID: {self.id} | Members: {self.members} | Representative: {self.representative} | Total members: {self.n_members} | Shared fitness: {self.fitness_shared}"
    
    def update_species_info(self) -> None:
        self.n_members = len(self.members)
        self.fitness_shared = np.sum([member.fitness for member in self.members])/self.n_members
    
    def add_member(self, individual: Individual) -> None:
        if individual not in self.members:
            self.members.append(individual)
            self.update_species_info()
        else:
            print(f"Warning: Individual {individual} already in species {self}.")

    def remove_member(self, individual: Individual) -> None:
        if individual in self.members:
            self.members.remove(individual)
            self.update_species_info()
        else:
            print(f"Warning: Individual {individual} not in species {self}.")

    def get_new_representative(self) -> None:
        self.representative = np.random.choice(self.members)
    
    def excess_connections(self, individual: Individual) -> int:
        return np.abs(len(individual.connections) - len(self.representative.connections))
    
    
    def disjoint_connections(self, individual: Individual, enabled_only: False) -> int:

        if enabled_only:
            individual_connections = set([individual.connections[i] for i in range(len(individual.connections)) if individual.enabled_connections[i]])
            representative_connections = set([self.representative.connections[i] for i in range(len(self.representative.connections)) if self.representative.enabled_connections[i]])

        else:
            individual_connections = set(individual.connections)
            representative_connections = set(self.representative.connections)

        return len(individual_connections.symmetric_difference(representative_connections))
    
    
    def compatability_distance(self, individual: Individual, enabled_only: bool = False) -> float:

        # (c1 * self.excess_connections(individual) + c2 * self.disjoint_connections(individual, enabled_only)) / n + c3 * np.abs(self.representative.fitness - individual.fitness)

        # Since each connection is uniquely attributed to a pair of nodes, we need only to compare the connections between the two individuals
        # In this instance we are only interested in topological similarity, so we measure compatability distance as the symmetric difference divided by total number of connections between them
        # This will result in a value between 0 and 1, where 0 is identical and 1 is completely different

        # Two individuals might be similar with respect to all connections, but may behave wildly differently is some of those connections are turned off
        # Therefore, we can also measure compatability distance with respect to only enabled connections
        # Still need to figure out how to weight this (default is kept as False)

        if enabled_only:
            individual_connections = set([individual.connections[i] for i in range(len(individual.connections)) if individual.enabled_connections[i]])
            representative_connections = set([self.representative.connections[i] for i in range(len(self.representative.connections)) if self.representative.enabled_connections[i]])

        else:
            individual_connections = set(individual.connections)
            representative_connections = set(self.representative.connections)

        return len(individual_connections.symmetric_difference(representative_connections))/(len(individual_connections) + len(representative_connections))


class Speciation(Species):
    def __init__(self, species: list[Species], delta_t: float = 0.5) -> list[Species]:
        self.species = sorted(species, key=lambda species: species.fitness_shared, reverse=True)
        self.delta_t = delta_t
        self.individuals = [individual for species in self.species for individual in species.members]
        self.representatives = [species.representative for species in self.species]
        unassigned_individuals = [individual for individual in self.individuals if individual not in self.representatives]
        while len(unassigned_individuals) > 0:
            individual = unassigned_individuals[0]
            for species in self.species:
                if species.compatability_distance(individual) < delta_t and individual not in species.members:
                    species.add_member(individual)
                    unassigned_individuals.remove(individual)
                elif species.compatability_distance(individual) >= delta_t and individual in species.members:
                    species.remove_member(individual)
            if individual in unassigned_individuals:
                new_species = super().__init__([individual])
                self.species.append(new_species)
                self.species = sorted(species, key=lambda species: species.fitness_shared, reverse=True)
                unassigned_individuals.remove(individual)
        return self.species


class Population:

    population_instances = []

    def __init__(self, species: list[Species]) -> None:
        self.species = species
        self.update_population_info()
        self.uuid = uuid.uuid4().hex
        Population.population_instances.append(self)
        self.id = '_'.join(['population', len(Population.population_instances) - 1, self.uuid[:4]])

    def __repr__(self) -> str:
        return f"Population ID: {self.id} | Individuals: {self.individuals} | Species: {self.species} | Total individuals: {self.population_size} | Total species: {self.n_species} | Total fitness: {self.total_fitness} | Average fitness: {self.average_fitness}"

    def update_population_info(self) -> None:
        self.species_fitness = [species.fitness_shared for species in self.species]
        self.total_fitness_shared = np.sum(self.species_fitness)
        self.n_species = len(self.species)
        self.individuals = [individual for species in self.species for individual in species.members]
        self.population_size = len(self.individuals)
        self.total_fitness = np.sum([individual.fitness for individual in self.individuals])
        self.average_fitness = self.total_fitness/self.population_size
    

class CustomPopulationInitiliser(Population):
    def __init__(self, population_size: int, individuals: list[Individual]):
        self.population_size = population_size
        self.individuals = individuals

    def __repr__(self) -> str:
        return f"Population size: {self.population_size} | Individuals: {self.individuals}"
    
    def generate_initial_population(self) -> None:
        if self.population_size == len(self.individuals):
            population = self.individuals

        elif self.population_size < len(self.individuals):
            population = list(np.random.choice(self.individuals, self.population_size, replace=False))

        elif self.population_size > len(self.individuals):
            population = self.individuals
            while len(population) < self.population_size:
                population.append(np.random.choice(self.individuals).copy())

        else:
            raise ValueError("Population size must be a positive integer")
        
        species_0 = Species(population)
        species = Speciation([species_0])
        super().__init__(species)


class RandomPopulationInitiliser(Population):
    def __init__(self, population_size: int, possible_nodes: list[Node], initialisation_type: str = 'repeat', connection_density: float = 0.5):
        self.population_size = population_size
        self.possible_nodes = possible_nodes # a sample of nodes from which to generate the individuals in the population, can sample nodes from layers.sample_nodes()
        self.initialisation_type = initialisation_type
        self.connection_density = connection_density
        if len(self.possible_nodes) < 2:
            raise ValueError("There must be at least two nodes to generate a random population")
        if self.initialisation_type not in ['repeat', 'unique']:
            raise ValueError("Initialisation type must be either 'repeat' or 'unique'")
        self.generate_initial_population()
        
    def __repr__(self) -> str:
        return f"Population size: {self.population_size} | Initialisation type: {self.initialisation_type} | Connection density: {self.connection_density} | Possible nodes: {self.possible_nodes}"
    
    def generate_initial_population(self) -> None:
        
        if self.initialisation_type == 'repeat':
            graph = RandomGraph(self.possible_nodes, self.connection_density)
            population = [Individual(graph.connections) for _ in range(self.population_size)]
        else:
            population = [Individual(RandomGraph(self.possible_nodes, self.connection_density).connections) for _ in range(self.population_size)]
        species_0 = Species(population)
        species = Speciation([species_0])
        super().__init__(species)
        


class Mutation:
    def __init__(self, population: Population, layers: SearchSpace, p_mutation: float) -> None:
        self.population = population
        self.layers = layers
        self.p_mutation = p_mutation

    def __repr__(self) -> str:
        return f"Mutation probability: {self.p_mutation}"
    
    def mutate_add_node(self, individual: Individual) -> None:

        new_node_layer_type = self.layers.get_random_layer_type(self.layers.local_layers)
        new_node_attributes = self.layers.get_random_layer_attributes(new_node_layer_type)

        node_in = individual.get_random_node(add_node_point=True, node_point='start')

        if node_in == 'start':
            node_out = individual.get_random_node()
            node_in = self.layers.assign_node_layer(new_node_layer_type, new_node_attributes)
            individual.add_connection(Connection(node_in, node_out, True))

        else:
            node_out = individual.get_random_node(add_node_point=True, node_point='end')
            if node_out == 'end':
                node_out = self.layers.assign_node_layer(new_node_layer_type, new_node_attributes)
                individual.add_connection(Connection(node_in, node_out, True))
            else:
                split_connection = [connection for connection in individual.connections if connection.node_in == node_in and connection.node_out == node_out][0]
                individual.delete_connection(split_connection)
                new_node = self.layers.assign_node_layer(new_node_layer_type, new_node_attributes)
                individual.add_connection(Connection(node_in, new_node, True))
                individual.add_connection(Connection(new_node, node_out, True))
        

    def mutate_add_connection(self, individual: Individual) -> None:
        possible_nodes = individual.nodes
        max_possible_connections = len(list(combinations(len(possible_nodes), 2)))
        current_connections = [(connection.node_in, connection.node_out) for connection in individual.connections]
        population_connections = [(connection.node_in, connection.node_out) for connection in Connection.connection_instances]
        if len(set(individual.connections)) < max_possible_connections:

            while True:
                node_in = individual.get_random_node(possible_nodes)
                node_out = individual.get_random_node(possible_nodes)

                if node_in != node_out and (node_in, node_out) not in current_connections:
                    
                    if (node_in, node_out) in population_connections:
                        connection = Connection.connection_instances[population_connections.index((node_in, node_out))]
                    else:
                        connection = Connection(node_in, node_out, True)

                    test_graph = Graph(individual.connections + [connection])

                    if not test_graph.check_recursion(node_in) or not test_graph.check_recursion(node_out):
                        individual.add_connection(connection)
                        break

        else:
            print("Warning: Reached maximum number of possible connections. Skipping mutation.")


    def mutate_switch_connection(self, individual: Individual) -> None:
        is_continuous = False
        combinations = list(product(individual.start_nodes, individual.end_nodes))
        switched_connections = []
        while len(switched_connections) < len(individual.connections) and not is_continuous:
            connection = individual.get_random_connection()

            if connection not in switched_connections:
                switched_connections.append(connection)
                individual.switch_connection(connection)
                
                for node_start, node_end in combinations:
                    if individual.check_continuity(node_start, node_end):
                        is_continuous = True
                        break

                if not is_continuous:
                    individual.switch_connection(connection)

        if len(switched_connections) == len(individual.connections) and not is_continuous:
            print("Warning: Switched all connections and did not find a continuous graph. Skipping mutation.")

    
    def mutate(self, individual: Individual, p_array: list[float]) -> None:
        mutation_type = np.random.choice(['add_node', 'add_connection', 'switch_connection'], p = p_array)
        if 'add_node' == mutation_type:
            self.mutate_add_node(individual)

        elif 'add_connection' == mutation_type:
            self.mutate_add_connection(individual)

        else:
            self.mutate_switch_connection(individual)

    def selection(self) -> None:
        selected_individuals = [individual for individual in self.population.individuals if np.random.uniform() < self.p_mutation]
        for individual in selected_individuals:
            self.mutate(individual)


class Crossover:
    def __init__(self, population: Population, p_crossover: float) -> None:
        self.population = population
        self.p_crossover = p_crossover

    def __repr__(self) -> str:
        return f"Crossover probability: {self.p_crossover}"
    
    def assign_offspring_count(self) -> None:
        species_sorted = sorted(self.population.species, key=lambda x: x.fitness_shared, reverse=True)
        offspring_remaining = self.population.population_size
        for species in species_sorted:

            max_offspring = len(list(combinations(species.members, 2)))

            if max_offspring <= 2:
                n_offspring = np.min([max_offspring + 2, int(np.ceil(species.fitness_shared / self.population.total_fitness_shared * self.population.population_size))])

            else:
                n_offspring = np.min([max_offspring, int(np.ceil(species.fitness_shared / self.population.total_fitness_shared * self.population.population_size))])

            if offspring_remaining - n_offspring < 0:
                n_offspring = offspring_remaining
                offspring_remaining = 0

            else:
                offspring_remaining -= n_offspring

            species.n_offspring = n_offspring

        return offspring_remaining # Need to confirm that this is always 0


    def crossover(self, individual_1: Individual, individual_2: Individual) -> Individual:
        fittest_individual = individual_1 if individual_1.fitness > individual_2.fitness else individual_2
        all_connections = list(set(individual_1.connections + individual_2.connections))
        new_connections = []
        new_nodes_in = []
        new_nodes_out = []
        for connection in all_connections:
            if connection in individual_2.connections and connection in individual_1.connections:
                new_connections.append(connection)
                new_nodes_in.append(connection.node_in)
                new_nodes_out.append(connection.node_out)

            else:
                inhereted_connection = connection if connection in fittest_individual.connections else None
                if inhereted_connection is not None:
                    new_connections.append(inhereted_connection)
                    new_nodes_in.append(inhereted_connection.node_in)
                    new_nodes_out.append(inhereted_connection.node_out)

        new_individual = Individual(new_connections)
        return new_individual
    

    def generate_offspring(self, mutation: Mutation) -> list:
        remainder = self.assign_offspring_count()
        offspring_remaining = self.population.population_size
        new_population = []
        current_species = self.population.species.copy()
        for species in current_species:
            if species.n_offspring > 0:
                if len(species.members) == 1 and species.n_offspring == 2:
                    species.add_member(mutation.mutate(species.members[0]))
                    

                elif len(species.members) == 2:
                    new_individual = self.crossover(species.members[0], species.members[1])
                    
                    if species.n_offspring == 2:
                        least_fit_individual = species.members[0] if species.members[0].fitness < species.members[1].fitness else species.members[1]
                        species.remove_member(least_fit_individual)
                        mutation.mutate(species.members[0])

                    else:
                        species.remove_member(species.members[0])
                        species.remove_member(species.members[1])

                    species.add_member(new_individual)

                elif len(species.members) > 2:
                    selected_pairs = []
                    new_individuals = []
                    p_selection = [1/len(species.members) * individual.fitness / species.fitness_shared for individual in species.members]
                    while len(selected_pairs) < species.n_offspring:
                        individual_1, individual_2 = np.random.choice(species.members, 2, replace=False, p=p_selection)
                        if set([individual_1, individual_2]) not in selected_pairs:
                            selected_pairs.append(set([individual_1, individual_2]))
                            new_individuals.append(self.crossover(individual_1, individual_2))

                    species.members = new_individuals

                offspring_remaining -= len(species.members)
                new_population.append(species)

        while offspring_remaining > 0:
            species = np.random.choice(new_population)
            species.add_member(mutation.mutate(np.random.choice(species.members)))
            offspring_remaining -= 1

        return new_population
    


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
            # elif node.node_type == 'Input':
            #     self.graph.layers[index] = Input(**node.attributes)
            else:
                raise ValueError("Node type must be one of 'Conv2D', 'BatchNormalization', 'Dense', 'MaxPooling2D', 'AveragePooling2D', 'SpatialDropout2D', 'GlobalAveragePooling2D', 'Flatten', 'Identity', 'Input'")
            
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



class EvolveBlock:
    def __init__(self, n_pop) -> None:
        pass # self, species: list, node_start: Node, node_end: Node, id: str, generation: str


    def run_evolution(self):
        self.population = NewPopulation(self)
