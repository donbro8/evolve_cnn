import numpy as np
from itertools import product, combinations
import yaml
from keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, AveragePooling2D, Flatten, Dense, BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from keras.utils import plot_model, to_categorical
from tensorflow import Tensor
import uuid

np.random.seed(0)


class SearchSpace:
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

    def __repr__(self) -> str:
        return str(self.layers)

    def get_random_layer_type(self, layer_types: list) -> str:
        return np.random.choice(layer_types)
    
    def get_random_layer_attributes(self, layer_type: str) -> Node:
        if layer_type not in self.layers.keys():
            raise ValueError(f"Invalid layer type. Must be one of {self.layers.keys()}")
        return np.random.choice(self.layers[layer_type])

    
class LayersLoader:
    @staticmethod
    def load_layers_from_yaml(file_path: str) -> SearchSpace:
        with open(file_path, "r") as f:
            layers = yaml.safe_load(f)

        return SearchSpace(layers)
    

class Node:

    node_instances = []

    def __init__(self, node_type: str, attributes: dict = {}) -> None:
        
        self.node_type = node_type
        self.attributes = attributes
        self.input_layer = None
        self.layer = None
        self.graph = None
        self.uuid = uuid.uuid4().hex
        Node.node_instances.append(self)
        self.id = '_'.join(['node', self.node_type.lower(), str(len(Node.node_instances)), self.uuid[:4]])

    def __repr__(self) -> str:
        return f"({self.id}, {self.node_type}, {self.attributes})"


class Connection:

    connection_instances = []

    def __init__(self, node_in: Node, node_out: Node, enabled: bool = True) -> None:
        self.node_in = node_in
        self.node_out = node_out
        self.enabled = enabled
        self.uuid = uuid.uuid4().hex
        Connection.connection_instances.append(self)
        self.id = '_'.join(['connection', self.node_in.id, self.node_out.id, str(len(Connection.connection_instances)), self.uuid[:4]])

    def __repr__(self) -> str:
        return f"({self.node_in}, {self.node_out}, {self.enabled})"



class Graph:

    graph_instances = []

    def __init__(self, connections: list[Connection]) -> None:
        self.connections = connections
        self.uuid = uuid.uuid4().hex
        Graph.graph_instances.append(self)
        self.id = '_'.join(['graph', str(len(Graph.graph_instances)), self.uuid[:4]])
        self.update_graph_info()

    def __repr__(self) -> str:
        return f"{self.connections}"
    
    def update_graph_info(self) -> None:
        self.nodes_in = [connection.node_in for connection in self.connections]
        self.nodes_out = [connection.node_out for connection in self.connections]
        self.nodes = list(set(self.nodes_in + self.nodes_out))
        self.start_nodes = [connection.node_in for connection in self.connections if connection.node_in not in self.nodes_out]
        self.end_nodes = [connection.node_out for connection in self.connections if connection.node_out not in self.nodes_in]
        self.valid_nodes = self.update_valid_nodes(self.start_nodes, self.end_nodes)
        self.input_nodes = []
        for node in self.start_nodes:
            self.input_nodes += self.get_node_inputs(node, visited=[])
        self.order_input_nodes()

    def order_input_nodes(self) -> None:
        ordered_nodes = list(set(self.start_nodes))
        remaining_nodes = [node for node in set(self.input_nodes) if node not in ordered_nodes]
        while len(remaining_nodes) > 0:
            node = remaining_nodes.pop(0)
            if set(node.input_layer).issubset(set(ordered_nodes)):
                ordered_nodes.append(node)
            else:
                remaining_nodes.append(node)
        self.input_nodes = ordered_nodes

    def add_connection(self, connection: Connection) -> None:
        if connection not in self.connections:
            self.connections.append(connection)
            self.update_graph_info()
        else:
            raise Warning(f"Connection {connection} already exists in graph {self.id} - cannot add duplicate connection")

    def delete_connection(self, connection: Connection) -> None:
        if connection in self.connections:
            self.connections.remove(connection)
            self.update_graph_info()
        else:
            raise Warning(f"Connection {connection} does not exist in graph {self.id} - cannot delete connection")


    def depth_first_search(self, node: Node, visited: set = set(), connected_nodes: set = set(), enabled_only: bool = True) -> list:
        if node not in visited:
            visited.add(node)
            for connection in self.connections:
                if connection.node_in == node and (connection.enabled or not enabled_only):
                    connected_nodes.add(connection.node_out)
                    self.depth_first_search(connection.node_out, visited, connected_nodes)
        return connected_nodes
    
    
    def breadth_first_search(self, node: Node, enabled_only: bool = True) -> list:
        queue = [node]
        visited = [node]
        while len(queue) > 0:
            node = queue.pop(0)
            for connection in self.connections:
                if connection.node_in == node and (connection.enabled or not enabled_only):
                    visited.append(connection.node_out)
                    queue.append(connection.node_out)
        return visited
    
    
    def get_node_neighbours_in(self, node: Node, enabled_only: bool = True) -> list:
        return [connection.node_in for connection in self.connections if connection.node_out == node and (connection.enabled or not enabled_only)]


    def get_node_neighbours_out(self, node: Node, enabled_only: bool = True) -> list:
        return [connection.node_out for connection in self.connections if connection.node_in == node and (connection.enabled or not enabled_only)]


    def get_node_inputs(self, node: Node, visited: list = [], enabled_only: bool = True) -> list:
        neighbours_out = self.get_node_neighbours_out(node = node, enabled_only = enabled_only)
        if node not in visited:
            for neighbour in neighbours_out:
                if neighbour not in self.valid_nodes:
                    continue
                neighbours_in = self.get_node_neighbours_in(node = neighbour, enabled_only = enabled_only)
                neighbour.input_layer = [node_in for node_in in neighbours_in if node_in in self.valid_nodes]
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
        valid_nodes = set(start_nodes + end_nodes)
        node_combinations = list(combinations(valid_nodes, 2))
        for node_start, node_end in node_combinations:
            for node in self.nodes:
                if self.check_continuity(node_start, node) and self.check_continuity(node, node_end):
                    valid_nodes.add(node)
        return valid_nodes
    
    def get_random_connection(self, enabled_only: bool = True) -> Connection:
        connections = [connection for connection in self.connections if connection.enabled or not enabled_only]
        return np.random.choice(connections)
    
    def get_random_node(self) -> Node:
        return np.random.choice(self.nodes)
    

class RandomGraph(Graph):
    def __init__(self, nodes: list[Node], connection_density: float = 0.5) -> None:
        if len(nodes) > 1:
            self.nodes = nodes
            self.connections = []
            possible_connections = list(combinations(self.nodes, 2))
            max_possible_connections = len(possible_connections)
            np.random.shuffle(possible_connections)
            while len(self.connections) < np.ceil(connection_density*max_possible_connections):
                node_in, node_out = possible_connections.pop(0)
                connection = Connection(node_in, node_out, enabled = True)
                self.connections.append(connection)
            return super().__init__(self.connections)
        else:
            raise ValueError(f"Cannot create random graph with {len(nodes)} nodes - must have at least 2 nodes")


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
        return super().__init__(self.connections, name = 'combined_graph')
        


class Individual:

    individual_instances = []

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.fitness = 0.5
        self.uuid = uuid.uuid4().hex
        Individual.individual_instances.append(self)
        self.id = '_'.join(['individual', len(Individual.individual_instances), self.uuid[:4]])

    def __repr__(self) -> str:
        return str(self.graph)


    # def generate_minimal_individual(self, node_start: Node, node_end: Node) -> None:
    #     self.graph = Graph([Connection(node_start, node_end, True)])
    #     self.__init__(self.id, self.graph)


    # def possible_connections(self, nodes: list) -> list:
    #     return [(node_in, node_out) for node_in, node_out in product(self.graph.nodes, nodes) if node_in != node_out]
    
    # def max_connections(self, nodes: list) -> int:
    #     n_nodes = len(set(nodes))
    #     return int((n_nodes - 1)*n_nodes/2)


    # Generate random graph now instead of random individual
    # Pass random graph to individual once generated
    # def generate_random_individual(self, node_start: Node, node_end: Node, nodes: list, n_connections: int) -> None:

    #     self.graph = Graph([])

    #     possible_connections = self.possible_connections([node_start] + nodes, nodes + [node_end])

    #     n_nodes = len(set([node_start] + nodes + [node_end]))

    #     max_possible_connections = self.max_connections([node_start] + nodes + [node_end])

    #     if n_connections > max_possible_connections:
    #         print(f"Warning: Number of connections ({n_connections}) exceeds maximum possible ({max_possible_connections}). Setting number of connections to maximum possible.")
    #         n_connections = (np.min([n_connections, int((n_nodes - 1)*n_nodes/2)]))

    #     while n_connections > 0:

    #         if n_connections == 1 and not self.graph.check_continuity(node_start, node_end) and len(self.graph.connections) > 0:
    #             connection = np.random.choice(self.graph.connections)
    #             self.graph.delete_connection(connection)
    #             possible_connections.append((connection.node_in, connection.node_out))
    #             n_connections += 1

    #         else:

    #             node_in, node_out = possible_connections[np.random.randint(0, len(possible_connections))]

    #             if ((node_in, node_out) not in [(self.graph.nodes_in[i], self.graph.nodes_out[i]) for i in range(len(self.graph.connections))]):

    #                 self.graph.add_connection(Connection(node_in, node_out, True))

    #                 if self.graph.check_recursion(node_in) or self.graph.check_recursion(node_out) or not self.graph.check_continuity(node_start, node_end):
    #                     self.graph.delete_connection(self.graph.connections[-1])

    #                 else:
    #                     possible_connections.remove((node_in, node_out))
    #                     n_connections -= 1

    #     self.__init__(self.id, self.graph)


    # def get_random_connection(self) -> Connection:
    #     return np.random.choice(self.graph.connections)
    
    # def get_random_node(self, nodes: list) -> Node:
    #     return np.random.choice(nodes)


class Species:

    species_instances = []

    def __init__(self, members: list[Individual]) -> None:
        self.members = members
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
        return np.abs(len(individual.graph.connections) - len(self.representative.graph.connections))
    
    
    def disjoint_connections(self, individual: Individual, enabled_only: False) -> int:

        if enabled_only:
            individual_connections = set([connection for connection in individual.graph.connections if connection.enabled])
            representative_connections = set([connection for connection in self.representative.graph.connections if connection.enabled])

        else:
            individual_connections = set(individual.graph.connections)
            representative_connections = set(self.representative.graph.connections)

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
            individual_connections = set([connection for connection in individual.graph.connections if connection.enabled])
            representative_connections = set([connection for connection in self.representative.graph.connections if connection.enabled])

        else:
            individual_connections = set(individual.graph.connections)
            representative_connections = set(self.representative.graph.connections)

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

    # def get_innovation_number(self, connection: Connection) -> None:
    #     if connection not in self.connections:
    #         connection.innovation_number = len(self.connections) + 1
    #     else:
    #         connection.innovation_number = self.connections.index(connection)
    
class CustomPopulationInitiliser(Population):
    def __init__(self, population_size: int, graphs: list[Graph]):
        self.population_size = population_size
        self.graphs = graphs

    def __repr__(self) -> str:
        return f"Population ID: {self.id} | Individuals: {self.individuals} | Species: {self.species} | Total individuals: {self.population_size} | Total species: {self.n_species} | Total fitness: {self.total_fitness} | Average fitness: {self.average_fitness}"
    
    def generate_initial_population(self) -> None:
        population = []
        while len(population) < self.population_size:
            population.append(Individual(np.random.choice(self.graphs)))
        species_0 = Species(population)
        species = Speciation([species_0])
        super().__init__(species)


class RandomPopulationInitiliser(Population):
    def __init__(self, population_size: int, nodes: list[Node], initialisation_type: str = 'repeat', connection_density: float = 0.5):
        self.population_size = population_size
        self.nodes = nodes
        self.initialisation_type = initialisation_type
        self.connection_density = connection_density
        if len(self.nodes) < 2:
            raise ValueError("There must be at least two nodes to generate a random population")
        if self.initialisation_type not in ['repeat', 'unique']:
            raise ValueError("Initialisation type must be either 'repeat' or 'unique'")
        self.generate_initial_population()
        
    def __repr__(self) -> str:
        return f"Population ID: {self.id} | Individuals: {self.individuals} | Species: {self.species} | Total individuals: {self.population_size} | Total species: {self.n_species} | Total fitness: {self.total_fitness} | Average fitness: {self.average_fitness}"
    
    def generate_initial_population(self) -> None:
        
        if self.initialisation_type == 'repeat':
            graph = RandomGraph(self.nodes, self.connection_density)
            population = [Individual(graph) for _ in range(self.population_size)]
        else:
            population = [Individual(RandomGraph(self.nodes, self.connection_density)) for _ in range(self.population_size)]
        species_0 = Species(population)
        species = Speciation([species_0])
        super().__init__(species)

# class RandomGraph(Graph):
#     def __init__(self, nodes: list[Node], connection_density: float = 0.5) -> None:
#         if len(nodes) > 1:
#             self.nodes = nodes
#             self.connections = []
#             possible_connections = list(combinations(self.nodes, 2))
#             max_possible_connections = len(possible_connections)
#             np.random.shuffle(possible_connections)
#             while len(self.connections) < np.ceil(connection_density*max_possible_connections):
#                 node_in, node_out = possible_connections.pop(0)
#                 connection = Connection(node_in, node_out, enabled = True)
#                 self.connections.append(connection)
#             return super().__init__(self.connections)
#         else:
#             raise ValueError(f"Cannot create random graph with {len(nodes)} nodes - must have at least 2 nodes")



# class NewPopulation(Population):
#     def __init__(self, population_size: int, initialisation_type: str = 'minimal', nodes: list = [], n_connections: int = 0) -> None:
#         self.individuals = []
#         self.population_size = population_size
#         self.base_id = f"s{self.node_start.id}.e{self.node_end.id}.g{0}.p{0}"
#         new_pop_id = self.base_id + f".s{0}"

#         if initialisation_type == 'minimal':
#             individual = Individual(new_pop_id + f".i{0}")
#             individual.generate_minimal_individual(self.node_start, self.node_end)

#             for i in range(n_individuals):
#                 self.individuals.append(Individual(new_pop_id + f".i{i + 1}", individual.graph))

#         elif initialisation_type == 'random_individual':
#             if n_connections == 0:
#                 raise ValueError("Number of connections must be specified for random initialisation.")
            
#             elif len(nodes) == 0:
#                 raise ValueError("List of nodes must be specified for random initialisation.")
            
#             else:
#                 individual = Individual(new_pop_id + f".i{0}")
#                 individual.generate_random_individual(self.node_start, self.node_end, nodes, n_connections)

#                 for i in range(n_individuals):
#                     self.individuals.append(Individual(new_pop_id + f".i{i + 1}", individual.graph))

#         elif initialisation_type == 'random_population':
#             if n_connections == 0:
#                 raise ValueError("Number of connections must be specified for random initialisation.")
            
#             elif len(nodes) == 0:
#                 raise ValueError("List of nodes must be specified for random initialisation.")
            
#             else:
#                 for i in range(n_individuals):
#                     individual = Individual(new_pop_id + f".i{i + 1}")
#                     individual.generate_random_individual(self.node_start, self.node_end, nodes, n_connections)
#                     self.individuals.append(Individual(new_pop_id + f".i{i + 1}", individual.graph))

#         else:
#             raise ValueError("Initialisation type must be either 'minimal', 'random_individual' or 'random_population'.")
        
#         self.population_size = n_individuals
#         self.species = [Species(self.individuals, new_pop_id)]
#         return Population(self.species, self.node_start, self.node_end, '0', 0)
        


class Mutation:
    def __init__(self, population: Population, layers: SearchSpace, p_mutation: float) -> None:
        self.population = population
        self.layers = layers
        self.p_mutation = p_mutation

    def __repr__(self) -> str:
        return f"Mutation probability: {self.p_mutation}"
    
    def mutate_add_node(self, individual: Individual) -> None:
        split_connection = individual.get_random_connection()
        node_in = split_connection.node_in
        node_out = split_connection.node_out
        individual.graph.delete_connection(split_connection)

        layer_type = self.layers.get_random_layer_type(self.layers.local_layers)
        new_node_attributes = self.layers.get_random_layer_attributes(layer_type)
        new_node = self.layers.add_node(layer_type, new_node_attributes)

        connection_before = Connection(node_in, new_node, True)
        individual.graph.add_connection(connection_before)

        connection_after = Connection(new_node, node_out, True)
        individual.graph.add_connection(connection_after)
        

    def mutate_add_connection(self, individual: Individual) -> None:
        possible_nodes = individual.graph.nodes
        max_possible_connections = individual.max_connections(possible_nodes)
        current_connections = [(connection.node_in, connection.node_out) for connection in individual.graph.connections]
        population_connections = [(connection.node_in, connection.node_out) for connection in self.population.connections]
        if len(set(individual.graph.connections)) < max_possible_connections:

            while True:
                node_in = individual.get_random_node(possible_nodes)
                node_out = individual.get_random_node(possible_nodes)

                if node_in != node_out and node_in != individual.node_start and node_out != individual.node_end and (node_in, node_out) not in current_connections:
                    
                    if (node_in, node_out) in population_connections:
                        connection = self.population.connections[population_connections.index((node_in, node_out))]
                    else:
                        connection = Connection(node_in, node_out, True)

                    individual.graph.add_connection(connection)

                    if individual.graph.check_recursion(node_in) or individual.graph.check_recursion(node_out):
                        individual.graph.delete_connection(connection)
                        continue

                    else:
                        break

    def mutate_switch_connection(self, individual: Individual) -> None:
        for connection in individual.graph.connections:
            connection.enabled = not connection.enabled
            if not connection.enabled and not individual.graph.check_continuity(individual.node_start, individual.node_end):
                connection.enabled = not connection.enabled
            else:
                break
    
    def mutate(self, individual: Individual) -> None:
        mutation_type = np.random.choice(['add_node', 'add_connection', 'switch_connection'])
        if 'add_node' in mutation_type:
            self.mutate_add_node(individual)

        elif 'add_connection' in mutation_type:
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
    
    # def subpopulation_selection(self, species: Species, proportion: float, p_fitness: float) -> list:
    #     n_individuals = int(np.ceil(len(species.individuals) * proportion))
    #     if np.random.uniform() < p_fitness:
    #         subpopulation = sorted(species.individuals, key=lambda x: x.fitness, reverse=True)[:n_individuals]

    #     else:
    #         subpopulation = list(np.random.choice(species.individuals, n_individuals, replace=False))
    #     return subpopulation
    
    def assign_offspring_count(self) -> None:
        species_sorted = sorted(self.population.species, key=lambda x: x.fitness_shared, reverse=True)
        offspring_remaining = self.population.population_size
        for species in species_sorted:

            max_offspring = len(list(combinations(species.individuals, 2)))

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

        self.population.offspring_remaining = offspring_remaining


    def crossover(self, individual_1: Individual, individual_2: Individual) -> Individual:
        fittest_individual = individual_1 if individual_1.fitness > individual_2.fitness else individual_2
        all_connections = list(set(individual_1.graph.connections + individual_2.graph.connections))
        new_connections = []
        new_nodes_in = []
        new_nodes_out = []
        for connection in all_connections:
            if connection in individual_2.graph.connections and connection in individual_1.graph.connections:
                new_connections.append(connection)
                new_nodes_in.append(connection.node_in)
                new_nodes_out.append(connection.node_out)

            else:
                inhereted_connection = connection if connection in fittest_individual.graph.connections else None
                if inhereted_connection is not None:
                    new_connections.append(inhereted_connection)
                    new_nodes_in.append(inhereted_connection.node_in)
                    new_nodes_out.append(inhereted_connection.node_out)

            
        new_graph = Graph(new_connections)
        new_individual = Individual(f"{individual_1.id}x{individual_2.id}", new_graph)
        return new_individual
    

    def generate_offspring(self, mutation: Mutation) -> list:
        self.assign_offspring_count()
        offspring_remaining = self.population.population_size
        new_population = []
        current_species = self.population.species.copy()
        for species in current_species:
            if species.n_offspring > 0:
                if len(species.individuals) == 1 and species.n_offspring == 2:
                    species.add_individual(mutation.mutate(species.individuals[0]))
                    

                elif len(species.individuals) == 2:
                    new_individual = self.crossover(species.individuals[0], species.individuals[1])
                    
                    if species.n_offspring == 2:
                        least_fit_individual = species.individuals[0] if species.individuals[0].fitness < species.individuals[1].fitness else species.individuals[1]
                        species.remove_individual(least_fit_individual)
                        mutation.mutate(species.individuals[0])

                    else:
                        species.remove_individual(species.individuals[0])
                        species.remove_individual(species.individuals[1])

                    species.add_individual(new_individual)

                elif len(species.individuals) > 2:
                    selected_pairs = []
                    new_individuals = []
                    p_selection = [1/len(species.individuals) * individual.fitness / species.fitness_shared for individual in species.individuals]
                    while len(selected_pairs) < species.n_offspring:
                        individual_1, individual_2 = np.random.choice(species.individuals, 2, replace=False, p=p_selection)
                        if set([individual_1, individual_2]) not in selected_pairs:
                            selected_pairs.append(set([individual_1, individual_2]))
                            new_individuals.append(self.crossover(individual_1, individual_2))

                    species.individuals = new_individuals

                offspring_remaining -= len(species.individuals)
                new_population.append(species)

        while offspring_remaining > 0:
            species = np.random.choice(new_population)
            species.add_individual(mutation.mutate(np.random.choice(species.individuals)))
            offspring_remaining -= 1

        return new_population
    


class BuildLayer(Layer):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.graph = graph
        self.input_nodes = self.graph.get_node_inputs(self.graph.node_start, visited=[], enabled_only=True)
        self.params = 0
        for node in self.input_nodes:
            if node.node_type == 'Conv2D':
                node.layer = Conv2D(**node.attributes)
            elif node.node_type == 'BatchNormalization':
                node.layer = BatchNormalization()
            elif node.node_type == 'Dense':
                node.layer = Dense(**node.attributes)
            elif node.node_type == 'MaxPooling2D':
                node.layer = MaxPooling2D(**node.attributes)
            elif node.node_type == 'AveragePooling2D':
                node.layer = AveragePooling2D(**node.attributes)
            elif node.node_type == 'SpatialDropout2D':
                node.layer = SpatialDropout2D(**node.attributes)
            elif node.node_type == 'GlobalAveragePooling2D':
                node.layer = GlobalAveragePooling2D()
            elif node.node_type == 'Flatten':
                node.layer = Flatten()
            elif node.node_type == 'Output':
                pass
            else:
                raise ValueError("Node type must be one of 'Conv2D', 'BatchNormalization', 'Dense', 'MaxPooling2D', 'AveragePooling2D', 'SpatialDropout2D', 'GlobalAveragePooling2D', 'Flatten', 'Output'")
            
    def get_layers(self):
        layers = []
        for node in self.input_nodes:
            if hasattr(node, 'layer'):
                layers.append(node.layer)
        return layers
    
    def count_params(self):
        params = 0
        for layer in self.get_layers():
            params += layer.count_params()
        self.params = params
    
    def call(self, inputs):
        x = inputs
        for node in self.input_nodes:
            if len(node.input_layer) > 1 if node.input_layer != None else False:
                x = Concatenate()([node.layer(x) for node in node.input_layer])
            x = node.layer(x)      
        return x
    

class BuildModel(Model):
    def __init__(self, graphs: list[Graph]) -> None:
        super().__init__()
        self.graphs = graphs
        for graph in graphs:
            setattr(self, graph.node_start.id, BuildLayer(graph))


    def count_params(self):
        params = 0
        for graph in self.graphs:
            getattr(self, graph.node_start.id).count_params()
            params += getattr(self, graph.node_start.id).params
        self.params = params
        return params

    
    def call(self, inputs):
        x = inputs
        for graph in self.graphs:
            x = getattr(self, graph.node_start.id)(x)
        return x



class EvolveBlock:
    def __init__(self, n_pop) -> None:
        pass # self, species: list, node_start: Node, node_end: Node, id: str, generation: str


    def run_evolution(self):
        self.population = NewPopulation(self)
