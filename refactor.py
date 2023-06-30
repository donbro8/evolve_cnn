import numpy as np
import graphviz
from itertools import product, combinations
import yaml
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, AveragePooling2D, Flatten, Dense, BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Layer, Model
from tensorflow import Tensor
import uuid
from keras.datasets import mnist, cifar10

np.random.seed(0)

class Node:
    def __init__(self, id: str, node_type: str, attributes: dict = {}) -> None:
        self.id = id
        self.node_type = node_type
        self.attributes = attributes
        self.input_layer = None
        self.layer = None
        self.uuid = uuid.uuid4().hex

    def __repr__(self) -> str:
        return f"({self.id}, {self.node_type}, {self.attributes})"
    

class SearchSpace:
    def __init__(self, layers: list) -> None:
        self.layers = {}
        layers = layers.copy()
        for layer in layers:
            layer_type = layer.pop("layer_type")
            id_count = 1
            if "parameter_set_type" in layer.keys():
                parameter_set_type = layer["parameter_set_type"]
                for key, value in parameter_set_type.items():
                    if value == "continuous":
                        rounding = layer["rounding"]
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

        self.local_layers = [layer for layer in self.layers.keys() if layer["location"] == "local"]
        self.global_layers = [layer for layer in self.layers.keys() if layer["location"] == "global"]

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


class Connection:
    def __init__(self, node_in: Node, node_out: Node, enabled: bool = True) -> None:
        self.node_in = node_in
        self.node_out = node_out
        self.enabled = enabled
        self.id = node_in.id + "_" + node_out.id
        self.uuid = uuid.uuid4().hex

    def __repr__(self) -> str:
        return f"({self.node_in}, {self.node_out}, {self.enabled})"


class Graph:
    def __init__(self, connections: list[Connection]) -> None:
        self.connections = connections
        self.nodes_in = [connection.node_in for connection in connections]
        self.nodes_out = [connection.node_out for connection in connections]
        self.nodes = list(set(self.nodes_in + self.nodes_out))
        self.node_start = self.nodes_in[0]
        self.node_end = self.nodes_out[-1]
        self.valid_nodes = self.update_valid_nodes(self.node_start, self.node_end)
        self.uuid = uuid.uuid4().hex

    def __repr__(self) -> str:
        return f"{self.connections}"

    def add_connection(self, connection: Connection) -> None:
        self.nodes_in.append(connection.node_in)
        self.nodes_out.append(connection.node_out)
        self.connections.append(connection)
        self.nodes = list(set(self.nodes_in + self.nodes_out))

    def delete_connection(self, connection: Connection) -> None:
        self.nodes_in.remove(connection.node_in)
        self.nodes_out.remove(connection.node_out)
        self.connections.remove(connection)
        self.nodes = list(set(self.nodes_in + self.nodes_out))


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


    def check_continuity(self, node_start: Node, node_end: Node) -> bool:

        connected_nodes = self.depth_first_search(node_start, visited = set(), connected_nodes = set(), enabled_only = True)

        return node_end in connected_nodes
    
    
    def check_recursion(self, node: Node) -> bool:

        connected_nodes = self.depth_first_search(node, visited = set(), connected_nodes = set(), enabled_only = True)

        return node in connected_nodes
    

    def update_valid_nodes(self, node_start: Node, node_end: Node) -> list:

        valid_nodes = set([node_start, node_end])

        for node in self.nodes:

            if self.check_continuity(node_start, node) and self.check_continuity(node, node_end):

                valid_nodes.add(node)

        return valid_nodes

class VisualiseBlock:
    def __init__(self, graph: Graph, name: str, rankdir: str = 'LR', size: str = '10,5') -> None:
        self.graph = graph
        self.output_graph = graphviz.Digraph(name = name, format = "pdf")
        self.output_graph.attr(rankdir = rankdir, size = size)
        self.node_order = self.graph.breadth_first_search(self.graph.node_start, enabled_only = False)
        self.valid_nodes = self.graph.update_valid_nodes(self.graph.node_start, self.graph.node_end)

    def __repr__(self) -> str:
        return str(self.graph)
    
    def draw_block(self) -> None:

        for node in self.node_order:
            if node in self.valid_nodes:

                if node.node_type == "input" or node.node_type == "output":
                    self.output_graph.attr(node.id, shape='box', color = 'black', label = node.node_type)
                                           
                elif node.node_type == 'Convolutional':
                    self.output_graph.attr(node.id, shape='doublecircle', color = 'red', label = node.node_type[0]+'\n'+str(node.attributes['filters'])+str(node.attributes['kernel_size']))

                elif node.node_type == 'Pooling':
                    self.output_graph.attr(node.id, shape='trapezium', color = 'darkviolet', label = node.node_type[0]+'\n'+node.attributes['type'][0]+str(node.attributes['pool_size']), orientation='-90')

                elif node.node_type == 'Dropout':
                    self.output_graph.attr(node.id, shape='circle', color = 'deepskyblue', label = node.node_type[0]+'\n'+str(node.attributes['rate']))

                else:
                    raise ValueError("Node type not recognised")

            # If the node is not a valid node then it is still drawn but greyed out
            else:
                self.output_graph.attr(node.id, shape='circle', color = 'lightgray', label = node.node_type[0])

        # Draw the connections between the defined nodes
        i = 1
        for connection in self.graph.connections:
            if connection.enabled and connection.node_in in self.valid_nodes and connection.node_out in self.valid_nodes:
                self.output_graph.edge(connection.node_in.id, connection.node_out.id, style = 'solid', label = str(i))

            else:
                self.output_graph.edge(connection.node_in.id, connection.node_out.id, color = 'lightgray', style = 'dashed', label = str(i))

            i += 1

        self.output_graph.render(directory = 'graphs', format = 'pdf').replace('\\', '/')



class Individual:
    def __init__(self, id: str, graph: Graph) -> None:
        self.graph = graph
        self.id = id
        self.uuid = uuid.uuid4().hex
        self.fitness = 0.5

    def __repr__(self) -> str:
        return str(self.graph)


    def generate_minimal_individual(self, node_start: Node, node_end: Node) -> None:
        self.graph = Graph([Connection(node_start, node_end, True)])
        self.__init__(self.id, self.graph)


    def possible_connections(self, nodes: list) -> list:
        return [(node_in, node_out) for node_in, node_out in product(self.graph.nodes, nodes) if node_in != node_out]
    
    def max_connections(self, nodes: list) -> int:
        n_nodes = len(set(nodes))
        return int((n_nodes - 1)*n_nodes/2)


    def generate_random_individual(self, node_start: Node, node_end: Node, nodes: list, n_connections: int) -> None:

        self.graph = Graph([])

        possible_connections = self.possible_connections([node_start] + nodes, nodes + [node_end])

        n_nodes = len(set([node_start] + nodes + [node_end]))

        max_possible_connections = self.max_connections([node_start] + nodes + [node_end])

        if n_connections > max_possible_connections:
            print(f"Warning: Number of connections ({n_connections}) exceeds maximum possible ({max_possible_connections}). Setting number of connections to maximum possible.")
            n_connections = (np.min([n_connections, int((n_nodes - 1)*n_nodes/2)]))

        while n_connections > 0:

            if n_connections == 1 and not self.graph.check_continuity(node_start, node_end) and len(self.graph.connections) > 0:
                connection = np.random.choice(self.graph.connections)
                self.graph.delete_connection(connection)
                possible_connections.append((connection.node_in, connection.node_out))
                n_connections += 1

            else:

                node_in, node_out = possible_connections[np.random.randint(0, len(possible_connections))]

                if ((node_in, node_out) not in [(self.graph.nodes_in[i], self.graph.nodes_out[i]) for i in range(len(self.graph.connections))]):

                    self.graph.add_connection(Connection(node_in, node_out, True))

                    if self.graph.check_recursion(node_in) or self.graph.check_recursion(node_out) or not self.graph.check_continuity(node_start, node_end):
                        self.graph.delete_connection(self.graph.connections[-1])

                    else:
                        possible_connections.remove((node_in, node_out))
                        n_connections -= 1

        self.__init__(self.id, self.graph)


    def get_random_connection(self) -> Connection:
        return np.random.choice(self.graph.connections)
    
    def get_random_node(self, nodes: list) -> Node:
        return np.random.choice(nodes)


class Species:
    def __init__(self, individuals: list, id: str) -> None:
        self.individuals = individuals
        self.n_individuals = len(individuals)
        self.fitness_shared = np.sum([individual.fitness for individual in self.individuals])/self.n_individuals
        self.id = id
        self.uuid = uuid.uuid4().hex
        self.representative = np.random.choice(individuals)


    def __repr__(self) -> str:
        return f"Species ID: {self.id} | Members: {self.individuals} | Representative: {self.representative} | Total individuals: {self.n_individuals} | Shared fitness: {self.fitness_shared}"
    
    
    def add_individual(self, individual: Individual) -> None:
        self.individuals.append(individual)
        self.n_individuals += 1
        self.fitness_shared = np.sum([individual.fitness for individual in self.individuals])/self.n_individuals

    def remove_individual(self, individual: Individual) -> None:
        self.individuals.remove(individual)
        self.n_individuals -= 1
        self.fitness_shared = np.sum([individual.fitness for individual in self.individuals])/self.n_individuals

    def get_new_representative(self) -> None:
        self.representative = np.random.choice(self.individuals)
    
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
    
    
    def compatability_distance(self, individual: Individual, enabled_only: bool = True) -> float:

        # (c1 * self.excess_connections(individual) + c2 * self.disjoint_connections(individual, enabled_only)) / n + c3 * np.abs(self.representative.fitness - individual.fitness)

        # Since each connection is uniquely attributed to a pair of nodes, we need only to compare the connections between the two individuals
        # In this instance we are only interested in topological similarity, so we measure compatability distance as the symmetric difference divided by total number of connections between them
        # This will result in a value between 0 and 1, where 0 is identical and 1 is completely different

        # Two individuals might be similar with respect to all connections, but may behave wildly differently is some of those connections are turned off
        # Therefore, we can also measure compatability distance with respect to only enabled connections
        # Still need to figure out how to weight this

        if enabled_only:
            individual_connections = set([connection for connection in individual.graph.connections if connection.enabled])
            representative_connections = set([connection for connection in self.representative.graph.connections if connection.enabled])

        else:
            individual_connections = set(individual.graph.connections)
            representative_connections = set(self.representative.graph.connections)

        return len(individual_connections.symmetric_difference(representative_connections))/(len(individual_connections) + len(representative_connections))



class Population:
    def __init__(self, species: list, node_start: Node, node_end: Node, id: str, generation: str) -> None:
        self.species = species
        self.species_fitness = [species.fitness_shared for species in self.species]
        self.total_fitness_shared = np.sum(self.species_fitness)
        self.individuals = [individual for species in self.species for individual in species.individuals]
        self.population_size = len(self.individuals)
        self.node_start = node_start
        self.node_end = node_end
        self.id = id
        self.uuid = uuid.uuid4().hex
        self.generation = generation
        self.base_id = f"s{self.node_start.id}.e{self.node_end.id}.g{self.generation}.p{self.id}"
        self.n_species = len(species.individuals)
        self.connections = list(set([connection for individual in self.individuals for connection in individual.graph.connections]))
        self.nodes = list(set([node for individual in self.individuals for node in individual.graph.nodes]))

        for connection in self.connections:
            self.get_innovation_number(connection)


    def __repr__(self) -> str:
        return f"Population ID: {self.id} | Start node: {self.node_start.id} | End node: {self.node_end.id} | Number of species: {self.n_species}"
    
    
    def get_innovation_number(self, connection: Connection) -> None:
        if connection not in self.connections:
            connection.innovation_number = len(self.connections) + 1
        else:
            connection.innovation_number = self.connections.index(connection)

    def generate_new_population(self, n_individuals: int, initialisation_type: str = 'minimal', nodes: list = [], n_connections: int = 0) -> None:

        self.individuals = []
        new_pop_id = self.base_id + f".s{0}"

        if initialisation_type == 'minimal':
            individual = Individual(new_pop_id + f".i{0}")
            individual.generate_minimal_individual(self.node_start, self.node_end)

            for i in range(n_individuals):
                self.individuals.append(Individual(new_pop_id + f".i{i + 1}", individual.graph))

        elif initialisation_type == 'random_individual':
            if n_connections == 0:
                raise ValueError("Number of connections must be specified for random initialisation.")
            
            elif len(nodes) == 0:
                raise ValueError("List of nodes must be specified for random initialisation.")
            
            else:
                individual = Individual(new_pop_id + f".i{0}")
                individual.generate_random_individual(self.node_start, self.node_end, nodes, n_connections)

                for i in range(n_individuals):
                    self.individuals.append(Individual(new_pop_id + f".i{i + 1}", individual.graph))

        elif initialisation_type == 'random_population':
            if n_connections == 0:
                raise ValueError("Number of connections must be specified for random initialisation.")
            
            elif len(nodes) == 0:
                raise ValueError("List of nodes must be specified for random initialisation.")
            
            else:
                for i in range(n_individuals):
                    individual = Individual(new_pop_id + f".i{i + 1}")
                    individual.generate_random_individual(self.node_start, self.node_end, nodes, n_connections)
                    self.individuals.append(Individual(new_pop_id + f".i{i + 1}", individual.graph))

        else:
            raise ValueError("Initialisation type must be either 'minimal', 'random_individual' or 'random_population'.")
        
        self.population_size = n_individuals
        self.species = [Species(self.individuals, new_pop_id)]
        self.__init__(self.species, self.node_start, self.node_end, self.id, '0')


    def speciation(self, delta_t: int) -> None:
        species_sorted = sorted(self.species, key=lambda species: species.fitness_shared, reverse=True)
        representatives = [species.representative for species in species_sorted]
        unassigned_individuals = [individual for individual in self.individuals if individual not in representatives]
        
        while len(unassigned_individuals) > 0:
            individual = unassigned_individuals[0]

            for species in species_sorted:
                if species.compatability_distance(individual) < delta_t and individual not in species.individuals:
                    species.add_individual(individual)
                    unassigned_individuals.remove(individual)


                elif species.compatability_distance(individual) > delta_t and individual in species.individuals:
                    species.remove_individual(individual)


            if individual in unassigned_individuals:
                species_sorted.append(Species([individual], self.base_id + f".s{len(species_sorted)}"))
                unassigned_individuals.remove(individual)

        
        species_sorted = [Species(species.individuals, self.base_id + f".s{i}") for i, species in enumerate(species_sorted)]
        self.__init__(species_sorted, self.node_start, self.node_end, self.id, self.generation + 1)
        


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
    


class InputLayer(Layer):
    def __init__(self, input_shape: tuple, n_filters: int = 16, kernel_size: tuple = (3,3), activation: str = 'relu', padding: str = 'same') -> None:
        super().__init__()
        self.input_layer = Input(shape=input_shape)
        self.conv = Conv2D(n_filters, kernel_size, activation, padding)
        self.batch_norm = BatchNormalization()

    def call(self, inputs: Tensor) -> Tensor:
        x = self.input_layer(inputs)
        x = self.conv(x)
        x = self.batch_norm(x)
        return x



class CellBlock(Layer):
    def __init__(self, nodes: list[Node]) -> None:
        super().__init__()
        self.nodes = nodes
        for node in self.nodes:
            if node.node_type == 'Convolutional':
                filters = node.attributes["filters"]
                kernel_size = (node.attributes["kernel_size"], node.attributes["kernel_size"])
                strides = (node.attributes["strides"], node.attributes["strides"])
                padding = node.attributes["padding"]
                activation = node.attributes["activation"]
                node.layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)

            elif node.node_type == 'Pooling':
                pool_size = (node.attributes["pool_size"], node.attributes["pool_size"])
                strides = (node.attributes["strides"], node.attributes["strides"])
                padding = node.attributes["padding"]

                if node.attributes["type"] == "AveragePooling2D" or node.attributes["type"].lower() in "average":
                    node.layer = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)
                
                elif node.attributes["type"] == "MaxPooling2D" or node.attributes["type"].lower() in "maximum":
                    node.layer = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)

                else:
                    raise ValueError("Invalid pooling type. Must be either AveragePooling2D or MaxPooling2D")
                
            elif node.node_type == 'Dropout':
                node.layer = SpatialDropout2D(rate = node.attributes["rate"])

            else:
                raise ValueError("Invalid node type. Must be either Convolutional, Pooling or Dropout")
            
    def call(self, inputs: Tensor) -> Tensor:
        x = inputs
        for node in self.nodes:
            if len(node.input_layer) == 1:
                x = node.layer(x)

            else:
                x = Concatenate()([node.layer for node in node.input_layer])
                x = node.layer(x)

        return x

class Output(Layer):
    def __init__(self, n_classes:int) -> None:
        super().__init__()
        self.batch_norm = BatchNormalization()
        self.global_avg = GlobalAveragePooling2D()
        self.output_layer = Dense(n_classes, activation='softmax')

    def call(self, inputs: Tensor) -> Tensor:
        x = self.batch_norm(inputs)
        x = self.global_avg(x)
        x = self.output_layer(x)
        return x
    

class Network(Model):
    def __init__(self, input_shape: tuple, nodes: list[Node], n_classes: int, n_cells: int) -> None:
        super().__init__()
        self.input_layer = InputLayer(input_shape)
        self.cell_block = CellBlock(nodes)
        self.output = Output(n_classes)
        self.n_cells = n_cells

    def call(self, inputs: Tensor) -> Tensor:
        x = self.input_layer(inputs)
        for _ in range(self.n_cells):
            x = self.cell_block(x)
        x = self.output(x)
        return x
    
