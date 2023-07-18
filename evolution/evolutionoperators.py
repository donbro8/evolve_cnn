from population import Individual, Population
from searchspace import SearchSpace
from graphs import Connection, Graph
from itertools import combinations, product
import numpy as np

class Mutation:
    def __init__(self, population: Population, layers: SearchSpace, mutation_probability: float) -> None:
        self.population = population
        self.layers = layers
        self.mutation_probability = mutation_probability

    def __repr__(self) -> str:
        return f"Mutation probability: {self.p_mutation}"
    
    def mutate_add_node(self, individual: Individual) -> None:

        new_node_layer_type = self.layers.get_random_layer_type(self.layers.local_layers)
        new_node_attributes = self.layers.get_random_layer_attributes(new_node_layer_type)

        while True:
            node_1 = individual.get_random_node()
            node_2 = individual.get_random_node()
            if node_1 != node_2:
                break
        
        if individual.order_nodes.index(node_1) < individual.order_nodes.index(node_2):
            node_in = node_1
            node_out = node_2

        else:
            node_in = node_2
            node_out = node_1

        split_connection = [connection for connection in individual.connections if connection.node_in == node_in and connection.node_out == node_out][0]
        individual.delete_connection(split_connection)
        new_node = self.layers.assign_node_layer(new_node_layer_type, new_node_attributes)
        individual.add_connection(Connection(node_in, new_node))
        individual.add_connection(Connection(new_node, node_out))
        

    def mutate_add_connection(self, individual: Individual) -> None:
        possible_nodes = individual.nodes
        max_possible_connections = len(list(combinations(possible_nodes, 2)))
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

    
    def mutate(self, individual: Individual, p_array: list[float] = [0.5, 0.5, 0.0]) -> None:
        mutation_type = np.random.choice(['add_node', 'add_connection', 'switch_connection'], p = p_array)
        if 'add_node' == mutation_type:
            self.mutate_add_node(individual)

        elif 'add_connection' == mutation_type:
            self.mutate_add_connection(individual)

        else:
            self.mutate_switch_connection(individual)

    def mutate_population(self, p_array: list[float] = [0.5, 0.5, 0.0]) -> None:
        selected_individuals = [individual for individual in self.population.individuals if np.random.uniform() < self.mutation_probability]
        for individual in selected_individuals:
            self.mutate(individual, p_array)


class Crossover:
    def __init__(self, population: Population) -> None:
        self.population = population

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
    

    def generate_offspring(self, layers: SearchSpace) -> list:
        self.layers = layers
        mutation = Mutation(self.population, self.layers, mutation_probability=1.0)
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