import numpy as np
from copy import deepcopy
import networkx as nx
import uuid
import itertools
from math import fsum


class Individual:

    individual_instances = set()
    path_instances = list()

    def __init__(self, edges: list[tuple[str,str]] = [('input', 'output')]) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges, weight = 1.0)
        self.fitness = 0.5
        self.age = 0
        self.__class__.individual_instances.add(self)
        self.id = len(self.__class__.individual_instances)

    def __repr__(self) -> str:
        return f"Individual {self.id} | Fitness: {self.fitness}"
    
    def enabled_edges(self, G: nx.DiGraph) -> list[tuple[str,str, float]]:
        return [(a, b, 1.0) for a,b in G.edges() if G.get_edge_data(a,b)['weight'] == 1]
    
    def valid_paths(self, G: nx.DiGraph, source: str = 'input', target: str = 'output') -> list[list[str]]:
        return list(nx.all_simple_paths(G, source, target))
    
    def convert_path_to_base_gene(self, path: list[str]) -> list[str]:
        return ['_'.join([node.split('_')[0],str(self.graph.nodes[node])]) for node in path]
    
    def get_path_genes(self, source: str = 'input', target: str = 'output') -> list[list[str]]:
        path_genes = []
        all_complete_paths = self.valid_paths(self.graph, source, target)
        for path in all_complete_paths:
            base_path = self.convert_path_to_base_gene(path)
            if base_path not in self.__class__.path_instances:
                self.__class__.path_instances.append(base_path)
            path_genes.append(base_path)
        return path_genes

    def valid_nodes(self, G: nx.DiGraph, source: str = 'input', target: str = 'output') -> list[str]:
        paths = self.valid_paths(G, source, target)
        return list(set([node for path in paths for node in path]))
    
    def reduced_graph(self, G: nx.DiGraph, source: str = 'input', target: str = 'output') -> nx.DiGraph:
        connected_graph = nx.DiGraph()
        connected_graph.add_weighted_edges_from(self.enabled_edges(G))
        valid_nodes = self.valid_nodes(connected_graph, source, target)
        reduced_graph = nx.DiGraph()
        reduced_graph.add_weighted_edges_from([(a, b, 1.0) for a,b in connected_graph.edges() if a in valid_nodes and b in valid_nodes])
        return reduced_graph

    def maximum_possible_edges(self, G: nx.DiGraph) -> int:
        number_of_nodes = len(G.nodes())
        return int(number_of_nodes * (number_of_nodes - 1)/2)
    
    def add_edge(self, G: nx.DiGraph) -> None:
        if len(G.edges()) < self.maximum_possible_edges(G):
            nodes = list(nx.topological_sort(G))
            while True:
                if np.random.rand() < 0.5:
                    node_in = np.random.choice(nodes[:-1])
                    node_out = np.random.choice(nodes[nodes.index(node_in) + 1:])
                else:
                    node_out = np.random.choice(nodes[1:])
                    node_in = np.random.choice(nodes[:nodes.index(node_out)])
                if not G.has_edge(node_in, node_out):
                    G.add_edge(node_in, node_out, weight = 1.0)
                    break
        else:
            print("Warning: Maximum number of edges reached.")
    
    def add_node(self, G: nx.DiGraph, node: tuple[str, dict]) -> None:
        if node[0] not in G.nodes():
            G.add_node(node[0], **node[1])
            edge = list(G.edges())[np.random.randint(len(G.edges()))]
            node_in = edge[0]
            node_out = edge[1]
            G.remove_edge(node_in, node_out)
            G.add_edge(node_in, node[0], weight = 1.0)
            G.add_edge(node[0], node_out, weight = 1.0)
        else:
            print(f"Warning: Node {node[0]} already exists.")
    
    def switch_edge_weight(self, G: nx.DiGraph, source = 'input', target = 'output') -> None:
        edges = list(G.edges())
        switched = False
        while len(edges) > 0 and not switched:
            edge = edges.pop(np.random.randint(len(edges)))
            if G[edge[0]][edge[1]]['weight'] == 1.0:
                G[edge[0]][edge[1]]['weight'] = 0.0
                reduced_graph = self.reduced_graph(G, source, target)
                for path in nx.all_simple_paths(reduced_graph, source, target):
                    if source in path and target in path:
                        switched = True
                        break
                    else:
                        G[edge[0]][edge[1]]['weight'] = 1.0
            else:
                G[edge[0]][edge[1]]['weight'] = 1.0
                break
    
    def ordered_nodes(self, G: nx.DiGraph) -> list[str]:
        return list(nx.topological_sort(G))
    
    def random_individual(self, G: nx.DiGraph, predefined_nodes: list[tuple[str, dict]], minimum_connection_density = 0.75) -> None:
        while True:
            max_edges = self.maximum_possible_edges(G)
            n_edges = len(G.edges())
            edge_density = n_edges/max_edges
            if len(predefined_nodes) == 0 and minimum_connection_density < edge_density:
                break

            else:
                if minimum_connection_density < edge_density:
                    node = predefined_nodes.pop(np.random.randint(len(predefined_nodes)))
                    self.add_node(G, node)

                else:
                    self.add_edge(G)



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
                    'pool_size':[(2,2)],
                    'strides':[(1,1)],
                    'padding':['same']
                    },
                'MaxPooling2D':{
                    'pool_size':[(2,2)],
                    'strides':[(1,1)],
                    'padding':['same']
                    },
                'SpatialDropout2D':{
                    'rate':[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
            print("Warning: No sample probabilities provided.")
            print("__Sampling uniformly from search layers.")
            sample_probabilities = np.array([1]*len(self.layer_types))/len(self.layer_types)

        else:
            if len(sample_probabilities) != len(self.layer_types):
                raise ValueError("Sample probabilities must be the same length as the number of layer types in the search space.")

            elif fsum(sample_probabilities) != 1.0:
                print("Warning: Sample probabilities must sum to 1.0.")
                print("__Normalising sample probabilities.")
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

class Species:

    species_instances = []

    def __init__(
            self,
            individuals: list[Individual],
            start_generation: int = 0
    ) -> None:
        self.members = individuals
        self.representative = np.random.choice(self.members)
        self.shared_fitness = np.sum([member.fitness for member in self.members])/len(self.members)
        self.start_generation = start_generation
        self.id = 'species_' + str(len(self.__class__.species_instances) + 1) + '_g' + str(self.start_generation)
        self.__class__.species_instances.append(self)

    def add_member(self, individual: Individual) -> None:
        if individual not in self.members:
            self.members.append(individual)
        else:
            print(f"Warning: Individual {individual} already in species {self}.")


    def remove_member(self, individual: Individual) -> None:
        if individual in self.members:
            self.members.remove(individual)
        else:
            print(f"Warning: Individual {individual} not in species {self}.")


    def update_representative(self) -> None:
        self.representative = np.random.choice(self.members)

    def update_shared_fitness(self) -> None:
        self.shared_fitness = np.sum([member.fitness for member in self.members])/len(self.members)

    
    def species_age(self, generation: int) -> int:
        return generation - self.start_generation
    
    
    def sorensen_dice(self, A: set, B: set) -> float:
        return 2*len(A.intersection(B))/(len(A) + len(B))
    
    def similarity(self, individual: Individual, c1: float = 0.5, c2: float = 1.0) -> float:
        connection_similarity = self.sorensen_dice(set(self.representative.graph.edges()), set(individual.graph.edges()))
        path_similarity = self.sorensen_dice(set(self.representative.get_path_genes()), set(individual.get_path_genes()))
        return (c1*connection_similarity + c2*path_similarity)/(c1 + c2)
    
    def path_to_edges(self, path: list[str]) -> list[tuple[str,str]]:
        return [tuple(path[i:i+2]) for i in range(len(path) - 1)]
    

    def selection(self, n: int = 2) -> list[Individual]:
        p_selection = np.array([member.fitness for member in self.members])/np.sum([member.fitness for member in self.members])
        return np.random.choice(self.members, size = n, replace = False, p=p_selection)
    
    
    def crossover(self, individual_1: Individual, individual_2: Individual) -> Individual:

        individual_1_path_genes = individual_1.get_path_genes()
        individual_1_path_genes_set = set([tuple(path) for path in individual_1_path_genes])
        individual_1_paths = list(nx.all_simple_paths(individual_1.graph, 'input', 'output'))

        individual_2_path_genes = individual_2.get_path_genes()
        individual_2_path_genes_set = set([tuple(path) for path in individual_2_path_genes])
        individual_2_paths = list(nx.all_simple_paths(individual_2.graph, 'input', 'output'))

        fitness_probabilities = np.array([individual_1.fitness, individual_2.fitness])/np.sum([individual_1.fitness, individual_2.fitness])
        

        shared_path_genes = list(individual_1_path_genes_set.intersection(individual_2_path_genes_set))

        min_paths = np.random.randint(max([1, len(shared_path_genes)]), max([1, np.random.choice([len(individual_1_paths), len(individual_2_paths)], p = fitness_probabilities)])  + 1)
        
        offspring_edges = []
        gene_edges = []

        n_paths = 0

        if len(shared_path_genes) > 0:
            for path_gene in shared_path_genes:
                if np.random.rand() < 0.5:
                    path_index = individual_1_path_genes.index(list(path_gene))
                    offspring_edges += self.path_to_edges(individual_1_paths[path_index])
                    gene_edges += self.path_to_edges(individual_1_path_genes[path_index])
                    
                
                else:
                    path_index = individual_2_path_genes.index(list(path_gene))
                    offspring_edges += self.path_to_edges(individual_2_paths[path_index])
                    gene_edges += self.path_to_edges(individual_2_path_genes[path_index])
                    
                n_paths += 1

        disjoint_paths = individual_1_path_genes_set.symmetric_difference(individual_2_path_genes_set)
        individual_1_disjoint_path_genes = list(individual_1_path_genes_set.intersection(disjoint_paths))
        individual_2_disjoint_path_genes = list(individual_2_path_genes_set.intersection(disjoint_paths))

        disjoint_paths = list(disjoint_paths)
        
        while True:

            if len(disjoint_paths) == 0 or min_paths <= n_paths or (len(individual_1_disjoint_path_genes) == 0 and len(individual_2_disjoint_path_genes) == 0):
                break

            
            else:
            
                if len(individual_1_disjoint_path_genes) > 0 and len(individual_2_disjoint_path_genes) > 0:

                    r = np.random.rand()
                
                    if r < fitness_probabilities[0]:
                        random_path = list(individual_1_disjoint_path_genes[np.random.randint(len(individual_1_disjoint_path_genes))])
                        path_index = individual_1_path_genes.index(random_path)
                        offspring_edges += self.path_to_edges(individual_1_paths[path_index])
                        gene_edges += self.path_to_edges(individual_1_path_genes[path_index])
                        disjoint_paths.remove(tuple(random_path))
                        individual_1_disjoint_path_genes.remove(tuple(random_path))
                        n_paths += 1

                    elif r >= fitness_probabilities[0]:
                        random_path = list(individual_2_disjoint_path_genes[np.random.randint(len(individual_2_disjoint_path_genes))])
                        path_index = individual_2_path_genes.index(random_path)
                        offspring_edges += self.path_to_edges(individual_2_paths[path_index])
                        gene_edges += self.path_to_edges(individual_2_path_genes[path_index])
                        disjoint_paths.remove(tuple(random_path))
                        individual_2_disjoint_path_genes.remove(tuple(random_path))
                        n_paths += 1

                else:
                
                    if len(individual_1_disjoint_path_genes) > 0 and len(individual_2_disjoint_path_genes) == 0:
                        random_path = list(individual_1_disjoint_path_genes[np.random.randint(len(individual_1_disjoint_path_genes))])
                        path_index = individual_1_path_genes.index(random_path)
                        offspring_edges += self.path_to_edges(individual_1_paths[path_index])
                        gene_edges += self.path_to_edges(individual_1_path_genes[path_index])
                        disjoint_paths.remove(tuple(random_path))
                        individual_1_disjoint_path_genes.remove(tuple(random_path))
                        n_paths += 1

                    elif len(individual_1_disjoint_path_genes) == 0 and len(individual_2_disjoint_path_genes) > 0:
                        random_path = list(individual_2_disjoint_path_genes[np.random.randint(len(individual_2_disjoint_path_genes))])
                        path_index = individual_2_path_genes.index(random_path)
                        offspring_edges += self.path_to_edges(individual_2_paths[path_index])
                        gene_edges += self.path_to_edges(individual_2_path_genes[path_index])
                        disjoint_paths.remove(tuple(random_path))
                        individual_2_disjoint_path_genes.remove(tuple(random_path))
                        n_paths += 1


        individual_1_edges = individual_1.graph.edges()
        individual_2_edges = individual_2.graph.edges()

        offspring = Individual()
        offspring.graph.remove_edge('input', 'output')
        for i in range(len(offspring_edges)):
            node_in = offspring_edges[i][0]
            node_out = offspring_edges[i][1]
            
            if offspring_edges[i] in individual_1_edges:

                if node_in not in offspring.graph.nodes():
                    offspring.graph.add_node(node_in, **individual_1.graph.nodes[node_in])

                if node_out not in offspring.graph.nodes():
                    offspring.graph.add_node(node_out, **individual_1.graph.nodes[node_out])

                offspring.graph.add_edge(node_in, node_out, weight = individual_1.graph[node_in][node_out]['weight'])
            
            elif offspring_edges[i] in individual_2_edges:

                if node_in not in offspring.graph.nodes():
                    offspring.graph.add_node(node_in, **individual_2.graph.nodes[node_in])

                if node_out not in offspring.graph.nodes():
                    offspring.graph.add_node(node_out, **individual_2.graph.nodes[node_out])

                offspring.graph.add_edge(node_in, node_out, weight = individual_2.graph[node_in][node_out]['weight'])
            
            else:
                raise ValueError("Invalid offspring edge.")
            
        return offspring
    
    
class Population:

    def __init__(
                self, 
                population_size: int=10,
                initialisation_type: str = 'minimal',
                search_space: SearchSpace = None
                ) -> None:
        assert initialisation_type in ['minimal', 'random'], "Initialisation type must be either 'minimal' or 'random'."
        if initialisation_type == 'random':
            assert search_space is not None, "Search space must be provided for random initialisation."
            self.search_space = search_space
        self.population_size = population_size
        self.initialisation_type = initialisation_type
        self.species = []
        
    def minimal_initialisation(self) -> None:
        self.population = [Individual() for _ in range(self.population_size)]
        self.species = [Species(self.population)]

    def random_initialisation(self, n_node_samples: int = 10, sample_probabilties = None, minimum_connection_density: float = 0.75) -> None:
        self.population = [Individual() for _ in range(self.population_size)]
        for individual in self.population:
            samples = self.search_space.sample_from_search_space(n_samples = n_node_samples, sample_probabilities = sample_probabilties)
            individual.random_individual(individual.graph, predefined_nodes=samples, minimum_connection_density = minimum_connection_density)
        self.species = [Species(self.population)]


    def reset_species(self) -> None:
        for species in self.species:
            for individual in species.members:
                if individual != species.representative:
                    species.remove_member(individual)
    
    def speciation(self, generation: int, c1: float = 0.5, c2: float = 1.0, similarity_threshold: float = 0.5, maximum_species_proportion: float = 0.2) -> None:
        self.reset_species()
        for individual in self.population:
            species_found = False
            for species in self.species:
                if species.similarity(individual, c1, c2) >= similarity_threshold:
                    species_found = True
                    if individual != species.representative:
                        species.add_member(individual)
                        break
            if not species_found:
                self.species.append(Species([individual], start_generation=generation))
                if len(self.species) > maximum_species_proportion*self.population_size:
                    self.species.remove(sorted(self.species, key = lambda x: x.shared_fitness)[0])
        for species in self.species:
            species.update_representative()
            species.update_shared_fitness()

    def remove_oldest_individual(self) -> None:
        self.population.remove(sorted(self.population, key = lambda x: x.age)[-1])
    
    def remove_worst_individuals(self, percentage_population: float = 0.5) -> None:
        number_of_individuals_to_remove = int(percentage_population*self.population_size)
        number_of_individuals_to_keep = self.population_size - number_of_individuals_to_remove
        self.population = sorted(self.population, key = lambda x: x.fitness, reverse = True)[:number_of_individuals_to_keep]
        for species in self.species:
            species.members = [member for member in species.members if member in self.population]
    
    def get_species_age(self, generation: int) -> list[int]:
        return [species.species_age(generation) for species in self.species]
    
    def maximum_unique_pairs(self, n: int = 2) -> int:
        return int(n*(n-1)/2)
    
    def generate_offspring(self, percentage_offspring: float = 0.5) -> None:
        
        shared_fitness = np.array([species.shared_fitness for species in self.species])/np.sum([species.shared_fitness for species in self.species])

        next_gen_species_count = np.round(self.population_size * shared_fitness, 0).astype(int)
        n_offspring = np.floor(next_gen_species_count*percentage_offspring).astype(int)

        leftover_offspring = self.population_size*percentage_offspring - np.sum(n_offspring)

        n_offspring = n_offspring + np.random.multinomial(leftover_offspring, shared_fitness)
        
        for i in range(len(self.species)):

            max_pairs = self.maximum_unique_pairs(len(self.species[i].members))

            if max_pairs < n_offspring[i]:
                
                # Get every combination of parent pairs
                parent_list = list(itertools.combinations(self.species[i].members, 2))

                for parents in parent_list:
                    self.species[i].add_member(self.species[i].crossover(parents[0], parents[1]))

                while len(self.species[i].members) < next_gen_species_count[i]:
                    self.species[i].add_member(Individual())

            # If there are enough/more than enough possible combinations, then we remove the worst performing
            # individuals and use every parent combination to generate offspring
            else:
                min_pairs = self.maximum_unique_pairs(n_offspring[i])
                n_members_to_keep = np.random.randint(min_pairs, max_pairs)
                self.species[i].members = sorted(self.species[i].members, key = lambda x: x.fitness, reverse = True)[:n_members_to_keep]

                parent_list = list(itertools.combinations(self.species[i].members, 2))

                while len(self.species[i].members) < next_gen_species_count[i]:
                    parents = parent_list.pop(np.random.randint(len(parent_list)))
                    self.species[i].add_member(self.species[i].crossover(parents[0], parents[1]))