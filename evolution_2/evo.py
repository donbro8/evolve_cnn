import numpy as np
import pandas as pd
from copy import deepcopy
import networkx as nx
import uuid
import itertools
from math import fsum
from itertools import accumulate
import datetime
import os
import pickle
from dataclasses import dataclass
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, MaxPooling2D, AveragePooling2D, SpatialDropout2D, GlobalAveragePooling2D, Flatten, Lambda, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras import backend 
import time

class Individual:

    individual_instances = set()
    path_instances = list()

    def __init__(self, edges: list[tuple[str,str]] = [('input', 'output')]) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges, weight = 1.0)
        self.fitness = 0.5
        self.is_trained = False
        self.age = 0
        self.__class__.individual_instances.add(self)
        self.id = len(self.__class__.individual_instances)
        self.initialise_evolution_tracker()

    def __repr__(self) -> str:
        return f"Individual {self.id} | Fitness: {self.fitness}"
    
    def initialise_evolution_tracker(
        self,
        species_id: str = None,
        species_start_generation: int = None,
        species_representative_id: int = None,
        species_similarity_score: float = None,
        species_shared_fitness: float = None,
        species_number_of_members: int = None,
        is_offspring: bool = False,
        offspring_of: tuple[int, int] = None,
        number_of_paths_inherited: int = None,
        crossover_shared_with: list[int] = [],
        offspring_generated: list[int] = [],
        is_mutated: bool = False,
        node_mutation: tuple[str, str, str] = None,
        connection_mutation: tuple[str, str] = None,
        switch_mutation: tuple[str, str, int] = None,
        training_history: dict = None,
        training_time: float = None,
        training_accuracy: float = None,
        training_loss: float = None,
        validation_accuracy: float = None,
        validation_loss: float = None,
        number_of_params: int = None
    ) -> None:
        self.species_id = species_id
        self.species_start_generation = species_start_generation
        self.species_representative_id = species_representative_id
        self.species_similarity_score = species_similarity_score
        self.species_shared_fitness = species_shared_fitness
        self.species_number_of_members = species_number_of_members
        self.is_offspring = is_offspring
        self.offspring_of = offspring_of
        self.number_of_paths_inherited = number_of_paths_inherited
        self.crossover_shared_with = crossover_shared_with
        self.offspring_generated = offspring_generated
        self.is_mutated = is_mutated
        self.node_mutation = node_mutation
        self.connection_mutation = connection_mutation
        self.switch_mutation = switch_mutation
        self.training_history = training_history
        self.training_time = training_time
        self.training_accuracy = training_accuracy
        self.training_loss = training_loss
        self.validation_accuracy = validation_accuracy
        self.validation_loss = validation_loss
        self.number_of_params = number_of_params
        
    
    
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
        for node in valid_nodes:
            nx.set_node_attributes(reduced_graph, {node:G.nodes[node]})
        return reduced_graph

    def maximum_possible_edges(self, G: nx.DiGraph) -> int:
        number_of_nodes = len(G.nodes())
        return int(number_of_nodes * (number_of_nodes - 1)/2)
    
    def add_edge(self, G: nx.DiGraph) -> None:
        if len(G.edges()) < self.maximum_possible_edges(G):
            self.is_built = False
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
                    self.connection_mutation = (node_in, node_out)
                    self.is_mutated = True
                    self.is_trained = False
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
            self.node_mutation = (node_in, node_out, node[0])
            self.is_mutated = True
            self.is_trained = False
        else:
            print(f"Warning: Node {node[0]} already exists.")
    
    def switch_edge_weight(self, G: nx.DiGraph, source = 'input', target = 'output') -> None:
        edges = list(G.edges())
        switched = False
        while len(edges) > 0 and not switched:
            edge = edges.pop(np.random.randint(len(edges)))
            if G[edge[0]][edge[1]]['weight'] == 1.0:
                G[edge[0]][edge[1]]['weight'] = 0.0
                try:
                    reduced_graph = self.reduced_graph(G, source, target)
                    for path in nx.all_simple_paths(reduced_graph, source, target):
                        if source in path and target in path:
                            switched = True
                            self.switch_mutation = (edge[0], edge[1], 0)
                            self.is_mutated = True
                            self.is_trained = False
                            break

                except:
                    print("warning: no paths to switch")
                
                G[edge[0]][edge[1]]['weight'] = 1.0

            else:
                G[edge[0]][edge[1]]['weight'] = 1.0
                self.switch_mutation = (edge[0], edge[1], 1)
                self.is_mutated = True
                self.is_trained = False
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


    def fitness_function(self, x, y, y_max, y_limit: int = 10000, beta:float = 0.5):
        return np.max([0, x - beta * (y/np.min([y_max, y_limit]))])



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
                    }#,
                # 'MaxPooling2D':{
                #     'pool_size':[(3,3)],
                #     'strides':[(1,1)],
                #     'padding':['same']
                #     },
                # 'SpatialDropout2D':{
                #     'rate':[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                #     }
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
            print(f"Warning: Individual {individual} already in species {self.id}.")


    def remove_member(self, individual: Individual) -> None:
        if individual in self.members:
            self.members.remove(individual)
        else:
            print(f"Warning: Individual {individual} not in species {self.id}.")


    def update_representative(self) -> None:
        self.representative = np.random.choice(self.members)

    def update_shared_fitness(self) -> None:
        self.shared_fitness = np.sum([member.fitness for member in self.members])/len(self.members)

    def update_individual_species_info(self, individual: Individual) -> None:
        individual.species_id = self.id
        individual.species_start_generation = self.start_generation
        individual.species_representative_id = self.representative.id
        individual.species_similarity_score = self.similarity(individual)
        individual.species_shared_fitness = self.shared_fitness
        individual.species_number_of_members = len(self.members)
    
    def species_age(self, generation: int) -> int:
        return generation - self.start_generation
    
    
    def sorensen_dice(self, A: set, B: set) -> float:
        return 2*len(A.intersection(B))/(len(A) + len(B))
    
    def similarity(self, individual: Individual, c1: float = 0.5, c2: float = 1.0) -> float:
        connection_similarity = self.sorensen_dice(set(self.representative.graph.edges()), set(individual.graph.edges()))
        path_similarity = self.sorensen_dice(set([tuple(path) for path in self.representative.get_path_genes()]), set([tuple(path) for path in individual.get_path_genes()]))
        return (c1*connection_similarity + c2*path_similarity)/(c1 + c2)
    
    def path_to_edges(self, path: list[str]) -> list[tuple[str,str]]:
        return [tuple(path[i:i+2]) for i in range(len(path) - 1)]
    

    def selection(self, n: int = 2) -> list[Individual]:
        p_selection = np.array([member.fitness for member in self.members])/np.sum([member.fitness for member in self.members])
        return np.random.choice(self.members, size = n, replace = False, p=p_selection)
    
    
    def crossover(self, individual_1: Individual, individual_2: Individual) -> Individual:

        individual_1.crossover_shared_with.append(individual_2.id)
        individual_2.crossover_shared_with.append(individual_1.id)

        individual_1_path_genes = individual_1.get_path_genes()
        individual_1_path_genes_set = set([tuple(path) for path in individual_1_path_genes])
        individual_1_paths = list(nx.all_simple_paths(individual_1.graph, 'input', 'output'))

        individual_2_path_genes = individual_2.get_path_genes()
        individual_2_path_genes_set = set([tuple(path) for path in individual_2_path_genes])
        individual_2_paths = list(nx.all_simple_paths(individual_2.graph, 'input', 'output'))

        fitness_probabilities = np.array([individual_1.fitness, individual_2.fitness])/np.sum([individual_1.fitness, individual_2.fitness])
        

        shared_path_genes = list(individual_1_path_genes_set.intersection(individual_2_path_genes_set))

        min_paths = np.random.randint(max([1, len(shared_path_genes)]), max([2, np.random.choice([len(individual_1_paths), len(individual_2_paths)], p = fitness_probabilities)])  + 1)
        
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

        offspring.offspring_of = (individual_1.id, individual_2.id)
        offspring.is_offspring = True
        offspring.number_of_paths_inherited = n_paths
        individual_1.offspring_generated.append(offspring.id)
        individual_2.offspring_generated.append(offspring.id)    
        
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
        for species in self.species:
            species.update_representative()
        self.reset_species()
        for individual in self.population:
            species_found = False
            for species in self.species:
                similarity_score = species.similarity(individual, c1, c2)
                if similarity_score >= similarity_threshold:
                    species_found = True
                    if individual != species.representative:
                        species.add_member(individual)
                    break
            if not species_found:
                self.species.append(Species([individual], start_generation=generation))
                if len(self.species) > maximum_species_proportion*self.population_size:
                    self.species.remove(sorted(self.species, key = lambda x: x.shared_fitness)[0])
        for species in self.species:
            species.update_shared_fitness()
            for individual in species.members:
                species.update_individual_species_info(individual)
    

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
    
    def generate_offspring(self, offspring_proportion: float = 0.5) -> None:
        
        shared_fitness = np.array([species.shared_fitness for species in self.species])/np.sum([species.shared_fitness for species in self.species])

        next_gen_species_count = np.round(self.population_size * shared_fitness, 0).astype(int)
        n_offspring = np.floor(next_gen_species_count*offspring_proportion).astype(int)

        leftover_offspring = self.population_size*offspring_proportion - np.sum(n_offspring)

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
                self.species[i].members = sorted(self.species[i].members, key = lambda x: x.fitness, reverse = True)

                parent_list = sorted(list(itertools.combinations(self.species[i].members, 2)), key = lambda x: x[0].fitness + x[1].fitness, reverse = True)

                n_generated = 0
                
                while n_generated < n_offspring[i]:
                    self.species[i].remove_member(self.species[i].members[-1])
                    parents = parent_list.pop(0)
                    self.species[i].add_member(self.species[i].crossover(parents[0], parents[1]))
                    n_generated += 1


class BuildLayer(Layer):

    # Need to pass in reduced graph to ensure only valid graph built
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
        measure_time: bool = True
    ):
        model.compile(
            optimizer = optimizer,
            loss = loss,
            metrics = metrics
        )
        if measure_time:
            start_time = time.time()
        history = model.fit(
            x = training_data[0],
            y = training_data[1],
            batch_size = batch_size,
            epochs = epochs,
            verbose = verbose,
            validation_data = validation_data
        )
        if measure_time:
            end_time = time.time()
            history.history['training_time'] = end_time - start_time
        return history
    

@dataclass
class ExperimentTracker:

    date_time: str
    run_number: int
    generation: int
    phase_name: str
    phase_threshold: float
    individual_mutation_rate: float
    node_mutation_prob: float
    connection_mutation_prob: float
    switch_mutation_prob: float
    individual: Individual

    def __post_init__(self):
        self.individual_id: int = self.individual.id
        self.individual_edges: list[tuple[str,str,dict]] = [edge for edge in self.individual.graph.edges(data = True)]
        self.individual_nodes: list[tuple[str,dict]] = [node for node in self.individual.graph.nodes(data = True)]
        self.individual_age: int = self.individual.age
        self.individual_fitness: float = self.individual.fitness
        self.individual_is_trained: bool = self.individual.is_trained
        self.species_id: str = self.individual.species_id
        self.species_start_generation: int = self.individual.species_start_generation
        self.species_representative_id: str = self.individual.species_representative_id
        self.species_similarity_score: float = self.individual.species_similarity_score
        self.species_shared_fitness: float = self.individual.species_shared_fitness
        self.species_number_of_members: int = self.individual.species_number_of_members
        self.is_offspring: bool = self.individual.is_offspring
        self.offspring_of: tuple[int,int] = self.individual.offspring_of
        self.number_of_paths_inherited: int = self.individual.number_of_paths_inherited
        self.crossover_shared_with: list[int] = self.individual.crossover_shared_with
        self.offspring_generated: list[int] = self.individual.offspring_generated
        self.is_mutated: bool = self.individual.is_mutated
        self.node_mutation: tuple[str,str,str] = self.individual.node_mutation
        self.connection_mutation: tuple[str,str] = self.individual.connection_mutation
        self.switch_mutation: tuple[str,str,int] = self.individual.switch_mutation
        self.training_history: dict = self.individual.training_history
        self.training_time: float = self.individual.training_time
        self.training_accuracy: float = self.individual.training_accuracy
        self.training_loss: float = self.individual.training_loss
        self.validation_accuracy: float = self.individual.validation_accuracy
        self.validation_loss: float = self.individual.validation_loss
        self.number_of_params: int = self.individual.number_of_params
    

class Evolution():

    def __init__(
        self,
        search_space: SearchSpace,
        input_graph: nx.DiGraph,
        output_graph: nx.DiGraph,
        reduction_cell_graph: nx.DiGraph,
        run_train_data: tuple[np.ndarray, np.ndarray], 
        run_validation_data: tuple[np.ndarray, np.ndarray],
        population_size: int = 10,
        initialisation_type: str = 'minimal',
        generations: int = 2,
        offspring_proportion: float = 0.5,
        phases: dict = {
            'rapid_expansion':{
                'generation_percentage':0.1,
                'individual_mutation_rate':1.0,
                'mutation_type_rate':[0.8,0.2,0.0]
            },
            'steady_growth':{
                'generation_percentage':0.7,
                'individual_mutation_rate':0.3,
                'mutation_type_rate':[0.4,0.4,0.2]
            },
            'local_exploration':{
                'generation_percentage':0.2,
                'individual_mutation_rate':0.8,
                'mutation_type_rate':[0.0,0.5,0.5]
            }
        },
        normal_cell_repeats: int = 3,
        substructure_repeats: int = 3,
        parameter_limit: int = 100000,
        complexity_penalty: float = 0.5,
        number_of_runs: int = 1,
        seed: int = 42

    ) -> None:
        self.search_space = search_space
        self.input_graph = input_graph
        self.output_graph = output_graph
        self.reduction_cell_graph = reduction_cell_graph
        self.population_size = population_size
        self.initialisation_type = initialisation_type
        self.generations = generations
        self.offspring_proportion = offspring_proportion
        self.phases = phases
        self.phase_names = list(phases.keys())
        self.phase_thresholds = np.array(list(accumulate([self.phases[phase]['generation_percentage']*self.generations for phase in self.phase_names])))
        self.normal_cell_repeats = normal_cell_repeats
        self.substructure_repeats = substructure_repeats
        self.run_train_data = run_train_data
        self.run_validation_data = run_validation_data
        self.parameter_limit = parameter_limit
        self.complexity_penalty = complexity_penalty
        self.number_of_runs = number_of_runs
        self.seed = seed

    
    def update_experiment_tracker(
        self,
        run_number: int,
        generation: int,
        phase_name: str,
        phase_threshold: float,
        individual_mutation_rate: float,
        node_mutation_prob: float,
        connection_mutation_prob: float,
        switch_mutation_prob: float,
        individuals: list[Individual],
        run_tracker: dict = {},
        save_to_file: bool = False,
        location: str = 'experiments',
        run_type: str = 'evolution'
    ) -> dict:
        
        run_id = 'r' + str(run_number) + '_g' + str(generation)

        run_tracker[run_id] = {
            'r' + str(run_number) + '_g' + str(generation) + '_i' + str(i):ExperimentTracker(
                date_time = datetime.datetime.now(),
                run_number = run_number,
                generation = generation,
                phase_name = phase_name,
                phase_threshold = phase_threshold,
                individual_mutation_rate = individual_mutation_rate,
                node_mutation_prob = node_mutation_prob,
                connection_mutation_prob = connection_mutation_prob,
                switch_mutation_prob = switch_mutation_prob,
                individual = individuals[i]
            )

            for i in range(len(individuals))
        }

        if save_to_file:
            self.pickle_experiment_tracker(run_tracker[run_id], filename = run_id, location = location, run_type = run_type)

        return run_tracker
    
    
    def pickle_experiment_tracker(self, evolution_tracker: dict, filename: str, location: str = 'experiments', run_type: str = 'evolution') -> None:

        if '.pkl' not in filename:
            filename += '.pkl'

        cwd = os.getcwd()
        path_dir = os.path.join(cwd, location, run_type)
        path_file = os.path.join(path_dir, filename)

        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        with open(path_file, 'wb') as f:
            pickle.dump(evolution_tracker, f)


    def pickle_to_pandas_dataframe(self, experiment_folder_path: str) -> pd.DataFrame:

        no_df = True

        for file_name in os.listdir(experiment_folder_path):
            path = os.path.join(experiment_folder_path, file_name)
            loaded_data = pickle.load(open(path, 'rb'))

            for key in loaded_data.keys():
                if no_df:
                    df = pd.DataFrame(data = {key:[value] for key, value in loaded_data[key].__dict__.items()}, index = [0])
                    no_df = False

                else:
                    df = pd.concat([df, pd.DataFrame(data = {key:[value] for key, value in loaded_data[key].__dict__.items()}, index = [df.shape[0]])])
        return df
    
    
    
    def single_evolutionary_run(self, run_number: int):

        # Initialise population
        print(f"Initialising a {self.initialisation_type} population of size {self.population_size} for run {run_number} of {self.number_of_runs}.")
        population = Population(population_size = self.population_size, initialisation_type = self.initialisation_type, search_space = self.search_space)
        population.minimal_initialisation()

        # Initialise phase
        phase_number = 0
        phase = self.phase_names[phase_number]
        phase_threshold = self.phase_thresholds[phase_number]
        individual_mutation_rate = self.phases[phase]['individual_mutation_rate']
        mutation_type_rate = self.phases[phase]['mutation_type_rate']
        maximum_params = 0

        evolution_tracker = self.update_experiment_tracker(
            run_number = run_number,
            generation = 0,
            phase_name = phase,
            phase_threshold = phase_threshold,
            individual_mutation_rate = individual_mutation_rate,
            node_mutation_prob = mutation_type_rate[0],
            connection_mutation_prob = mutation_type_rate[1],
            switch_mutation_prob = mutation_type_rate[2],
            individuals = population.population,
            save_to_file = True
        )

        print(f"Generation 0 of {self.generations} in phase {phase}.")

        for generation in range(1, self.generations + 1):

            # Update phase if threshold has been reached
            if generation >= phase_threshold:

                print('Phase threshold reached. Moving to next phase...')
                phase_number += 1
                phase = self.phase_names[phase_number]
                phase_threshold = self.phase_thresholds[phase_number]
                individual_mutation_rate = self.phases[phase]['individual_mutation_rate']
                mutation_type_rate = self.phases[phase]['mutation_type_rate']

            print(f"Generation {generation} of {self.generations} in phase {phase} with individual mutation rate of {individual_mutation_rate} and mutation type probabilities {list(zip(['node', 'connection', 'switch'], mutation_type_rate))}.")
            
            print(f"Generating offspring...")
            population.generate_offspring(offspring_proportion=self.offspring_proportion)

            print("Mutating population")
            for i in range(len(population.population)):
                individual = population.population[i]
                individual.is_mutated = False
                if np.random.rand() < individual_mutation_rate:
                    mutation_type = np.random.choice(['node', 'connection', 'switch'], p = mutation_type_rate)
                    print(f"Attempting {mutation_type} mutation for individual {individual.id}: {i + 1} of {len(population.population)}.")
                    if mutation_type == 'node':
                        node = self.search_space.sample_from_search_space(n_samples = 1)[0]
                        individual.add_node(individual.graph, node)
                    elif mutation_type == 'connection':
                        individual.add_edge(individual.graph)
                    else:
                        individual.switch_edge_weight(individual.graph)

                if individual.is_trained == False:
                    
                    print(f"Training individual {individual.id}: {i + 1} of {len(population.population)}.")
                    normal_cell_graph = individual.reduced_graph(individual.graph)
                    model_compiler = ModelCompiler(
                        input_graph=self.input_graph, 
                        output_graph=self.output_graph, 
                        normal_cell_graph=normal_cell_graph, 
                        reduction_cell_graph=self.reduction_cell_graph, 
                        normal_cell_repeats=self.normal_cell_repeats, 
                        substructure_repeats=self.substructure_repeats
                        )
                    model = model_compiler.build_model(input_shape = self.run_train_data[0].shape[1:])
                    for layer in model.layers:
                        if 'NC' in layer.name:
                            num_params = sum(backend.count_params(p) for p in layer.trainable_weights)
                            maximum_params = np.max([num_params, maximum_params])
                            break

                    history = model_compiler.train_model(training_data = self.run_train_data, validation_data = self.run_validation_data, model = model)
                    individual.training_history = history.history
                    individual.is_trained = True
                    individual.training_time = history.history['training_time']
                    individual.training_accuracy = history.history['accuracy'][-1]
                    individual.training_loss = history.history['mse'][-1]
                    individual.validation_accuracy = history.history['val_accuracy'][-1]
                    individual.validation_loss = history.history['val_mse'][-1]
                    individual.number_of_params = num_params
                    individual.fitness = individual.fitness_function(history.history['val_accuracy'][-1], num_params, maximum_params, self.parameter_limit, self.complexity_penalty)

                individual.age += 1

            print("Applying speciation to new population")
            population.speciation(generation)


            evolution_tracker = self.update_experiment_tracker(
                run_number = run_number,
                generation = generation,
                phase_name = phase,
                phase_threshold = phase_threshold,
                individual_mutation_rate = individual_mutation_rate,
                node_mutation_prob = mutation_type_rate[0],
                connection_mutation_prob = mutation_type_rate[1],
                switch_mutation_prob = mutation_type_rate[2],
                individuals = population.population,
                save_to_file = True
            )

        return population

    

    def single_random_run(self, run_number: int, minum_node_samples: int = 1, maximum_node_samples: int = 20, lowest_connection_density: float = 0.25):
        # Initialise population
        print(f"Initialising a random population of size {self.population_size} for run {run_number} of {self.number_of_runs}.")
        population = [Individual() for _ in range(self.population_size)]
        i = 0
        maximum_params = 0
        for individual in population:
            n_samples = np.random.randint(minum_node_samples, maximum_node_samples)
            samples = self.search_space.sample_from_search_space(n_samples = n_samples)
            connection_density = np.random.rand()*(1 - lowest_connection_density) + lowest_connection_density
            individual.random_individual(individual.graph, predefined_nodes=samples, minimum_connection_density = connection_density)
            normal_cell_graph = individual.reduced_graph(individual.graph)
            model_compiler = ModelCompiler(
                        input_graph=self.input_graph, 
                        output_graph=self.output_graph, 
                        normal_cell_graph=normal_cell_graph, 
                        reduction_cell_graph=self.reduction_cell_graph, 
                        normal_cell_repeats=self.normal_cell_repeats, 
                        substructure_repeats=self.substructure_repeats
                        )
            
            print(f"Training individual {individual.id}: {i + 1} of {self.population_size}.")
            model = model_compiler.build_model(input_shape = self.run_train_data[0].shape[1:])
            for layer in model.layers:
                if 'NC' in layer.name:
                    num_params = sum(backend.count_params(p) for p in layer.trainable_weights)
                    maximum_params = np.max([num_params, maximum_params])
                    break

            history = model_compiler.train_model(training_data = self.run_train_data, validation_data = self.run_validation_data, model = model)
            individual.training_history = history.history
            individual.is_trained = True
            individual.training_time = history.history['training_time']
            individual.training_accuracy = history.history['accuracy'][-1]
            individual.training_loss = history.history['mse'][-1]
            individual.validation_accuracy = history.history['val_accuracy'][-1]
            individual.validation_loss = history.history['val_mse'][-1]
            individual.number_of_params = num_params
            individual.fitness = individual.fitness_function(history.history['val_accuracy'][-1], num_params, maximum_params, self.parameter_limit, self.complexity_penalty)
            i += 1
        
        random_run_tracker = self.update_experiment_tracker(
            run_number = run_number,
            generation = 0,
            phase_name = None,
            phase_threshold = None,
            individual_mutation_rate = None,
            node_mutation_prob = None,
            connection_mutation_prob = None,
            switch_mutation_prob = None,
            individuals = population,
            save_to_file = True,
            run_type = 'random'
        )
        
        for generation in range(1, self.generations + 1):
            print(f"Generation {generation} of {self.generations}.")
            for i in range(np.max([1, int(self.population_size*self.offspring_proportion)])):
                population = sorted(population, key = lambda x: x.fitness, reverse = True)
                min_fitness = population[-1].fitness


                individual = Individual()
                n_samples = np.random.randint(minum_node_samples, maximum_node_samples)
                samples = self.search_space.sample_from_search_space(n_samples = n_samples)
                connection_density = np.random.rand()*(1 - lowest_connection_density) + lowest_connection_density
                individual.random_individual(individual.graph, predefined_nodes=samples, minimum_connection_density = connection_density)
                normal_cell_graph = individual.reduced_graph(individual.graph)
                model_compiler = ModelCompiler(
                            input_graph=self.input_graph, 
                            output_graph=self.output_graph, 
                            normal_cell_graph=normal_cell_graph, 
                            reduction_cell_graph=self.reduction_cell_graph, 
                            normal_cell_repeats=self.normal_cell_repeats, 
                            substructure_repeats=self.substructure_repeats
                            )
                
                print(f"Training individual {individual.id}: {i + 1} of {np.max([1, int(self.population_size*self.offspring_proportion)])}.")
                model = model_compiler.build_model(input_shape = self.run_train_data[0].shape[1:])
                for layer in model.layers:
                    if 'NC' in layer.name:
                        num_params = sum(backend.count_params(p) for p in layer.trainable_weights)
                        maximum_params = np.max([num_params, maximum_params])
                        break

                history = model_compiler.train_model(training_data = self.run_train_data, validation_data = self.run_validation_data, model = model)
                individual.training_history = history.history
                individual.is_trained = True
                individual.training_time = history.history['training_time']
                individual.training_accuracy = history.history['accuracy'][-1]
                individual.training_loss = history.history['mse'][-1]
                individual.validation_accuracy = history.history['val_accuracy'][-1]
                individual.validation_loss = history.history['val_mse'][-1]
                individual.number_of_params = num_params
                individual.fitness = individual.fitness_function(history.history['val_accuracy'][-1], num_params, maximum_params, self.parameter_limit, self.complexity_penalty)
                
                if individual.fitness > min_fitness:
                    population[-1] = individual


            random_run_tracker = self.update_experiment_tracker(
                run_number = run_number,
                generation = generation,
                phase_name = None,
                phase_threshold = None,
                individual_mutation_rate = None,
                node_mutation_prob = None,
                connection_mutation_prob = None,
                switch_mutation_prob = None,
                individuals = population,
                save_to_file = True,
                run_type = 'random'
            )

        return population
    



    def run_multiple_experiments(self, experiment_type: str = 'evolution') -> None:
        assert experiment_type in ['evolution', 'random'], "Experiment type must be either 'evolution' or 'random'."
        np.random.seed(self.seed)
        experiments = []
        for run_number in range(1, self.number_of_runs + 1):
            print(f"Starting {experiment_type} run {run_number} of {self.number_of_runs}.")
            if experiment_type == 'evolution':
                population = self.single_evolutionary_run(run_number)

            else:
                population = self.single_random_run(run_number)
            experiments.append(population)
        return experiments