import yaml
import os
from gblock.functions import print_function, load_yaml, print_to_log
import numpy as np
from keras.datasets import mnist, cifar10
from sklearn.model_selection import train_test_split
import pprint
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, AveragePooling2D, Flatten, Dense, BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D
from keras.utils.vis_utils import plot_model
from keras.models import Model
from copy import deepcopy
from data_preprocessing.data_loader import *
import logging
import networkx as nx
import graphviz
import json
from itertools import combinations

import sys 



class BlockNEAT:

    def __init__(self, parameter_path):

        print_to_log('BlockNEAT')
        
        self.parameters = load_yaml(parameter_path)

        self.general_parameters = self.parameters['general']

        self.data_parameters = self.parameters['data']

        self.model_parameters = self.parameters['model']

        self.neat_parameters = self.parameters['neat']

        if self.general_parameters['seed_value'] != None:
            np.random.seed(self.general_parameters['seed_value'])



    def load_data(self, use_case = None, **kwargs):

        """
        Load predefined use case data for the model
        """

        print_to_log('BlockNEAT.load_data')

        print_to_log('Loading data for use case: {}'.format(use_case))

        possible_use_cases = ['mnist', 'cifar10', 'cifar100', 'gibbon']

        if use_case == None or use_case not in possible_use_cases:
            raise ValueError('Please specify a valid use case from the following: {} or supply own data to model'.format(possible_use_cases))
        
        elif use_case == 'mnist':

            (X_train, Y_train), (X_test, Y_test) = load_mnist_data()

        elif use_case == 'cifar10':

            (X_train, Y_train), (X_test, Y_test) = load_cifar10_data()

        elif use_case == 'cifar100':

            (X_train, Y_train), (X_test, Y_test) = load_cifar100_data()

        elif use_case == 'gibbon':

            if len(kwargs) > 0:

                try:
                    (X_train, Y_train), (X_test, Y_test) = load_gibbon_data(**kwargs)

                except ValueError as e:
                    print_to_log(e)
            else:
                (X_train, Y_train), (X_test, Y_test) = load_gibbon_data()

        return (X_train, Y_train), (X_test, Y_test)
    

    
    
    def generate_initial_population(self, path_to_minimal_individual, population_size = None):

        """
        Generate the minimal structure population for the blocks        
        """

        print_to_log('BlockNEAT.generate_initial_population')

        if self.neat_parameters['block']['population_size'] != None:
            self.population_size = self.neat_parameters['block']['population_size']

        elif population_size != None:
            self.population_size = population_size

        else:
            raise ValueError('Please specify a population size for the blocks')

        population = {'individuals':{}}

        minimal_individual = load_yaml(path_to_minimal_individual)

        minimal_individual['meta_data']['connections'][0] = tuple(minimal_individual['meta_data']['connections'][0])

        for i in range(self.population_size):

            population['individuals']['individual_' + str(i + 1)] = minimal_individual

        nodes = list(minimal_individual['nodes'].keys())

        connections = [('node_input', 'node_output', 'connection_1')]

        self.nodes = nodes
        self.connections = connections

        print_to_log('Generating initial population of size {} given the minimal structure at path {}.'.format(self.population_size, path_to_minimal_individual))
        
        return population #, nodes, connections
    


    # Probably not needed, better to mutate the new offspring immediately after it is generated
    def mutation_selection(self, population, mutation_rate = 0.01):

        """
        Select the individuals to be mutated
        """

        print_to_log('BlockNEAT.mutation_selection')

        individuals = list(population.keys())

        mutate_index = np.where(np.random.random(self.population_size) < mutation_rate)[0]

        print_to_log('Mutation selection: Mutating the following individuals: {}'.format(individuals[mutate_index]))

        return individuals[mutate_index]
    


    def define_new_node(self):

        """
        Add a new node to the individual
        """

        print_to_log('BlockNEAT.mutation_selection.mutate_add_node.define_new_node')

        # Select a random layer to add the node to
        layer_type = np.random.choice(list(self.model_parameters.keys())) # Think about using probability distribution for conv, pool, dropout (p =[0.8, 0.1, 0.1])

        # define the new node
        if layer_type == 'convolution':
            kernel = np.random.choice(self.model_parameters['convolution']['kernel'])
            filter = np.random.choice(self.model_parameters['convolution']['filter'])
            padding = np.random.choice(self.model_parameters['convolution']['padding'])
            node_attr = {'kernel':kernel, 'filter':filter, 'padding':padding}
            node_id = 'node_c_k{}_f{}_p{}'.format(kernel, filter, padding)
            # try:
            #     node_id = 'node_c' + str(max([int(item.replace('node_c', '')) for item in self.nodes if 'node_c' in item]) + 1)
            # except:
            #     node_id = 'node_c1'


        elif layer_type == 'pooling':
            pool_type = np.random.choice(self.model_parameters['pooling']['type'])
            size = np.random.choice(self.model_parameters['pooling']['size'])
            node_attr = {'type':pool_type, 'size':size, 'padding':'same'}
            node_id = 'node_p_t{}_s{}'.format(pool_type, size)
            # try:
            #     node_id = 'node_p' + pool_type[0] + str(max([int(item.replace('node_p' + pool_type[0], '')) for item in self.nodes if 'node_p' + pool_type[0] in item]) + 1)
            # except:
            #     node_id = 'node_p' + pool_type[0] + '1'


        elif layer_type == 'dropout':
            dropout_rate = np.round(np.random.random()*(np.max(self.model_parameters['dropout']['rate']) - np.min(self.model_parameters['dropout']['rate'])) + np.min(self.model_parameters['dropout']['rate']), 2)
            node_attr = {'rate':dropout_rate}
            node_id = 'node_d_r{}'.format(dropout_rate)
            # try:
            #     node_id = 'node_d' + str(max([int(item.replace('node_d', '')) for item in self.nodes if 'node_d' in item]) + 1)
            # except:
            #     node_id = 'node_d1'

        else:
            raise ValueError('Layer type not recognised')
        
        print_to_log('Define new node: adding a new node of type {} with attributes {} and id {}'.format(layer_type, node_attr, node_id))

        return node_attr, layer_type, node_id
    

    
    def get_innovation_number(self, node_in_connection, node_out_connection):

        """
        Determine the innovation number and name for a connection
        """

        print_to_log('BlockNEAT.mutation_selection.get_innovation_number')

        if (node_in_connection, node_out_connection) not in [nodes[0:-1] for nodes in self.connections]:
            innovation_number = len(self.connections) + 1
            innovation_type = 'new'
            

        else:
            innovation_number = [i for i in range(len(self.connections)) if self.connections[i][0:-1] == (node_in_connection, node_out_connection)][0] + 1
            innovation_type = 'existing'

        connection_name = f'connection_{innovation_number}'

        print_to_log('Innovation number: Connection {} between nodes {} and {} has innovation number {} and is a {} connection.'.format(connection_name, node_in_connection, node_out_connection, innovation_number, innovation_type))

        return innovation_number, connection_name, innovation_type
    
    

    def update_nodes_local(self, individual):

        """
        For each node, update the local information (immediate connections)
        """

        print_to_log('BlockNEAT.mutation_selection.update_nodes_local')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

        connections = list(individual_copy['connections'].keys())

        nodes = list(individual_copy['nodes'].keys())

        for node in nodes:

            print_to_log('Updating local information for node {}'.format(node))

            individual_copy['nodes'][node]['connections_in'] = []
            individual_copy['nodes'][node]['connections_in_enabled'] = []
            individual_copy['nodes'][node]['nodes_in'] = []
            individual_copy['nodes'][node]['n_connections_in'] = 0


            individual_copy['nodes'][node]['connections_out'] = []
            individual_copy['nodes'][node]['connections_out_enabled'] = []
            individual_copy['nodes'][node]['nodes_out'] = []
            individual_copy['nodes'][node]['n_connections_out'] = 0

            for conn in connections:

                print_to_log('Updating local information for connection {}'.format(conn))

                if individual_copy['connections'][conn]['out'] == node:

                    individual_copy['nodes'][node]['connections_in'].append(conn)
                    individual_copy['nodes'][node]['connections_in_enabled'].append(individual_copy['connections'][conn]['enabled'])
                    individual_copy['nodes'][node]['nodes_in'].append(individual_copy['connections'][conn]['in'])
                    individual_copy['nodes'][node]['n_connections_in'] += 1

                elif individual_copy['connections'][conn]['in'] == node:

                    individual_copy['nodes'][node]['connections_out'].append(conn)
                    individual_copy['nodes'][node]['connections_out_enabled'].append(individual_copy['connections'][conn]['enabled'])
                    individual_copy['nodes'][node]['nodes_out'].append(individual_copy['connections'][conn]['out'])
                    individual_copy['nodes'][node]['n_connections_out'] += 1

        return individual_copy
    

    
    def depth_first_search(self, individual, node, visited = [], connected_nodes = [], direction = 'forward', enabled_only = True): 
        """
        Depth first search algorithm follows the path of enabled connections from a source node.
        If direction = 'forward', the algorithm follows all output connections and nodes until it reaches the output or a node with no output connections.
        Conversely, if direction = 'backward', the algorithm follows all input connections and nodes until it reaches the input or a node with no input connections.
        The algorithm returns a list of all nodes visited on the path.
        """

        print_to_log('BlockNEAT.mutation_selection.update_nodes_global.depth_first_search')

        base_conn = 'connections_'
        base_node = 'nodes_'

        if direction == 'forward':

            base_dir = 'out'

        elif direction == 'backward':

            base_dir = 'in'

        else:
            raise ValueError('Direction must be either forward or backward')

        if enabled_only:

            enabled_label = '_enabled'
            conn_dir = base_conn + base_dir + enabled_label

        else:

            enabled_label = ''
        
        node_dir = base_node + base_dir
        
        if node not in visited:

            print_to_log('Depth first search: visiting node {}'.format(node))
            
            visited.append(node)

            if enabled_only:
                indexes = individual['nodes'][node][conn_dir]

            else:
                indexes = [True]*len(individual['nodes'][node][node_dir])
        
            for neighbour in np.array(individual['nodes'][node][node_dir])[indexes]:

                print_to_log('Depth first search: visiting neighbouring node {}'.format(neighbour))
                
                connected_nodes.append(neighbour)
                
                self.depth_first_search(individual, neighbour, visited, connected_nodes, direction, enabled_only)

        
        return list(set(connected_nodes))
    
    

    def update_nodes_global(self, individual):

        """
        For each node, update the global information (all connections)
        """

        print_to_log('BlockNEAT.mutation_selection.update_nodes_global')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

        nodes = list(individual_copy['nodes'].keys())

        for node in nodes:

            print_to_log('Updating global information for node {}'.format(node))

            individual_copy['nodes'][node]['preceding_nodes'] = self.depth_first_search(individual_copy, node, visited = [], connected_nodes = [], direction = 'backward')
            individual_copy['nodes'][node]['following_nodes'] = self.depth_first_search(individual_copy, node, visited = [], connected_nodes = [], direction = 'forward')

            if len(individual_copy['nodes'][node]['following_nodes']) > 0 or node == 'node_output':
                individual_copy['nodes'][node]['connectedness'] = len(individual_copy['nodes'][node]['preceding_nodes']) / (len(individual_copy['nodes'])  - 1)

            else:
                individual_copy['nodes'][node]['connectedness'] = 0 

        return individual_copy
    

    def breadth_first_search(self, individual, node = 'node_input'): 
        
        queue = [node]

        visited = [node]
        
        while len(queue) != 0:

            node = queue[0]

            for neighbour in individual['nodes'][node]['nodes_out']:

                if neighbour not in visited:

                    queue.append(neighbour)

                    visited.append(neighbour)

            
            queue.remove(node)

        visited.remove('node_output')
        visited.append('node_output')

        return visited
    

    
    def update_meta_data(self, individual):

        """
        Update the meta data for the individual
        """

        print_to_log('BlockNEAT.mutation_selection.update_meta_data')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

        individual_copy['meta_data']['connections'] = []
        individual_copy['meta_data']['connections_enabled'] = []
        individual_copy['meta_data']['n_convolution'] = 0
        individual_copy['meta_data']['n_pooling'] = 0
        individual_copy['meta_data']['n_dropout'] = 0

        for conn in individual_copy['connections']:

            print_to_log('Updating meta data for connection {}'.format(conn))

            individual_copy['meta_data']['connections'].append((individual_copy['connections'][conn]['in'], individual_copy['connections'][conn]['out'], conn))
            individual_copy['meta_data']['connections_enabled'].append(individual_copy['connections'][conn]['enabled'])


        for node in individual_copy['nodes']:

            print_to_log('Updating meta data for node {}'.format(node))

            if individual_copy['nodes'][node]['type'] == 'convolution':
                individual_copy['meta_data']['n_convolution'] += 1

            elif individual_copy['nodes'][node]['type'] == 'pooling':
                individual_copy['meta_data']['n_pooling'] += 1

            elif individual_copy['nodes'][node]['type'] == 'dropout':
                individual_copy['meta_data']['n_dropout'] += 1
        
        individual_copy['meta_data']['n_nodes'] = len(individual_copy['nodes'])
        individual_copy['meta_data']['n_connections'] = len(individual_copy['connections'])
        individual_copy['meta_data']['n_enabled'] = sum(individual_copy['meta_data']['connections_enabled'])
        individual_copy['meta_data']['node_list'] = list(individual_copy['nodes'].keys())

        return individual_copy
    
    
    def mutate_add_node(self, individual):

        """
        Add a new node to the individual
        """

        print_to_log('BlockNEAT.mutation_selection.mutate_add_node')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

        # Select a random layer to add the node to
        node_attr, layer_type, node_id = self.define_new_node()

        # Get random connection to split using the node
        split_connection = np.random.choice(list(individual_copy['connections'].keys()))

        print_to_log('Node mutate: splitting {}'.format(split_connection))

        # The nodes that are connected by the addition of the new node for innovation number tracking
        node_in = individual_copy['connections'][split_connection]['in']
        node_out = individual_copy['connections'][split_connection]['out']

        print_to_log('Node mutate: adding node {} between {} and {}'.format(node_id, node_in, node_out))

        # Get the innovation numbers for the two connections formed by the addition of the new node
        innovation_number_in, connection_name_in, innovation_type_in = self.get_innovation_number(node_in, node_id)

        if innovation_type_in == 'new':
            self.connections.append((node_in, node_id)  + (connection_name_in,))

        innovation_number_out, connection_name_out, innovation_type_out = self.get_innovation_number(node_id, node_out)

        if innovation_type_out == 'new':
            self.connections.append((node_id, node_out) + (connection_name_out,))

        if node_id not in self.nodes:
            self.nodes.append(node_id)

        # Delete the split connection
        del individual_copy['connections'][split_connection]


        # Update the connection in information
        individual_copy['connections'][connection_name_in] = {
                                                                'in':node_in, 
                                                                'out':node_id,
                                                                'enabled':True, 
                                                                'innovation':innovation_number_in
        }
        
        # Update the connection out information
        individual_copy['connections'][connection_name_out] = {
                                                                'in':node_id, 
                                                                'out':node_out,
                                                                'enabled':True, 
                                                                'innovation':innovation_number_out
        }
        
        # Update the new node information
        individual_copy['nodes'][node_id] = {
                                            'type':layer_type,
                                            'attributes':node_attr,
                                            'n_connections_in':1,
                                            'connections_in':[connection_name_in],
                                            'connections_in_enabled':[True],
                                            'nodes_in':[node_in],
                                            'preceding_nodes':[],
                                            'n_connections_out':1,
                                            'connections_out':[connection_name_out],
                                            'connections_out_enabled':[True],
                                            'nodes_out':[node_out],
                                            'following_nodes':[],
                                            'connectedness':0
        }

        # self.individual_copy_updated = individual_copy
        
        print_to_log('Node mutate: updating local information for individual')
        
        # Update local information for all nodes
        individual_copy = self.update_nodes_local(individual_copy)

        # self.individual_copy_local = individual_copy

        print_to_log('Node mutate: updating global information for individual')

        # Update global information for all nodes
        individual_copy = self.update_nodes_global(individual_copy)

        print_to_log('Node mutate: updating meta data for individual')

        # Update meta data
        individual_copy = self.update_meta_data(individual_copy)

        return individual_copy
    


    def mutate_add_connection(self, individual):

        """
        Add a new connection to the individual
        """

        print_to_log('BlockNEAT.mutation_selection.mutate_add_connection')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

        nodes = individual_copy['meta_data']['node_list']

        # Exclude output node in possible nodes for input to connection
        possible_nodes_in = [node for node in nodes if node != 'node_output']

        # Select a random node going into connection
        node_in = np.random.choice(possible_nodes_in) # maybe look at adding a probablistic bias to this based on relative depth/connectedness


        # Compile node exclusion list and then determine possible nodes that can be used for output connection
        # Node exclude list inlcudes: 
        # - input node (no inputs allowed), 
        # - node_in (no recursion on itself), 
        # - nodes_out of node_in (already connected to), 
        # - preceding nodes of node_in (no recursion on preceding nodes)

        preceding_nodes = self.depth_first_search(individual, node_in, visited = [], connected_nodes = [], direction = 'backward', enabled_only = False)
        excluded_nodes = ['node_input'] + [node_in] + individual_copy['nodes'][node_in]['nodes_out'] + preceding_nodes
        possible_nodes_out = [node for node in nodes if node not in excluded_nodes]

        
        # If no possible nodes out, continue to next individual
        if len(possible_nodes_out) == 0:
            return None

        # Select a random node out
        else:

            node_out = np.random.choice(possible_nodes_out)

            print_to_log('Connection mutate: adding connection between {} and {}'.format(node_in, node_out))

            # Get the innovation numbers for the two connections formed by the addition of the new node
            innovation_number, connection_name, innovation_type = self.get_innovation_number(node_in, node_out)


            if innovation_type == 'new':
                self.connections.append((node_in, node_out)  + (connection_name,))


            # Update the connection in information
            individual_copy['connections'][connection_name] = {
                                                                    'in':node_in, 
                                                                    'out':node_out,
                                                                    'enabled':True, 
                                                                    'innovation':innovation_number
            }


            print_to_log('Connection mutate: updating local information for individual')
            
            # Update local information for all nodes
            individual_copy = self.update_nodes_local(individual_copy)

            print_to_log('Connection mutate: updating global information for individual')

            # Update global information for all nodes
            individual_copy = self.update_nodes_global(individual_copy)

            print_to_log('Connection mutate: updating meta data for individual')

            # Update meta data
            individual_copy = self.update_meta_data(individual_copy)

            return individual_copy
        
    

    def check_continuity(self, individual):

        """
        Check if the individual is continuous
        """

        following_nodes = individual['nodes']['node_input']['following_nodes']
        preceding_nodes = individual['nodes']['node_output']['preceding_nodes']

        if 'node_output' in following_nodes and 'node_input' in preceding_nodes:
            return True

        else:
            return False
        

    
    def mutate_switch_connection(self, individual):

        """
        Switch the enabled/disabled status of a random connection
        """

        print_to_log('BlockNEAT.mutation_selection.mutate_switch_connection')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

        # Get list of connections
        connections = list(individual_copy['connections'].keys())

        # Select a random connection
        switch_connection = np.random.choice(connections)

        # Get nodes connected by the connection
        node_in = individual_copy['connections'][switch_connection]['in']
        node_out = individual_copy['connections'][switch_connection]['out']

        print_to_log('Switch mutate: switching connection {} between {} and {}'.format(switch_connection, node_in, node_out))

        # Switch the enabled/disabled status of the connection
        switch_value = not individual_copy['connections'][switch_connection]['enabled']
        individual_copy['connections'][switch_connection]['enabled'] = switch_value
        individual_copy['nodes'][node_in]['connections_out_enabled'][individual_copy['nodes'][node_in]['connections_out'].index(switch_connection)] = switch_value
        individual_copy['nodes'][node_out]['connections_in_enabled'][individual_copy['nodes'][node_out]['connections_in'].index(switch_connection)] = switch_value
        individual_copy['meta_data']['connections_enabled'][individual_copy['meta_data']['connections'].index((node_in, node_out, switch_connection))] = switch_value
            
        print_to_log('Switch mutate: updating local information for individual')
        
        # Update local information for all nodes
        individual_copy = self.update_nodes_local(individual_copy)

        print_to_log('Switch mutate: updating global information for individual')

        # Update global information for all nodes
        individual_copy = self.update_nodes_global(individual_copy)

        print_to_log('Switch mutate: updating meta data for individual')

        # Update meta data
        individual_copy = self.update_meta_data(individual_copy)

        # Check if the individual is continuous
        if self.check_continuity(individual_copy):

            print_to_log('Switch mutate: individual is continuous')

            return individual_copy
        
        else:

            print_to_log('Switch mutate: individual is not continuous')
            
            return None
        

    def crossover(self, parents, population, path_to_minimal_individual):
        """
        Crossover two or more parents to produce an offspring
        """

        print_to_log('BlockNEAT.crossover_selection.crossover')

        offspring = load_yaml(path_to_minimal_individual)

        offspring['connections'] = {}

        # Get all innovations/connections and all unique innovations/connections given the parent list
        all_innovations = []
        all_fitnesses = []
        compatability_scores = []

        for parent_1 in parents:
            for parent_2 in parents:
                if parent_1 != parent_2:
                    compatability_scores.append(self.compatibility_distance(population['individuals'][parent_1], population['individuals'][parent_2]))

        if np.sum(compatability_scores) > 0:

            parents = [population['individuals'][parent] for parent in parents]

            for parent in parents:

                all_fitnesses.append(parent['scores']['fitness'])

                for connection in list(parent['connections'].keys()):

                    all_innovations.append(connection)
            
            all_innovations_set = list(set(all_innovations))

            print_to_log('Crossover: all innovations: {}'.format(all_innovations_set))
            # Loop through all unique innovations
            for innovation in all_innovations_set:

                print_to_log('Crossover: innovation: {}'.format(innovation))

                # Count the number of times the innovation appears in all innovations
                n_occurances = len([inno for inno in all_innovations if inno == innovation])

                print_to_log('Crossover: number of occurances: {} and length of parents {}'.format(n_occurances, len(parents)))
                
                # Check if the number of appearances match the number of parents or if the fitnesses are all the same
                if n_occurances == len(parents) or len(set(all_fitnesses)) == 1:

                    # If the innovation exists in all parents then randomly select a parent to inherit the connection from
                    gene_parent = np.random.choice(parents)

                # If the innovation does not appear in all parents, select the parent with the highest fitness
                else:
                    max_fitness = 0
                    for parent in parents:
                        if parent['scores']['fitness'] >= max_fitness:
                            max_fitness = parent['scores']['fitness']
                            gene_parent = parent

                print_to_log('Crossover: gene parent: {}'.format(gene_parent))

                
                # Inherit the connection from the selected parent

                try:
                    
                    inherit_connection = gene_parent['connections'][innovation]
                    offspring['connections'][innovation] = inherit_connection

                    # Get node in and out of connection
                    node_in = inherit_connection['in']
                    node_out = inherit_connection['out']

                    # Get the node information
                    offspring['nodes'][node_in] = gene_parent['nodes'][node_in]
                    offspring['nodes'][node_out] = gene_parent['nodes'][node_out]

                    # print('Crossover: inherited connection {} from parent'.format(innovation))
                    
                # If there is no connection to inherit (no corresponding innovation), then skip to the next innovation, i.e. inherit nothing
                except Exception as e:
                    print_to_log(e)
                    print_to_log('Crossover: no connection to inherit. Continuing to next innovation')
                    continue

            # pprint.pprint(offspring)

            # Update local information for all nodes
            offspring = self.update_nodes_local(offspring)

            # Update global information for all nodes
            offspring = self.update_nodes_global(offspring)

            # Update meta data
            offspring = self.update_meta_data(offspring)

            # Check if the individual is continuous
            if self.check_continuity(offspring):
                return offspring
            
            else:
                return None
            
        else:
            return population['individuals'][parents[0]]
        


    def build_layers(self, individual, inputs):

        """
        Define each layer/node in an individual/block given a set of attricutes for each.
        """

        print_to_log('BlockNEAT.build_layers')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

         # Loop through all nodes
        for node in individual_copy['nodes']:

            # # If the layer already exists, then skip the node
            # try:
            #     individual_copy['nodes'][node]['layer']

            # # If the layer does not exist, then define the layer
            # except:
                    
            # If the node is the input node, then set the layer equal to the inputs
            if node == 'node_input':

                individual_copy['nodes'][node]['attributes']['layer_object'] = inputs
                individual_copy['nodes'][node]['layer'] = inputs
                individual_copy['nodes'][node]['concat_list'] = []
                individual_copy['nodes'][node]['concat_list_names'] = []

            # If the node is the output node, then set the layer equal to None since it is defined by the previous incoming layer(s)
            elif node == 'node_output':
                individual_copy['nodes'][node]['layer'] = None
                individual_copy['nodes'][node]['concat_list'] = []
                individual_copy['nodes'][node]['concat_list_names'] = []

            # If the node is anything other than the input node or the output node

                # If the node is a convolution node
            elif individual_copy['nodes'][node]['type'] == 'convolution':

                # Get the attributes of the convolution node
                filters = individual_copy['nodes'][node]['attributes']['filter']
                kernel = (individual_copy['nodes'][node]['attributes']['kernel'], individual_copy['nodes'][node]['attributes']['kernel'])
                padding = individual_copy['nodes'][node]['attributes']['padding']

                # Set the layer to a convolution layer
                individual_copy['nodes'][node]['attributes']['layer_object'] = Conv2D(filters = filters, kernel_size = kernel, padding = padding, name = node) 

                # Functional layer is empty
                individual_copy['nodes'][node]['layer'] = None
                individual_copy['nodes'][node]['concat_list'] = []
                individual_copy['nodes'][node]['concat_list_names'] = []

            # If the node is a pooling node
            elif individual_copy['nodes'][node]['type'] == 'pooling':

                # Get the attributes of the pooling node
                pool_size = (individual_copy['nodes'][node]['attributes']['size'], individual_copy['nodes'][node]['attributes']['size'])
                pool_type = individual_copy['nodes'][node]['attributes']['type']
                padding = individual_copy['nodes'][node]['attributes']['padding']

                # Set the layer to a pooling layer
                if pool_type == 'max':
                    individual_copy['nodes'][node]['attributes']['layer_object'] = MaxPooling2D(pool_size = pool_size, strides=(1, 1), padding = padding, name = node)

                elif pool_type == 'average':
                    individual_copy['nodes'][node]['attributes']['layer_object'] = AveragePooling2D(pool_size = pool_size, strides=(1, 1), padding = padding, name = node)

                else:
                    raise Exception('Invalid pooling type')
                
                # Functional layer is empty
                individual_copy['nodes'][node]['layer'] = None
                individual_copy['nodes'][node]['concat_list'] = []
                individual_copy['nodes'][node]['concat_list_names'] = []

            # If the node is a dropout node
            elif individual_copy['nodes'][node]['type'] == 'dropout':

                # Get the attributes of the dropout node
                rate = individual_copy['nodes'][node]['attributes']['rate']

                # Set the layer to a dropout layer
                individual_copy['nodes'][node]['attributes']['layer_object'] = SpatialDropout2D(rate = rate, name = node)

                # Functional layer is empty
                individual_copy['nodes'][node]['layer'] = None
                individual_copy['nodes'][node]['concat_list'] = []
                individual_copy['nodes'][node]['concat_list_names'] = []

            else:
                raise Exception('Invalid node type')
                
        return individual_copy
    

    def valid_nodes(self, individual):

        """
        Return a list of nodes that are valid to be defined, the nodes that form part of a continuous connection between input and output.
        """

        valid_nodes = ['node_input', 'node_output']

        for node in list(individual['nodes'].keys()):

            if 'node_input' in individual['nodes'][node]['preceding_nodes'] and 'node_output' in individual['nodes'][node]['following_nodes']:

                valid_nodes.append(node)

        return valid_nodes
    


    def build_block(self, individual, node = 'node_input', valid_nodes = None, visited_nodes = [], visited_neighbours = []):

        """
        Build a block given a set of attributes for each node.
        """

        print_to_log('BlockNEAT.build_block')

        print_to_log('Visited nodes: {}'.format(visited_nodes))

        if valid_nodes == None:
            
            # Make a copy of the individual
            individual = deepcopy(individual)

            # Get the valid nodes
            valid_nodes = self.valid_nodes(individual)

        if node not in visited_nodes:

            # Check if the layer object is defined
            try:
                # Output layer is defined by the nodes coming into it, so it is ignored
                if node != 'node_output':
                    individual['nodes'][node]['attributes']['layer_object']
                else:
                    pass

            # Raise exception if the layer object has not been defined
            except:
                raise Exception(f'The layer for {node} is not defined. Try running build_layers first.')
            

            # Get neighbouring nodes where the connection is enabled
            neighbouring_nodes = list(np.array(individual['nodes'][node]['nodes_out'])[individual['nodes'][node]['connections_out_enabled']])

            for neighbour in neighbouring_nodes:

                visited_neighbours.append(neighbour)

                if neighbour not in valid_nodes:

                    continue

                connections_in_enabled = list(np.array(individual['nodes'][neighbour]['connections_in'])[individual['nodes'][neighbour]['connections_in_enabled']])

                valid_nodes_in = [individual['connections'][conn]['in'] for conn in connections_in_enabled if individual['connections'][conn]['in'] in valid_nodes]

                n_valid_nodes_in = len(valid_nodes_in)

                print_to_log('Node: {} | Neighbouring nodes: {} | Neighbour: {} | Valid nodes in: {} | n valid nodes in: {}'.format(node, neighbouring_nodes, neighbour, valid_nodes_in, n_valid_nodes_in))

                if n_valid_nodes_in == 1:

                    if neighbour == 'node_output':

                        # Set the output node to the preceding node's layer
                        individual['nodes'][neighbour]['layer'] = individual['nodes'][node]['layer']
                        individual['nodes'][neighbour]['attributes']['layer_object'] = individual['nodes'][node]['attributes']['layer_object']
                    
                    # If the neighbouring node is not the output node
                    else:

                        # Define the neighbouring node's layer with the preceding node's layer as input
                        individual['nodes'][neighbour]['layer'] = individual['nodes'][neighbour]['attributes']['layer_object'](individual['nodes'][node]['layer'])

                        print_to_log('--- Running recursion with node {} as the new node'.format(neighbour))
                        # Recursively call the function to define the layers for the neighbouring node, i.e. depth first search while the neighbouring node has only one incoming connection
                        self.build_block(individual, neighbour, valid_nodes, visited_nodes)


                # If the neighbouring node has more than one incoming connection
                elif n_valid_nodes_in > 1:

                    print_to_log('--- Adding layer of node {} to the list of layers to concatenate for node {}'.format(node, neighbour))

                    print_to_log('--- Concat list names for node {} with added node {}: {} + {}'.format(neighbour, node, individual['nodes'][neighbour]['concat_list_names'],  [node]))
                    
                    individual['nodes'][neighbour]['concat_list'] = individual['nodes'][neighbour]['concat_list'] + [individual['nodes'][node]['layer']]
                    individual['nodes'][neighbour]['concat_list_names'] = individual['nodes'][neighbour]['concat_list_names'] + [node]

                    # If all the layers for the input nodes to the neighbouring node have been defined and added to the list of layers to concatenate
                    if n_valid_nodes_in == len(individual['nodes'][neighbour]['concat_list']):

                        print_to_log('------ All layers for node {} have been added to the list of layers to concatenate'.format(neighbour))

                        # If the neighbouring node is the output node define it as a concatenation layer
                        if neighbour == 'node_output':
                            individual['nodes'][neighbour]['attributes']['layer_object'] = Concatenate(name = 'node_output')(individual['nodes'][neighbour]['concat_list'])
                            individual['nodes'][neighbour]['layer'] = individual['nodes'][neighbour]['attributes']['layer_object']

                        # Concatenate the layers in the list of layers to concatenate and give it as input to define the neighbouring node's layer
                        else:
                            individual['nodes'][neighbour]['layer'] = individual['nodes'][neighbour]['attributes']['layer_object'](Concatenate(name = '_'.join(individual['nodes'][neighbour]['concat_list_names']))(individual['nodes'][neighbour]['concat_list']))

                        print_to_log('--- Running recursion with node {} as the new node'.format(neighbour))
                        # Recursively call the function to define the layers for the next neighbouring node.
                        self.build_block(individual, neighbour, valid_nodes, visited_nodes)
                
                else:

                    raise Exception('Invalid number of incoming connections')
                
            visited_nodes.append(node)
        
        # Return the updated individual
        return individual



    def draw_block(self, individual, path, name, rankdir = 'LR', size = '10,5'):

        """
        Draw a block represented by an individual.
        """

        # Define the graph and its attributes
        graph = graphviz.Digraph(name)
        graph.attr(rankdir = rankdir, size = size)

        # Order the nodes using breadth first search algorithm
        nodes_order = self.breadth_first_search(individual)

        # Get the valid nodes
        valid_nodes = self.valid_nodes(individual)

        # Loop through all nodes
        for node in nodes_order:

            # If the node is a valid node give it a colour, shape and label depending on its type
            if node in valid_nodes:

                if node.split('_')[1][0] == 'i' or node.split('_')[1][0] == 'o':
                    graph.node(node, shape='box', color = 'black', label = node.split('_')[1])

                elif node.split('_')[1][0] == 'c':
                    graph.node(node, shape='doublecircle', color = 'red', label = node.split('_')[1])

                elif node.split('_')[1][0] == 'p':
                    graph.node(node, shape='trapezium', color = 'darkviolet', label = node.split('_')[1], orientation='-90')

                else:
                    graph.node(node, shape='circle', color = 'deepskyblue', label = node.split('_')[1])

            # If the node is not a valid node then it is still drawn but greyed out
            else:
                graph.node(node, shape='circle', color = 'lightgray', label = node.split('_')[1])

        # Draw the connections between the defined nodes
        for conn in individual['connections']:
            if individual['connections'][conn]['enabled'] and individual['connections'][conn]['in'] in valid_nodes and individual['connections'][conn]['out'] in valid_nodes:
                graph.edge(individual['connections'][conn]['in'], individual['connections'][conn]['out'], style = 'solid', label = conn.split('_')[1])
            else:
                graph.edge(individual['connections'][conn]['in'], individual['connections'][conn]['out'], color = 'lightgray', style = 'dashed', label = conn.split('_')[1])

        # Return the graph
        graph.render(directory = path + name, format = 'pdf').replace('\\', '/') # , format = 'pdf'

        individual['meta_data']['graph_path'] = path + name + '.pdf'

        return individual



    def excess_genes(self, individual1, individual2):

        """
        Number of excess genes between two individuals.

        Returns the absolute value of the difference between the number of connections in the two individuals.
        """

        return np.abs(len(individual1['connections']) - len(individual2['connections']))
    

    def disjoint_genes(self, individual1, individual2):

        """
        The number of mismatched (disjoint) genes between two individuals.

        Returns the length of the symmetric difference between the two individuals' connections (same as innovations).
        """

        return len(set(individual1['connections']).symmetric_difference(set(individual2['connections'])))


    def disjoint_enabled_genes(self, individual1, individual2):

        """
        The number of mismatched (disjoint) enabled genes between two individuals.

        Returns the length of the symmetric difference between the two individuals' enabled connections.
        """

        return len(set([conn for conn in individual1['connections'] if individual1['connections'][conn]['enabled']]).symmetric_difference(set([conn for conn in individual2['connections'] if individual2['connections'][conn]['enabled']])))
    

    def compatibility_distance(self, individual1, individual2, c1 = 1.0, c2 = 1.0, c3 = 1.0):

        """
        Compatibility distance between two individuals given the excess genes, disjoint genes and disjoint enabled genes.

        Returns the weighted sum of the excess genes, disjoint genes and disjoint enabled genes.
        """

        # Modified compatibility distance https://nn.cs.utexas.edu/soft-view.php?SoftID=4

        # Original is multiplied by 1/N where N is the number of genes in the larger genome

        N = max(len(individual1['connections']), len(individual2['connections']))

        return 1/N * (c1*self.excess_genes(individual1, individual2) + c2*self.disjoint_genes(individual1, individual2) + c3*self.disjoint_enabled_genes(individual1, individual2))


    
    
    def speciation(self, population, delta_t = 3.0):

        """
        The population is divided into species based on the compatibility distance between individuals.
        """

        print_to_log('Starting speciation ...')

        population_copy = deepcopy(population)


        # Get all individuals in the population ordered by their species
        try:
            individuals = [item for sublist in [population_copy['species'][s]['members'] for s in population_copy['species'].keys()] for item in sublist]
            print_to_log(f'Ordered list of individuals based on species: {individuals}')
        # If that fails then no species has been defined (i.e. first generation), so group them all into species_1
        except:
            individuals = list(population_copy['individuals'].keys())
            population_copy['species'] = {'species_1': {'members': individuals}}
            print_to_log(f'No species yet, grouping all individuals into species_1: {individuals}')


        assigned_individuals = []

        species = population_copy['species'].keys()

        # Loop through all species and get a representative individual for each
        for s in species:

            print_to_log(f'Getting representative from {s}')

            print_to_log(f'Possible individuals in {s}: {species}')

            representative = np.random.choice(population_copy['species'][s]['members'])
            population_copy['species'][s]['representative'] = representative
            population_copy['species'][s]['members'] = [representative]
            population_copy['individuals'][representative]['meta_data']['species'] = s

            assigned_individuals.append(representative)

        unassigned_individuals = [individual for individual in individuals if individual not in assigned_individuals]

        # Loop through all individuals 
        while len(unassigned_individuals) > 0:

            print_to_log('.Population: {}'.format(population_copy))

            print_to_log(f'Unassigned individuals: {unassigned_individuals}')

            print_to_log(f'Assigned individuals: {assigned_individuals}')

            individual = unassigned_individuals.pop(0)

            assigned_individuals.append(individual)

            print_to_log(f'__Selected individual to be assigned: {individual}')

            incompatability_count = 0

            # Loop through all species
            for s in species:

                print_to_log(f'____Checking if {individual} belongs in species {s}')

                # Get representative individual for the current species
                representative = population_copy['species'][s]['representative']

                # # If the current individual is not equal to the representative
                # if individual != representative:

                print_to_log(f'______Comparing {individual} to representative ({representative})')

                compatability_distance = self.compatibility_distance(population_copy['individuals'][individual], population_copy['individuals'][representative])

                print_to_log(f'______Compatability distance {compatability_distance} compared to delta {delta_t}')
                
                # If the compatibility distance between the current individual and the representative is less than delta_t
                if compatability_distance < delta_t:

                    # Add the current individual to the species and update its meta data
                    population_copy['species'][s]['members'].append(individual)
                    population_copy['individuals'][individual]['meta_data']['species'] = s

                    print_to_log('______Adding {} to species: {}'.format(individual,s))
                    break

                # If the compatibility distance between the current individual and the representative is greater than or equal to delta_t
                else:

                    # Increment the incompatability count
                    incompatability_count += 1
                    continue

            if incompatability_count == len(species):
                # Create a new species with the current individual as the representative, add the current individual to the species and update its meta data
                new_species = 'species_{}'.format(len(population_copy['species']) + 1)
                population_copy['species'][new_species] = {'representative': individual, 'members': [individual]}
                population_copy['individuals'][individual]['meta_data']['species'] = new_species

                print_to_log('______Creating new species {} with {}'.format(new_species, individual))


        # Return the speciated population
        return population_copy
    

    def fitness_sharing(self, individual_i, population):

        """
        The fitness of an indidivudal is calculated by dividing the individuals fitness by the number of members in the species.
        """

        # From NEAT paper
        # f_i_prime = f_i / sum_{j=1}^{N} sh(d(i,j))
        # sh(d(i,j)) = 1 if d(i,j) < delta_t
        # sh(d(i,j)) = 0 if d(i,j) >= delta_t
        # Therefore, fitness is only shared within the species since all outside of species have a distance of 0
        # Reduces to f_i_prime = f_i / N_species

        population_copy = deepcopy(population)

        # Get the individual fitness, species and members for the individual
        fitness_i = population_copy['individuals'][individual_i]['scores']['fitness']
        species = population_copy['individuals'][individual_i]['meta_data']['species']
        members = population_copy['species'][species]['members']

        print_to_log(f'Individual: {individual_i}')
        print_to_log(f'Fitness: {fitness_i}')
        print_to_log(f'Species: {species}')
        print_to_log(f'Members: {members}')
        print_to_log(f'Number of members: {len(members)}')

        # Calculate the shared fitness
        population_copy['individuals'][individual_i]['scores']['fitness_shared'] = fitness_i / len(members)

        print_to_log(f'Shared fitness (fitness/n_members): {population_copy["individuals"][individual_i]["scores"]["fitness_shared"]}')

        return population_copy



    def offspring_proportion(self, population):

        """
        The number of offspring to be generated by each species given the sum of the shared fitness of the species, the sum of the shared fitness for the population and the population size.
        """

        print_to_log('Calculating offspring proportion ...')

        population_copy = deepcopy(population)

        # Calculate the sum of the shared fitness for the entire population
        population_fitness_sum = np.sum(np.array([population_copy['individuals'][individual]['scores']['fitness_shared'] for individual in population_copy['individuals'].keys()]))

        print_to_log(f'Population fitness sum: {population_fitness_sum}')
        
        species = list(population_copy['species'].keys())

        n_members = np.array([len(population_copy['species'][s]['members']) for s in species])

        n_species = len(species)

        # n_individuals = self.population_size

        n_species_next_gen = n_species # int(np.min([np.floor(n_individuals / 2), n_species])) # Number of species cannot be greater than half the population size

        fitness_species_sums = [np.sum(np.array([population_copy['individuals'][individual]['scores']['fitness_shared'] for individual in population_copy['species'][s]['members']])) for s in species]
        
        fitness_species_sums = np.where(n_members <= 2, 0, fitness_species_sums)
        
        if np.sum(fitness_species_sums) == 0:
            arg_sorted_members = np.argsort(n_members)
            n_members = np.array(n_members)[arg_sorted_members]
            species = np.array(species)[arg_sorted_members]
            fitness_species_sums = np.array(fitness_species_sums)[arg_sorted_members]

        else:
            arg_sorted_fitness_species_sums = np.argsort(fitness_species_sums)
            n_members = np.array(n_members)[arg_sorted_fitness_species_sums]
            species = np.array(species)[arg_sorted_fitness_species_sums]
            fitness_species_sums = np.array(fitness_species_sums)[arg_sorted_fitness_species_sums]

        species = species[:n_species_next_gen]
        fitness_species_sums = fitness_species_sums[:n_species_next_gen]

        # fitness_species_sums = [np.sum(np.array([population_copy['individuals'][individual]['scores']['fitness_shared'] for individual in population_copy['species'][s]['members']])) for s in species]

        n_offspring = np.where(n_members < 2, 1, 0)
        n_offspring = np.where(n_members == 2, 2, n_offspring) # where there are two individuals in a species, two offspring are generated
        # n_offspring = np.where(fitness_species_sums == 0, 1, n_offspring)

        print_to_log(f'Number of species in {species} is {n_species}')
        print_to_log('Species fitness sums: {}'.format(fitness_species_sums))

        # n_members = [8, 4, 3, 2, 1, 1, 1, 1, 1, 1]
        # fitness_species_sums = [0.51, 0.5, 0.2, 0, 0, 0, 0, 0, 0, 0]
        # population_fitness_sum = sum(fitness_species_sums)

        # n_offspring = np.where(np.array(n_members) <= 2, 1, 0)
        # print(n_offspring)

        # n_remaining = np.sum(n_members) - np.sum(n_offspring)

        # for i in range(len(n_offspring)):
        #     if n_offspring[::-1][i] == 0:
        #         n_offspring[::-1][i] = np.ceil(fitness_species_sums[::-1][i] / population_fitness_sum * n_remaining)
        #         n_remaining -= n_offspring[::-1][i]

        #         if i == len(n_offspring) - 1:
        #             n_offspring[::-1][i] += n_remaining

        # print(n_offspring)
        # print(sum(n_offspring))

        assigned = sum(n_offspring)
        unassigned = self.population_size - assigned
        
        # Loop through all species
        for i in range(n_species_next_gen):
                
            if fitness_species_sums[i] == 0 and n_members[i] > 2:
                n_offspring[i] = n_members[i]

            else:
                if n_offspring[i] == 0:
                    n_offspring[i] = np.ceil(fitness_species_sums[i] / np.sum(fitness_species_sums) * unassigned)
                    unassigned -= n_offspring[i]

                    if i == n_species_next_gen - 1:
                        n_offspring[i] += unassigned

            population_copy['species'][species[i]]['n_offspring'] = n_offspring[i]

        print('species: ', species)
        print('n_offspring: ', n_offspring)

            # Assign the number of offspring to be generated by the current species
            # in descending order of fitness, i.e. prioritise the highest fitness species
            # Subtract the number of offspring assigned to the current species from the number of unassigned offspring
            # If the total n_offspring assigned is less than the population size after ordered proportioning, 
            # then distribute the remaning offspring evenly across the remaining species
            # This could lead to an excat same proportion as was input 

            # s = species[i]

            # print_to_log(f'Calculating offspring proportion for species: {s}')

            # # If the current species has less than 3 members, then only one offspring will be generated
            # if len(population_copy['species'][s]['members']) <= 2:
            #     print_to_log('Species has two or less members, so one offspring will be generated')
            #     population_copy['species'][s]['n_offspring'] = 1
            
            # # If the current species is the last species and the total number of offspring assigned so far is less than the population size
            # elif i == len(species) - 1 and assigned < self.population_size:
            #     print_to_log('Last species, so assigning remaining offspring')
            #     population_copy['species'][s]['n_offspring'] = unassigned
            #     assigned += unassigned
            #     unassigned = 0

            # # If there is only 1 species in the population, then all individuals are selected for reproduction
            # elif n_species == 1:
            #     print_to_log('Only one species in population, so all individuals will be selected for reproduction')
            #     population_copy['species'][s]['n_offspring'] = len(population_copy['individuals'].keys())


            # # If the population fitness sum is 0, then the number of offspring is the same as the number of members in the current species
            # elif population_fitness_sum == 0:
            #     print_to_log('Population fitness sum is 0, so proportion stays the same as before')
            #     population_copy['species'][s]['n_offspring'] = len(population_copy['species'][s]['members'])
            
            # # Otherwise, the number of offspring is calculated as the floor of the sum of the shared fitness for the current species divided by the sum of the shared fitness for the population multiplied by the population size
            # else:
            #     print_to_log(f'Calculating offspring proportion given (species fitness sum (population size {self.population_size})*{fitness_species_sums})/({population_fitness_sum} and population fitness sum)')
            #     population_copy['species'][s]['n_offspring'] = np.floor(fitness_species_sums[i] / population_fitness_sum) * self.population_size

            # assigned += population_copy['species'][s]['n_offspring']
            # unassigned -= population_copy['species'][s]['n_offspring']

        return population_copy

    
    def get_subpopulation(self, population, proportion = 2/3, fitness_based_probability = 1, n = 2):

        """
        A subpopulation is generated from the population by selecting a proportion of the population from each species.
        """

        population_copy = deepcopy(population)

        # Define empty dictionary for subpopulation
        subpopulation = {}

        # Loop through all species
        for s in population_copy['species'].keys():

            # Get the members of the current species
            members = population_copy['species'][s]['members']

            # Calculate the number of members to be selected from the current species given the proportion
            sub_species_size = int(np.round(len(members) * proportion))

            # Number of combinations
            n_combinations = len(list(combinations(members[0:sub_species_size], n)))

            if n_combinations >= population_copy['species'][s]['n_offspring']:

                # Get random number between 0 and 1
                r = np.random.random()

                # If the sub_species_size is greater than 0 and the random number is less than the fitness based probability
                if sub_species_size > 0 and r < fitness_based_probability:

                    # Sort the members of the current species by their shared fitness
                    species_fitnesses_arg_sort = np.argsort(np.array([population_copy['individuals'][individual]['scores']['fitness_shared'] for individual in members]))

                    # Select the top sub_species_size members from the current species, based on shared fitness and add them to the subpopulation for the current species
                    subpopulation[s] = {'members': np.array(members)[species_fitnesses_arg_sort][-sub_species_size:].tolist()}

                # If the sub_species_size is greater than 0 or the random number is greater than or equal to the fitness based probability
                else:
                    # Randomly select sub_species_size members from the current species and add them to the subpopulation for the current species
                    subpopulation[s] = {'members': np.random.choice(population_copy['species'][s]['members'], size = sub_species_size, replace = False).tolist()}

            else:
                values = [0]

                while True:
                    if len(list(combinations(values, n))) < population_copy['species'][s]['n_offspring']:
                        values.append(values[-1] + 1)

                    else:
                        break

                sub_species_size = len(values)

                # Get random number between 0 and 1
                r = np.random.random()

                # If the sub_species_size is greater than 0 and the random number is less than the fitness based probability
                if r < fitness_based_probability:

                    # Sort the members of the current species by their shared fitness
                    species_fitnesses_arg_sort = np.argsort(np.array([population_copy['individuals'][individual]['scores']['fitness_shared'] for individual in members]))

                    # Select the top sub_species_size members from the current species, based on shared fitness and add them to the subpopulation for the current species
                    subpopulation[s] = {'members': np.array(members)[species_fitnesses_arg_sort][-sub_species_size:].tolist()}

                # If the sub_species_size is greater than 0 or the random number is greater than or equal to the fitness based probability
                else:
                    # Randomly select sub_species_size members from the current species and add them to the subpopulation for the current species
                    subpopulation[s] = {'members': np.random.choice(population_copy['species'][s]['members'], size = sub_species_size, replace = False).tolist()}

        # Return the subpopulation
        return subpopulation
    
    
    def parent_selection(self, subpopulation, species, n = 2): # Ignoring fitness based selection for now

        """
        n parents are selected from the subpopulation of the current species.
        """

        return np.random.choice(subpopulation[species]['members'], size = n, replace = False).tolist()
    
    
    def get_validation_data(self, X, y, validation_split = None):

        """
        Split the data into training and validation sets.
        """

        if validation_split is None:
            validation_split = self.data_parameters['test_val_split']

        X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = validation_split, random_state = self.general_parameters['seed_value'])

        # X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
        # X_val  = X_val.reshape(X_val.shape[0], 28, 28, 1).astype('float32') / 255

        # # convert class vectors to binary class matrices
        # Y_train = to_categorical(Y_train, 10)
        # Y_val = to_categorical(Y_val, 10)

        return X_train, X_val, Y_train, Y_val

    
    def compile_network(self, individual, name = 'block_cnn'):

        """
        Define the model of the individual.
        """

        # For now, just a simple CNN block (output should be dynamic given the number of targets)
        flatten = Flatten()(individual['nodes']['node_output']['layer'])
        dense = Dense(128, activation = 'relu')(flatten) 
        output = Dense(10, activation = 'softmax')(dense)

        model = Model(inputs = individual['nodes']['node_input']['layer'], outputs = output, name = name)

        return model
    

    def train_model(self, individual, model, X_train, y_train, X_val, y_val):

        """
        Train the model of the individual and return the individual with the training history.
        """

        model.compile(loss = 'categorical_crossentropy', 
                                                 optimizer = 'adam', 
                                                 metrics = ['accuracy', 'mse'])
                                                #  , 
                                                #  val_metrics = ['accuracy', 'loss']), 'f1_score'

        history = model.fit(X_train, y_train, 
                                             epochs = self.general_parameters['epochs'], 
                                             batch_size = self.general_parameters['batch_size'], 
                                             verbose = self.general_parameters['verbose'], 
                                             validation_data = (X_val, y_val))
        
        individual['scores']['history'] = history.history

        return individual
    

    def count_block_params(self, model):

        """
        Count the number of parameters in the block.
        """

        params = 0

        for layer in model.layers:

            if layer.name == 'node_input':

                start = True

            if start == True:

                params += layer.count_params()

            if layer.name == 'flatten':

                start = False

        return params


    
    def fitness(self, individual, model, parameter_constraint = None, beta = 1):

        """
        Calculate the fitness of the individual given the number of parameters in the block and the latest validation accuracy.
        """

        block_params = self.count_block_params(model)
        val_accuracy = individual['scores']['history']['val_accuracy'][-1]

        self.block_params.append(block_params)
        self.val_accuracies.append(val_accuracy)

        if parameter_constraint is None:
            beta = 0 # Not currently including the block parameters in the fitness score
            parameter_constraint = block_params

        elif parameter_constraint is not None:
            parameter_constraint = parameter_constraint

        else:
            parameter_constraint = np.mean(list(set(self.block_params)))

        weight = beta/parameter_constraint

        print_to_log('Val accuracy: {}'.format(val_accuracy))
        print_to_log('Weighted block params: {}'.format(weight * block_params))
        print_to_log('Fitness: {}'.format(np.max([0, val_accuracy - weight * block_params])))

        return np.max([0, val_accuracy - weight * block_params]) 
    

    def delete_keys_from_dict(self, dictionary, label):

        modified_dict = {}
        for key, value in dictionary.items():
            if label not in key:
                if isinstance(value, dict):
                    modified_dict[key] = self.delete_keys_from_dict(value, label)
                else:
                    modified_dict[key] = value  # or copy.deepcopy(value) if a copy is desired for non-dicts.
        return modified_dict


    def evolve_block(self, X, y, save_path = os.getcwd() + '/data/evolution/', run = 1):

        
        print_to_log(50*'_')
        print_to_log('Evolution starting...')
        print_to_log(50*'_')

        generations = self.neat_parameters['block']['generation_limit']
        initial_mutation_probability = self.neat_parameters['block']['mutation_probability']
        minimum_mutation_probability = self.neat_parameters['block']['minimum_probability']

        

        self.population = self.generate_initial_population(os.getcwd() + '/gblock/minimal_individual_structure.yaml', self.neat_parameters['block']['population_size'])

        decay_constant = 1/self.population_size * np.log(minimum_mutation_probability/initial_mutation_probability)
        
        self.block_params = []
        self.val_accuracies = []
        
        X_train, X_val, y_train, y_val = self.get_validation_data(X, y)

        if os.path.exists(save_path + 'run_' + str(run) + '/'):
            shutil.rmtree(save_path + 'run_' + str(run) + '/')

        dir = save_path + 'run_' + str(run) + '/' + 'generation_' 

        os.makedirs(dir + str(0), exist_ok = True)

        for g in range(1, generations + 1):

            dir_gen = dir + str(g)

            print_to_log('____________Directory: {}'.format(dir))

            print_to_log('Generation: {} of {}'.format(g, generations))

            self.population = self.speciation(self.population) # delta_t is the compatibility threshold (add to parameters)

            print_to_log('Population: ' + str(self.population))

            for individual in self.population['individuals'].keys():

                if individual != 'species':

                    self.population = self.fitness_sharing(individual, self.population)

            self.population = self.offspring_proportion(self.population)

            with open(dir + str(g - 1) + '/population.txt', 'w') as f:

                f.write(str(self.population))

            subpopulation = self.get_subpopulation(self.population, proportion = 1, fitness_based_probability = 1) # proportion and fitness_based_probability are parameters to be reduced

            new_population = {'individuals': {}, 'species': {}}

            i = 0

            for species in subpopulation.keys():

                # offspring_count = 0
                # offspring_none_count = 0

                new_population['species'][species] = {'members': []}

                parents_selected = []

                print('Species: {}'.format(species))

                n_offspring = self.population['species'][species]['n_offspring']

                while len(new_population['species'][species]['members']) < n_offspring:

                    print('n_members: ', len(new_population['species'][species]['members']))

                    print('n_offspring: ', self.population['species'][species]['n_offspring'])

                    if len(subpopulation[species]['members']) == 1:
                        parents = [subpopulation[species]['members'][0]]

                    else:
                        parents = self.parent_selection(subpopulation, species, n = 2)

                    
                    # In order for this to work with a proportion smaller than 1, the number of combinations in
                    # the subpopulation needs to be greater than or equal to the number of offspring to be generated
                    # Including this check will ensure that the same parents are not selected twice, thus leading to duplicate 
                    # offspring being generated
                    # For now we use a proportion of 1, so it should not be an issue

                    # Also, if the number of members in the species is equal to one then the same parent will be selected twice
                    # Thus leading to the exact same offspring being generated
                    # Add a check to bypass this issue (species size stays the same)
                    # Two parents in a species can only produce one offspring (species shrinks)
                    # Three parents can only produce three offspring (species size stays the same)
                    # Four parents can only produce six offspring, etc. (potential for species to grow)

                    print('Parents: {}'.format(parents))
                    print('Parents selected: {}'.format(parents_selected))
                    if set(parents) not in parents_selected or (set(parents) in parents_selected and len(subpopulation[species]['members']) == 2):

                        
                        if set(parents) not in parents_selected and len(parents) > 1:
                            parents_selected.append(set(parents))
                            offspring = self.crossover(parents, self.population, os.getcwd() + '/gblock/minimal_individual_structure.yaml')

                        
                        else:
                            if len(parents) == 1:
                                parents = [parents[0]]
                                parents_selected.append(set(parents[0]))
                                offspring = deepcopy(self.population['individuals'][parents[0]])

                            else:
                                fittest_parent = parents[np.argmax([self.population['individuals'][parent]['scores']['fitness'] for parent in parents])]
                                parents = [fittest_parent]
                                parents_selected.append(set(parents))
                                offspring = deepcopy(self.population['individuals'][fittest_parent]) # Copy of fittest parent

                        
                        if offspring is not None:

                            # offspring_count += 1

                            mutation_probability = initial_mutation_probability * np.exp(decay_constant * g)

                            i += 1

                            species_name = species.split('_')[-1]

                            file_name = f'g{g}_i{i}_s{species_name}_p' + '_'.join([p.split('_')[-1] for p in parents])

                            individual_id = 'individual_{}'.format(i)

                            new_population['species'][species]['members'].append(individual_id)

                            # Repetition of code (can move to function)
                            r_node_mutate = np.random.uniform()

                            if r_node_mutate < mutation_probability:

                                mutate_offspring = self.mutate_add_node(offspring)

                                if mutate_offspring is not None:

                                    offspring = mutate_offspring

                            r_connection_mutate = np.random.uniform()

                            if r_connection_mutate < mutation_probability:
                                    
                                    mutate_offspring = self.mutate_add_connection(offspring)

                                    if mutate_offspring is not None:

                                        offspring = mutate_offspring

                            r_switch_connection_mutate = np.random.uniform()

                            # Only start switching connections after 50% of generations have passed and is inversely probable when compared to node and connection mutation
                            # The goal is to favour growth in the start and efficiency search toward the end
                            if r_switch_connection_mutate < (initial_mutation_probability - mutation_probability) and g/(generations + 1) > 0.5:
                                    
                                    mutate_offspring = self.mutate_switch_connection(offspring)

                                    if mutate_offspring is not None:

                                        offspring = mutate_offspring

                            
                            offspring = self.build_layers(offspring, inputs = Input(shape = (28, 28, 1), name = 'node_input'))

                            offspring = self.build_block(offspring, node = 'node_input', valid_nodes = None, visited_nodes = [], visited_neighbours = [])

                            print_to_log(50*'-')
                            print_to_log('Offspring: {}'.format(str(offspring)))
                            
                            offspring = self.draw_block(offspring, path = dir_gen + '/', name = file_name)

                            if offspring['meta_data']['n_convolution'] > 0:

                                model = self.compile_network(offspring, name = file_name)

                                self.model = model

                                self.offspring = offspring

                                offspring = self.train_model(offspring, model, X_train, y_train, X_val, y_val)
                            
                                offspring['scores']['fitness'] = self.fitness(offspring, model)
                            
                            new_population['individuals'][individual_id] = offspring

                            new_population['individuals'][individual_id]['meta_data']['species'] = species

                            new_population['individuals'][individual_id]['meta_data']['parents'] = set(parents)

                        # else:

                        #     offspring_none_count += 1

                    # if offspring_count + offspring_none_count == n_offspring:

                    #     self.population['species'][species]['n_offspring'] = offspring_count

                    #     n_offspring = offspring_count
                        
            self.previous_population = self.population
            
            self.population = new_population
