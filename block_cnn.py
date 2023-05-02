import yaml
import os
from gblock.functions import print_function, load_yaml
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



class BlockNEAT:

    def __init__(self, parameter_path):

        print('BlockNEAT')
        
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

        print('BlockNEAT.load_data')

        print('Loading data for use case: {}'.format(use_case))

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
                    print(e)
            else:
                (X_train, Y_train), (X_test, Y_test) = load_gibbon_data()

        return (X_train, Y_train), (X_test, Y_test)
    

    
    
    def generate_initial_population(self, path_to_minimal_individual, population_size = None):

        """
        Generate the minimal structure population for the blocks        
        """

        print('BlockNEAT.generate_initial_population')

        if self.neat_parameters['block']['population_size'] != None:
            self.population_size = self.neat_parameters['block']['population_size']

        elif population_size != None:
            self.population_size = population_size

        else:
            raise ValueError('Please specify a population size for the blocks')

        population = {}

        minimal_individual = load_yaml(path_to_minimal_individual)

        minimal_individual['meta_data']['connections'][0] = tuple(minimal_individual['meta_data']['connections'][0])

        for i in range(self.population_size):

            population['individual_' + str(i + 1)] = minimal_individual

        nodes = list(minimal_individual['nodes'].keys())

        connections = [('node_input', 'node_output', 'connection_1')]

        self.nodes = nodes
        self.connections = connections

        print('Generating initial population of size {} given the minimal structure at path {}.'.format(self.population_size, path_to_minimal_individual))
        
        return population #, nodes, connections
    


    def mutation_selection(self, population, mutation_rate = 0.01):

        """
        Select the individuals to be mutated
        """

        print('BlockNEAT.mutation_selection')

        individuals = list(population.keys())

        mutate_index = np.where(np.random.random(self.population_size) < mutation_rate)[0]

        print('Mutation selection: Mutating the following individuals: {}'.format(individuals[mutate_index]))

        return individuals[mutate_index]
    


    def define_new_node(self):

        """
        Add a new node to the individual
        """

        print('BlockNEAT.mutation_selection.mutate_add_node.define_new_node')

        # Select a random layer to add the node to
        layer_type = np.random.choice(list(self.model_parameters.keys())) # Think about using probability distribution for conv, pool, dropout (p =[0.8, 0.1, 0.1])

        # define the new node
        if layer_type == 'convolution':
            kernel = np.random.choice(self.model_parameters['convolution']['kernel'])
            filter = np.random.choice(self.model_parameters['convolution']['filter'])
            padding = np.random.choice(self.model_parameters['convolution']['padding'])
            node_attr = {'kernel':kernel, 'filter':filter, 'padding':padding}
            try:
                node_id = 'node_c' + str(max([int(item.replace('node_c', '')) for item in self.nodes if 'node_c' in item]) + 1)
            except:
                node_id = 'node_c1'


        elif layer_type == 'pooling':
            pool_type = np.random.choice(self.model_parameters['pooling']['type'])
            size = np.random.choice(self.model_parameters['pooling']['size'])
            node_attr = {'type':pool_type, 'size':size, 'padding':'same'}
            try:
                node_id = 'node_p' + pool_type[0] + str(max([int(item.replace('node_p' + pool_type[0], '')) for item in self.nodes if 'node_p' + pool_type[0] in item]) + 1)
            except:
                node_id = 'node_p' + pool_type[0] + '1'


        elif layer_type == 'dropout':
            dropout_rate = np.round(np.random.random()*(np.max(self.model_parameters['dropout']['rate']) - np.min(self.model_parameters['dropout']['rate'])) + np.min(self.model_parameters['dropout']['rate']), 2)
            node_attr = {'rate':dropout_rate}
            try:
                node_id = 'node_d' + str(max([int(item.replace('node_d', '')) for item in self.nodes if 'node_d' in item]) + 1)
            except:
                node_id = 'node_d1'

        else:
            raise ValueError('Layer type not recognised')
        
        print('Define new node: adding a new node of type {} with attributes {} and id {}'.format(layer_type, node_attr, node_id))

        return node_attr, layer_type, node_id
    

    
    def get_innovation_number(self, node_in_connection, node_out_connection):

        """
        Determine the innovation number and name for a connection
        """

        print('BlockNEAT.mutation_selection.get_innovation_number')

        if (node_in_connection, node_out_connection) not in [nodes[0:-1] for nodes in self.connections]:
            innovation_number = len(self.connections) + 1
            innovation_type = 'new'
            

        else:
            innovation_number = [i for i in range(len(self.connections)) if self.connections[i][0:-1] == (node_in_connection, node_out_connection)][0] + 1
            innovation_type = 'existing'

        connection_name = f'connection_{innovation_number}'

        print('Innovation number: Connection {} between nodes {} and {} has innovation number {} and is a {} connection.'.format(connection_name, node_in_connection, node_out_connection, innovation_number, innovation_type))

        return innovation_number, connection_name, innovation_type
    
    

    def update_nodes_local(self, individual):

        """
        For each node, update the local information (immediate connections)
        """

        print('BlockNEAT.mutation_selection.update_nodes_local')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

        connections = list(individual_copy['connections'].keys())

        nodes = list(individual_copy['nodes'].keys())

        for node in nodes:

            print('Updating local information for node {}'.format(node))

            individual_copy['nodes'][node]['connections_in'] = []
            individual_copy['nodes'][node]['connections_in_enabled'] = []
            individual_copy['nodes'][node]['nodes_in'] = []
            individual_copy['nodes'][node]['n_connections_in'] = 0


            individual_copy['nodes'][node]['connections_out'] = []
            individual_copy['nodes'][node]['connections_out_enabled'] = []
            individual_copy['nodes'][node]['nodes_out'] = []
            individual_copy['nodes'][node]['n_connections_out'] = 0

            for conn in connections:

                print('Updating local information for connection {}'.format(conn))

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

        print('BlockNEAT.mutation_selection.update_nodes_global.depth_first_search')

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

            print('Depth first search: visiting node {}'.format(node))
            
            visited.append(node)

            if enabled_only:
                indexes = individual['nodes'][node][conn_dir]

            else:
                indexes = [True]*len(individual['nodes'][node][node_dir])
        
            for neighbour in np.array(individual['nodes'][node][node_dir])[indexes]:

                print('Depth first search: visiting neighbouring node {}'.format(neighbour))
                
                connected_nodes.append(neighbour)
                
                self.depth_first_search(individual, neighbour, visited, connected_nodes, direction, enabled_only)

        
        return list(set(connected_nodes))
    
    

    def update_nodes_global(self, individual):

        """
        For each node, update the global information (all connections)
        """

        print('BlockNEAT.mutation_selection.update_nodes_global')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

        nodes = list(individual_copy['nodes'].keys())

        for node in nodes:

            print('Updating global information for node {}'.format(node))

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

        print('BlockNEAT.mutation_selection.update_meta_data')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

        individual_copy['meta_data']['connections'] = []
        individual_copy['meta_data']['connections_enabled'] = []
        individual_copy['meta_data']['n_convolution'] = 0
        individual_copy['meta_data']['n_pooling'] = 0
        individual_copy['meta_data']['n_dropout'] = 0

        for conn in individual_copy['connections']:

            print('Updating meta data for connection {}'.format(conn))

            individual_copy['meta_data']['connections'].append((individual_copy['connections'][conn]['in'], individual_copy['connections'][conn]['out'], conn))
            individual_copy['meta_data']['connections_enabled'].append(individual_copy['connections'][conn]['enabled'])


        for node in individual_copy['nodes']:

            print('Updating meta data for node {}'.format(node))

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

        print('BlockNEAT.mutation_selection.mutate_add_node')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

        # Select a random layer to add the node to
        node_attr, layer_type, node_id = self.define_new_node()

        # Get random connection to split using the node
        split_connection = np.random.choice(list(individual_copy['connections'].keys()))

        print('Node mutate: splitting {}'.format(split_connection))

        # The nodes that are connected by the addition of the new node for innovation number tracking
        node_in = individual_copy['connections'][split_connection]['in']
        node_out = individual_copy['connections'][split_connection]['out']

        print('Node mutate: adding node {} between {} and {}'.format(node_id, node_in, node_out))

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
        
        print('Node mutate: updating local information for individual')
        
        # Update local information for all nodes
        individual_copy = self.update_nodes_local(individual_copy)

        # self.individual_copy_local = individual_copy

        print('Node mutate: updating global information for individual')

        # Update global information for all nodes
        individual_copy = self.update_nodes_global(individual_copy)

        print('Node mutate: updating meta data for individual')

        # Update meta data
        individual_copy = self.update_meta_data(individual_copy)

        return individual_copy
    


    def mutate_add_connection(self, individual):

        """
        Add a new connection to the individual
        """

        print('BlockNEAT.mutation_selection.mutate_add_connection')

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

            print('Connection mutate: adding connection between {} and {}'.format(node_in, node_out))

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


            print('Connection mutate: updating local information for individual')
            
            # Update local information for all nodes
            individual_copy = self.update_nodes_local(individual_copy)

            print('Connection mutate: updating global information for individual')

            # Update global information for all nodes
            individual_copy = self.update_nodes_global(individual_copy)

            print('Connection mutate: updating meta data for individual')

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

        print('BlockNEAT.mutation_selection.mutate_switch_connection')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

        # Get list of connections
        connections = list(individual_copy['connections'].keys())

        # Select a random connection
        switch_connection = np.random.choice(connections)

        # Get nodes connected by the connection
        node_in = individual_copy['connections'][switch_connection]['in']
        node_out = individual_copy['connections'][switch_connection]['out']

        print('Switch mutate: switching connection {} between {} and {}'.format(switch_connection, node_in, node_out))

        # Switch the enabled/disabled status of the connection
        switch_value = not individual_copy['connections'][switch_connection]['enabled']
        individual_copy['connections'][switch_connection]['enabled'] = switch_value
        individual_copy['nodes'][node_in]['connections_out_enabled'][individual_copy['nodes'][node_in]['connections_out'].index(switch_connection)] = switch_value
        individual_copy['nodes'][node_out]['connections_in_enabled'][individual_copy['nodes'][node_out]['connections_in'].index(switch_connection)] = switch_value
        individual_copy['meta_data']['connections_enabled'][individual_copy['meta_data']['connections'].index((node_in, node_out, switch_connection))] = switch_value
            
        print('Switch mutate: updating local information for individual')
        
        # Update local information for all nodes
        individual_copy = self.update_nodes_local(individual_copy)

        print('Switch mutate: updating global information for individual')

        # Update global information for all nodes
        individual_copy = self.update_nodes_global(individual_copy)

        print('Switch mutate: updating meta data for individual')

        # Update meta data
        individual_copy = self.update_meta_data(individual_copy)

        # Check if the individual is continuous
        if self.check_continuity(individual_copy):

            print('Switch mutate: individual is continuous')

            return individual_copy
        
        else:

            print('Switch mutate: individual is not continuous')
            
            return None
        

    def crossover(self, parents):

        """
        Crossover two or more parents to produce an offspring
        """

        print('BlockNEAT.crossover_selection.crossover')

        offspring = load_yaml('/Users/Donovan/Documents/Masters/masters-ed02/clean_code/gblock/minimal_individual_structure.yaml')

        offspring['connections'] = {}

        # Get all innovations/connections and all unique innovations/connections given the parent list
        all_innovations = []
        all_fitnesses = []

        for parent in parents:

            all_fitnesses.append(parent['scores']['fitness'])

            for connection in list(parent['connections'].keys()):

                all_innovations.append(connection)
        
        all_innovations_set = list(set(all_innovations))

        print('Crossover: all innovations: {}'.format(all_innovations))

        # Loop through all unique innovations
        for innovation in all_innovations_set:

            print('Crossover: innovation: {}'.format(innovation))

            # Count the number of times the innovation appears in all innovations
            n_occurances = len([inno for inno in all_innovations if inno == innovation])

            print('Crossover: number of occurances: {} and length of parents {}'.format(n_occurances, len(parents)))
            
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

            print('Crossover: gene parent: {}'.format(gene_parent))

            
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
                print(e)

                print('Crossover: no connection to inherit. Continuing to next innovation')
                continue

        pprint.pprint(offspring)

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
        


    def build_layers(self, individual, inputs):

        """
        Define each layer/node in an individual/block given a set of attricutes for each.
        """

        print('BlockNEAT.build_layers')

        # Make a copy of the individual
        individual_copy = deepcopy(individual)

         # Loop through all nodes
        for node in individual_copy['nodes']:

            # If the layer already exists, then skip the node
            try:
                individual_copy['nodes'][node]['layer']

            # If the layer does not exist, then define the layer
            except:
                    
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

        print('BlockNEAT.build_block')

        print('Visited nodes:', visited_nodes)



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

                print('Node: {} | Neighbouring nodes: {} | Neighbour: {} | Valid nodes in: {} | n valid nodes in: {}'.format(node, neighbouring_nodes, neighbour, valid_nodes_in, n_valid_nodes_in))

                if n_valid_nodes_in == 1:

                    if neighbour == 'node_output':

                        # Set the output node to the preceding node's layer
                        individual['nodes'][neighbour]['layer'] = individual['nodes'][node]['layer']
                        individual['nodes'][neighbour]['attributes']['layer_object'] = individual['nodes'][node]['attributes']['layer_object']
                    
                    # If the neighbouring node is not the output node
                    else:

                        # Define the neighbouring node's layer with the preceding node's layer as input
                        individual['nodes'][neighbour]['layer'] = individual['nodes'][neighbour]['attributes']['layer_object'](individual['nodes'][node]['layer'])

                        print('--- Running recursion with node {} as the new node'.format(neighbour))
                        # Recursively call the function to define the layers for the neighbouring node, i.e. depth first search while the neighbouring node has only one incoming connection
                        self.build_block(individual, neighbour, valid_nodes, visited_nodes)


                # If the neighbouring node has more than one incoming connection
                elif n_valid_nodes_in > 1:

                    print('--- Adding layer of node {} to the list of layers to concatenate for node {}'.format(node, neighbour))

                    print('--- Concat list names for node {} with added node {}: {} + {}'.format(neighbour, node, individual['nodes'][neighbour]['concat_list_names'],  [node]))
                    
                    individual['nodes'][neighbour]['concat_list'] = individual['nodes'][neighbour]['concat_list'] + [individual['nodes'][node]['layer']]
                    individual['nodes'][neighbour]['concat_list_names'] = individual['nodes'][neighbour]['concat_list_names'] + [node]

                    # If all the layers for the input nodes to the neighbouring node have been defined and added to the list of layers to concatenate
                    if n_valid_nodes_in == len(individual['nodes'][neighbour]['concat_list']):

                        print('------ All layers for node {} have been added to the list of layers to concatenate'.format(neighbour))

                        # If the neighbouring node is the output node define it as a concatenation layer
                        if neighbour == 'node_output':
                            individual['nodes'][neighbour]['attributes']['layer_object'] = Concatenate(name = 'node_output')(individual['nodes'][neighbour]['concat_list'])
                            individual['nodes'][neighbour]['layer'] = individual['nodes'][neighbour]['attributes']['layer_object']

                        # Concatenate the layers in the list of layers to concatenate and give it as input to define the neighbouring node's layer
                        else:
                            individual['nodes'][neighbour]['layer'] = individual['nodes'][neighbour]['attributes']['layer_object'](Concatenate(name = '_'.join(individual['nodes'][neighbour]['concat_list_names']))(individual['nodes'][neighbour]['concat_list']))

                        print('--- Running recursion with node {} as the new node'.format(neighbour))
                        # Recursively call the function to define the layers for the next neighbouring node.
                        self.build_block(individual, neighbour, valid_nodes, visited_nodes)
                
                else:

                    raise Exception('Invalid number of incoming connections')
                
            visited_nodes.append(node)
        
        # Return the updated individual
        return individual


    def draw_block(self, individual, name, rankdir = 'LR', size = '10,5'):

        graph = graphviz.Digraph(name)
        graph.attr(rankdir = rankdir, size = size)

        nodes_order = self.breadth_first_search(individual)

        valid_nodes = self.valid_nodes(individual)

        for node in nodes_order:

            if node in valid_nodes:

                if node.split('_')[1][0] == 'i' or node.split('_')[1][0] == 'o':
                    graph.node(node, shape='box', color = 'black', label = node.split('_')[1])

                elif node.split('_')[1][0] == 'c':
                    graph.node(node, shape='doublecircle', color = 'red', label = node.split('_')[1])

                elif node.split('_')[1][0] == 'p':
                    graph.node(node, shape='trapezium', color = 'darkviolet', label = node.split('_')[1], orientation='-90')

                else:
                    graph.node(node, shape='circle', color = 'deepskyblue', label = node.split('_')[1])

            else:
                graph.node(node, shape='circle', color = 'lightgray', label = node.split('_')[1])

        
        for conn in individual['connections']:
            if individual['connections'][conn]['enabled'] and individual['connections'][conn]['in'] in valid_nodes and individual['connections'][conn]['out'] in valid_nodes:
                graph.edge(individual['connections'][conn]['in'], individual['connections'][conn]['out'], style = 'solid', label = conn.split('_')[1])
            else:
                graph.edge(individual['connections'][conn]['in'], individual['connections'][conn]['out'], color = 'lightgray', style = 'dashed', label = conn.split('_')[1])

        graph.view()


    def excess_genes(self, individual1, individual2):

        return np.abs(len(individual1['connections']) - len(individual2['connections']))
    

    def disjoint_genes(self, individual1, individual2):

        return len(set(individual1['connections']).symmetric_difference(set(individual2['connections'])))


    def disjoint_enabled_genes(self, individual1, individual2):

        return len(set([conn for conn in individual1['connections'] if individual1['connections'][conn]['enabled']]).symmetric_difference(set([conn for conn in individual2['connections'] if individual2['connections'][conn]['enabled']])))
    

    def compatibility_distance(self, c1, c2, c3, individual1, individual2):

        # Modified compatibility distance https://nn.cs.utexas.edu/soft-view.php?SoftID=4

        # Original is multiplied by 1/N where N is the number of genes in the larger genome

        # N = max(len(individual1['connections']), len(individual2['connections']))

        return c1*self.excess_genes(individual1, individual2) + c2*self.disjoint_genes(individual1, individual2) + c3*self.disjoint_enabled_genes(individual1, individual2)


    
    
    def speciation(self, population, delta_t, c1 = 1, c2 = 1, c3 = 1):

        population_copy = deepcopy(population)

        try:
            individuals = [item for sublist in [population_copy['species'][s]['members'] for s in population_copy['species'].keys()] for item in sublist]

        except:
            individuals = list(population_copy.keys())
            population_copy['species'] = {'species_1': {'members': individuals}}
        

        for s in population_copy['species']:

            representative = np.random.choice(population_copy['species'][s]['members'])
            population_copy['species'][s]['representative'] = representative
            population_copy['species'][s]['members'] = [representative]
            population_copy[representative]['meta_data']['species'] = s


        for individual in individuals:

            for s in population_copy['species']:

                representative = population_copy['species'][s]['representative']

                if individual != representative:

                    if self.compatibility_distance(c1, c2, c3, population_copy[individual], population_copy[representative]) < delta_t:
                        population_copy['species'][s]['members'].append(individual)
                        population_copy[individual]['meta_data']['species'] = s

                    else:
                        new_species = 'species_{}'.format(len(population_copy['species']) + 1)
                        population_copy['species'][new_species] = {'representative': individual, 'members': [individual]}
                        population_copy[individual]['meta_data']['species'] = new_species
                        break

        return population_copy
    

    def fitness_sharing(self, individual_i, population):

        # From NEAT paper
        # f_i_prime = f_i / sum_{j=1}^{N} sh(d(i,j))
        # sh(d(i,j)) = 1 if d(i,j) < delta_t
        # sh(d(i,j)) = 0 if d(i,j) >= delta_t
        # Therefore, fitness is only shared within the species since all outside of species have a distance of 0
        # Reduces to f_i_prime = f_i / N_species

        population_copy = deepcopy(population)

        fitness_i = population_copy[individual_i]['scores']['fitness']
        species = population_copy[individual_i]['meta_data']['species']
        members = population_copy['species'][species]['members']

        population_copy[individual_i]['scores']['fitness_shared'] = fitness_i / len(members)

        return population_copy



    def offspring_proportion(self, population):

        population_copy = deepcopy(population)

        population_fitness_sum = np.sum(np.array([population_copy[individual]['scores']['fitness_shared'] for individual in population_copy.keys()]))

        for s in population_copy['species']:

            fitness_species_sum = np.sum(np.array([population_copy[individual]['scores']['fitness_shared'] for individual in population_copy['species'][s]['members']]))

            population_copy['species'][s]['n_offspring'] = np.floor(fitness_species_sum / population_fitness_sum) * self.population_size

        return population_copy

    
    def get_subpopulation(self, population, proportion = 1/3, fitness_based_probability = 1):

        population_copy = deepcopy(population)

        subpopulation = {}

        for s in population_copy['species']:

            members = population_copy['species'][s]['members']

            sub_species_size = int(np.round(len(members) * proportion))

            r = np.random.random()

            if sub_species_size > 0 and r < fitness_based_probability:

                species_fitnesses_arg_sort = np.argsort(np.array([population_copy[individual]['scores']['fitness_shared'] for individual in members]))

                subpopulation[s] = {'members': np.array(members)[species_fitnesses_arg_sort][-sub_species_size:].tolist()}

            else:

                subpopulation[s] = {'members': np.random.choice(population_copy['species'][s]['members'], size = sub_species_size, replace = False).tolist()}

        return subpopulation
    
    
    def parent_selection(self, subpopulation, species, n = 2): # Ignoring fitness based selection for now
        return np.random.choice(subpopulation[species]['members'], size = n, replace = False).tolist()
    
    
    def get_validation_data(self, X, y, validation_split = None):

        if validation_split is None:
            validation_split = self.data_parameters['test_val_split']

        return train_test_split(X, y, test_size = validation_split, random_state = self.data_parameters['seed_value'])

    
    def compile_model(self, individual, name = 'block_cnn'):

        return Model(inputs = individual['nodes']['node_input'], outputs = individual['nodes']['node_output'], name = name)