from dataclasses import dataclass
from evolution.population import Individual, Population
from evolution.search_space import SearchSpace
from evolution.network import ModelCompiler
import networkx as nx
import numpy as np
import datetime
from itertools import accumulate
import os
import pickle
import pandas as pd
from tensorflow.keras import backend
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from IPython.display import clear_output

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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
        complexity_penalty: float = 0.2,
        number_of_runs: int = 1,
        batch_size: int = 32, 
        epochs: int = 2, 
        verbose: int = 1,
        optimizer: str = 'adam',
        loss: str = 'categorical_crossentropy',
        metrics: list[str] = ['accuracy','mse'],
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
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
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
            filename +=  '.pkl'

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
    
    
    def train_individual(self, individual: Individual, maximum_params) -> None:

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

        individual.number_of_params = num_params
        
        try:
            if num_params <= self.parameter_limit:
                
                history = model_compiler.train_model(
                    training_data = self.run_train_data, 
                    validation_data = self.run_validation_data, 
                    model = model,
                    batch_size = self.batch_size,
                    epochs = self.epochs, 
                    verbose = self.verbose,
                    optimizer = self.optimizer,
                    loss = self.loss,
                    metrics = self.metrics
                    )
                
                
                individual.training_history = history.history
                individual.is_trained = True
                individual.training_time = history.history['training_time']
                individual.training_accuracy = history.history['accuracy'][-1]
                individual.training_loss = history.history['mse'][-1]
                individual.validation_accuracy = history.history['val_accuracy'][-1]
                individual.validation_loss = history.history['val_mse'][-1]
                individual.fitness = individual.fitness_function(history.history['val_accuracy'][-1], num_params, self.parameter_limit, self.complexity_penalty)

            else:
                print("Individual exceeds parameter limit. Fitness is set to zero.")
                individual.training_history = None
                individual.training_time = None
                individual.training_accuracy = None
                individual.training_loss = None
                individual.validation_accuracy = None
                individual.validation_loss = None
                individual.fitness = 0
    
        except Exception as e:
            if 'RESOURCE_EXHAUSTED:  OOM when allocating tensor' in str(e):
                print("Resource exhausted. Fitness is set to zero.")
                individual.training_history = None
                individual.training_time = None
                individual.training_accuracy = None
                individual.training_loss = None
                individual.validation_accuracy = None
                individual.validation_loss = None
                individual.fitness = 0

        return maximum_params
    
    # def single_generation(self, population: Population, )
    
    
    def single_evolutionary_run(self, run_number: int):

        # Initialise population
        print(f"Initialising a {self.initialisation_type} population of size {self.population_size} for run {run_number} of {self.number_of_runs}.")
        population = Population(population_size = self.population_size, initialisation_type = self.initialisation_type, search_space = self.search_space)
        population.minimal_initialisation()
        i = 0
        maximum_params = 1

        # Initialise phase
        phase_number = 0
        phase = self.phase_names[phase_number]
        phase_threshold = self.phase_thresholds[phase_number]
        individual_mutation_rate = self.phases[phase]['individual_mutation_rate']
        mutation_type_rate = self.phases[phase]['mutation_type_rate']

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
            if generation > phase_threshold:

                print('Phase threshold reached. Moving to next phase...')
                phase_number += 1
                phase = self.phase_names[phase_number]
                phase_threshold = self.phase_thresholds[phase_number]
                individual_mutation_rate = self.phases[phase]['individual_mutation_rate']
                mutation_type_rate = self.phases[phase]['mutation_type_rate']

            print(f"Generation {generation} of {self.generations} in phase {phase} with individual mutation rate of {individual_mutation_rate} and mutation type probabilities {list(zip(['node', 'connection', 'switch'], mutation_type_rate))}.")

            if generation > 1:
                print("Applying speciation to new population")
                population.speciation(generation)

                print(f"Generating offspring...")
                population.generate_offspring(offspring_proportion=self.offspring_proportion)
            
            self.population = population.population

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
                    print(f"Attempting training of individual {individual.id}: {i + 1} of {len(population.population)}.")
                    
                    maximum_params = self.train_individual(individual, maximum_params)

                individual.age += 1

            self.population = population.population

            self.population = population.population

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

            clear_output(wait=True)

        return population

    

    def single_random_run(self, run_number: int, minum_node_samples: int = 1, maximum_node_samples: int = 21):
        # Initialise population
        print(f"Initialising a random population of size {self.population_size} for run {run_number} of {self.number_of_runs}.")
        population = [Individual() for _ in range(self.population_size)]
        i = 0
        maximum_params = 0
        for individual in population:
            n_samples = np.random.randint(minum_node_samples, maximum_node_samples)
            samples = self.search_space.sample_from_search_space(n_samples = n_samples)
            connection_density = np.random.rand()*0.4 + 0.1
            individual.random_individual(individual.graph, predefined_nodes=samples, minimum_connection_density = connection_density)
            maximum_params = self.train_individual(individual, maximum_params)

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
            population = sorted(population, key = lambda x: x.fitness, reverse = True)[:int(np.floor(self.population_size*self.offspring_proportion))]
            while len(population) < self.population_size:

                individual = Individual()
                n_samples = np.random.randint(minum_node_samples, maximum_node_samples)
                samples = self.search_space.sample_from_search_space(n_samples = n_samples)
                connection_density = np.random.rand()*0.4 + 0.1
                individual.random_individual(individual.graph, predefined_nodes=samples, minimum_connection_density = connection_density)
                maximum_params = self.train_individual(individual, maximum_params)
                
                population.append(individual)


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

            clear_output(wait=True)

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