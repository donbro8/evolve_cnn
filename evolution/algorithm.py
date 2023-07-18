from population import Individual, Population, Speciation, CustomPopulationInitialiser
from evolutionoperators import Mutation, Crossover
from modelbuilder import TrainModel
from graphs import Graph
from searchspace import SearchSpace
import numpy as np

class EvolveBlock:
    def __init__(self, population_size: int, generations: int, mutation_probability: float, crossover_probability: float, layers: SearchSpace) -> None:
        self.population_size = population_size
        self.generations = generations
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.layers = layers

    def generate_initial_population(self, initial_individuals: list[Individual]) -> Population:
        population = CustomPopulationInitialiser(self.population_size, initial_individuals)
        return population
    

    def run_genetic_operators(self, population: Population, mutation_type_probability: list[float] = [0.5, 0.5, 0.0]) -> None:

        new_species = Speciation(population.species).species
        new_population = Population(new_species)

        if np.random.uniform() < self.crossover_probability:
            crossover = Crossover(new_population)
            crossover.assign_offspring_count()
            new_population.species = crossover.generate_offspring(self.layers)
            new_population.update_population_info()


        if self.mutation_probability > 0:
            mutation = Mutation(new_population, self.layers, self.mutation_probability)
            mutation.mutate_population(mutation_type_probability)
            new_population.update_population_info()

        return new_population
    
    def run_evaluation(self, population: Population, input_graph: Graph, output_graph: Graph, train_data: tuple[np.ndarray], test_data: tuple[np.ndarray], validation_data: tuple[np.ndarray]) -> Population:
        for individual in population.individuals:
            graphs = [input_graph, individual, output_graph]
            model = TrainModel(graphs, train_data, test_data, validation_data)
            individual.fitness = model.evaluate_model()
        population.update_population_info()
        return population

    
    def run_evolution(self, initial_individuals: list[Individual], input_graph: Graph, output_graph: Graph, train_data: tuple[np.ndarray], test_data: tuple[np.ndarray], validation_data: tuple[np.ndarray]) -> Population:
        self.population = self.generate_initial_population(initial_individuals)
        
        for generation in range(self.generations):

            print(f"Generation {generation + 1} of {self.generations}")
            self.population = self.run_genetic_operators(self.population)
            self.population = self.run_evaluation(self.population, input_graph, output_graph, train_data, test_data, validation_data)

        return self.population