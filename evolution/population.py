from graphs import Graph, Connection
import numpy as np
import uuid

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
        return self.id
    
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
        self.id = '_'.join(['species', str(len(Species.species_instances) - 1), self.uuid[:4]])


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

        if enabled_only:
            individual_connections = set([individual.connections[i] for i in range(len(individual.connections)) if individual.enabled_connections[i]])
            representative_connections = set([self.representative.connections[i] for i in range(len(self.representative.connections)) if self.representative.enabled_connections[i]])

        else:
            
            individual_connections = set(individual.connections)
            representative_connections = set(self.representative.connections)

        return len(individual_connections.symmetric_difference(representative_connections))/(len(individual_connections) + len(representative_connections))


class Speciation:
    def __init__(self, input_species: list[Species], delta_t: float = 0.5) -> list[Species]:
        self.species = sorted(input_species, key=lambda x: x.fitness_shared, reverse=True)
        self.delta_t = delta_t
        self.individuals = [individual for species in self.species for individual in species.members]
        self.representatives = [species.representative for species in self.species]
        unassigned_individuals = [individual for individual in self.individuals if individual not in self.representatives]
        while len(unassigned_individuals) > 0:
            individual = unassigned_individuals[0]
            for species in self.species:
                if species.compatability_distance(individual) < delta_t:
                    if individual not in species.members:
                        species.add_member(individual)
                    unassigned_individuals.remove(individual)
                elif species.compatability_distance(individual) >= delta_t and individual in species.members:
                    species.remove_member(individual)
                species.update_species_info()
            if individual in unassigned_individuals:
                new_species = Species([individual])
                self.species.append(new_species)
                self.species = sorted(self.species, key=lambda x: x.fitness_shared, reverse=True)
                unassigned_individuals.remove(individual)


class Population:

    population_instances = []

    def __init__(self, species: list[Species]) -> None:
        self.species = species
        self.update_population_info()
        self.uuid = uuid.uuid4().hex
        Population.population_instances.append(self)
        self.id = '_'.join(['population', str(len(Population.population_instances) - 1), self.uuid[:4]])

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
    

class CustomPopulationInitialiser(Population):
    def __init__(self, population_size: int, individuals: list[Individual]):
        self.population_size = population_size
        self.individuals = individuals
        self.generate_initial_population()

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
        super().__init__(species.species)