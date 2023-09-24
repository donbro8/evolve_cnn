import pickle
import numpy as np
import sys
from evolution.population import Individual

experiment = sys.argv[1]
run_number = int(sys.argv[2])

population = pickle.load(open(f'population_{experiment}.pkl', 'rb'))
evolution = pickle.load(open(f'evolution_{experiment}.pkl', 'rb'))


evolution.generation += 1

if experiment == 'evolution':
    # Update phase if threshold has been reached
    if evolution.generation > evolution.phase_threshold:

        print('Phase threshold reached. Moving to next phase...')
        evolution.phase_number += 1
        evolution.phase = evolution.phase_names[evolution.phase_number]
        evolution.phase_threshold = evolution.phase_thresholds[evolution.phase_number]
        evolution.individual_mutation_rate = evolution.phases[evolution.phase]['individual_mutation_rate']
        evolution.mutation_type_rate = evolution.phases[evolution.phase]['mutation_type_rate']

    print(f"Generation {evolution.generation} of {evolution.generations} in phase {evolution.phase} with individual mutation rate of {evolution.individual_mutation_rate} and mutation type probabilities {list(zip(['node', 'connection', 'switch'], evolution.mutation_type_rate))}.")
    
    if evolution.generation > 1:

        print(f"Generating offspring...")
        population.generate_offspring(offspring_proportion=evolution.offspring_proportion)

    print("Mutating population")
    for i in range(len(population.population)):
        individual = population.population[i]
        individual.is_mutated = False
        if np.random.rand() < evolution.individual_mutation_rate:
            mutation_type = np.random.choice(['node', 'connection', 'switch'], p = evolution.mutation_type_rate)
            print(f"Attempting {mutation_type} mutation for individual {individual.id}: {i + 1} of {len(population.population)}.")
            if mutation_type == 'node':
                node = evolution.search_space.sample_from_search_space(n_samples = 1)[0]
                individual.add_node(individual.graph, node)
            elif mutation_type == 'connection':
                individual.add_edge(individual.graph)
            else:
                individual.switch_edge_weight(individual.graph)

        if individual.is_trained == False:
            print(f"Attempting training of individual {individual.id}: {i + 1} of {len(population.population)}.")
            
            evolution.maximum_params = evolution.train_individual(individual, evolution.maximum_params)

        individual.age += 1

    print("Applying speciation to new population")
    population.speciation(evolution.generation)

    evolution_tracker = evolution.update_experiment_tracker(
        run_number = run_number,
        generation = evolution.generation,
        phase_name = evolution.phase,
        phase_threshold = evolution.phase_threshold,
        individual_mutation_rate = evolution.individual_mutation_rate,
        node_mutation_prob = evolution.mutation_type_rate[0],
        connection_mutation_prob = evolution.mutation_type_rate[1],
        switch_mutation_prob = evolution.mutation_type_rate[2],
        individuals = population.population,
        save_to_file = True
    )

else:
    minum_node_samples = 1 
    maximum_node_samples = 21

    print(f"Generation {evolution.generation} of {evolution.generations}.")
    
    i = 1

    if evolution.generation == 1:
      for individual in population.population:
        n_samples = np.random.randint(minum_node_samples, maximum_node_samples)
        samples = evolution.search_space.sample_from_search_space(n_samples = n_samples)
        connection_density = np.random.rand()*(1 - 0.9*((n_samples - minum_node_samples)/(maximum_node_samples - minum_node_samples)))
        individual.random_individual(individual.graph, predefined_nodes=samples, minimum_connection_density = connection_density)

        print(f"Attempting training of individual {individual.id}: {i} of {evolution.population_size}.")
        
        evolution.maximum_params = evolution.train_individual(individual, evolution.maximum_params)

        i += 1
    
    else:
      population.population = sorted(population.population, key = lambda x: x.fitness, reverse = True)[:int(np.floor(evolution.population_size*evolution.offspring_proportion))]
    
      individuals_to_train = evolution.population_size - len(population.population)
      
      while len(population.population) < evolution.population_size:

          individual = Individual()
          population.individual_instances.add(individual)
          individual.id = len(population.individual_instances)
          n_samples = np.random.randint(minum_node_samples, maximum_node_samples)
          samples = evolution.search_space.sample_from_search_space(n_samples = n_samples)
          connection_density = np.random.rand()*(1 - 0.9*((n_samples - minum_node_samples)/(maximum_node_samples - minum_node_samples)))
          individual.random_individual(individual.graph, predefined_nodes=samples, minimum_connection_density = connection_density)
          
          print(f"Attempting training of individual {individual.id}: {i} of {individuals_to_train}.")
          
          evolution.maximum_params = evolution.train_individual(individual, evolution.maximum_params)
          
          population.population.append(individual)

          i += 1


    random_run_tracker = evolution.update_experiment_tracker(
        run_number = run_number,
        generation = evolution.generation,
        phase_name = None,
        phase_threshold = None,
        individual_mutation_rate = None,
        node_mutation_prob = None,
        connection_mutation_prob = None,
        switch_mutation_prob = None,
        individuals = population.population,
        save_to_file = True,
        run_type = 'random'
    )
    



with open(f'population_{experiment}.pkl', 'wb') as f:
    pickle.dump(population, f)

with open(f'evolution_{experiment}.pkl', 'wb') as f:
    pickle.dump(evolution, f)
