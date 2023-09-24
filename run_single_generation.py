import pickle
import numpy as np
import sys

population = pickle.load('population.pkl')
evolution = pickle.load('evolution.pkl')

experiment = sys.argv[1]
run_number = sys.argv[2]

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
        
        maximum_params = evolution.train_individual(individual, maximum_params)

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


with open('population.pkl', 'wb') as f:
    pickle.dump(population, f)

with open('evolution.pkl', 'wb') as f:
    pickle.dump(evolution, f)
