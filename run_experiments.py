from utils import *
from evolution.evolution import *
from evolution.network import *
from evolution.population import *
from evolution.search_space import *
import traceback

experiment_params = load_config('configs/experiment_params.yaml')
print(experiment_params)

(X_train, Y_train), (X_test, Y_test) = dataset_loader(experiment_params['experiment_dataset'])
print('Raw data shapes: ', X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

(X_train_sub, Y_train_sub), (X_train_extra, Y_train_extra) = data_subsample(X_train, Y_train, subsample_size = 0.1, seed = 42, stratify = True)
print('Subsampled data shapes: ', X_train_sub.shape, Y_train_sub.shape, X_train_extra.shape, Y_train_extra.shape)

(X_train_evo, Y_train_evo), (X_val_evo, Y_val_evo) = data_subsample(X_train_sub, Y_train_sub, subsample_size = 0.5, seed = 42, stratify = True)
print('Experiment train/validation split data shapes: ', X_train_evo.shape, Y_train_evo.shape, X_val_evo.shape, Y_val_evo.shape)

graph_config = load_config('configs/network_graphs.yaml')
network_graphs = update_output_classes(build_network_graphs(graph_config), Y_train_evo.shape[-1])
print('Macrostructure network graphs: ', list(network_graphs.keys()))

layer_config = update_layer_config(load_config('configs/search_space.yaml'))
search_space = SearchSpace(layer_config)
print(search_space.layer_types)

mutation_phases = load_config('configs/mutation_phases.yaml')
print(list(mutation_phases.keys()))



evolution = Evolution(
    search_space,
    input_graph=network_graphs['input_graph'],
    output_graph=network_graphs['output_graph'],
    reduction_cell_graph=network_graphs['reduction_cell_graph'],
    run_train_data = (X_train_evo, Y_train_evo),
    run_validation_data = (X_val_evo, Y_val_evo),
    population_size = experiment_params['population_size'],
    initialisation_type = 'minimal',
    generations = experiment_params['generations'],
    offspring_proportion = experiment_params['offspring_proportion'],
    phases = mutation_phases,
    normal_cell_repeats = experiment_params['normal_cell_repeats'],
    substructure_repeats = experiment_params['substructure_repeats'],
    parameter_limit = experiment_params['parameter_limit'],
    complexity_penalty = experiment_params['complexity_penalty'],
    number_of_runs = experiment_params['number_of_runs'],
    seed = experiment_params['seed'],
    batch_size = experiment_params['batch_size'], 
    epochs = experiment_params['epochs'], 
    verbose = experiment_params['verbose'],
    optimizer = experiment_params['optimizer'],
    loss = experiment_params['loss'],
    metrics = experiment_params['metrics']
)



try:
    experiment_types = ['evolution', 'random']
    outputs = []
    for experiment in experiment_types:
        running_experiment = experiment
        outputs.append(evolution.run_multiple_experiments(experiment_type = running_experiment))
except Exception as e:
    traceback_info = traceback.format_exc()
    send_email_alert(subject=f"Experiment '{running_experiment}' failed", message=f"Exception: {e} \n {traceback_info}")