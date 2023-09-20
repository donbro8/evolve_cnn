import smtplib, ssl
import yaml
import networkx as nx
import graphviz
from sklearn.model_selection import train_test_split
import keras
import json
import datetime
from evolution.network import ModelCompiler
import pandas as pd
import os
import pickle

def load_config(path: str):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def send_email_alert(subject: str, message: str):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"

    config = load_config('configs/alerter.yaml')

    sender_email = config['sender_email']
    receiver_email = config['receiver_email']
    password = config['password']

    context = ssl.create_default_context()

    message = f"""\
        Subject: {subject}

        Message: {message}""" 
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
    return message

def build_network_graphs(graphs: dict):
    network_graphs = {}
    for graph in graphs.keys():
        if graphs[graph] is not None:
            network_graphs[graph] = nx.DiGraph()
            for node in graphs[graph]['nodes'].keys():
                attributes = graphs[graph]['nodes'][node]
                for attribute in attributes.keys():
                    if isinstance(attributes[attribute], list):
                        attributes[attribute] = tuple(attributes[attribute])
                network_graphs[graph].add_node(node, **attributes)
            for edge in graphs[graph]['edges'].keys():
                node_1, node_2 = graphs[graph]['edges'][edge][0], graphs[graph]['edges'][edge][1]
                network_graphs[graph].add_edge(node_1, node_2, weight = 1.0)
    return network_graphs


def update_output_classes(network_graphs: dict, n_output_classes: int):
    for graph in network_graphs.keys():
        if graph == 'output_graph':
            for node in network_graphs[graph].nodes:
                if network_graphs[graph].out_degree(node) == 0:
                    network_graphs[graph].nodes[node]['units'] = n_output_classes
    return network_graphs


def convert_to_list_of_tuple(x: list[list]):
    if isinstance(x, list):
        if all([isinstance(i, list) for i in x]):
            return list([tuple(i) for i in x])
    return x

def update_layer_config(layer_config: dict):
    for layer in layer_config.keys():
        for attribute in layer_config[layer].keys():
            layer_config[layer][attribute] = convert_to_list_of_tuple(layer_config[layer][attribute])
    return layer_config

def dataset_loader(dataset_name: str = 'mnist'):
    if dataset_name == 'mnist':
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    elif dataset_name == 'cifar10':
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    elif dataset_name == 'cifar100':
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar100.load_data()
    elif dataset_name == 'fashion_mnist':
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()

    # reshape to be [samples][width][height][channels]
    if len(X_train.shape) == 3:
        X_train = X_train.reshape(X_train.shape + (1,)).astype('float32') / 255
        X_test = X_test.reshape(X_test.shape + (1,)).astype('float32') / 255

    # one hot encode outputs
    Y_train = keras.utils.to_categorical(Y_train)
    Y_test = keras.utils.to_categorical(Y_test)

    return (X_train, Y_train), (X_test, Y_test)

def data_subsample(X, Y, subsample_size: float = 0.1, seed: int = 42, stratify: bool = True):
    if stratify:
        X_sub, X_extra, Y_sub, Y_extra = train_test_split(X, Y, train_size = subsample_size, random_state = seed, stratify = Y)
    else:
        X_sub, X_extra, Y_sub, Y_extra = train_test_split(X, Y, train_size = subsample_size, random_state = seed)
    return (X_sub, Y_sub), (X_extra, Y_extra)

def paretoset(df, field1, field2, minimize1 = True, minimize2 = True):
    df_pareto = df.copy()
    df_pareto = df_pareto.dropna(subset = [field1, field2])
    df_pareto['pareto'] = 1

    for i in range(len(df_pareto)):
        for j in range(len(df_pareto)):
            if i != j:

                if minimize1 and minimize2:

                    if df_pareto[field1].iloc[i] > df_pareto[field1].iloc[j]:
                        if df_pareto[field2].iloc[i] > df_pareto[field2].iloc[j]:
                            df_pareto['pareto'].iloc[i] = 0

                elif minimize1 and not minimize2:
                    if df_pareto[field1].iloc[i] > df_pareto[field1].iloc[j]:
                        if df_pareto[field2].iloc[i] < df_pareto[field2].iloc[j]:
                            df_pareto['pareto'].iloc[i] = 0

                elif not minimize1 and minimize2:
                    if df_pareto[field1].iloc[i] < df_pareto[field1].iloc[j]:
                        if df_pareto[field2].iloc[i] > df_pareto[field2].iloc[j]:
                            df_pareto['pareto'].iloc[i] = 0

                elif not minimize1 and not minimize2:
                    if df_pareto[field1].iloc[i] < df_pareto[field1].iloc[j]:
                        if df_pareto[field2].iloc[i] < df_pareto[field2].iloc[j]:
                            df_pareto['pareto'].iloc[i] = 0

    return df_pareto


def build_blocks(block_type = 'resnet'):
    if block_type == 'resnet':
        resnet_block = nx.DiGraph()   

        resnet_block.add_node('input')
        resnet_block.add_node('output')
        resnet_block.add_node('Conv2D_1', filters = 32, kernel_size = (3,3), padding = 'same')
        resnet_block.add_node('Conv2D_2', filters = 32, kernel_size = (3,3), padding = 'same')
        resnet_block.add_node('Conv2D_3', filters = 32, kernel_size = (1,1), padding = 'same')
        resnet_block.add_node('BatchNormalization_1')
        resnet_block.add_node('BatchNormalization_2')
        resnet_block.add_node('ReLU_1')
        resnet_block.add_node('ReLU_2')

        resnet_block.add_edge('input', 'Conv2D_1', weight = 1.0)
        resnet_block.add_edge('Conv2D_1', 'BatchNormalization_1', weight = 1.0)
        resnet_block.add_edge('BatchNormalization_1', 'ReLU_1', weight = 1.0)
        resnet_block.add_edge('ReLU_1', 'Conv2D_2', weight = 1.0)
        resnet_block.add_edge('Conv2D_2', 'BatchNormalization_2', weight = 1.0)
        resnet_block.add_edge('BatchNormalization_2', 'ReLU_2', weight = 1.0)
        resnet_block.add_edge('Conv2D_3', 'ReLU_2', weight = 1.0)
        resnet_block.add_edge('ReLU_2', 'output', weight = 1.0)
        return resnet_block

    elif block_type == 'inception':
        inception_block = nx.DiGraph()

        inception_block.add_node('input')
        inception_block.add_node('output')
        inception_block.add_node('Conv2D_1', filters = 32, kernel_size = (1,1), activation = 'relu', padding = 'same')
        inception_block.add_node('Conv2D_2', filters = 32, kernel_size = (1,1), activation = 'relu', padding = 'same')
        inception_block.add_node('Conv2D_3', filters = 32, kernel_size = (1,1), activation = 'relu', padding = 'same')
        inception_block.add_node('Conv2D_4', filters = 32, kernel_size = (1,1), activation = 'relu', padding = 'same')
        inception_block.add_node('Conv2D_5', filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same')
        inception_block.add_node('Conv2D_6', filters = 32, kernel_size = (5,5), activation = 'relu', padding = 'same')
        inception_block.add_node('MaxPooling2D_1', pool_size = (3,3), strides = (1,1), padding = 'same')

        inception_block.add_edge('input', 'Conv2D_1', weight = 1.0)
        inception_block.add_edge('input', 'Conv2D_2', weight = 1.0)
        inception_block.add_edge('input', 'Conv2D_3', weight = 1.0)
        inception_block.add_edge('input', 'MaxPooling2D_1', weight = 1.0)
        inception_block.add_edge('Conv2D_2', 'Conv2D_5', weight = 1.0)
        inception_block.add_edge('Conv2D_3', 'Conv2D_6', weight = 1.0)
        inception_block.add_edge('MaxPooling2D_1', 'Conv2D_4', weight = 1.0)
        inception_block.add_edge('Conv2D_1', 'output', weight = 1.0)
        inception_block.add_edge('Conv2D_4', 'output', weight = 1.0)
        inception_block.add_edge('Conv2D_5', 'output', weight = 1.0)
        inception_block.add_edge('Conv2D_6', 'output', weight = 1.0)
        return inception_block
    
    elif block_type == None:
        no_block = nx.DiGraph()

        no_block.add_node('input')
        no_block.add_node('output')

        no_block.add_edge('input', 'output', weight = 1.0)
        return no_block
    
    else:
        print('Block type not recognised. Please choose from: resnet, inception')




def post_training_analysis(
        normal_cells: dict,
        network_graphs: dict,
        datasets: list[str] = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100'], 
        normal_cell_repeats: int=1, 
        substructure_repeats: int=1,
        epochs: int = 1,
        save_results: bool = False,
        file_name: str = None):
    
    model_results = {key:{} for key in datasets}

    for dataset in model_results.keys():
        print(f'Running analysis on {dataset}...')
        (X_train, Y_train), (X_test, Y_test) = dataset_loader(dataset)
        network_graphs = update_output_classes(network_graphs, Y_train.shape[-1])
        for key, value in normal_cells.items():
            print(f'Running analysis on block {key}... | Block {list(normal_cells.keys()).index(key) + 1}/{len(normal_cells)}')
            model_compiler = ModelCompiler(
                input_graph=network_graphs['input_graph'], 
                output_graph=network_graphs['output_graph'], 
                normal_cell_graph=value, 
                reduction_cell_graph=network_graphs['reduction_cell_graph'], 
                normal_cell_repeats=normal_cell_repeats, 
                substructure_repeats=substructure_repeats
            )
            model = model_compiler.build_model(input_shape = X_train.shape[1:])
            history = model_compiler.train_model(
                            training_data = (X_train, Y_train), 
                            validation_data = (X_test, Y_test), 
                            model = model,
                            batch_size = 32,
                            epochs = epochs, 
                            verbose = 1,
                            optimizer = 'adam',
                            loss = 'categorical_crossentropy',
                            metrics = ['accuracy','mse']
            )
            history.history['test_accuracy'] = model.evaluate(X_test, Y_test)[1]
            model_results[dataset][key] = history.history

    if save_results:
        if file_name is None:
            file_name = f'model_results_n{normal_cell_repeats}_s{substructure_repeats}{datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")}.json'
        with open(file_name, 'w') as fp:
            json.dump(model_results, fp)
    return model_results


def plot_model_graph(model_graph: nx.DiGraph, filename: str = None, directory: str = None, rankdir:str = 'LR'):
    dot = graphviz.Digraph(graph_attr={'rankdir':rankdir,'fontname':'computer modern roman','stagger': 'true'})
    for node in model_graph.nodes:
        node_type = node.split('_')[0]
        node_attributes = model_graph.nodes[node]
        if len(node_attributes) == 0:
            node_attributes_label = ''
        else:
            node_attributes_label = '<br/>' + '<br/>'.join([f'{key.replace("_"," ")}: {value}' for key,value in node_attributes.items()])
        dot.node(node, label = '<<B>' + node_type.title() + '</B>' + node_attributes_label + '>', fontsize = '16')

    for edge in model_graph.edges:
        if model_graph.edges[edge]['weight'] == 1.0:
            dot.edge(edge[0], edge[1], style = 'solid')

        else:
            dot.edge(edge[0], edge[1], style = 'dashed')
    if filename is None:
        filename = 'model_graph'
    dot.render(filename=filename, directory = directory, view=True)


def pickle_to_pandas_dataframe(experiment_folder_path: str) -> pd.DataFrame:

    no_df = True

    for file_name in os.listdir(experiment_folder_path):
        if file_name != '.ipynb_checkpoints':
            path = os.path.join(experiment_folder_path, file_name)
            loaded_data = pickle.load(open(path, 'rb'))

            for key in loaded_data.keys():
                if no_df:
                    df = pd.DataFrame(data = {key:[value] for key, value in loaded_data[key].__dict__.items()}, index = [0])
                    no_df = False

                else:
                    df = pd.concat([df, pd.DataFrame(data = {key:[value] for key, value in loaded_data[key].__dict__.items()}, index = [df.shape[0]])])
    return df
