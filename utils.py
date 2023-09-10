import smtplib, ssl
import yaml
import networkx as nx
from sklearn.model_selection import train_test_split
import keras

def load_config(path: str):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def send_email_alert(subject: str, message: str):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"

    config = load_config('alerter.yaml')

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