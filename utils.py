import smtplib, ssl
import yaml
import networkx as nx
import graphviz
from sklearn.model_selection import train_test_split
import keras
from keras import backend
import json
import datetime
from evolution.network import ModelCompiler
import pandas as pd
import os
import pickle
from pyvis.network import Network
from IPython.core.display import display, HTML
import numpy as np
import glob


def load_config(path: str):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def send_email_alert(subject: str, message: str):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"

    config = load_config("configs/alerter.yaml")

    sender_email = config["sender_email"]
    receiver_email = config["receiver_email"]
    password = config["password"]

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
            for node in graphs[graph]["nodes"].keys():
                attributes = graphs[graph]["nodes"][node]
                for attribute in attributes.keys():
                    if isinstance(attributes[attribute], list):
                        attributes[attribute] = tuple(attributes[attribute])
                network_graphs[graph].add_node(node, **attributes)
            for edge in graphs[graph]["edges"].keys():
                node_1, node_2 = (
                    graphs[graph]["edges"][edge][0],
                    graphs[graph]["edges"][edge][1],
                )
                network_graphs[graph].add_edge(node_1, node_2, weight=1.0)
    return network_graphs


def update_output_classes(network_graphs: dict, n_output_classes: int):
    for graph in network_graphs.keys():
        if graph == "output_graph":
            for node in network_graphs[graph].nodes:
                if network_graphs[graph].out_degree(node) == 0:
                    network_graphs[graph].nodes[node]["units"] = n_output_classes
    return network_graphs


def convert_to_list_of_tuple(x: list[list]):
    if isinstance(x, list):
        if all([isinstance(i, list) for i in x]):
            return list([tuple(i) for i in x])
    return x


def update_layer_config(layer_config: dict):
    for layer in layer_config.keys():
        for attribute in layer_config[layer].keys():
            layer_config[layer][attribute] = convert_to_list_of_tuple(
                layer_config[layer][attribute]
            )
    return layer_config


def dataset_loader(dataset_name: str = "mnist"):
    if dataset_name == "mnist":
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    elif dataset_name == "cifar10":
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    elif dataset_name == "cifar100":
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar100.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()

    # reshape to be [samples][width][height][channels]
    if len(X_train.shape) == 3:
        X_train = X_train.reshape(X_train.shape + (1,)).astype("float32") / 255
        X_test = X_test.reshape(X_test.shape + (1,)).astype("float32") / 255

    # one hot encode outputs
    Y_train = keras.utils.to_categorical(Y_train)
    Y_test = keras.utils.to_categorical(Y_test)

    return (X_train, Y_train), (X_test, Y_test)


def data_subsample(
    X, Y, subsample_size: float = 0.1, seed: int = 42, stratify: bool = True
):
    if stratify:
        X_sub, X_extra, Y_sub, Y_extra = train_test_split(
            X, Y, train_size=subsample_size, random_state=seed, stratify=Y
        )
    else:
        X_sub, X_extra, Y_sub, Y_extra = train_test_split(
            X, Y, train_size=subsample_size, random_state=seed
        )
    return (X_sub, Y_sub), (X_extra, Y_extra)


def paretoset(df, field1, field2, minimize1=True, minimize2=True):
    df_pareto = df.copy()
    df_pareto = df_pareto.dropna(subset=[field1, field2])
    df_pareto["pareto"] = 1

    for i in range(len(df_pareto)):
        for j in range(len(df_pareto)):
            if i != j:
                if minimize1 and minimize2:
                    if df_pareto[field1].iloc[i] > df_pareto[field1].iloc[j]:
                        if df_pareto[field2].iloc[i] > df_pareto[field2].iloc[j]:
                            df_pareto["pareto"].iloc[i] = 0

                elif minimize1 and not minimize2:
                    if df_pareto[field1].iloc[i] > df_pareto[field1].iloc[j]:
                        if df_pareto[field2].iloc[i] < df_pareto[field2].iloc[j]:
                            df_pareto["pareto"].iloc[i] = 0

                elif not minimize1 and minimize2:
                    if df_pareto[field1].iloc[i] < df_pareto[field1].iloc[j]:
                        if df_pareto[field2].iloc[i] > df_pareto[field2].iloc[j]:
                            df_pareto["pareto"].iloc[i] = 0

                elif not minimize1 and not minimize2:
                    if df_pareto[field1].iloc[i] < df_pareto[field1].iloc[j]:
                        if df_pareto[field2].iloc[i] < df_pareto[field2].iloc[j]:
                            df_pareto["pareto"].iloc[i] = 0

    return df_pareto


def build_blocks(block_type="resnet"):
    if block_type == "resnet":
        resnet_block = nx.DiGraph()

        resnet_block.add_node("input")
        resnet_block.add_node("output")
        resnet_block.add_node(
            "Conv2D_1", filters=32, kernel_size=(3, 3), padding="same"
        )
        resnet_block.add_node(
            "Conv2D_2", filters=32, kernel_size=(3, 3), padding="same"
        )
        resnet_block.add_node(
            "Conv2D_3", filters=32, kernel_size=(1, 1), padding="same"
        )
        resnet_block.add_node("BatchNormalization_1")
        resnet_block.add_node("BatchNormalization_2")
        resnet_block.add_node("ReLU_1")
        resnet_block.add_node("ReLU_2")

        resnet_block.add_edge("input", "Conv2D_1", weight=1.0)
        resnet_block.add_edge("Conv2D_1", "BatchNormalization_1", weight=1.0)
        resnet_block.add_edge("BatchNormalization_1", "ReLU_1", weight=1.0)
        resnet_block.add_edge("ReLU_1", "Conv2D_2", weight=1.0)
        resnet_block.add_edge("Conv2D_2", "BatchNormalization_2", weight=1.0)
        resnet_block.add_edge("BatchNormalization_2", "ReLU_2", weight=1.0)
        resnet_block.add_edge("Conv2D_3", "ReLU_2", weight=1.0)
        resnet_block.add_edge("ReLU_2", "output", weight=1.0)
        return resnet_block

    elif block_type == "inception":
        inception_block = nx.DiGraph()

        inception_block.add_node("input")
        inception_block.add_node("output")
        inception_block.add_node(
            "Conv2D_1",
            filters=32,
            kernel_size=(1, 1),
            activation="relu",
            padding="same",
        )
        inception_block.add_node(
            "Conv2D_2",
            filters=32,
            kernel_size=(1, 1),
            activation="relu",
            padding="same",
        )
        inception_block.add_node(
            "Conv2D_3",
            filters=32,
            kernel_size=(1, 1),
            activation="relu",
            padding="same",
        )
        inception_block.add_node(
            "Conv2D_4",
            filters=32,
            kernel_size=(1, 1),
            activation="relu",
            padding="same",
        )
        inception_block.add_node(
            "Conv2D_5",
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )
        inception_block.add_node(
            "Conv2D_6",
            filters=32,
            kernel_size=(5, 5),
            activation="relu",
            padding="same",
        )
        inception_block.add_node(
            "MaxPooling2D_1", pool_size=(3, 3), strides=(1, 1), padding="same"
        )

        inception_block.add_edge("input", "Conv2D_1", weight=1.0)
        inception_block.add_edge("input", "Conv2D_2", weight=1.0)
        inception_block.add_edge("input", "Conv2D_3", weight=1.0)
        inception_block.add_edge("input", "MaxPooling2D_1", weight=1.0)
        inception_block.add_edge("Conv2D_2", "Conv2D_5", weight=1.0)
        inception_block.add_edge("Conv2D_3", "Conv2D_6", weight=1.0)
        inception_block.add_edge("MaxPooling2D_1", "Conv2D_4", weight=1.0)
        inception_block.add_edge("Conv2D_1", "output", weight=1.0)
        inception_block.add_edge("Conv2D_4", "output", weight=1.0)
        inception_block.add_edge("Conv2D_5", "output", weight=1.0)
        inception_block.add_edge("Conv2D_6", "output", weight=1.0)
        return inception_block

    elif block_type == None:
        no_block = nx.DiGraph()

        no_block.add_node("input")
        no_block.add_node("output")

        no_block.add_edge("input", "output", weight=1.0)
        return no_block

    else:
        print("Block type not recognised. Please choose from: resnet, inception")


def post_training_analysis(
    normal_cells: dict,
    network_graphs: dict,
    datasets: list[str] = ["mnist", "fashion_mnist", "cifar10", "cifar100"],
    normal_cell_repeats: int = 1,
    substructure_repeats: int = 1,
    epochs: int = 1,
    save_results: bool = False,
    file_name: str = None,
):
    model_results = {key: {} for key in datasets}

    for dataset in model_results.keys():
        print(f"Running analysis on {dataset} dataset...")
        (X_train, Y_train), (X_test, Y_test) = dataset_loader(dataset)
        network_graphs = update_output_classes(network_graphs, Y_train.shape[-1])
        for key, value in normal_cells.items():
            print(
                f"Running analysis on block {key}... | Block {list(normal_cells.keys()).index(key) + 1}/{len(normal_cells)}"
            )
            model_compiler = ModelCompiler(
                input_graph=network_graphs["input_graph"],
                output_graph=network_graphs["output_graph"],
                normal_cell_graph=value,
                reduction_cell_graph=network_graphs["reduction_cell_graph"],
                normal_cell_repeats=normal_cell_repeats,
                substructure_repeats=substructure_repeats,
            )
            model = model_compiler.build_model(input_shape=X_train.shape[1:])
            
            for layer in model.layers:
              if 'NC' in layer.name:
                  num_params = sum(backend.count_params(p) for p in layer.trainable_weights)
                  break
            
            history = model_compiler.train_model(
                training_data=(X_train, Y_train),
                validation_data=(X_test, Y_test),
                model=model,
                batch_size=32,
                epochs=epochs,
                verbose=1,
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy", "mse"],
            )

            history.history["test_accuracy"] = model.evaluate(X_test, Y_test)[1]
            history.history["number_of_params"] = num_params
            history.history["normal_cell_repeats"] = normal_cell_repeats
            history.history["substructure_repeats"] = substructure_repeats
            model_results[dataset][key] = history.history

    if save_results:
        if file_name is None:
            file_name = f'model_results_n{normal_cell_repeats}_s{substructure_repeats}{datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")}.json'
        with open(file_name, "w") as fp:
            json.dump(model_results, fp)
    return model_results


def plot_model_graph(
    model_graph: nx.DiGraph,
    filename: str = None,
    directory: str = None,
    rankdir: str = "LR",
):
    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": rankdir,
            "fontname": "computer modern roman",
            "stagger": "true",
        }
    )
    for node in model_graph.nodes:
        node_type = node.split("_")[0]
        node_attributes = model_graph.nodes[node]
        if len(node_attributes) == 0:
            node_attributes_label = ""
        else:
            node_attributes_label = "<br/>" + "<br/>".join(
                [
                    f'{key.replace("_"," ")}: {value}'
                    for key, value in node_attributes.items()
                ]
            )
        dot.node(
            node,
            label="<<B>" + node_type.title() + "</B>" + node_attributes_label + ">",
            fontsize="16",
        )

    for edge in model_graph.edges:
        if model_graph.edges[edge]["weight"] == 1.0:
            dot.edge(edge[0], edge[1], style="solid")

        else:
            dot.edge(edge[0], edge[1], style="dashed")
    if filename is None:
        filename = "model_graph"
    dot.render(filename=filename, directory=directory, view=True)


def pickle_to_pandas_dataframe(experiment_folder_path: str) -> pd.DataFrame:
    no_df = True

    for file_name in os.listdir(experiment_folder_path):
        if file_name != ".ipynb_checkpoints":
            path = os.path.join(experiment_folder_path, file_name)
            loaded_data = pickle.load(open(path, "rb"))

            for key in loaded_data.keys():
                if no_df:
                    df = pd.DataFrame(
                        data={
                            key: [value]
                            for key, value in loaded_data[key].__dict__.items()
                        },
                        index=[0],
                    )
                    no_df = False

                else:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                data={
                                    key: [value]
                                    for key, value in loaded_data[key].__dict__.items()
                                },
                                index=[df.shape[0]],
                            ),
                        ]
                    )
    return df


def prepare_tree(df):
    tree_data = {}
    df_temp = df.copy().sort_values(["generation", "individual_id"])
    df_temp = df_temp.replace({np.nan: None})
    generation = df_temp["generation"].min()
    while generation < df_temp["generation"].max():
        tree_data[generation] = {}
        df_gen = df_temp[df_temp["generation"] == generation]
        for i in range(len(df_gen)):
            individual_id = df_gen["individual_id"].iloc[i]
            species_id = df_gen["species_id"].iloc[i]
            fitness = df_gen["individual_fitness"].iloc[i]
            if species_id != None:
                species_id = species_id.split("_")[1]
            else:
                species_id = "X"
            if generation == 0:
                is_in_previous_gen = False
            else:
                df_gen_sub_one = df_temp[df_temp["generation"] == generation - 1]
                is_in_previous_gen = individual_id in list(
                    df_gen_sub_one["individual_id"]
                )
            if is_in_previous_gen:
                parents = None
                node_mutation = df_gen["node_mutation"].iloc[i]
                connection_mutation = df_gen["connection_mutation"].iloc[i]
                switch_mutation = df_gen["switch_mutation"].iloc[i]
                if node_mutation != None:
                    mutation_type = "node"
                elif connection_mutation != None:
                    mutation_type = "connection"
                elif switch_mutation != None:
                    mutation_type = "switch"
                else:
                    mutation_type = None
            else:
                mutation_type = None
                parents = df_gen["offspring_of"].iloc[i]

            tree_data[generation][individual_id] = {
                "is_in_previous_gen": is_in_previous_gen,
                "mutation_type": mutation_type,
                "parents": parents,
                "species_id": species_id,
                "fitness": fitness,
            }
        generation += 1
    return tree_data



def plot_tree_data(tree_data, rankdir: str = "LR"):
    dot = graphviz.Digraph(
        graph_attr={
            "rankdir": rankdir,
            "fontname": "computer modern roman",
            "stagger": "true",
        }
    )
    for key, value in tree_data.items():
        for individual, info in value.items():
            species_id = tree_data[key][individual]["species_id"]
            dot.node(
                f"I{individual}_G{key}", label=f"I{individual}_G{key}_S{species_id}"
            )
            if tree_data[key][individual]["is_in_previous_gen"]:
                mutation_type = tree_data[key][individual]["mutation_type"]
                if mutation_type != None:
                    if mutation_type == "node":
                        dot.edge(
                            f"I{individual}_G{key - 1}",
                            f"I{individual}_G{key}",
                            color="gray",
                            style="dotted",
                        )
                    elif mutation_type == "connection":
                        dot.edge(
                            f"I{individual}_G{key - 1}",
                            f"I{individual}_G{key}",
                            color="gray",
                            style="solid",
                        )
                    elif mutation_type == "switch":
                        dot.edge(
                            f"I{individual}_G{key - 1}",
                            f"I{individual}_G{key}",
                            color="gray",
                            style="dashed",
                        )
            else:
                parents = tree_data[key][individual]["parents"]
                if parents != None:
                    dot.edge(
                        f"I{parents[0]}_G{key - 1}",
                        f"I{individual}_G{key}",
                        style="tapered",
                        arrowhead="none",
                    )
                    dot.edge(
                        f"I{parents[1]}_G{key - 1}",
                        f"I{individual}_G{key}",
                        style="tapered",
                        arrowhead="none",
                    )
    dot.render("tree", view=True)


def plot_tree_data_pyviz(tree_data, rankdir: str = "LR"):
    net = Network(directed=True, notebook=True, cdn_resources="in_line")

    # groups = []

    for key, value in tree_data.items():
        for individual, info in value.items():
            species_id = tree_data[key][individual]["species_id"]
            fitness = tree_data[key][individual]["fitness"]

            net.add_node(
                f"I{individual}_G{key}",
                label=f"I{individual}_G{key}_S{species_id}",
                level=int(key),
                size=fitness * 50,
            )  # , group = str(species_id)

            # groups.append(str(species_id))

            if tree_data[key][individual]["is_in_previous_gen"]:
                mutation_type = tree_data[key][individual]["mutation_type"]

                if mutation_type != None:
                    if mutation_type == "node":
                        net.add_edge(
                            f"I{individual}_G{key - 1}",
                            f"I{individual}_G{key}",
                            color="gray",
                            style="dotted",
                        )

                    elif mutation_type == "connection":
                        net.add_edge(
                            f"I{individual}_G{key - 1}",
                            f"I{individual}_G{key}",
                            color="gray",
                            style="solid",
                        )

                    elif mutation_type == "switch":
                        net.add_edge(
                            f"I{individual}_G{key - 1}",
                            f"I{individual}_G{key}",
                            color="gray",
                            style="dashed",
                        )

            else:
                parents = tree_data[key][individual]["parents"]

                if parents != None:
                    net.add_edge(
                        f"I{parents[0]}_G{key - 1}",
                        f"I{individual}_G{key}",
                        style="tapered",
                        arrowhead="none",
                    )
                    net.add_edge(
                        f"I{parents[1]}_G{key - 1}",
                        f"I{individual}_G{key}",
                        style="tapered",
                        arrowhead="none",
                    )

    # groups = list(set(groups))

    # groups_dict =

    # "edges": {
    #   "color": {
    #     "inherit": true
    #   },
    #   "smooth": false
    # },
    # "nodes": {
    #   "shape": "dot",
    #   "color":{
    #     "background":"gray",
    #     "border":"black"
    #   }
    # },
    # "configure": {
    #   "enabled": true,
    #   "filter": "layout"
    # },

    net.set_options(
        """var options = {
        "layout": {
          "hierarchical": {
            "enabled": true,
            "levelSeparation": 500,
            "nodeSpacing": 20,
            "treeSpacing": 20,
            "direction": "LR",
            "sortMethod": "directed",
            "shakeTowards": "roots"
          }
        },
        "physics": {
          "hierarchicalRepulsion": {
            "centralGravity": 0
          },
          "minVelocity": 0.75,
          "solver": "hierarchicalRepulsion"
        }
      }"""
    )

    # net.add_event_listener("selectNode", callback_function)

    net.show("mygraph.html")
    display(HTML("mygraph.html"))

    # "layout": {
    #   "hierarchical": {
    #     "enabled": true,
    #     "levelSeparation": -393,
    #     "nodeSpacing": 30,
    #     "treeSpacing": 40,
    #     "parentCentralization": false,
    #     "direction": "DU",
    #     "sortMethod": "directed",
    #     "shakeTowards": "roots"
    #   }


def build_training_graphs(df_pareto_evolution, df_pareto_random, experiments = ['evolution','random'], sample_size = 10, seed = 42):

    assert 'pareto' in df_pareto_evolution.columns and 'pareto' in df_pareto_random.columns
    
    np.random.seed(seed)

    

    training_graphs = {
        'basic_block':build_blocks(block_type = None),
        'resnet':build_blocks(block_type = 'resnet'),
        'inception':build_blocks(block_type = 'inception')
    }

    for experiment in experiments:

      if experiment == 'evolution':

        df_temp = df_pareto_evolution.copy()[df_pareto_evolution['pareto'] == 1]
        
        min_fitness = df_temp['individual_fitness'].min()
        max_fitness = df_temp['individual_fitness'].max()
        step_size = max_fitness/sample_size
        ranges = list(zip(np.arange(min_fitness, max_fitness,step_size).round(2), np.arange(min_fitness + step_size, max_fitness + step_size,step_size).round(2)))[::-1]

      else:

        df_temp = df_pareto_random.copy()[df_pareto_random['pareto'] == 1]
        min_fitness = df_temp['individual_fitness'].min()
        max_fitness = df_temp['individual_fitness'].max()
        step_size = max_fitness/sample_size
        ranges = list(zip(np.arange(min_fitness, max_fitness,step_size).round(2), np.arange(min_fitness + step_size, max_fitness + step_size,step_size).round(2)))[::-1]

      individuals_to_assign = sample_size

      for val_acc_range in ranges:

        df_range = df_temp[(df_temp['individual_fitness'] > val_acc_range[0]) & (df_temp['individual_fitness'] <= val_acc_range[1])]
        
        
        if len(df_range) > 0:
            individual = df_range.sample(n = 1, random_state=42)
            print(type(individual))
            print(individual.iloc[0]["individual_id"])
            key = f'I{individual.iloc[0]["individual_id"]}_G{individual.iloc[0]["generation"]}_{experiment[0].upper()}'
            print(key)
            if key not in training_graphs.keys():
                training_graphs[key] = individual.iloc[0]['individual'].graph
                individuals_to_assign -= 1
                ranges.remove(val_acc_range)

      while individuals_to_assign > 0:
        for val_acc_range in ranges:
          range_mean = (val_acc_range[0] + val_acc_range[1])/2
          df_diff = df_temp.copy()
          df_diff['diff'] = (df_temp['individual_fitness']  - range_mean).abs()
          individual = df_diff.sort_values(by=['diff']).groupby(by=['individual_id']).head(1)
          key = f'I{individual.iloc[0]["individual_id"]}_G{individual.iloc[0]["generation"]}_{experiment[0].upper()}'
          if key not in training_graphs.keys():
              training_graphs[key] = individual.iloc[0]['individual'].graph
              individuals_to_assign -= 1
              ranges.remove(val_acc_range)

    return training_graphs

def convert_model_results(path):
    all_data = []
    for file in [f for f in glob.glob(os.path.join(path,'*')) if '.pkl' in f]:
        data = pickle.load(open(file, 'rb'))
        for dataset, experiment_data in data.items():
            for block, training_history in experiment_data.items():
                all_data += [
                      {
                          "dataset":dataset,
                          "block_id":block,
                          "loss":training_history["loss"][epoch],
                          "accuracy":training_history["accuracy"][epoch],
                          "mse":training_history["mse"][epoch],
                          "val_loss":training_history["val_loss"][epoch],
                          "val_accuracy":training_history["val_accuracy"][epoch],
                          "val_mse":training_history["val_mse"][epoch],
                          "training_time":training_history["training_time"],
                          "timeout_reached":training_history["timeout_reached"],
                          "test_accuracy":training_history["test_accuracy"],
                          "number_of_params":training_history["number_of_params"],
                          "normal_cell_repeats":training_history["normal_cell_repeats"],
                          "substructure_repeats":training_history["substructure_repeats"],
                      }
                      for epoch in range(len(training_history["loss"]))
                ]

    return pd.DataFrame.from_records(all_data)