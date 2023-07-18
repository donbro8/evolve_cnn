import uuid
import numpy as np
from itertools import product

class Node:

    node_instances = []

    def __init__(self, node_type: str, attributes: dict = {}) -> None:
        self.node_type = node_type
        self.attributes = attributes
        self.uuid = uuid.uuid4().hex
        Node.node_instances.append(self)
        self.id = '_'.join(['node', self.node_type.lower(), str(len(Node.node_instances)), self.uuid[:4]])

    def __repr__(self) -> str:
        return f"(ID: {self.id} | Node type: {self.node_type} | Node attributes: {self.attributes})"
    
    def copy(self) -> None:
        return Node(self.node_type, self.attributes)
    

class Connection:

    connection_instances = []

    def __init__(self, node_in: Node, node_out: Node) -> None: # , enabled: bool = True
        existing_connections = [(connection.node_in, connection.node_out) for connection in Connection.connection_instances]
        if (node_in, node_out) not in existing_connections:
            self.node_in = node_in
            self.node_out = node_out
            self.uuid = uuid.uuid4().hex
            Connection.connection_instances.append(self)
            self.id = '_'.join(['connection', self.node_in.id, self.node_out.id, str(len(Connection.connection_instances)), self.uuid[:4]])
        else:
            print(f"Connection with node in {node_in} and node out {node_out} already exists. Returning existing connection.")
            index = existing_connections.index((node_in, node_out))
            self = Connection.connection_instances[index]


    def __repr__(self) -> str:
        return f"(ID: {self.id} | Node in: {self.node_in} | Node out: {self.node_out})"


class Graph:

    graph_instances = []

    def __init__(self, connections: list[Connection], all_enabled: bool = True, enabled_connections: list[bool] = []) -> None:
        self.connections = connections
        if all_enabled:
            self.enabled_connections = [True for _ in range(len(self.connections))]
        elif not all_enabled and len(enabled_connections) == len(self.connections):
            self.enabled_connections = enabled_connections
        else:
            raise ValueError(f"Invalid enabled connections list. Must be a list of booleans with length equal to the number of connections in the graph")
        self.uuid = uuid.uuid4().hex
        Graph.graph_instances.append(self)
        self.id = '_'.join(['graph', str(len(Graph.graph_instances)), self.uuid[:4]])
        self.update_graph_info()

    def __repr__(self) -> str:
        return f"ID: {self.id}"
    
    def update_graph_info(self) -> None:
        self.nodes_in = [connection.node_in for connection in self.connections]
        self.nodes_out = [connection.node_out for connection in self.connections]
        self.nodes = list(set(self.nodes_in + self.nodes_out))
        self.node_inputs = [[] for _ in range(len(self.nodes))]
        self.layers = [None for _ in range(len(self.nodes))]
        self.start_nodes = [connection.node_in for connection in self.connections if connection.node_in not in self.nodes_out]
        self.end_nodes = [connection.node_out for connection in self.connections if connection.node_out not in self.nodes_in]
        self.valid_nodes = self.update_valid_nodes(self.start_nodes, self.end_nodes)
        self.node_clusters = []
        for node in self.start_nodes:
            self.node_clusters += self.get_node_inputs(node, visited=[])
        self.order_nodes = self.order_input_nodes()

    def order_input_nodes(self) -> None:
        ordered_nodes = list(set([node for node in self.start_nodes if node in self.valid_nodes]))
        remaining_nodes = [node for node in set(self.node_clusters) if node not in ordered_nodes]
        while len(remaining_nodes) > 0:
            node = remaining_nodes.pop(0)
            index = self.nodes.index(node)
            if set(self.node_inputs[index]).issubset(set(ordered_nodes)):
                ordered_nodes.append(node)
            else:
                remaining_nodes.append(node)
        return ordered_nodes
        
    def add_connection(self, connection: Connection, enabled: bool = True) -> None:
        if connection not in self.connections:
            self.connections.append(connection)
            self.enabled_connections.append(enabled)
            self.update_graph_info()
        else:
            raise Warning(f"Connection {connection} already exists in graph {self.id} - cannot add duplicate connection")

    def delete_connection(self, connection: Connection) -> None:
        if connection in self.connections:
            index = self.connections.index(connection)
            self.enabled_connections.pop(index)
            self.connections.pop(index)
            self.update_graph_info()
        else:
            raise Warning(f"Connection {connection} does not exist in graph {self.id} - cannot delete connection")

    def switch_connection(self, connection: Connection) -> None:
        if connection in self.connections:
            index = self.connections.index(connection)
            self.enabled_connections[index] = not self.enabled_connections[index]
            self.update_graph_info()
        else:
            raise Warning(f"Connection {connection} does not exist in graph {self.id} - cannot switch connection")
    
    
    def depth_first_search(self, node: Node, visited: set = set(), connected_nodes: set = set(), enabled_only: bool = True) -> list:
        if node not in visited:
            visited.add(node)
            for i in range(len(self.connections)):
                if self.connections[i].node_in == node and (self.enabled_connections[i] or not enabled_only):
                    connected_nodes.add(self.connections[i].node_out)
                    self.depth_first_search(self.connections[i].node_out, visited, connected_nodes)
        return connected_nodes
    
    
    def breadth_first_search(self, node: Node, enabled_only: bool = True) -> list:
        queue = [node]
        visited = [node]
        while len(queue) > 0:
            node = queue.pop(0)
            for i in range(len(self.connections)):
                if self.connections[i].node_in == node and (self.enabled_connections[i] or not enabled_only):
                    visited.append(self.connections[i].node_out)
                    queue.append(self.connections[i].node_out)
        return visited
    
    
    def get_node_neighbours_in(self, node: Node, enabled_only: bool = True) -> list:
        return [self.connections[i].node_in for i in range(len(self.connections)) if self.connections[i].node_out == node and (self.enabled_connections[i] or not enabled_only)]


    def get_node_neighbours_out(self, node: Node, enabled_only: bool = True) -> list:
        return [self.connections[i].node_out for i in range(len(self.connections)) if self.connections[i].node_in == node and (self.enabled_connections[i] or not enabled_only)]

    def get_node_inputs(self, node: Node, visited: list = [], enabled_only: bool = True) -> list:
        neighbours_out = self.get_node_neighbours_out(node = node, enabled_only = enabled_only)
        if node not in visited and node in self.valid_nodes:
            for neighbour in neighbours_out:
                if neighbour not in self.valid_nodes:
                    continue
                neighbours_in = self.get_node_neighbours_in(node = neighbour, enabled_only = enabled_only)
                index = self.nodes.index(neighbour)
                self.node_inputs[index] = [node_in for node_in in neighbours_in if node_in in self.valid_nodes]
                self.get_node_inputs(neighbour, visited)
            visited.append(node)
        return visited[::-1]
    

    def check_continuity(self, node_start: Node, node_end: Node, enabled_only: bool = True) -> bool:
        connected_nodes = self.depth_first_search(node_start, visited = set(), connected_nodes = set(), enabled_only = enabled_only)
        return node_end in connected_nodes
    
    
    def check_recursion(self, node: Node, enabled_only = True) -> bool:
        connected_nodes = self.depth_first_search(node, visited = set(), connected_nodes = set(), enabled_only = enabled_only)
        return node in connected_nodes
    

    def update_valid_nodes(self, start_nodes: list[Node], end_nodes: list[Node]) -> list:
        valid_nodes = set()
        node_combinations = list(product(start_nodes, end_nodes))
        for node_start, node_end in node_combinations:
            if self.check_continuity(node_start, node_end):
                valid_nodes.add(node_start)
                valid_nodes.add(node_end)
                for node in self.nodes:
                    if self.check_continuity(node_start, node) and self.check_continuity(node, node_end):
                        valid_nodes.add(node)
        return valid_nodes
    
    def get_random_connection(self, enabled_only: bool = True) -> Connection:
        connections = [self.connections[i] for i in range(len(self.connections)) if self.enabled_connections[i] or not enabled_only]
        return np.random.choice(connections)
    
    def get_random_node(self, add_node_point: bool = False, node_point: str = 'start') -> Node:
        if add_node_point:
            possible_nodes = self.nodes + [node_point]
        else:
            possible_nodes = self.nodes
        return np.random.choice(possible_nodes)