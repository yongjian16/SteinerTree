import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import dgl
from sklearn.metrics import f1_score, precision_score, recall_score
from model import *
import os
import pandas as pd
import networkx as nx
import numpy as np

def load_graph(graph_file_name, feature_file_name, root_gamma, device):
    g = nx.read_graphml(graph_file_name)
    df_dataset = pd.read_csv(feature_file_name)
    G_nodes = [eval(v) for v in g.nodes()]
    df_G_nodes = pd.DataFrame(G_nodes, columns=["vid", "day"])
    df_dataset = df_dataset[df_dataset.set_index(['vid', 'day']).index.isin(df_G_nodes.set_index(['vid', 'day']).index)]
    df_features = pd.DataFrame(columns=df_dataset.columns)
    for i in range(len(G_nodes)):
        vid = df_G_nodes["vid"][i]
        day = df_G_nodes["day"][i]
        df_features = pd.concat([df_features, df_dataset[(df_dataset['vid'] == vid) & (df_dataset['day'] == day)]])
    df_features = df_features.drop(columns=['vid', 'pid', 'day', 'cdiff'])
    df_features.replace({False: 0, True: 1}, inplace=True)
    features = torch.tensor(df_features.values, dtype=torch.float32, requires_grad=False, device=device)
    root_neighbors = get_neighbors_for_root(G_nodes)
    nodes = g.nodes()
    edges = g.edges()
    adj = torch.tensor(nx.adjacency_matrix(g, weight=None).todense(), dtype=torch.float32, requires_grad=False, device=device)
    terminals, asymp, whether_terminal, whether_asymp, node_list, node_dict = node_preprocess(nodes)
    labels = torch.tensor(whether_terminal, dtype=torch.long, requires_grad=False, device=device)
    Ws = torch.tensor([edges[edge]["weight"] for edge in edges], dtype=torch.float32, requires_grad=False, device=device)
    graph = dgl.from_networkx(g).to(device)
    weight_tensor, norm = compute_loss_para(adj, device)

    edge_dict, edge_weight = add_root(edges, root_neighbors, root_gamma)

    del df_dataset
    del G_nodes
    del df_G_nodes
    del df_features
    del root_neighbors

    return graph, nodes, node_list, node_dict, labels, whether_asymp, features, terminals, asymp, edge_dict, edge_weight, Ws, adj, weight_tensor, norm

def compute_loss_para(adj, device):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm

def fix_encoder(encoder, status):
    for param in encoder.parameters():
        param.requires_grad = status

def update_node_probability(node_prob, predicted_node_prob, predicted_label, nodes, terminals, device):
    for i, node in enumerate(nodes):            
        if node in terminals:
            node_prob[node] = 1
        else:
            node_prob[node] = float(predicted_node_prob[i]) if predicted_label[i] == 1 else (1 - float(predicted_node_prob[i]))
    node_prob["root"] = 0
    return node_prob

def predict_node_probability(Y_pred_val, nodes, terminals, device):
    predicted_node_probability, predicted_label = torch.max(Y_pred_val.data, 1)
    node_prob = update_node_probability({}, predicted_node_probability, predicted_label, nodes, terminals, device)
    del predicted_node_probability
    del predicted_label
    return node_prob

def get_edge_info(edge_dict, edge_weight, edges):
    for edge in edges:
        if edge[0] not in edge_dict:
            edge_dict[edge[0]] = {edge[1]}
        else:
            edge_dict[edge[0]].add(edge[1])
        edge_weight[edge] = edges[edge]["weight"]
    return edge_dict, edge_weight

def add_root(edges, root_neighbors, root_gamma):
    edge_dict, edge_weight = get_edge_info({}, {}, edges)
    edge_dict["root"] = set()
    for neighbor in root_neighbors:
        edge_dict["root"].add(neighbor)
        edge_weight[("root", neighbor)] = root_gamma
    return edge_dict, edge_weight

def get_neighbors_for_root(G_nodes):
    root_neighbors = set()
    unique = set()
    sorted_nodes = sorted(G_nodes)
    for n in sorted_nodes:
        if n[0] not in unique:
            unique.add(n[0])
            root_neighbors.add(str(n))
    return root_neighbors

def node_preprocess(nodes):
    terminals = set()
    asymp = set()
    whether_terminal = []
    whether_asymp = []
    node_list = []
    node_dict = {}
    index = 0
    for node in nodes:
        if nodes[node]["terminal"]:
            terminals.add(node)
        if nodes[node]["ASYMP"]:
            asymp.add(node)
        whether_terminal.append(1 if nodes[node]["terminal"] else 0)    
        whether_asymp.append(1 if nodes[node]["ASYMP"] else 0)
        node_list.append(node)
        node_dict[node] = index
        index += 1
    return terminals, asymp, whether_terminal, whether_asymp, node_list, node_dict

def add_edge(solution, node, solution_nodes, edge_dict, edge_weight):
    # node is the selected next node, add an corresponding edge for the node, the other endpoint should be in the solution
    if solution_nodes == set():
        solution.append((node, None))
    else:
        minimum_edge = math.inf
        selected_end_point = None
        for end_point in solution_nodes:
            if end_point in edge_dict and node in edge_dict[end_point]:
                temp_edge_weight = edge_weight[(end_point, node)]
                if temp_edge_weight < minimum_edge:
                    minimum_edge = temp_edge_weight
                    selected_end_point = end_point
    return solution + [(node, selected_end_point)]

def get_all_neighbors(node, neighbors, edge_dict):
    return neighbors.union(edge_dict[node]) if node in edge_dict else neighbors

def state2tens(state, num_of_nodes, device, node_list):
    solution = state.partial_solution_nodes   
    xv = [[(1 if node_list[i] in solution else 0)] for i in range(num_of_nodes)]
    state_tsr = torch.tensor(xv, dtype=torch.float32, requires_grad=False, device=device)
    del xv
    return state_tsr

def is_state_final(state, terminals):
    solution_nodes = state.partial_solution_nodes
    return terminals <= solution_nodes

def get_next_neighbor_random(state, neighbors):
    candidates = neighbors - state.partial_solution_nodes
    if candidates == set():
        return None
    return random.sample(candidates, 1)[0]

def discount_rewards(rewards, obj_baseline):
    return [(reward - obj_baseline) for reward in rewards]

def init(num_of_features, Encoder_hidden1, Encoder_hidden2, Decoder_hidden3, FFN_hidden1, FFN_hidden2,
         FFN_out, graph, features, Ws, learning_rate, q_hidden1, q_hidden2, q_hidden3, q_out, INIT_LR, device, file_name=None):
    """
    Create or load the model
    """
    gae_encoder = GAEEncoder(num_of_features, Encoder_hidden1, Encoder_hidden2).to(device)
    gae_decoder = GAEDecoder(Encoder_hidden2, Decoder_hidden3).to(device)
    ffn_model = FeedforwardNeuralNetModel(Encoder_hidden2, FFN_hidden1, FFN_hidden2, FFN_out).to(device)

    fix_encoder(gae_encoder, False)
    org_encoder = gae_encoder(graph, features, Ws)
    decoder_only_optimizer = optim.Adam(gae_decoder.parameters(), lr=learning_rate)
    ffn_only_optimizer = optim.SGD(ffn_model.parameters(), lr=learning_rate)

    fix_encoder(gae_encoder, True)
    gae_optimizer = optim.Adam(list(gae_encoder.parameters()) + list(gae_decoder.parameters()), lr=learning_rate)
    ffn_optimizer = optim.SGD(list(gae_encoder.parameters()) + list(ffn_model.parameters()), lr=learning_rate)

    Q_net = QNet(3 * Encoder_hidden2, q_hidden1, q_hidden2, q_hidden3, q_out, Encoder_hidden2).to(device)
    optimizer = optim.Adam(list(Q_net.parameters()) + list(gae_encoder.parameters()), lr=INIT_LR)

    obj_baseline = None
    if file_name is not None:
        checkpoint = torch.load(file_name)

        gae_encoder.load_state_dict(checkpoint['gae_encoder'])

        gae_decoder.load_state_dict(checkpoint['gae_decoder'])
        decoder_only_optimizer.load_state_dict(checkpoint['decoder_only_optimizer'])
        gae_optimizer.load_state_dict(checkpoint['gae_optimizer'])

        ffn_model.load_state_dict(checkpoint['ffn_model'])
        ffn_only_optimizer.load_state_dict(checkpoint['ffn_only_optimizer'])
        ffn_optimizer.load_state_dict(checkpoint['ffn_optimizer'])

        Q_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        org_encoder = checkpoint["org_encoder"]
        obj_baseline = checkpoint["obj_baseline"]

    Q_func = QFunction(Q_net, optimizer)
    return Q_func, Q_net, optimizer, gae_encoder, gae_decoder, decoder_only_optimizer, gae_optimizer, ffn_model, ffn_only_optimizer, ffn_optimizer, org_encoder, obj_baseline, file_name is None

def checkpoint(model, optimizer,
                     episode, FOLDER_NAME, org_encoder, gae_encoder, gae_decoder, decoder_only_optimizer, gae_optimizer, ffn_model, ffn_only_optimizer, ffn_optimizer, obj_baseline):
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
        
    fname = os.path.join(FOLDER_NAME, "{}.tar".format(episode))

    torch.save({
        "org_encoder": org_encoder,
        'gae_encoder': gae_encoder.state_dict(),
        "decoder_only_optimizer": decoder_only_optimizer.state_dict(),
        "gae_optimizer": gae_optimizer.state_dict(),
        'gae_decoder': gae_decoder.state_dict(),
        'ffn_model': ffn_model.state_dict(),
        "ffn_only_optimizer": ffn_only_optimizer.state_dict(),
        "ffn_optimizer": ffn_optimizer.state_dict(),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "obj_baseline": obj_baseline
    }, fname)

def get_predicted_node_labels(solution_nodes, terminals, nodes):
    predicted_asymp_nodes = solution_nodes - terminals
    predicted_list = []
    for node in nodes:
        if node != "root":
            predicted_list.append(1 if node in predicted_asymp_nodes else 0)
    return predicted_list

def get_predicted_solution(encoder, node_list, node_dict, terminals, graph, edge_dict, edge_weight, Q_func, State, NR_NODES, device):
    start_node = "root"
    solution = [(start_node, None)]
    solution_nodes = {start_node}
    neighbors = edge_dict[start_node]
    current_state = State(partial_solution=solution, partial_solution_nodes=solution_nodes, neighbors=neighbors, length=1)
    current_state_tsr = state2tens(current_state, NR_NODES, device, node_list)
    while not is_state_final(current_state, terminals):
        next_node, est_reward = Q_func.get_best_action(graph, current_state, neighbors, encoder, node_list, current_state_tsr)
        solution = add_edge(solution, next_node, solution_nodes, edge_dict, edge_weight)
        solution_nodes = solution_nodes.union({next_node})
        neighbors = get_all_neighbors(next_node, neighbors, edge_dict)
        new_length = current_state.length + 1
        del current_state
        current_state = State(partial_solution=solution, partial_solution_nodes=solution_nodes, neighbors=neighbors, length=new_length)
        idx = node_dict[next_node]
        current_state_tsr[idx][0] = 1
    del current_state
    del current_state_tsr
    return solution, solution_nodes

def test_accuracy(terminals, asymp, nodes, current_epoch, edge_weight, predicted_solution, predicted_solution_nodes, asymp_labels):
    predicted_labels = get_predicted_node_labels(predicted_solution_nodes, terminals, nodes)
    f1 = f1_score(asymp_labels, predicted_labels)
    recall = recall_score(asymp_labels, predicted_labels)
    precision = precision_score(asymp_labels, predicted_labels)
    # print("epoch: {}, f1_score: {}, recall: {}, precision: {}".format(current_epoch, f1, recall, precision))
    filename = "epoch{}_f1{}_recall{}_precision{}.graphml".format(current_epoch, f1, recall, precision)
    create_solution_graph(predicted_solution, nodes, filename, terminals, asymp, edge_weight)
    return filename

def create_solution_graph(solution, nodes, name, terminals, asymp, edge_weight):
    solution_graph = nx.DiGraph()
    for node in nodes:
        solution_graph.add_node(node, terminal=(node in terminals), ASYMP=(node in asymp), nodeID=node)
    for edge in solution:
        if edge != ("root", None):
            weight = float(edge_weight[edge[::-1]])
            solution_graph.add_edge(edge[1], edge[0], weight=weight)
    nx.write_graphml(solution_graph, "results/" + name)

def save_plot_graph(x, y, graphname, filename, x_axis, y_axis):
    plt.plot(x, y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(graphname)
    plt.savefig(filename)
    plt.close()
    plt.cla()
    plt.clf()