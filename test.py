from input_data import *
from helper import *
import argparse

'''
1. Load models and Q function.
2. Generate a solution graph
3. Prune the graph by removing nodes without any terminals as descendants
'''


# check whether a node has any terminal as descendants
# if the node itself is a terminal, return True
def with_terminals(node, edge_dict, terminals):
    if node == "root" or node in terminals:
        return True
    if node in edge_dict:
        for neighbor in edge_dict[node]:
            if neighbor in terminals:
                return True
        found_in_neighbors = False
        for neighbor in edge_dict[node]:
            found_in_neighbors = found_in_neighbors or with_terminals(neighbor, edge_dict, terminals)
        return found_in_neighbors
    else:
        return False


def remove_nodes_without_terminals_as_descendants(nodes, edges, edge_dict, terminals, asymp, filename):
    solution_nodes = set()
    solution_graph = nx.DiGraph()
    for node in nodes:
        solution_graph.add_node(node, terminal=(node in terminals), ASYMP=(node in asymp), nodeID=node)
    for edge in edges:
        src = edge[0]
        tar = edge[1]
        if with_terminals(src, edge_dict, terminals) and with_terminals(tar, edge_dict, terminals):
            weight = edges[edge]["weight"]
            solution_graph.add_edge(src, tar, weight=weight)
            # only add nodes that have edges
            solution_nodes.add(src)
            solution_nodes.add(tar)
    nx.write_graphml(solution_graph, "results/pruned_" + filename)
    return solution_nodes


def prune_graph_and_test_accuracy(filename):
    graph = nx.read_graphml("results/" + filename)
    nodes = graph.nodes()
    edges = graph.edges()
    edge_dict, _ = get_edge_info({}, {}, edges)

    solution_nodes = remove_nodes_without_terminals_as_descendants(nodes, edges, edge_dict, terminals, asymp, filename)
    predict_asymp_labels = get_predicted_node_labels(solution_nodes, terminals, nodes)
    print("After pruning, f1_score:", f1_score(asymp_labels, predict_asymp_labels),
          "recall:", recall_score(asymp_labels, predict_asymp_labels),
          "precision:", precision_score(asymp_labels, predict_asymp_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-gamma', '--gamma', type=int, default=128,
                        help='gamma.')
    parser.add_argument('-alpha', '--alpha', type=int, default=1,
                        help='alpha.')
    parser.add_argument('-graph', '--graph', type=str, default="data/G500.graphml",
                        help='graph name.')
    parser.add_argument('-features', '--features', type=str, default="data/features.csv",
                        help='feature file name.')
    parser.add_argument('-modelname', '--modelname', type=str, default=None,
                        help='model name.')
    args = parser.parse_args()
    gamma = args.gamma
    alpha = args.alpha
    graph_file = args.graph
    feature_file = args.features
    model_name = args.modelname
    epoch = int(model_name.split("/")[1].split(".")[0])

    # seed everything for reproducible results first:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    graph, nodes, node_list, node_dict, labels, asymp_labels, features, terminals, asymp, edge_dict, edge_weight, Ws, adj, weight_tensor, norm = load_graph(
        graph_file, feature_file, gamma, device)
    num_of_features = features.size()[-1]
    NR_NODES = features.size()[0]  # Number of nodes N

    # Load module and Q-function
    Q_func, Q_net, _, gae_encoder, gae_decoder, _, _, ffn_model, _, _, _, _, _ = init(num_of_features, Encoder_hidden1,
                                                                                      Encoder_hidden2, Decoder_hidden3,
                                                                                      FFN_hidden1, FFN_hidden2,
                                                                                      FFN_out, graph, features, Ws,
                                                                                      learning_rate, q_hidden1,
                                                                                      q_hidden2, q_hidden3, q_out,
                                                                                      INIT_LR, device,
                                                                                      file_name=model_name)
    with torch.no_grad():
        encoder = gae_encoder(graph, features, Ws)
    predicted_solution, predicted_solution_nodes = get_predicted_solution(encoder, node_list, node_dict, terminals,
                                                                          graph, edge_dict, edge_weight,
                                                                          Q_func, State, NR_NODES, device)
    filename = test_accuracy(terminals, asymp, nodes, epoch, edge_weight, predicted_solution,
                             predicted_solution_nodes, asymp_labels)
    del encoder
    del predicted_solution
    del predicted_solution_nodes
    prune_graph_and_test_accuracy(filename)
