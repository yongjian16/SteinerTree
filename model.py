import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, MaxPooling, EdgeWeightNorm

class GAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim):
        super(GAEEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        layers = [GraphConv(self.in_dim, self.hidden1_dim, norm='none', activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)
        
        self.norm = EdgeWeightNorm(norm='both')
        
    def forward(self, g, features, edge_weight):
        norm_edge_weight = self.norm(g, edge_weight)
        x1 = self.layers[0](g, features, edge_weight=norm_edge_weight)
        x2 = self.layers[1](g, x1)
        return x2

class GAEDecoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super(GAEDecoder, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden)
    
    def forward(self,x):
        out1 = self.layer1(x)
        adj_rec = torch.sigmoid(torch.matmul(out1, out1.t()))
        return adj_rec

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim), 
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim), 
            nn.Softmax()
        )
    def forward(self, X):
        return self.net(X)

class QNet(nn.Module):   
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, emb_dim):
        super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim), 
            nn.LeakyReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden2_dim, hidden3_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden3_dim, output_dim)
        )
        self.maxpool = MaxPooling()
        
    def forward(self, graph, encoder, current_state_tsr):
        num_of_nodes = encoder.shape[0]
        in_state = torch.where(current_state_tsr > 0, encoder, torch.zeros_like(encoder))
        mu_in_state = F.relu(self.maxpool(graph, in_state).repeat(num_of_nodes, 1)) 
        # del in_state
        not_in_state = torch.where(current_state_tsr == 0, encoder, torch.zeros_like(encoder))
        mu_not_in_state = F.relu(self.maxpool(graph, not_in_state).repeat(num_of_nodes, 1))
        # del not_in_state
        Q_val = F.softmax(self.net(torch.cat((mu_in_state, mu_not_in_state, encoder), 1)), dim=0)
        # del mu_in_state
        # del mu_not_in_state
        return Q_val.squeeze(dim=1)

class QFunction():
    def __init__(self, model, optimizer):
        self.model = model  # The actual QNet
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()
    
    def predict(self, graph, encoder, current_state_tsr):
        with torch.no_grad():
            estimated_rewards = self.model(graph, encoder, current_state_tsr)
        return estimated_rewards
                
    def get_best_action(self, graph, state, neighbors, encoder, node_list, current_state_tsr):
        """ Computes the best (greedy) action to take from a given state
            Returns a tuple containing the ID of the next node and the corresponding estimated reward
        """
        estimated_rewards = self.predict(graph, encoder, current_state_tsr)
        sorted_reward_idx = estimated_rewards.argsort(descending=True)
        
        solution = state.partial_solution
        already_in = state.partial_solution_nodes

        for idx in sorted_reward_idx.tolist():
            actual_node = node_list[idx]
            if (state.length == 0 or solution[-1][0] != actual_node) and actual_node not in already_in and actual_node in neighbors:
                est_rewards = estimated_rewards[idx].item()
                del estimated_rewards
                del sorted_reward_idx
                del solution
                del already_in
                return actual_node, est_rewards