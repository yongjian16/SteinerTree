from input_data import *
from helper import *
import argparse

'''
1. pretrain GAE
2. pretrain FFN
3. pretrain GAE and FFN together
4. train GAE, FFN, and QNet
'''

def train_gae_model(num_of_iterations, optimizer, org_encoder=None):
    if org_encoder != None:
        encoder = org_encoder
    for i in range(num_of_iterations):
        if org_encoder == None:
            gae_encoder.train()
            encoder = gae_encoder(graph, features, Ws)
        gae_decoder.train()
        decoder = gae_decoder(encoder)
        gae_loss = norm * F.binary_cross_entropy(decoder.view(-1), adj.view(-1), weight=weight_tensor)
        optimizer.zero_grad()
        gae_loss.backward()
        optimizer.step()
        gae_loss.item()

def train_ffn_model(num_of_iterations, optimizer, org_encoder=None):
    if org_encoder != None:
        encoder = org_encoder
    for i in range(num_of_iterations):
        if org_encoder == None:
            gae_encoder.train()
            encoder = gae_encoder(graph, features, Ws)
        ffn_model.train()
        ffn_output = ffn_model(encoder)
        ffn_loss = F.binary_cross_entropy(ffn_output, target)
        optimizer.zero_grad()
        ffn_loss.backward()
        optimizer.step()
        ffn_loss.item()

def train_together(num_of_iterations=10000):
    for k in range(num_of_iterations):
        train_gae_model(1, gae_optimizer)
        train_ffn_model(1, ffn_optimizer)

def train(obj_baseline):
    losses = []
    loss_iterations = []
    gae_losses = []
    ffn_losses = []
    gae_ffn_iterations = []
    
    batch_rewards = []
    estimated_rewards = []
    batch_counter = 0
    for episode in range(rl):
        total_edge_weight = 0
        total_node_weight = 0
        total_reward = 0
        finished = False
        start_node = "root"
        solution = [(start_node, None)]
        solution_nodes = {start_node}
        neighbors = edge_dict[start_node]
        # current state (tuple and tensor)
        current_state = State(partial_solution=solution, partial_solution_nodes=solution_nodes, neighbors=neighbors, length=1)
        current_state_tsr = state2tens(current_state, NR_NODES, device, node_list)
        # current value of epsilon
        epsilon = max(MIN_EPSILON, (1-EPSILON_DECAY_RATE)**episode)
        # fix encoder
        if batch_counter == 0:
            fix_encoder(gae_encoder, False)
            fixed_encoder = gae_encoder(graph, features, Ws)
        # train gae
        gae_loss_for_each_iteration = train_gae_model(5, decoder_only_optimizer, fixed_encoder)
        gae_losses.append(gae_loss_for_each_iteration)
        # train ffn
        ffn_loss_for_each_iteration = train_ffn_model(5, ffn_only_optimizer, fixed_encoder)
        ffn_losses.append(ffn_loss_for_each_iteration)

        gae_ffn_iterations.append(episode)
        # encoder update in RL
        if batch_counter == 0:
            fix_encoder(gae_encoder, True) 
            encoder = gae_encoder(graph, features, Ws)
        # update node probability   
        with torch.no_grad():
            y_pred_val = ffn_model(encoder)
        node_prob = predict_node_probability(y_pred_val, nodes, terminals, device)
        del y_pred_val
        while not finished: # encoder should update
            if epsilon >= random.random(): # explore
                next_node = get_next_neighbor_random(current_state, neighbors)
            else: # exploit
                next_node, next_reward = Q_func.get_best_action(graph, current_state, neighbors, encoder, node_list, current_state_tsr)
                if episode % 1000 == 0:
                    print('Ep {} | next est reward: {}'.format(episode, next_reward))
                del next_reward
            # store estimated rewards
            if episode > 0:
                est_rewards = Q_func.model(graph, encoder, current_state_tsr)
                estimated_rewards.append(est_rewards[node_dict[next_node]])
                del est_rewards
            next_solution = add_edge(solution, next_node, solution_nodes, edge_dict, edge_weight)
            next_solution_nodes = solution_nodes.union({next_node})
            next_edge_weight = edge_weight[next_solution[-1][::-1]]
            next_node_weight = node_prob[next_node]
            reward = - next_edge_weight + alpha * next_node_weight
            next_neighbors = get_all_neighbors(next_node, neighbors, edge_dict)
            next_state = State(partial_solution=next_solution, partial_solution_nodes=next_solution_nodes, neighbors=next_neighbors, length=current_state.length+1)
            next_state_tsr = state2tens(next_state, NR_NODES, device, node_list)
            # update state and current solution
            del current_state
            del current_state_tsr
            del solution
            del neighbors
            del solution_nodes
            del next_node
            
            current_state = next_state
            current_state_tsr = next_state_tsr       
            solution = next_solution
            neighbors = next_neighbors
            solution_nodes = next_solution_nodes
            total_edge_weight += next_edge_weight
            total_node_weight += next_node_weight
            total_reward += reward
            
            del reward
            del next_edge_weight
            del next_node_weight

            # take a gradient step
            loss_val = None           
            if is_state_final(current_state, terminals):
                finished = True
                if episode == 0:
                    obj_baseline = total_reward
                else:
                    obj_rewards = [abs(obj_baseline - total_reward)/(current_state.length-1)] * (current_state.length-1)
                    batch_rewards += discount_rewards(obj_rewards, obj_baseline / current_state.length-1)
                    del obj_rewards
                    batch_counter += 1
                    if batch_counter == BATCH_SIZE:
                        reward_tensor = torch.tensor(batch_rewards, device=device) 
                        del batch_rewards
                        estimated_rewards_tsr = torch.stack(estimated_rewards).to(device)
                        del estimated_rewards
                        # Calculate loss
                        logprob = torch.log(estimated_rewards_tsr)
                        del estimated_rewards_tsr
                        selected_logprobs = reward_tensor * logprob
                        del reward_tensor
                        del logprob
                        
                        optimizer.zero_grad()
                        loss = -selected_logprobs.sum()
                        del selected_logprobs
                        
                        loss_val = loss.item()
                        # Calculate gradients
                        loss.backward()
                        # Apply gradients
                        optimizer.step()
                        
                        del loss

                        estimated_rewards = []
                        batch_rewards = []
                        batch_counter = 0

                        losses.append(loss_val)
                        loss_iterations.append(episode / 16)

                        del fixed_encoder
                        del encoder
                        
                # delete variables from the last iteration
                del current_state
                del current_state_tsr
                del solution
                del neighbors
                del solution_nodes
                del node_prob
                del total_reward
                        
        length = total_edge_weight - alpha * total_node_weight

        if episode % 5000 == 0:
            print('Ep %d. length = %.4f / epsilon = %.4f' % (
                episode, length, epsilon))
        
        if episode >= frequency and episode % frequency == 0:
            # save the model
            # save_plot_graph(loss_iterations, losses, "RL", "results/{}_rl_loss.png".format(episode), "Training iteration", "Loss")
            # save_plot_graph(gae_ffn_iterations, gae_losses, "GAE in RL", "results/{}_GAE_loss.png".format(episode), "Training iteration", "Loss")
            # save_plot_graph(gae_ffn_iterations, ffn_losses, "FFN in RL", "results/{}_FFN_loss.png".format(episode), "Training iteration", "Loss")
            checkpoint(Q_net, optimizer, episode, FOLDER_NAME, org_encoder, gae_encoder, gae_decoder, decoder_only_optimizer, gae_optimizer, ffn_model, ffn_only_optimizer, ffn_optimizer, obj_baseline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-gamma', '--gamma', type=int, default=128,
                        help= 'gamma.')
    parser.add_argument('-alpha', '--alpha', type=int, default=1,
                        help= 'alpha.')
    parser.add_argument('-graph', '--graph', type=str, default="data/G500.graphml",
                        help= 'graph name.')
    parser.add_argument('-features', '--features', type=str, default="data/features.csv",
                        help= 'feature file name.')
    parser.add_argument('-gae', '--gae', type=int, default=10000,
                        help= 'pretrain gae epochs.')
    parser.add_argument('-ffn', '--ffn', type=int, default=10000,
                        help= 'pretrain ffn epochs.')
    parser.add_argument('-pretrain', '--pretrain', type=int, default=100000,
                        help= 'pretrain gae and ffn together epochs.')
    parser.add_argument('-rl', '--rl', type=int, default=100001,
                        help= 'train rl epochs.')
    parser.add_argument('-modelname', '--modelname', type=str, default=None,
                        help= 'model name.')
    parser.add_argument('-batchsize', '--batchsize', type=int, default=16,
                        help= 'batch size.')
    parser.add_argument('-frequency', '--frequency', type=int, default=100,
                        help= 'how often do you want to save the model')
    args = parser.parse_args()
    gamma = args.gamma
    alpha = args.alpha
    graph_file = args.graph
    feature_file = args.features
    gae = args.gae
    ffn = args.ffn
    pretrain = args.pretrain
    rl = args.rl
    model_name = args.modelname
    BATCH_SIZE = args.batchsize
    frequency = args.frequency

    # seed everything for reproducible results first:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    graph, nodes, node_list, node_dict, labels, asymp_labels, features, terminals, asymp, edge_dict, edge_weight, Ws, adj, weight_tensor, norm = load_graph(
        graph_file, feature_file, gamma, device)
    num_of_features = features.size()[-1]
    NR_NODES = features.size()[0]  # Number of nodes N

    # Create module, optimizer, LR scheduler, and Q-function
    Q_func, Q_net, optimizer, gae_encoder, gae_decoder, decoder_only_optimizer,  gae_optimizer, ffn_model, ffn_only_optimizer, ffn_optimizer, org_encoder, obj_baseline, pre_train = init(num_of_features, Encoder_hidden1, Encoder_hidden2, Decoder_hidden3, FFN_hidden1, FFN_hidden2,
         FFN_out, graph, features, Ws, learning_rate, q_hidden1, q_hidden2, q_hidden3, q_out, INIT_LR, device, file_name=model_name)

    target = torch.zeros(NR_NODES, 2)
    target[range(target.shape[0]), labels]=1

    if pre_train:
        # train gae and ffn separately
        gae_loss_separately = train_gae_model(gae, decoder_only_optimizer)
        ffn_loss_separately = train_ffn_model(ffn, ffn_only_optimizer)
        # train together
        train_together(pretrain)
        # save the pre-trained models
        checkpoint(Q_net, optimizer, -1, FOLDER_NAME, org_encoder, gae_encoder, gae_decoder, decoder_only_optimizer, gae_optimizer, ffn_model, ffn_only_optimizer, ffn_optimizer, None)
    train(obj_baseline)