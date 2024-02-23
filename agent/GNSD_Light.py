from . import RLAgent
from common.registry import Registry
import math
import numpy as np
import os
import random
from collections import OrderedDict, deque
import gym
from torch.distributions import Categorical, Normal
from keras.utils import to_categorical
from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator
import torch
from torch import nn
import torch.nn.functional as F
import torch_scatter
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops

device = torch.device("cpu")

@Registry.register_model('GNSD-Light')
class LightAgent(RLAgent):
    #  TODO: test multiprocessing effect on agents or need deep copy here
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        """
        multi-agents in one model-> modify self.action_space, self.reward_generator, self.ob_generator here
        """
        #  general setting of world and model structure
        # TODO: different phases matching
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph

        self.world = world
        self.sub_agents = len(self.world.intersections)
        # TODO: support dynamic graph later
        self.edge_idx = torch.tensor(self.graph['sparse_adj'].T, dtype=torch.long)  # source -> target

        #  model parameters
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['setting'].param

        #  get generator for CoLightAgent
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"],
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        # TODO: add irregular control of signals in the future
        self.action_space = gym.spaces.Discrete(len(self.world.intersections[0].phases))

        if self.phase:
            # TODO: irregular ob and phase in the future
            if self.one_hot:
                self.ob_length = self.ob_generator[0][1].ob_length + len(self.world.intersections[0].phases)
            else:
                self.ob_length = self.ob_generator[0][1].ob_length + 1
        else:
            self.ob_length = self.ob_generator[0][1].ob_length

        self.get_attention = Registry.mapping['logger_mapping']['setting'].param['get_attention']
        self.rank = rank
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.her = Registry.mapping['model_mapping']['setting'].param['her']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.his_length = Registry.mapping['trainer_mapping']['setting'].param['his_length']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.model = self._build_model().to(device)
        self.target_model = self._build_model().to(device)
        self.update_target_network()
        self.criterion = nn.MSELoss(reduction='mean').to(device)
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.learning_rate,
                                       alpha=0.9, centered=False, eps=1e-7)

    def reset(self):
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        # queue metric
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        # delay metric
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"],
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

    def get_ob(self):
        x_obs = []  # sub_agents * lane_nums,
        for i in range(len(self.ob_generator)):
            x_obs.append((self.ob_generator[i][1].generate()) / self.vehicle_max)
        # construct edge information.
        length = set([len(i) for i in x_obs])
        if len(length) == 1:  # each intersections may has  different lane nums
            x_obs = np.array(x_obs, dtype=np.float32)
        else:
            x_obs = [np.expand_dims(x, axis=0) for x in x_obs]
        return x_obs

    def get_reward(self):
        # TODO: test output
        rewards_p = []
        rewards = []  # sub_agents
        for i in range(len(self.reward_generator)):  # 得到每个路口的平均值 [16]
            rewards_p.append(self.reward_generator[i][1].generate())
            rewards.append(self.reward_generator[i][1].generate_reward())
        rewards_p = np.squeeze(np.array(rewards_p)) * 12  # *12,即乘以了每个路口的12条道 【16】里面的每一项是每个路口12条道的总和
        rewards = np.squeeze(np.array(rewards))
        return [rewards_p], [rewards_p]
        # return [rewards_p], [rewards_p]

    def get_phase(self):
        # TODO: test phase output onehot/int
        phase = []  # sub_agents
        for i in range(len(self.phase_generator)):
            phase.append((self.phase_generator[i][1].generate()))
        phase = (np.concatenate(phase)).astype(np.int8)
        # phase = np.concatenate(phase, dtype=np.int8)
        return phase

    def get_queue(self):
        """
        get delay of intersection
        return: value(one intersection) or [intersections,](multiple intersections)
        """
        queue = []
        for i in range(len(self.queue)):
            queue.append((self.queue[i][1].generate()))
        tmp_queue = np.squeeze(np.array(queue))
        queue = np.sum(tmp_queue, axis=1 if len(tmp_queue.shape) == 2 else 0)
        return queue

    def get_delay(self):
        delay = []
        for i in range(len(self.delay)):
            delay.append((self.delay[i][1].generate()))
        delay = np.squeeze(np.array(delay))
        return delay  # [intersections,]

    def autoregreesive_act(self, observation, his_a, his_adj, model, batch_size, train):
        nodelayer_BFS = self.graph['nodelayer_BFS']
        edge_nei = torch.tensor(self.graph['Adj_matrix']).to(device)
        edge_pre = torch.tensor(self.graph['precursor_matrix']).to(device)
        nei_pos = torch.tensor(self.graph['nei_pos_matrix']).to(device)
        node_BFS = torch.tensor(self.graph['node_BFS']).to(device)
        his_pos_matrix = torch.tensor(self.graph['his_pos_matrix']).to(device)
        node_BFS_ = torch.zeros_like(node_BFS).to(device)
        for i in range(len(node_BFS)):
             node_BFS_[node_BFS[i]] = i
        node_BFS_ = F.one_hot(node_BFS_, num_classes=self.sub_agents)

        shifted_action = torch.zeros((batch_size, self.sub_agents+1, self.action_space.n+1)).to(device)
        output_action = torch.zeros((batch_size, self.sub_agents, 1), dtype=torch.long).to(device)
        shifted_action[:, -1, -1] = 1  # leader的前驱智能体是-1；前驱智能体不存在，所以0位标为1，（0位是无效位，后八位是有效位）,0位标志，代表这个动作为空

        padd_agent = torch.tensor([8]*his_a.size(0)).reshape(his_a.size(0), 1).to(device)
        his_a = torch.cat((his_a, padd_agent), dim=-1).to(device) # [batch*4,17]
        one_hot_his_a = F.one_hot(his_a, num_classes=self.action_space.n+1)

        his_a = torch.tensor(one_hot_his_a, dtype=torch.float32).reshape(batch_size, -1, one_hot_his_a.shape[-2], one_hot_his_a.shape[-1])

        for i, nodes in enumerate(nodelayer_BFS):
            out = model(observation, his_a, edge_nei, edge_pre, nei_pos, his_pos_matrix, node_BFS_, train)
            logit = out[:, nodes, :]

            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1)

            output_action[:, nodes, :] = action.unsqueeze(-1)
            if i + 1 < len(nodelayer_BFS):
                one_hot_action = torch.tensor(F.one_hot(action, num_classes=self.action_space.n), dtype=torch.float32)
                shifted_action[:, nodes, :-1] = one_hot_action

                his_a[:, :, nodes, :] = torch.roll(his_a[:, :, nodes, :], -1, dims=1)
                his_a[:, -1, nodes, :-1] = one_hot_action
                his_a[:, -1, nodes, -1] = 0

        return out, output_action

    def parallel_act(self, observation, actions, his_a, his_adj, train):
        actions = actions.view(self.batch_size, self.sub_agents)
        edge_nei = torch.tensor(self.graph['Adj_matrix']).to(device)
        edge_pre = torch.tensor(self.graph['precursor_matrix']).to(device)
        nei_pos = torch.tensor(self.graph['nei_pos_matrix']).to(device)
        node_BFS = torch.tensor(self.graph['node_BFS']).to(device)
        his_pos_matrix = torch.tensor(self.graph['his_pos_matrix']).to(device)

        node_BFS_ = torch.zeros_like(node_BFS)
        for i in range(len(node_BFS)):
            node_BFS_[node_BFS[i]] = i
        node_BFS_ = F.one_hot(node_BFS_, num_classes=self.sub_agents)

        one_hot_action = F.one_hot(actions.squeeze(-1), num_classes=self.action_space.n)  # (batch, n_agent, action_dim)
        shifted_action = torch.zeros((self.batch_size, self.sub_agents+1, self.action_space.n+1)).to(device)
        shifted_action[:, -1, -1] = 1  # leader的前驱智能体是-1；前驱智能体不存在，所以0位标为1，（0位是无效位，后八位是有效位）,0位标志，代表这个动作为空
        shifted_action[:, :-1, :-1] = one_hot_action


        padd_agent = torch.tensor([8]*his_a.size(0)).reshape(his_a.size(0), 1).to(device)
        his_a = torch.cat((his_a, padd_agent), dim=-1) # [batch*4,17]
        one_hot_his_a = F.one_hot(his_a, num_classes=self.action_space.n + 1)
        his_a = torch.tensor(one_hot_his_a, dtype=torch.float32).reshape(self.batch_size, -1, one_hot_his_a.shape[-2], one_hot_his_a.shape[-1])
        out = self.model(observation, his_a, edge_nei, edge_pre, nei_pos, his_pos_matrix, node_BFS_, train)
        return out

    def get_action(self, ob, his_a, his_adj, test=False):

        if not test:
            if np.random.rand() <= self.epsilon:
                return np.random.randint(0, self.action_space.n, self.sub_agents)
        observation = torch.tensor(ob, dtype=torch.float32).to(device)
        his_a = torch.tensor(his_a, dtype=torch.long).to(device)
        his_adj = torch.tensor(his_adj, dtype=torch.long).to(device)
        logit, output_action = self.autoregreesive_act(observation, his_a, his_adj, self.model, 1, train=False)

        return (output_action.squeeze()).clone().cpu().detach().numpy()

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def _build_model(self):
        model = Decoder(self.ob_length, self.action_space.n, self.sub_agents, self.his_length, **self.model_dict)
        return model

    def remember(self, last_obs, last_phase, actions, his_a, his_adj, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase, his_a, his_adj)))

    def _batchwise(self, samples):
        batch_list = []
        batch_list_p = []
        actions = []
        his_a = []
        his_adj = []
        rewards = []
        for item in samples:
            dp = item[1]
            state = torch.tensor(dp[0], dtype=torch.float32)
            batch_list.append(Data(x=state, edge_index=self.edge_idx))

            state_p = torch.tensor(dp[4], dtype=torch.float32)
            batch_list_p.append(Data(x=state_p, edge_index=self.edge_idx))
            rewards.append(dp[3])
            actions.append(dp[2])
            his_a.append(dp[6])
            his_adj.append(dp[7])
        batch_t = Batch.from_data_list(batch_list)
        batch_tp = Batch.from_data_list(batch_list_p)  # (agents*batch_size,obs_dim+agent_dim)
        # TODO reshape slow warning
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        his_a = torch.tensor(np.array(his_a), dtype=torch.long)
        his_adj = torch.tensor(np.array(his_adj), dtype=torch.long)
        rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        actions = actions.view(actions.shape[0] * actions.shape[1])
        his_a = his_a.view(his_a.shape[0] * his_a.shape[1],  his_a.shape[2])

        return batch_t, batch_tp, rewards, actions, his_a, his_adj

    def train(self):
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions, his_a, his_adj = self._batchwise(samples)
        out, _ = self.autoregreesive_act(b_tp.x.to(device), his_a.to(device), his_adj.to(device), self.target_model, self.batch_size, train=False)
        out = out.reshape(-1, self.action_space.n)
        target = rewards.to(device) + self.gamma * torch.max(out, dim=-1)[0]
        target_f = self.parallel_act(b_t.x.to(device), actions.to(device), his_a.to(device), his_adj.to(device), train=False).reshape(-1, self.action_space.n)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]

        deltas = target_f - self.parallel_act(b_t.x.to(device), actions.to(device), his_a.to(device), his_adj.to(device), train=True).reshape(-1, self.action_space.n)

        real_deltas = torch.where(deltas > 0, deltas, deltas * self.her)
        loss = torch.mean(torch.pow(real_deltas.to(device), 2)).to(device)
        loss.requires_grad_(True)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.clone().cpu().detach().numpy()

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                  'model', f'{e}_{2}.pt')
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_name, map_location='cpu'))
        self.target_model = self._build_model()
        self.target_model.load_state_dict(torch.load(model_name, map_location='cpu'))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        path = os.path.join(path, 'pos2')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)


class MultiHeadAttModel(nn.Module):
    def __init__(self, n_embd, dv, nv, n_agent):
        super(MultiHeadAttModel, self).__init__()
        self.n_embd = n_embd
        self.dv = dv
        self.nv = nv
        self.fcv = init_(nn.Linear(n_embd, dv * nv))
        self.fck = init_(nn.Linear(n_embd, dv * nv))
        self.fcq = init_(nn.Linear(n_embd, dv * nv))
        self.fcout = init_(nn.Linear(dv * nv, n_embd))
        self.n_agent = n_agent
        self.positional_encoding_nei = torch.tensor([3, 5, 7, 11, 13]) #第一项为需要mask的向量，其他为上下左右自己向量
        self.positional_encoding_pre = positional_encoding(5, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量

    def forward(self, q, k, adjs, adjs_pos=None):
        query, key = q, k
        n_agent = q.size(1)
        batch_size = q.size(0)
        d_k = k.size(-1)

        agent_repr = torch.unsqueeze(query, dim=2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2)  # [?, 16, 1, 1, 32]

        # hj*Ws
        neighbor_repr = torch.unsqueeze(key, dim=1)
        neighbor_repr = neighbor_repr.repeat(1, n_agent, 1, 1)  # [?, 16, 16, 32]

        if adjs_pos != None:
            flag = 'nei' if adjs_pos.size(-1) == 6 else 'pre'
            adjs_pos = adjs_pos.repeat(batch_size, 1, 1)
            adjs_pos = torch.reshape(adjs_pos, (-1, n_agent, adjs_pos.size(-2), 1, adjs_pos.size(-1))) #[256,16,32,1,6]
            positional_encoding = self.positional_encoding_nei if flag == 'nei' else self.positional_encoding_pre
            pos_encoding = positional_encoding.repeat(adjs_pos.size(0), adjs_pos.size(1), adjs_pos.size(2), 1, 1)
            pos_encoding = (torch.matmul(adjs_pos, pos_encoding)).squeeze()
            neighbor_repr = neighbor_repr + pos_encoding

        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]

        att = torch.matmul(agent_repr_head, neighbor_repr_head) / math.sqrt(d_k)
        mask = adjs
        mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(att.size(0), 1, 1, 1, 1)
        att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)


        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]

        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        out_concat = transpose_output(out, self.nv)
        out_concat = torch.squeeze(out_concat, dim=2)  # [?, 16, 32]
        out_concat = F.relu(self.fcout(out_concat))

        return out_concat

class MultiHeadAttModel_allin(nn.Module):
    def __init__(self, n_embd, dv, nv, n_agent, his_length):
        super(MultiHeadAttModel_allin, self).__init__()
        self.n_embd = n_embd
        self.action_dim = 8
        self.dv = dv
        self.nv = nv
        self.fcv = init_(nn.Linear(n_embd, dv * nv))
        self.fck = init_(nn.Linear(n_embd, dv * nv))
        self.fcq = init_(nn.Linear(n_embd, dv * nv))
        self.fcout = init_(nn.Linear(dv * nv, n_embd))
        self.his_action_encoder = nn.Sequential(init_(nn.Linear(self.action_dim + 1, self.n_embd, bias=False), activate=True), nn.GELU())
        self.n_agent = n_agent
        self.positional_encoding = positional_encoding(his_length, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量

    def forward(self, q, k, adjs_pre, adjs_pos=None):
        query, key = q, k
        n_agent = q.size(1)
        batch_size = q.size(0)
        d_k = k.size(-1)


        # hi*Wt
        agent_repr = torch.unsqueeze(query, dim=2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2)  # [?, 16, 1, 1, 32]

        neighbor_repr = torch.unsqueeze(key, dim=1)
        neighbor_repr = neighbor_repr.repeat(1, n_agent, 1, 1, 1)  # [?, 16, 16, 32]

        if adjs_pos != None:

            adjs_pos = adjs_pos.reshape(adjs_pos.shape[0], -1, n_agent+1).unsqueeze(0)
            adjs_pos = adjs_pos.repeat(batch_size, 1, 1, 1).unsqueeze(-2).type(torch.float32)

            neighbor_repr = torch.matmul(adjs_pos, neighbor_repr)
            neighbor_repr = self.his_action_encoder(neighbor_repr.squeeze(-2))
            positional_encoding = self.positional_encoding.unsqueeze(0).unsqueeze(1).repeat(neighbor_repr.size(0),neighbor_repr.size(1),1,1).to(device)
            neighbor_repr = neighbor_repr + positional_encoding

        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]

        att = torch.matmul(agent_repr_head, neighbor_repr_head) / math.sqrt(d_k)

        att = F.softmax(att, dim=-1)

        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]

        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        out_concat = transpose_output(out, self.nv)
        out_concat = torch.squeeze(out_concat, dim=2)  # [?, 16, 32]
        out_concat = F.relu(self.fcout(out_concat))

        return out_concat


class DecodeBlock_obs(nn.Module):

    def __init__(self, n_embd, dv, n_head, n_agent):
        super(DecodeBlock_obs, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttModel(n_embd, dv, n_head, n_agent)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU()
        )

    def forward(self, x, adjs_nei, nei_pos):
        x = x + self.attn(x, x, adjs_nei)
        return x


class DecodeBlock(nn.Module):

    def __init__(self, n_embd, dv, n_head, n_agent, his_length):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = MultiHeadAttModel_allin(n_embd, dv, n_head, n_agent, his_length)
        self.attn2 = MultiHeadAttModel(n_embd, dv, n_head, n_agent)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.Softmax(-1),
        )

    def forward(self, rep_enc, his_action_embeddings, adjs_pre, his_pos_matrix):
        x = self.attn1(q=rep_enc, k=his_action_embeddings, adjs_pre=adjs_pre, adjs_pos=his_pos_matrix)
        x = rep_enc + x
        return x


class Decoder(nn.Module):

    def __init__(self, input_dim, output_dim, n_node, his_length, **kwargs):
        super(Decoder, self).__init__()
        self.model_dict = kwargs
        self.action_dim = gym.spaces.Discrete(output_dim).n
        self.his_length = his_length

        self.n_embd = self.model_dict.get('NODE_EMB_DIM')[0]
        self.obs_dim = input_dim
        self.n_agent = n_node
        n_block = 1

        self.action_encoder = nn.Sequential(init_(nn.Linear(self.action_dim+1, self.n_embd, bias=False), activate=True), nn.GELU())
        self.his_action_encoder = nn.Sequential(init_(nn.Linear(self.action_dim+1, self.n_embd, bias=False), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(init_(nn.Linear(self.obs_dim, self.n_embd), activate=True), nn.GELU())
        self.positional_encoding = positional_encoding(self.n_agent, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量
        self.ln = nn.LayerNorm(self.n_embd)
        self.blocks_obs = nn.Sequential(*[DecodeBlock_obs(self.model_dict.get('INPUT_DIM_1')[i],
                                                          self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                                          self.model_dict.get('NUM_HEADS')[i],
                                                          self.n_agent) for i in range(n_block)])
        self.blocks = nn.Sequential(*[DecodeBlock(self.model_dict.get('INPUT_DIM_1')[i],
                                                  self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                                  self.model_dict.get('NUM_HEADS')[i],
                                                  self.n_agent, self.his_length) for i in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(self.n_embd, self.action_dim), activate=True))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def _forward(self, obs, his_action,adjs_nei, adjs_pre, nei_pos, his_pos_matrix, node_BFS):
        obs = torch.reshape(obs, (-1, self.n_agent, self.obs_dim))
        obs_emb = self.obs_encoder(obs)
        positional_encoding = torch.matmul(node_BFS.type(torch.float32), self.positional_encoding.to(device))
        positional_encoding = positional_encoding.repeat(obs_emb.size(0), 1, 1)
        obs_emb_pos = obs_emb + positional_encoding
        for block in self.blocks_obs:
            obs_embeddings = block.forward(obs_emb_pos, adjs_nei, nei_pos)
        for block in self.blocks:
            x = block(obs_embeddings, his_action, adjs_pre, his_pos_matrix)
        out = self.head(x)
        return out

    def forward(self, obs, his_action, adjs_nei, adjs_pre, nei_pos, his_pos_matrix,node_BFS, train=True):
        if train:
            return self._forward(obs, his_action, adjs_nei, adjs_pre, nei_pos, his_pos_matrix, node_BFS)
        else:
            with torch.no_grad():
                return self._forward(obs, his_action, adjs_nei, adjs_pre, nei_pos, his_pos_matrix, node_BFS)


def positional_encoding(max_seq_len, d_model):
    # 初始化位置编码矩阵
    position = torch.arange(0, max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(max_seq_len, d_model)
    # 使用正弦和余弦函数计算位置编码
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


