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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@Registry.register_model('bfs-hdgn')
class CoLightAgent(RLAgent):
    #  TODO: test multiprocessing effect on agents or need deep copy here
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        """
        multi-agents in one model-> modify self.action_space, self.reward_generator, self.ob_generator here
        """
        #  general setting of world and model structure
        # TODO: different phases matching
        # self.buffer_size = Registry.mapping['trainer_mapping']['trainer_setting'].param['buffer_size']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph

        self.world = world
        # self.sub_agents = 1
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
            # LaneVehicleGenerator通过这个生成的I
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        #  get reward generator for CoLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  get queue generator for CoLightAgent
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

        #  get delay generator for CoLightAgent
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
        # train parameters
        self.rank = rank
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.her = Registry.mapping['model_mapping']['setting'].param['her']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.his_length = Registry.mapping['trainer_mapping']['setting'].param['his_length']
        # self.epsilon = np.ones(self.sub_agents) * Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        # self.lr_decay = Registry.mapping['model_mapping']['setting'].param['lr_decay']
        # self.lr_min = Registry.mapping['model_mapping']['setting'].param['lr_min']
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
        # self.optimizer = optim.Adam(self.model.parameters(),
        #                                lr=self.learning_rate)

    def reset(self):
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        #  get reward generator for CoLightAgent
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
        leader = torch.tensor(nodelayer_BFS[0]).to(device)
        follow = torch.tensor(sorted(np.concatenate(nodelayer_BFS[1:])), dtype=torch.int64).to(device)
        his_pos_matrix = torch.tensor(self.graph['his_pos_matrix']).to(device)
        # follow = torch.sort(follow)
        # follow.sort()
        # follow = torch.flatten(torch.cat(nodelayer_BFS[1:]))
        node_BFS_ = torch.zeros_like(node_BFS).to(device)
        for i in range(len(node_BFS)):
             node_BFS_[node_BFS[i]] = i
        node_BFS_ = F.one_hot(node_BFS_, num_classes=self.sub_agents)

        # his_a = his_a.view(batch_size, -1, his_a.size(-1))
        shifted_action = torch.zeros((batch_size, self.sub_agents+1, self.action_space.n+1)).to(device)
        # his_aciton = torch.zeros((batch_size, self.sub_agents+1, 4, self.action_space.n+1)).to(device)
        output_q = torch.zeros((batch_size, self.sub_agents, self.action_space.n), dtype=torch.float32)
        output_action = torch.zeros((batch_size, self.sub_agents, 1), dtype=torch.long).to(device)
        shifted_action[:, -1, -1] = 1  # leader的前驱智能体是-1；前驱智能体不存在，所以0位标为1，（0位是无效位，后八位是有效位）,0位标志，代表这个动作为空

        '''处理his_a,转为one_hot,his_a不包括当前时刻动作，所以输入到网络里应该加上前驱当前时刻动作
            当前的his_a是所有路口的历史动作，加上前驱当前时刻的动作，作为网络的输入,不用每一轮挑出对应的前驱的历史动作，因为输入了前驱矩阵，在网络中再挑出来，且parallel_act无法每一轮挑'''
        padd_agent = torch.tensor([8]*his_a.size(0)).reshape(his_a.size(0), 1).to(device)
        his_a = torch.cat((his_a, padd_agent), dim=-1).to(device) # [batch*4,17]
        one_hot_his_a = F.one_hot(his_a, num_classes=self.action_space.n+1)
        # one_hot_his_action = [[0] + sublist for sublist in one_hot_his_a]

        his_a = torch.tensor(one_hot_his_a, dtype=torch.float32).reshape(batch_size, -1, one_hot_his_a.shape[-2], one_hot_his_a.shape[-1])

        # his_aciton = torch.concatenate((his_a, shifted_action), dim=-2)
        for i, nodes in enumerate(nodelayer_BFS):
            out = model(observation, his_a, edge_nei, edge_pre, nei_pos, his_pos_matrix, node_BFS_, train)
            logit = out[:, nodes, :]
            # output_q[:, nodes, :] = logit

            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1)

            output_action[:, nodes, :] = action.unsqueeze(-1)
            if i + 1 < len(nodelayer_BFS):
                # shifted_action[:, nodes, 0] = 0  # 这个位置的动作不为空了
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
        nodelayer_BFS = self.graph['nodelayer_BFS']
        leader = torch.tensor(nodelayer_BFS[0]).to(device)
        follow = torch.tensor(sorted(np.concatenate(nodelayer_BFS[1:])), dtype=torch.int64).to(device)

        node_BFS_ = torch.zeros_like(node_BFS)
        for i in range(len(node_BFS)):
            node_BFS_[node_BFS[i]] = i
        node_BFS_ = F.one_hot(node_BFS_, num_classes=self.sub_agents)

        one_hot_action = F.one_hot(actions.squeeze(-1), num_classes=self.action_space.n)  # (batch, n_agent, action_dim)
        shifted_action = torch.zeros((self.batch_size, self.sub_agents+1, self.action_space.n+1)).to(device)
        shifted_action[:, -1, -1] = 1  # leader的前驱智能体是-1；前驱智能体不存在，所以0位标为1，（0位是无效位，后八位是有效位）,0位标志，代表这个动作为空
        shifted_action[:, :-1, :-1] = one_hot_action

        # his_a[:, :, :, :] = torch.roll(his_a[:, :, :, :], -1, dims=1)
        # his_a[:, -1, :, :-1] = one_hot_action
        # his_a[:, -1, :, -1] = 0

        padd_agent = torch.tensor([8]*his_a.size(0)).reshape(his_a.size(0), 1).to(device)
        his_a = torch.cat((his_a, padd_agent), dim=-1) # [batch*4,17]
        one_hot_his_a = F.one_hot(his_a, num_classes=self.action_space.n + 1)
        his_a = torch.tensor(one_hot_his_a, dtype=torch.float32).reshape(self.batch_size, -1, one_hot_his_a.shape[-2], one_hot_his_a.shape[-1])
        out = self.model(observation, his_a, edge_nei, edge_pre, nei_pos, his_pos_matrix, node_BFS_, train)
        return out

    def get_action(self, ob, his_a, his_adj, test=False):
        """
        input are np.array here
        # TODO: support irregular input in the future
        :param ob: [agents, ob_length] -> [batch, agents, ob_length]
        :param phase: [agents] -> [batch, agents]
        :param test: boolean, exploit while training and determined while testing
        :return: [batch, agents] -> action taken by environment
        """
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
        # load onto tensor

        batch_list = []
        batch_list_p = []
        actions = []
        his_a = []
        his_adj = []
        rewards = []
        for item in samples:
            dp = item[1]
            state = torch.tensor(dp[0], dtype=torch.float32)
            # a_pre = torch.tensor(dp[6], dtype=torch.float32)
            # a_pre = torch.tensor(np.tile(dp[6], (len(dp[0]), 1)), dtype=torch.float32)
            batch_list.append(Data(x=state, edge_index=self.edge_idx))

            state_p = torch.tensor(dp[4], dtype=torch.float32)
            # a_pre = torch.tensor(dp[6], dtype=torch.float32)
            # a_pre_p = torch.tensor(np.tile(dp[7], (len(dp[4]), 1)), dtype=torch.float32)
            batch_list_p.append(Data(x=state_p, edge_index=self.edge_idx))
            rewards.append(dp[3])
            actions.append(dp[2])
            his_a.append(dp[6])
            his_adj.append(dp[7])
        batch_t = Batch.from_data_list(batch_list)
        batch_tp = Batch.from_data_list(batch_list_p)  # (agents*batch_size,obs_dim+agent_dim)
        # TODO reshape slow warning
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        # try:
        #     a = torch.tensor(np.array(actions), dtype=torch.long)
        # except:
        #     actions = torch.tensor(np.array(actions), dtype=torch.long)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        his_a = torch.tensor(np.array(his_a), dtype=torch.long)
        his_adj = torch.tensor(np.array(his_adj), dtype=torch.long)
        # 虽然sub_agents==1,但是依然是集中训练，所以，将获得所有智能体的信息，信息拉平
        # if self.sub_agents > 1:
        #     rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        #     actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here
        rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        actions = actions.view(actions.shape[0] * actions.shape[1])
        his_a = his_a.view(his_a.shape[0] * his_a.shape[1],  his_a.shape[2])
        # his_adj = his_adj.view(his_adj.shape[0] * his_adj.shape[1],  his_adj.shape[2])
        # rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        # actions = torch.tensor(np.array(actions), dtype=torch.long)
        # actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here

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
        # loss = self.criterion(self.model(x=b_t.x, edge_index=b_t.edge_index, train=True), target_f)

        deltas = target_f - self.parallel_act(b_t.x.to(device), actions.to(device), his_a.to(device), his_adj.to(device), train=True).reshape(-1, self.action_space.n)
        # deltas = target_f - self.model(x=b_t.x.to(device), edge_index=edge, train=True)

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
                                  'model', f'{e}_{0}.pt')
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_name, map_location='cpu'))
        self.target_model = self._build_model()
        self.target_model.load_state_dict(torch.load(model_name, map_location='cpu'))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        path = os.path.join(path, 'pos2')
        # path = os.path.join(path, '2')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)


# class PositionalEncoding(nn.Module):
#     def __init__(self, max_len, d_model=128, dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)  ## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
#         pe[:, 1::2] = torch.cos(position * div_term)  ##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
#         ## 上面代码获取之后得到的pe:[max_len*d_model]
#
#         ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
#         pe = pe.unsqueeze(0).transpose(0, 1)
#
#         self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以
#
#     def forward(self, x):
#         """
#         x: [seq_len, batch_size, d_model]
#         """
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

def softmax_mod(input, dim=None):
    if dim is None:
        dim = input.dim() - 1
        # 将输入张量减去每行的最大值，以防止指数溢出
    input_max, _ = torch.max(input, dim=dim, keepdim=True)
    input_exp = torch.exp(input)
    # 计算指定维度上的softmax分母，即每行元素的指数值之和
    denominator = torch.sum(input_exp, dim=dim, keepdim=True)
    # 执行softmax计算，除以分母得到最终结果
    output = input_exp / (denominator + 1)
    return output
# Define the softmax_one function with added one in the denominator , which helps to reduce
#the negative impact impact of tiny values in the softmax function and improves numerical stability
def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))



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
        # self.positional_encoding_nei = positional_encoding(6, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量
        self.positional_encoding_pre = positional_encoding(5, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量
        # self.positional_encoding = PositionalEncoding(4, self.n_embd)

    def forward(self, q, k, adjs, adjs_pos=None):
        query, key = q, k
        n_agent = q.size(1)
        batch_size = q.size(0)
        # Get the dimension of key vectors (needed for scaling the dot product of Q and k)
        d_k = k.size(-1)

        # create a 0 vector with same size as one quer vector in Q
        # zero_vector = torch.zeros(batch_size, 1, d_k).to(q.device)
        # query = torch.cat((zero_vector, q), dim=1)
        # key = torch.cat((zero_vector, k), dim=1)

        # hi*Wt
        agent_repr = torch.unsqueeze(query, dim=2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2)  # [?, 16, 1, 1, 32]
        shape1 = agent_repr_head.shape
        # print(shape1)

        # hj*Ws
        neighbor_repr = torch.unsqueeze(key, dim=1)
        neighbor_repr = neighbor_repr.repeat(1, n_agent, 1, 1)  # [?, 16, 16, 32]
        # shape2 = neighbor_repr.shape
        # print(shape2)
        # if adjs != None:
        # adjs = adjs.repeat(batch_size, 1, 1)
        # adjs = torch.reshape(adjs, (-1, n_agent, adjs.size(1), n_agent))
        # neighbor_repr = torch.matmul(adjs, neighbor_repr)  # [?, 16, 5, 32]

        if adjs_pos != None:

            flag = 'nei' if adjs_pos.size(-1) == 6 else 'pre'
            adjs_pos = adjs_pos.repeat(batch_size, 1, 1)
            adjs_pos = torch.reshape(adjs_pos, (-1, n_agent, adjs_pos.size(-2), 1, adjs_pos.size(-1))) #[256,16,32,1,6]
            # pos_encoding = positional_encoding(neighbor_repr.size(-2), neighbor_repr.size(-1))
            positional_encoding = self.positional_encoding_nei if flag == 'nei' else self.positional_encoding_pre
            pos_encoding = positional_encoding.repeat(adjs_pos.size(0), adjs_pos.size(1), adjs_pos.size(2), 1, 1)
            pos_encoding = (torch.matmul(adjs_pos, pos_encoding)).squeeze()
            # pos_encoding = F.pad(pos_encoding, (0, 0, 1, 0))
            neighbor_repr = neighbor_repr + pos_encoding
        shape3 = neighbor_repr.shape
        # adjs_pos.unsqueeze(-1).repeat(1, 1, 1, neighbor_repr.size(-1))
        # print(shape3)

        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]
        # print(neighbor_repr_head.shape)

        # att = torch.exp(torch.matmul(agent_repr_head, neighbor_repr_head))
        # att = att / (1 + torch.sum(att))
        att = torch.matmul(agent_repr_head, neighbor_repr_head) / math.sqrt(d_k)
        mask = adjs
        # mask = F.pad(mask, (1, 0))
        mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(att.size(0), 1, 1, 1, 1)
        att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)
        # att = softmax_one(att, dim=-1)  # [?, 16, 1, 1, 5]

        # print("att:",att.shape)
        att_record = torch.squeeze(att, dim=2)

        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]
        # print(neighbor_hidden_repr_head.shape)

        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        # print(out.shape)
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
        # self.positional_encoding_pre = positional_encoding(16, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量
        # self.positional_encoding = PositionalEncoding(4, self.n_embd)

    def forward(self, q, k, adjs_pre, adjs_pos=None):
        query, key = q, k
        n_agent = q.size(1)
        batch_size = q.size(0)
        # Get the dimension of key vectors (needed for scaling the dot product of Q and k)
        d_k = k.size(-1)

        # create a 0 vector with same size as one quer vector in Q
        # zero_vector = torch.zeros(batch_size, 1, d_k).to(q.device)
        # query = torch.cat((zero_vector, q), dim=1)
        # key = torch.cat((zero_vector, k), dim=1)

        # hi*Wt
        agent_repr = torch.unsqueeze(query, dim=2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2)  # [?, 16, 1, 1, 32]
        shape1 = agent_repr_head.shape
        # print(shape1)

        # hj*Ws
        # neighbor_repr = key.reshape(key.size(0), -1, key.size(-1)) # (batch_size, 4[his]*17[n_agent+1], 128)
        neighbor_repr = torch.unsqueeze(key, dim=1)
        neighbor_repr = neighbor_repr.repeat(1, n_agent, 1, 1, 1)  # [?, 16, 16, 32]
        # shape2 = neighbor_repr.shape
        # print(shape2)
        # if adjs != None:
        # adjs = adjs.repeat(batch_size, 1, 1)
        # adjs = torch.reshape(adjs, (-1, n_agent, adjs.size(1), n_agent))
        # neighbor_repr = torch.matmul(adjs, neighbor_repr)  # [?, 16, 5, 32]

        if adjs_pos != None:

            # flag = 'nei' if adjs_pos.size(-1) == 6 else 'pre'
            adjs_pos = adjs_pos.reshape(adjs_pos.shape[0], -1, n_agent+1).unsqueeze(0)
            adjs_pos = adjs_pos.repeat(batch_size, 1, 1, 1).unsqueeze(-2).type(torch.float32)
            # adjs_pos = torch.astype(torch.float32)
            # adjs_pos = torch.reshape(adjs_pos, (-1, n_agent, adjs_pos.size(-2), 1, adjs_pos.size(-1))) #[256,16,32,1,6]
            # pos_encoding = positional_encoding(neighbor_repr.size(-2), neighbor_repr.size(-1))
            # positional_encoding = self.positional_encoding_nei if flag == 'nei' else self.positional_encoding_pre
            # pos_encoding = positional_encoding.repeat(adjs_pos.size(0), adjs_pos.size(1), adjs_pos.size(2), 1, 1)
            # pos_encoding = (torch.matmul(adjs_pos, pos_encoding)).squeeze()
            # pos_encoding = F.pad(pos_encoding, (0, 0, 1, 0))
            try:
                neighbor_repr = torch.matmul(adjs_pos, neighbor_repr)
            except:
                aaaa = 0
            neighbor_repr = self.his_action_encoder(neighbor_repr.squeeze(-2))
            positional_encoding = self.positional_encoding.unsqueeze(0).unsqueeze(1).repeat(neighbor_repr.size(0),neighbor_repr.size(1),1,1).to(device)
            neighbor_repr = neighbor_repr + positional_encoding
        shape3 = neighbor_repr.shape
        # print(shape3)

        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]
        # print(neighbor_repr_head.shape)

        # att = torch.exp(torch.matmul(agent_repr_head, neighbor_repr_head))
        # att = att / (1 + torch.sum(att))
        att = torch.matmul(agent_repr_head, neighbor_repr_head) / math.sqrt(d_k)
        # mask = adjs_pre
        # # mask = F.pad(mask, (1, 0))
        # mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(att.size(0), 1, 1, 1, 4)
        # att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)
        # att = softmax_one(att, dim=-1)  # [?, 16, 1, 1, 5]

        # print("att:",att.shape)
        att_record = torch.squeeze(att, dim=2)

        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]
        # print(neighbor_hidden_repr_head.shape)

        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        # print(out.shape)
        out_concat = transpose_output(out, self.nv)
        out_concat = torch.squeeze(out_concat, dim=2)  # [?, 16, 32]
        out_concat = F.relu(self.fcout(out_concat))

        return out_concat
class MultiHeadAttModel_his(nn.Module):
    def __init__(self, n_embd, dv, nv, n_agent):
        super(MultiHeadAttModel_his, self).__init__()
        self.n_embd = n_embd
        self.dv = dv
        self.nv = nv
        self.fcv = init_(nn.Linear(n_embd, dv * nv))
        self.fck = init_(nn.Linear(n_embd, dv * nv))
        self.fcq = init_(nn.Linear(n_embd, dv * nv))
        self.fcout = init_(nn.Linear(dv * nv, n_embd))
        self.n_agent = n_agent
        self.positional_encoding = positional_encoding(4, self.n_embd)  # 第一项为需要mask的向量，其他为上下左右自己向量
        # self.positional_encoding = PositionalEncoding(4, self.n_embd)

    def forward(self, q, k, adjs):
        query, key = q, k
        n_agent = q.size(1)
        batch_size = q.size(0)
        # Get the dimension of key vectors (needed for scaling the dot product of Q and k)
        d_k = k.size(-1)
        adjs = adjs.reshape(-1, 4)

        # create a 0 vector with same size as one quer vector in Q
        # zero_vector = torch.zeros(batch_size, 1, d_k).to(q.device)
        # query = torch.cat((zero_vector, q), dim=1)
        # key = torch.cat((zero_vector, k), dim=1)

        # hi*Wt
        agent_repr = torch.unsqueeze(query, dim=2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2)  # [?, 16, 1, 1, 32]
        shape1 = agent_repr_head.shape
        # print(shape1)

        # hj*Ws
        neighbor_repr = key.permute(0, 2, 1, 3)
        positional_encoding = self.positional_encoding.repeat(batch_size, n_agent, 1, 1)
        neighbor_repr = neighbor_repr + positional_encoding
        # neighbor_repr = torch.unsqueeze(key, dim=1)
        # neighbor_repr = neighbor_repr.repeat(1, n_agent, 1, 1, 1)  # [?, 16, 16, 32]
        # shape2 = neighbor_repr.shape
        # print(shape2)
        # if adjs != None:
        # adjs = adjs.repeat(batch_size, 1, 1)
        # adjs = torch.reshape(adjs, (-1, n_agent, adjs.size(1), n_agent))
        # neighbor_repr = torch.matmul(adjs, neighbor_repr)  # [?, 16, 5, 32]

            # adjs_pos = torch.reshape(adjs_pos,(-1, n_agent, adjs_pos.size(-2), 1, adjs_pos.size(-1)))  # [256,16,32,1,6]
            # pos_encoding = positional_encoding(neighbor_repr.size(-2), neighbor_repr.size(-1))
        # pos_encoding = self.positional_encoding.repeat(n_agent, 1, 1)
            # pos_encoding = F.pad(pos_encoding, (0, 0, 1, 0))
        # neighbor_repr = neighbor_repr + pos_encoding
        shape3 = neighbor_repr.shape
        # print(shape3)

        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        # neighbor_repr_head = transpose_qkv(neighbor_repr_head.reshape(neighbor_repr_head.shape[0],neighbor_repr_head.shape[1],-1,neighbor_repr_head.shape[-1]), self.nv)
        # neighbor_repr_head = neighbor_repr_head.reshape(neighbor_repr_head.shape[0],neighbor_repr_head.shape[1], 4, -1,neighbor_repr_head.shape[-1])
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]
        # print(neighbor_repr_head.shape)

        # att = torch.exp(torch.matmul(agent_repr_head, neighbor_repr_head))
        # att = att / (1 + torch.sum(att))
        att = torch.matmul(agent_repr_head, neighbor_repr_head) / math.sqrt(d_k)
        mask = adjs
        # mask = F.pad(mask, (1, 0))
        mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(int(att.size(0)/mask.size(0)), att.size(1), 1, 1, 1)
        mask[:, -1, :, :, :] = torch.tensor([0, 0, 0, 1])
        att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)
        # att = softmax_one(att, dim=-1)  # [?, 16, 1, 1, 5]

        # print("att:",att.shape)
        att_record = torch.squeeze(att, dim=2)

        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]
        # print(neighbor_hidden_repr_head.shape)

        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        # print(out.shape)
        out_concat = transpose_output(out, self.nv)
        out_concat = torch.squeeze(out_concat, dim=2)  # [?, 16, 32]
        out_concat = F.relu(self.fcout(out_concat))

        return out_concat

# def add_pos(neighbor_repr):
#     pos_encoding = positional_encoding(neighbor_repr.size(-2), neighbor_repr.size(-1))
#     pos_encoding = pos_encoding.repeat(neighbor_repr.size(0), neighbor_repr.size(1), 1, 1)
#     pos = (torch.ones((pos_encoding.shape[0], pos_encoding.shape[1], pos_encoding.shape[2], pos_encoding.shape[3])) * 0)
#     pos_fill = (torch.ones((pos_encoding.shape[0], pos_encoding.shape[1], pos_encoding.shape[-1])) * 0)
#     graph = Registry.mapping['world_mapping']['graph_setting'].graph
#     node_precursor = graph['node_precursor']
#     for i, nodes in enumerate(node_precursor):
#         j = 0
#         a = [0, 1, 2, 3]
#         for node in nodes:
#             if node == -1:
#                 continue
#             elif node == i-1:
#                 pos[:, i, j, :] = pos_encoding[:, i, 0, :]
#                 a.remove(0)
#                 j += 1
#             elif node == i+1:
#                 pos[:, i, j, :] = pos_encoding[:, i, 1, :]
#                 a.remove(1)
#                 j += 1
#             elif node < i:
#                 pos[:, i, j, :] = pos_encoding[:, i, 2, :]
#                 a.remove(2)
#                 j += 1
#             else:
#                 pos[:, i, j, :] = pos_encoding[:, i, 3, :]
#                 a.remove(3)
#                 j += 1
#         for f in a:
#             pos[:, i, j, :] = pos_encoding[:, i, f, :]
#             j += 1
#         # while j < 4:
#         #     pos[:, i, j, :] = pos_fill
#         #     j += 1
#     neighbor_repr = neighbor_repr + pos
#
#     return neighbor_repr

# def positional_encoding(max_seq_len, d_model):
#     # 初始化位置编码矩阵
#     position = torch.arange(0, max_seq_len).unsqueeze(1).float()
#     div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
#     pe = torch.zeros(max_seq_len, d_model)
#     # 使用正弦和余弦函数计算位置编码
#     pe[:, 0::2] = torch.sin(position * div_term)
#     pe[:, 1::2] = torch.cos(position * div_term)
#
#     padd = torch.zeros(d_model)
#     # 增加一个维度，以便将位置编码添加到词嵌入中
#     # pe = pe.unsqueeze(0)
#     return pe
class DecodeBlock_obs(nn.Module):

    def __init__(self, n_embd, dv, n_head, n_agent):
        super(DecodeBlock_obs, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttModel(n_embd, dv, n_head, n_agent)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU()
            # init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, adjs_nei, nei_pos):
        x = x + self.attn(x, x, adjs_nei)
        # x = self.mlp(x)
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
        # x = self.ln1(x + self.attn1(x, x, adjs_mates))
        x = self.attn1(q=rep_enc, k=his_action_embeddings, adjs_pre=adjs_pre, adjs_pos=his_pos_matrix)

        # x = self.attn2(q=rep_enc, k=his_action_embeddings, adjs=adjs_pre, adjs_pos=adjs_pos)
        # x = self.mlp(x)
        x = rep_enc + x
        # x = rep_enc + self.attn2(q=rep_enc, k=x, adjs=adjs_pre)
        # x = self.ln3(x + self.mlp(x))
        return x


class Decoder(nn.Module):

    # def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, action_type='Discrete', dec_actor=False, share_actor=False):
    def __init__(self, input_dim, output_dim, n_node, his_length, **kwargs):
        super(Decoder, self).__init__()
        self.model_dict = kwargs
        self.action_dim = gym.spaces.Discrete(output_dim).n
        self.his_length = his_length
        # self.n_embd = n_embd
        # self.dec_actor = dec_actor
        # self.share_actor = share_actor
        # self.action_type = action_type
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
        # self.head = nn.Sequential(init_(nn.Linear(self.n_embd, self.n_embd), activate=True), nn.GELU(),
        #                           init_(nn.Linear(self.n_embd, self.action_dim)))
        self.head = nn.Sequential(init_(nn.Linear(self.n_embd, self.action_dim), activate=True))
        self.head_l = nn.Sequential(init_(nn.Linear(self.n_embd, self.action_dim), activate=True))
        self.head_f = nn.Sequential(init_(nn.Linear(self.n_embd, self.action_dim), activate=True))
        self.val_head = nn.Sequential(init_(nn.Linear(self.n_embd, self.n_embd), activate=True), nn.GELU(),
                                      nn.LayerNorm(self.n_embd),
                                      init_(nn.Linear(self.n_embd, 1)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def _forward(self, obs, his_action, adjs_nei, adjs_pre, nei_pos, his_pos_matrix, node_BFS):
        obs = torch.reshape(obs, (-1, self.n_agent, self.obs_dim))
        # action = torch.reshape(action, (-1, self.n_agent+1, self.action_dim+1))
        # out = torch.zeros(action.size(0), self.n_agent, self.action_dim)

        obs_emb = self.obs_encoder(obs)
        positional_encoding = torch.matmul(node_BFS.type(torch.float32), self.positional_encoding.to(device))
        positional_encoding = positional_encoding.repeat(obs_emb.size(0), 1, 1)
        obs_emb_pos = obs_emb + positional_encoding
        for block in self.blocks_obs:
            obs_embeddings = block.forward(obs_emb_pos, adjs_nei, nei_pos)
        # positional_encoding = self.positional_encoding.repeat(obs_embeddings.size(0), 1, 1)
        # obs_embeddings = obs_embeddings + positional_encoding
        # obs_embeddings_leader = obs_embeddings[:, leader, :]
        # obs_embeddings_follow = obs_embeddings[:, follow, :]
        # action_embeddings = self.action_encoder(action)
        # his_action_embeddings = self.action_encoder(his_action)
        # x = action_embeddings
        # x = self.ln(x)
        # x = x + positional_encoding
        # x = torch.cat((x, obs_embeddings[:, leader, :]), dim=-2)
        for block in self.blocks:
            x = block(obs_embeddings, his_action, adjs_pre, his_pos_matrix)
        # out_l = self.head(obs_embeddings_leader)
        out = self.head(x)
        # out[:, leader, :] = out_l
        # out[:, follow, :] = out_f
        # out = self.head(out)
        # val = self.val_head(x)

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

    # padd = torch.zeros(d_model).unsqueeze(0)
    # pe = torch.cat((padd, pe), dim=0)
    # 增加一个维度，以便将位置编码添加到词嵌入中
    # pe = pe.unsqueeze(0)
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


class ColightNet(nn.Module):
    # TODO input_dim要加上前驱的动作维度 输入的时候得改 input_dim!=obs_dim 而是 ==obs_dim+agents_num
    def __init__(self, input_dim, output_dim, n_node, **kwargs):
        super(ColightNet, self).__init__()
        self.model_dict = kwargs
        self.action_space = gym.spaces.Discrete(output_dim)
        self.features = input_dim
        self.module_list = nn.ModuleList()
        self.module_list_att = nn.ModuleList()
        # self.embedding_MLP = Embedding_MLP(self.features, layers=self.model_dict.get('NODE_EMB_DIM'))
        self.embedding_MLP = Embedding_MLP(self.features, layers=self.model_dict.get('NODE_EMB_DIM'))
        for i in range(self.model_dict.get('N_LAYERS')):
            block = MultiHeadAttModel(n_embd=self.model_dict.get('INPUT_DIM_1')[i],
                                      dv=self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                      d_out=self.model_dict.get('OUTPUT_DIM_1')[i],
                                      nv=self.model_dict.get('NUM_HEADS')[i],
                                      n_node=n_node,
                                      suffix=i)
            self.module_list.append(block)
        self.block_att_pre = MultiHeadAttention(d=8,
                                                dv=self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[0],
                                                d_out=8,
                                                nv=self.model_dict.get('NUM_HEADS')[0],
                                                n_node=n_node,
                                                suffix=0)
        for i in range(self.model_dict.get('N_LAYERS')):
            block_att = AdditiveAttention(d1=8,
                                          d2=8,
                                          dv=self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                          d_out=self.model_dict.get('OUTPUT_DIM_1')[i],
                                          suffix=i, dropout=0.1)
            self.module_list_att.append(block_att)
        self.embedding_MLP_l = Embedding_MLP_l(block.d_out, layers=self.model_dict.get('NODE_EMB_L_DIM'))
        self.embedding_MLP_f = Embedding_MLP_f(block_att.dv, layers=self.model_dict.get('NODE_EMB_L_DIM'))
        self.embedding_MLP_a = Embedding_MLP_a(16, layers=self.model_dict.get('NODE_EMB_L_DIM'))

        output_dict = OrderedDict()
        output_dict_f = OrderedDict()
        # output_dict_a = OrderedDict()

        # if len(self.model_dict['OUTPUT_LAYERS']) != 0:
        #     # TODO: dubug this branch
        #     for l_idx, l_size in enumerate(self.model_dict['OUTPUT_LAYERS']):
        #         name = f'output_{l_idx}'
        #         if l_idx == 0:
        #             h = nn.Linear(block.d_out, l_size)
        #         else:
        #             h = nn.Linear(self.model_dict.get('OUTPUT_LAYERS')[l_idx - 1], l_size)
        #         output_dict.update({name: h})
        #         name = f'relu_{l_idx}'
        #         output_dict.update({name: nn.ReLU})
        #     out = nn.Linear(self.model_dict['OUTPUT_LAYERS'][-1], self.action_space.n)
        # else:
        #     out = nn.Linear(block.d_out, self.action_space.n)
        out = nn.Linear(block.d_out, self.action_space.n)
        name = f'output'
        output_dict.update({name: out})
        out_f = nn.Linear(self.block_att_pre.d_out * 2, self.action_space.n)
        name = f'output_f'
        output_dict_f.update({name: out_f})
        # out_a = nn.Linear(self.action_space.n, block.d_out)
        # name = f'output_a'
        # output_dict_a.update({name: out_a})
        self.output_layer = nn.Sequential(output_dict)
        # self.output_layer_l = nn.Sequential(output_dict)
        self.output_layer_f = nn.Sequential(output_dict_f)
        # self.output_layer_a = nn.Sequential(output_dict_a)

    # x.size={batch*intersections,feature}
    def forward(self, x, edge_index, train=True):
        # 循环batch*intersections次
        h = self.embedding_MLP.forward(x, train)
        # TODO: implement att
        for mdl in self.module_list:
            h = mdl.forward(h, edge_index, train)
        h = self.embedding_MLP_l.forward(h, train)
        if train:
            out = self.compu_out(h, train)
        else:
            with torch.no_grad():
                out = self.compu_out(h, train)
        return out

    def compu_out(self, h, train):
        graph = Registry.mapping['world_mapping']['graph_setting'].graph
        precursor_matrix = torch.tensor(graph['precursor_matrix']).to(device)
        node_BFS = graph['node_BFS']
        temp_h = h.clone()
        temp_h = torch.reshape(temp_h, (-1, len(node_BFS), temp_h.shape[-1]))
        out = torch.zeros((temp_h.shape[0], temp_h.shape[1], temp_h.shape[2]))
        for i, node in enumerate(node_BFS):
            f_pre = self.block_att_pre.forward(temp_h, precursor_matrix, node, train)
            f_pre_temp = f_pre.detach()
            f = torch.cat([temp_h[:, node, :], f_pre_temp], dim=1)
            f = self.output_layer_f(f)
            out[:, node, :] = f
            temp_h[:, node, :] = F.softmax(f, dim=1)
        return out.view(-1, out.shape[-1])

    # a是选出的动作，a_是前驱智能体,这一层智能体的编号idx
    def comp_a2ve(self, hi, node_precursor, nodes):
        vector_list = (torch.ones((int(hi.shape[0] * len(nodes)), 5, 16)) * -1).to(device)
        pos_enc = self.positional_encoding(5, 8)
        pos_enc = pos_enc.repeat(hi.shape[0], 1, 1)
        i = 0
        for id in nodes:
            h_pre = (torch.ones((hi.shape[0], 5, 8)) * 0)
            for idx in node_precursor[id]:
                if idx == id - 1:  # left
                    h_pre[:, 0, :] = hi[:, idx, :]
                elif idx == id + 1:  # right
                    h_pre[:, 1, :] = hi[:, idx, :]
                elif idx < id:  # up
                    h_pre[:, 2, :] = hi[:, idx, :]
                else:  # down
                    h_pre[:, 3, :] = hi[:, idx, :]
            h_pre[:, 4, :] = hi[:, id, :]
            vector = torch.cat([h_pre, pos_enc], dim=2)

            vector_list[i::len(nodes), :, :] = vector.to(device)
            i += 1
        return vector_list

    def positional_encoding(self, max_seq_len, d_model):
        # 初始化位置编码矩阵
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)

        # 使用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个维度，以便将位置编码添加到词嵌入中
        # pe = pe.unsqueeze(0)
        return pe

    def one_hot_encode(self, arr):
        num_classes = 8
        arr_tensor = arr
        mask = arr_tensor == -1
        arr_tensor[mask] = 0

        encoded_arr = F.one_hot(arr_tensor, num_classes=num_classes)
        encoded_arr[mask.unsqueeze(-1).repeat(1, 1, 8)] = 0

        return encoded_arr


class Embedding_MLP(nn.Module):
    def __init__(self, in_size, layers):
        super(Embedding_MLP, self).__init__()
        constructor_dict = OrderedDict()
        for l_idx, l_size in enumerate(layers):
            name = f"node_embedding_{l_idx}"
            if l_idx == 0:
                h = nn.Linear(in_size, l_size)
                constructor_dict.update({name: h})
            else:
                h = nn.Linear(layers[l_idx - 1], l_size)
                constructor_dict.update({name: h})
            name = f"n_relu_{l_idx}"
            constructor_dict.update({name: nn.ReLU()})
        self.embedding_node = nn.Sequential(constructor_dict)

    def _forward(self, x):
        x = self.embedding_node(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class Embedding_MLP_l(nn.Module):
    def __init__(self, in_size, layers):
        super(Embedding_MLP_l, self).__init__()
        constructor_dict = OrderedDict()
        for l_idx, l_size in enumerate(layers):
            name = f"node_embedding_{l_idx}"
            if l_idx == 0:
                h = nn.Linear(in_size, l_size)
                constructor_dict.update({name: h})
                name = f"n_softmax_{l_idx}"
                constructor_dict.update({name: nn.Softmax()})
        self.embedding_node = nn.Sequential(constructor_dict)

    def _forward(self, x):
        x = self.embedding_node(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class Embedding_MLP_f(nn.Module):
    def __init__(self, in_size, layers):
        super(Embedding_MLP_f, self).__init__()
        constructor_dict = OrderedDict()
        for l_idx, l_size in enumerate(layers):
            name = f"node_embedding_{l_idx}"
            if l_idx == 0:
                h = nn.Linear(in_size, l_size)
                constructor_dict.update({name: h})
                # name = f"n_softmax_{l_idx}"
                # constructor_dict.update({name: nn.Softmax()})
        self.embedding_node = nn.Sequential(constructor_dict)

    def _forward(self, x):
        x = self.embedding_node(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class Embedding_MLP_a(nn.Module):
    def __init__(self, in_size, layers):
        super(Embedding_MLP_a, self).__init__()
        constructor_dict = OrderedDict()
        for l_idx, l_size in enumerate(layers):
            name = f"node_embedding_{l_idx}"
            if l_idx == 0:
                h = nn.Linear(in_size, l_size)
                constructor_dict.update({name: h})
                name = f"n_tanh_{l_idx}"
                constructor_dict.update({name: nn.Tanh()})
        self.embedding_node = nn.Sequential(constructor_dict)

    def _forward(self, x):
        x = self.embedding_node(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 3, 1, 2, 4)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3], X.shape[4])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2], X.shape[3])
    X = X.permute(0, 2, 3, 1, 4)
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d, dv, d_out, nv, n_node, suffix):
        super(MultiHeadAttention, self).__init__()
        self.d = d
        self.dv = dv
        self.d_out = d_out
        self.nv = nv
        self.suffix = suffix
        self.fcv = nn.Linear(d * 2, dv * nv)
        self.fck = nn.Linear(d * 2, dv * nv)  # 乘以2 因为加了位置编码
        self.fcq = nn.Linear(d, dv * nv)
        self.fcout = nn.Linear(dv * nv, d_out)
        self.n_node = n_node

    def comp_a2ve(self, node):
        graph = Registry.mapping['world_mapping']['graph_setting'].graph
        node_precursor = graph['node_precursor']
        num = 4
        pos_enc = self.positional_encoding(num, 8)
        padd = torch.tensor([0] * 8, dtype=torch.float32)
        h_enc = []
        for idx in node_precursor[node]:
            if idx == -1:
                break
            if idx == node - 1:  # left
                h_enc.append(pos_enc[0, :])
            elif idx == node + 1:  # right
                h_enc.append(pos_enc[1, :])
            elif idx < node:  # up
                h_enc.append(pos_enc[2, :])
            else:  # down
                h_enc.append(pos_enc[3, :])
        while len(h_enc) < num:
            h_enc.append(padd)
        return torch.stack(h_enc)

    def positional_encoding(self, max_seq_len, d_model):
        # 初始化位置编码矩阵
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)

        # 使用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def _forward(self, obs, adj, node):
        # obs = torch.reshape(obs, (-1, self.n_node, self.d_out))
        adjs = adj.repeat(obs.size(0), 1, 1)
        adjs = torch.reshape(adjs, (-1, self.n_node, adjs.size(1), self.n_node))
        adjs = adjs[:, node, :, :]
        # hi*Wt
        agent_repr = obs[:, node, :]
        agent_repr = agent_repr.unsqueeze(1).unsqueeze(2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=1)  # [?, 16, 1, 1, 32]
        shape1 = agent_repr_head.shape
        # print(shape1)

        # hj*Ws
        # neighbor_repr = torch.unsqueeze(obs, dim=1)
        # neighbor_repr = neighbor_repr.repeat(1, self.n_node, 1, 1)  # [?, 16, 16, 32]
        # shape2 = neighbor_repr.shape
        neighbor_repr = obs
        # print(shape2)
        neighbor_repr = torch.matmul(adjs, neighbor_repr)  # [?, 16, 5, 32]
        neighbor_repr = neighbor_repr.unsqueeze(1)
        shape3 = neighbor_repr.shape
        # print(shape3)

        # 添加位置编码
        # mask_pos = torch.sum(adj[node, :, :], dim=-1)
        # pos_enc = self.positional_encoding(mask_pos, adjs.size(1), 8)
        pos_enc = self.comp_a2ve(node)
        pos_enc = (pos_enc.repeat(neighbor_repr.shape[0], 1, 1)).unsqueeze(1)
        neighbor_repr = torch.cat([neighbor_repr, pos_enc], dim=-1)

        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]
        # print(neighbor_repr_head.shape)

        # att = torch.exp(torch.matmul(agent_repr_head, neighbor_repr_head))
        # att = att / (1 + torch.sum(att))
        att = F.softmax(torch.matmul(agent_repr_head, neighbor_repr_head), dim=4)  # [?, 16, 1, 1, 5]
        # print("att:",att.shape)
        att_record = torch.squeeze(att, dim=2)

        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]
        # print(neighbor_hidden_repr_head.shape)

        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        # print(out.shape)
        out_concat = transpose_output(out, self.nv)
        out_concat = torch.squeeze(out_concat)  # [?, 16, 32]
        out_concat = F.relu(self.fcout(out_concat))

        return out_concat.reshape(-1, out_concat.shape[-1])

    def forward(self, x, edge_index, node, train=True):
        if train:
            return self._forward(x, edge_index, node)
        else:
            with torch.no_grad():
                return self._forward(x, edge_index, node)


# class MultiHeadAttention(nn.Module):
#     def __init__(self, d1, d2, dv, d_out, nv, suffix):
#         super(MultiHeadAttention, self).__init__()
#         self.d1 = d1
#         self.d2 = d2
#         self.dv = dv
#         self.d_out = d_out
#         self.nv = nv
#         self.suffix = suffix
#         self.fcv = nn.Linear(d2, dv * nv)
#         self.fck = nn.Linear(d2, dv * nv)
#         self.fcq = nn.Linear(d1, dv * nv)
#         self.fcout = nn.Linear(dv * nv, d_out)
#
#
#     def _forward(self, x, y, nodes):
#         x = torch.reshape(x, (-1, len(nodes), self.d1)) #[?,6,8]
#         y = torch.reshape(y, (-1, len(nodes), 5, self.d2)) #[?,6,4,16]
#         # hi*Wt
#         agent_repr = torch.unsqueeze(x, dim=2)
#         agent_repr_head = F.relu(self.fcq(agent_repr))
#         agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
#         agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2)  # [?, 16, 1, 1, 32]
#         shape1 = agent_repr_head.shape
#         # print(shape1)
#
#         # hj*Ws
#         neighbor_repr = y
#         shape2 = neighbor_repr.shape
#         # print(shape2)
#         # neighbor_repr = torch.matmul(adjs, neighbor_repr)  # [?, 16, 5, 32]
#         # shape3 = neighbor_repr.shape
#         # print(shape3)
#         neighbor_repr_head = F.relu(self.fck(neighbor_repr))
#         neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
#         neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]
#         # print(neighbor_repr_head.shape)
#
#         att = F.softmax(torch.matmul(agent_repr_head, neighbor_repr_head),dim=4)  # [?, 16, 1, 1, 5]
#         # print("att:",att.shape)
#         # att_record = torch.squeeze(att, dim=2)
#
#         neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
#         neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
#         neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]
#         # print(neighbor_hidden_repr_head.shape)
#
#         out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
#         # print(out.shape)
#         out_concat = transpose_output(out, self.nv)
#         out_concat = torch.squeeze(out_concat, dim=2)  # [?, 16, 32]
#         out_concat = F.relu(self.fcout(out_concat))
#
#         return out_concat.reshape(-1, out_concat.shape[2])
#
#     def forward(self, x, y, nodes, train=True):
#         if train:
#             return self._forward(x, y, nodes)
#         else:
#             with torch.no_grad():
#                 return self._forward(x, y, nodes)
#
#     # def forward(self, x, y, train=True):
#     #     if train:
#     #         return self._forward(x, y)
#     #     else:
#     #         with torch.no_grad():
#     #             return self._forward(x, y)
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, d1, d2, dv, d_out, suffix, dropout):
        super(AdditiveAttention, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.dv = dv
        self.d_out = d_out
        self.suffix = suffix
        self.fcq = nn.Linear(d1, dv, bias=False)
        self.fck = nn.Linear(d2, dv, bias=False)
        self.fcv = nn.Linear(d2, dv, bias=False)
        self.fcu = nn.Linear(dv, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.Softmax = nn.Softmax(dim=1)
        # self.fcout = nn.Linear(dv * nv, d_out)

    def _forward(self, x, y, nodes):
        x = torch.reshape(x, (-1, len(nodes), self.d1))  # [?,6,8]
        y = torch.reshape(y, (-1, len(nodes), 5, self.d2))  # [?,6,4,16]
        queries, keys = self.fcq(x), self.fck(y)
        value = y
        # hi*Wt
        features = queries.unsqueeze(2) + keys
        features = torch.tanh(features)
        scores = self.fcu(features).squeeze(-1)
        attention = self.Softmax(scores)
        attention = torch.unsqueeze(attention, dim=2)
        attention = torch.unsqueeze(attention, dim=3)
        # attention = self.dropout(attention)
        # out = torch.mean(torch.matmul(attention.unsqueeze(2), value).unsqueeze(2), dim=2)

        value = F.relu(self.fcv(value))
        value = torch.unsqueeze(value, dim=2)
        out = torch.mean(torch.matmul(attention, value), dim=2)
        # neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        # neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]
        # # print(neighbor_hidden_repr_head.shape)
        #
        # out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        # # print(out.shape)
        # out_concat = transpose_output(out, self.nv)
        # out_concat = torch.squeeze(out_concat, dim=2)  # [?, 16, 32]
        # out_concat = F.relu(self.fcout(out_concat))

        return out.squeeze(2)

    def forward(self, x, y, nodes, train=True):
        if train:
            return self._forward(x, y, nodes)
        else:
            with torch.no_grad():
                return self._forward(x, y, nodes)
