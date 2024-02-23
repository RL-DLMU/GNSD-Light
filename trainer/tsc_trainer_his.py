import os
from queue import Queue
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from trainer.base_trainer import BaseTrainer

@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''
    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        # self.epoch = Registry.mapping['trainer_mapping']['setting'].param['epoch']
        # self.iterations = Registry.mapping['trainer_mapping']['setting'].param['iterations']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.his_length = Registry.mapping['trainer_mapping']['setting'].param['his_length']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        # self.iterations = int(self.buffer_size/self.batch_size)
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later
        
        # TODO: support other dataset in the future
        self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log') + '_DTL.log'
                                     )

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards','rewards_p','queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards','rewards_p','queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        # graph = Registry.mapping['world_mapping']['graph_setting'].graph
        # node_BFS = graph['node_BFS']
        self.agents = []
        agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, i))

        # for magd agents should share information 
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)

    # @profile
    def train(self):
        '''
        train
        Train the agent(s).

        :param: None
        :return: None
        '''
        '''写入csv'''
        name = self.agents[0].model_dict['name']
        '''追加进csv'''
        total_decision_num = 0
        flush = 0
        reward_list = []
        reward_l_list = []
        loss_list = []
        real_average_travel_time_list = []
        b = [0] * len(self.world.intersections)
        self.ag = self.agents[0]
        for e in range(self.episodes):
            # TODO: check this reset agent
            self.metric.clear()

            last_obs = self.env.reset()  # agent * [sub_agent, feature]
            his_a = deque(maxlen=self.his_length)
            his_action = deque(maxlen=self.his_length)
            his_adj = deque(maxlen=self.his_length)
            his_adj_ = deque(maxlen=self.his_length)

            for a in self.agents:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        while len(his_action) < self.his_length:
                            his_action.appendleft(np.array([8] * self.ag.sub_agents))
                            his_adj_.appendleft(0)

                        actions = []
                        for idx, ag in enumerate(self.agents):
                            try:
                                actions.append(ag.get_action(last_obs[idx], his_action, his_adj_, test=False))
                            except:
                                actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
                        actions = np.stack(actions)  # [agent, intersections]
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])

                    his_a.append(actions.flatten())
                    his_adj.append(1)
                    actions_prob = []
                    for idx, intersection in enumerate(self.world.intersections):
                        actions_prob.append(self.ag.get_action_prob(last_obs[0], last_phase[0]))

                    rewards_list = []
                    rewards_p_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, rewards_p, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                        rewards_p_list.append(np.stack(rewards_p))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    rewards_p = np.mean(rewards_p_list, axis=0)
                    self.metric.update(rewards, rewards_p)

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])

                    his_action = his_a.copy()
                    his_adj_ = his_adj.copy()
                    while len(his_action) < self.his_length:
                        his_action.appendleft(np.array([8]*self.ag.sub_agents))
                        his_adj_.appendleft(0)

                    for idx, ag in enumerate(self.agents):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], list(his_action), list(his_adj_), actions_prob[idx], rewards[idx],
                                    obs[idx], cur_phase[idx], dones[idx], f'{e}_{i // self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                    total_decision_num += 1
                    last_obs = obs
                self.learning_rate = 0
                if total_decision_num > self.learning_start and\
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    self.ag.e = e
                    cur_loss_q = np.stack([self.ag.train()])  # TODO: training

                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    self.ag.update_target_network()

                if all(dones):
                    break

            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            rewards_l, rewards_f = self.metric.rewards_each()
            self.logger.info("step:{}/{}, q_loss:{}, rewards_p:{}, rewards:{}, rewards_l:{},queue:{}, delay:{}, travel_time:{}, throughput:{}".format(i, self.steps,\
                mean_loss, self.metric.rewards_p(), self.metric.rewards(), rewards_l, self.metric.queue(), self.metric.delay(), self.metric.real_average_travel_time(), int(self.metric.throughput())))

            if e % self.save_rate == 0:
                self.ag.save_model(e=e)
            for j in range(len(self.world.intersections)):
                b[j] = self.metric.lane_queue()[0][j]
            if e % 20 == 0:
                for j in range(len(self.world.intersections)):
                    print(j, b[j])
                print(sorted(b,reverse=True))
                print("========================")
            if self.test_when_train:
                real_average_travel_time = self.train_test(e)
                rewards_l, rewards_f = self.metric.rewards_each()
                reward_list.append(self.metric.rewards())
                reward_l_list.append(rewards_l)
                real_average_travel_time_list.append(real_average_travel_time)
                loss_list.append(mean_loss)

        self.ag.save_model(e=self.episodes)




    def train_test(self, e):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        obs = self.env.reset()
        self.metric.clear()
        for a in self.agents:
            a.reset()

        i = 0
        his_a = deque(maxlen=self.his_length)
        his_action = deque(maxlen=self.his_length)
        his_adj = deque(maxlen=self.his_length)
        his_adj_ = deque(maxlen=self.his_length)
        while i < self.test_steps:
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                while len(his_action) < self.his_length:
                    his_action.appendleft(np.array([8] * self.ag.sub_agents))
                    his_adj_.appendleft(0)
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], his_action, his_adj_, test=True))

                actions = np.stack(actions)  # [agent, intersections]
                rewards_p_list = []
                rewards_list = []
                his_a.append(actions.flatten())
                his_adj.append(1)
                his_action = his_a.copy()
                his_adj_ = his_adj.copy()
                while len(his_action) < self.his_length:
                    his_action.appendleft(np.array([8] * self.ag.sub_agents))
                    his_adj_.appendleft(0)
                for _ in range(self.action_interval):
                    obs, rewards, rewards_p, dones, _ = self.env.step(actions.flatten())  # make sure action is [intersection]
                    i += 1
                    rewards_list.append(np.stack(rewards))
                    rewards_p_list.append(np.stack(rewards_p))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                rewards_p = np.mean(rewards_p_list, axis=0)
                self.metric.update(rewards, rewards_p)
            if all(dones):
                break
        rewards_l, rewards_f = self.metric.rewards_each()
        mega_l = Registry.mapping['model_mapping']['setting'].param['mega_l']
        mega_f = Registry.mapping['model_mapping']['setting'].param['mega_f']
        her = Registry.mapping['model_mapping']['setting'].param['her']
        self.logger.info("Test step:{}/{}, travel time :{}, rewards_p:{}, rewards:{}, rewards_l:{}, queue:{}, delay:{}, throughput:{}".format(\
            e, self.episodes, self.metric.real_average_travel_time(), self.metric.rewards_p(), self.metric.rewards(), rewards_l,\
            self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
        self.writeLog("TEST", e, self.metric.real_average_travel_time(),\
            mega_l, mega_f, her, self.metric.rewards(), rewards_l, self.metric.queue(), self.metric.queue_l(), self.metric.delay(), self.metric.throughput())


        return self.metric.real_average_travel_time()

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        if not drop_load:
            [ag.load_model(self.episodes) for ag in self.agents]
        for e in range(20):

            self.metric.clear()

            attention_mat_list = []
            obs = self.env.reset()
            for a in self.agents:
                a.reset()
            self.ag = self.agents[0]
            his_a = deque(maxlen=5)
            his_action = deque(maxlen=5)
            his_adj = deque(maxlen=5)
            his_adj_ = deque(maxlen=5)
            for i in range(self.test_steps):
                if i % self.action_interval == 0:
                    while len(his_action) < 5:
                        his_action.appendleft(np.array([8] * self.ag.sub_agents))
                        his_adj_.appendleft(0)

                    actions = []
                    for idx, ag in enumerate(self.agents):
                        actions.append(ag.get_action(obs[idx], his_action, his_adj_, test=True))
                    actions = np.stack(actions)  # [agent, intersections]
                    rewards_p_list = []
                    rewards_list = []
                    his_a.append(actions.flatten())
                    his_adj.append(1)
                    his_action = his_a.copy()
                    his_adj_ = his_adj.copy()
                    while len(his_action) < 5:
                        his_action.appendleft(np.array([8] * self.ag.sub_agents))
                        his_adj_.appendleft(0)
                    for _ in range(self.action_interval):
                        obs, rewards, rewards_p, dones, _ = self.env.step(actions.flatten())  # make sure action is [intersection]
                        i += 1
                        rewards_list.append(np.stack(rewards))
                        rewards_p_list.append(np.stack(rewards_p))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    rewards_p = np.mean(rewards_p_list, axis=0)
                    self.metric.update(rewards, rewards_p)

                    if all(dones):
                        break
                    rewards_l, rewards_f = self.metric.rewards_each()
                    mega_l = Registry.mapping['model_mapping']['setting'].param['mega_l']
                    mega_f = Registry.mapping['model_mapping']['setting'].param['mega_f']
                    her = Registry.mapping['model_mapping']['setting'].param['her']

                    self.logger.info("Test step:{}/{}, travel time :{}, rewards_p:{}, rewards:{}, rewards_l:{}, queue:{}, delay:{}, throughput:{}".format( \
                            int(i/10), self.episodes, self.metric.real_average_travel_time(), self.metric.rewards_p(),self.metric.rewards(), rewards_l, \
                            self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))

                    self.writeLog("TEST", int(i/10), self.metric.real_average_travel_time(), \
                                  mega_l, mega_f, her, self.metric.rewards(), rewards_l, self.metric.queue(), self.metric.queue_l(), self.metric.delay(), self.metric.throughput())
        return self.metric

    def writeLog(self, mode, step, travel_time, mega_l, mega_f, her, cur_rwd, cur_rwd_l, cur_queue, cur_queue_l,
                 cur_delay, cur_throughput, none_='''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''):
        none_
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % mega_l + '\t' + "%.1f" % mega_f + "\t" + "%.1f" % her + '\t' + \
              "%.2f" % cur_rwd + "\t" + "%.2f" % cur_rwd_l + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_queue_l + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

