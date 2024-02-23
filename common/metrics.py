import numpy as np
from common.registry import Registry


class Metrics(object):
    '''
    Register Metric for evaluating model performance. Currently support reward, queue length, delay(approximate or real), throughput and travel time. 
    - Average travel time (travel time): The average time that each vehicle spent on traveling within 
    the network, including waiting time and actual travel time. A smaller travel time means the better performance.
    - Queue length (queue): The average queue length over time, where the queue length at time t 
    is the sum of the number of vehicles waiting on lanes. A smaller queue length means the better performance.
    - Approximated delay (delay): Averaged difference between the current speed of vehicle and the 
    maximum speed limit of this lane over all vehicles, calculated from 1 - (sum_i^n(v_i)/(n*v_max)), where n is the 
    number of vehicles on the lane, v_i is the speed of vehicle i and v_max is the maximum allowed speed. 
    A smaller delay means the better performance.
    - Throughput: Number of vehicles that have finished their trips until current simulation step. A larger
    throughput means the better performance.
    '''
    def __init__(self, lane_metrics, world_metrics, world, agents):
        # must take record of rewards
        self.world = world
        self.agents = agents
        self.decision_num = 0
        self.lane_metric_List = lane_metrics
        self.lane_metrics = dict()

        for k in self.lane_metric_List:
            j_ = []
            for i in range(len(agents)):
                j = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
                j_.append(j)
            j_ = np.array(j_)
            self.lane_metrics.update({k: j_})
        self.lane_metrics['rewards'] = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        self.lane_metrics['rewards_p'] = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        self.lane_metrics.update({k: np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32) for k in self.lane_metric_List})
        self.world_metrics = world_metrics

    def update(self, rewards=None, rewards_p=None):
        '''
        update
        Recalculate metrics.

        :param rewards: reward name
        :return: None
        '''
        if rewards is not None:
            self.lane_metrics['rewards'] += np.squeeze(rewards)
            # self.lane_metrics['rewards'] += rewards.flatten()
        if rewards_p is not None:
            self.lane_metrics['rewards_p'] += np.squeeze(rewards_p)
        if 'delay' in self.lane_metrics.keys():
            # self.lane_metrics['delay'] += (np.stack(np.array([self.agents[0].get_delay()], dtype=np.float32))).flatten()
            self.lane_metrics['delay'] += (np.stack(np.array([ag.get_delay() for ag in self.agents], dtype=np.float32)))
        if 'queue' in self.lane_metrics.keys():
            # self.lane_metrics['queue'] += (np.stack(np.array([self.agents[0].get_queue()], dtype=np.float32))).flatten()
            self.lane_metrics['queue'] += (np.stack(np.array([ag.get_queue() for ag in self.agents], dtype=np.float32)))

        self.decision_num += 1

    def clear(self):
        '''
        clear
        Reset metrics.

        :param: None
        :return: None
        '''
        for k in self.lane_metric_List:
            j_ = []
            for i in range(len(self.agents)):
                j = np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
                j_.append(j)
            j_ = np.array(j_)
            self.lane_metrics.update({k: j_})
        # self.lane_metrics['rewards'] =  np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32)
        # self.lane_metrics = {k : np.array([0 for _ in range(len(self.world.intersections))], dtype=np.float32) for k in self.lane_metric_List}
        self.decision_num = 0

    def delay(self):
        '''
        delay
        Calculate vehicle delay.

        :param: None
        :return: real delay or approximate delay
        '''
        # real_delay
        if 'delay' not in self.lane_metrics.keys():
            return self.world.get_real_delay()
        
        # apx_delay
        else:
            try:
                result = self.lane_metrics['delay']
                return np.sum(result) / (self.decision_num * len(self.world.intersections))
            except KeyError:
                print(('apx delay is not recorded in lane_metrics, please add it into the list'))
                return None

    # def lane_delay(self):
    #     try:
    #         result = self.lane_metrics['delay']
    #         return result / self.decision_num 
    #     except KeyError:
    #         print('lane delay is not recorded in lane_metrics, please add it into the list')
    #         return None

    def queue(self):
        '''
        queue
        Calculate total queue length of all lanes.

        :param: None
        :return: total queue length
        '''
        # 每个路口的平均queue长度
        try:
            # queue_l = 0
            # leader_num = 0
            # leader = Registry.mapping['world_mapping']['graph_setting'].graph['node_leadid']
            result = self.lane_metrics['queue']
            # for i, queue in result:
            #     if leader[i] == 1:
            #         queue_l += queue
            #         leader_num += 1
            # queue_l = queue_l / (self.decision_num * leader_num)
            # queue_all = np.sum(result) / (self.decision_num * len(self.world.intersections))
            return np.sum(result) / (self.decision_num * len(self.world.intersections))
        except KeyError:
            print('queue in not recorded in lane_metrics, please add it into the list')
            return None


    def queue_l(self):
        '''
        queue
        Calculate total queue length of all lanes.

        :param: None
        :return: total queue length
        '''
        # 每个路口的平均queue长度
        try:
            queue_l = 0
            leader_num = 0
            leader = Registry.mapping['world_mapping']['graph_setting'].graph['node_leadid']
            result = self.lane_metrics['queue']
            for i, queue in enumerate(result[0]):
                if leader[i] == 1:
                    queue_l += queue
                    leader_num += 1
            return queue_l / (self.decision_num * leader_num)
        except KeyError:
            print('queue in not recorded in lane_metrics, please add it into the list')
            return None

    def lane_queue(self):
        '''
        lane_queue
        Calculate average queue length of lanes.

        :param: None
        :return: average queue length of lanes
        '''
        # 每个路口的总queue长度
        try:
            result = self.lane_metrics['queue']
            return result / self.decision_num
        except KeyError:
            print(('queue in not recorded in lane_metrics, please add it into the list'))
            return None

    def queue_each(self):
        '''
        lane_queue
        Calculate average queue length of lanes.

        :param: None
        :return: average queue length of lanes
        '''
        # 各个路口的queue长度
        try:
            result = self.lane_metrics['queue']
            return self.each(result)
            # graph = Registry.mapping['world_mapping']['graph_setting'].graph
            # node_leadid = graph['node_leadid']
            # result = self.lane_metrics['queue']
            # result_l = []
            # result_f = []
            # for idx, result_i in enumerate(result[0]):
            #     if node_leadid[idx] == 1:
            #         result_l .append(result_i)
            #     else:
            #         result_f.append(result_i)
            # result_l = sum(result_l)/len(result_l)
            # result_f = sum(result_f) / len(result_f)
            # return result_l / self.decision_num, result_f / self.decision_num
        except KeyError:
            print(('queue in not recorded in lane_metrics, please add it into the list'))
            return None

    def each(self, result):

        graph = Registry.mapping['world_mapping']['graph_setting'].graph
        node_leadid = graph['node_leadid']
        result_l = []
        result_f = []
        for idx, result_i in enumerate(result[0]):
            if node_leadid[idx] == 1:
                result_l.append(result_i)
            else:
                result_f.append(result_i)
        result_l = sum(result_l) / len(result_l)
        result_f = sum(result_f) / len(result_f)
        return result_l / self.decision_num, result_f / self.decision_num

    def rewards(self):
        '''
        rewards
        Calculate total rewards of all lanes.

        :param: None
        :return: total rewards
        '''
        result = self.lane_metrics['rewards']
        return np.sum(result[0]) / self.decision_num

    def rewards_p(self):
        '''
        rewards
        Calculate total rewards of all lanes.

        :param: None
        :return: total rewards
        '''
        result = self.lane_metrics['rewards_p']
        return np.sum(result[0]) / self.decision_num

    def rewards_each(self):
        '''
        rewards
        Calculate total rewards of all lanes.

        :param: None
        :return: total rewards
        '''
        result = self.lane_metrics['rewards']
        return self.each(result)
        # graph = Registry.mapping['world_mapping']['graph_setting'].graph
        # node_leadid = graph['node_leadid']
        # result = self.lane_metrics['rewards']
        # result_l = []
        # result_f = []
        # for idx, result_i in enumerate(result[0]):
        #     if node_leadid[idx] == 1:
        #         result_l.append(result_i)
        #     else:
        #         result_f.append(result_i)
        # result_l = sum(result_l) / len(result_l)
        # result_f = sum(result_f) / len(result_f)
        # return result_l / self.decision_num, result_f / self.decision_num
    
    def lane_rewards(self):
        '''
        lane_rewards
        Calculate average reward of lanes.

        :param: None
        :return: average reward of lanes
        '''
        result = self.lane_metrics['rewards']
        return result / self.decision_num
    
    def throughput(self):
        '''
        throughput
        Calculate throughput.

        :param: None
        :return: current throughput
        '''
        return self.world.get_cur_throughput()

    def throughput(self):
        '''
        throughput
        Calculate throughput.

        :param: None
        :return: current throughput
        '''
        return self.world.get_cur_throughput()
    
    def real_average_travel_time(self):
        '''
        real_average_travel_time
        Calculate average travel time.

        :param: None
        :return: average_travel_time
        '''
        return self.world.get_average_travel_time()
    

    
