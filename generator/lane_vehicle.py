import numpy as np
from . import BaseGenerator
from world import world_cityflow, world_sumo #, world_openengine


class LaneVehicleGenerator(BaseGenerator):
    '''
    Generate state or reward based on statistics of lane vehicles.

    :param world: World object
    :param I: Intersection object
    :param fns: list of statistics to get, currently support "lane_count", "lane_waiting_count" , "lane_waiting_time_count", "lane_delay" and "pressure". 
        "lane_count": get number of running vehicles on each lane. 
        "lane_waiting_count": get number of waiting vehicles(speed less than 0.1m/s) on each lane. 
        "lane_waiting_time_count": get the sum of waiting time of vehicles on the lane since their last action. 
        "lane_delay": the delay of each lane: 1 - lane_avg_speed/speed_limit.
    :param in_only: boolean, whether to compute incoming lanes only. 
    :param average: None or str, None means no averaging, 
        "road" means take average of lanes on each road, 
        "all" means take average of all lanes.
    :param negative: boolean, whether return negative values (mostly for Reward).
    '''
    def __init__(self, world, I, fns, in_only=False, average=None, negative=False):
        self.world = world
        self.I = I

        # get lanes of intersections
        self.lanes = []
        # 寻找到了roads是从I中得到的，所以得去I处赵I中的in_roads或roads是如何得到的，I在world_cityflow的Intersection中的insert_road中
        if in_only:
            roads = I.in_roads
        else:
            roads = I.roads

        # ---------------------------------------------------------------------
        # # resort roads order to NESW
        # if self.I.lane_order_cf != None or self.I.lane_order_sumo != None:
        #     tmp = []
        #     if isinstance(world, world_sumo.World):
        #         for x in ['N', 'E', 'S', 'W']:
        #             if self.I.lane_order_sumo[x] != -1:
        #                 tmp.append(roads[self.I.lane_order_sumo[x]])
        #             # else:
        #             #     tmp.append('padding_roads')
        #         roads = tmp

        #         # TODO padding roads into 12 dims
        #         for r in roads:
        #             if not self.world.RIGHT:
        #                 tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
        #             else:
        #                 tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
        #             self.lanes.append(tmp)

        #     elif isinstance(world, world_cityflow.World):
        #         for x in ['N', 'E', 'S', 'W']:
        #             if self.I.lane_order_cf[x] != -1:
        #                 tmp.append(roads[self.I.lane_order_cf[x]])
        #             # else:
        #             #     tmp.append('padding_roads')
        #         roads = tmp

        #         # TODO padding roads into 12 dims
        #         for road in roads:
        #             from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
        #             self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])

        #     else:
        #         raise Exception('NOT IMPLEMENTED YET')
        
        # else:
        #     if isinstance(world, world_sumo.World):
        #         for r in roads:
        #             if not self.world.RIGHT:
        #                 tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
        #             else:
        #                 tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
        #             self.lanes.append(tmp)
        #             # TODO: rank lanes by lane ranking [0,1,2], assume we only have one digit for ranking
        #     elif isinstance(world, world_cityflow.World):
        #         for road in roads:
        #             from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
        #             self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])
            
        #     else:
        #         raise Exception('NOT IMPLEMENTED YET')


            

        # ---------------------------------------------------------------------------------------------------------------
        # TODO: register it in Registry
        if isinstance(world, world_sumo.World):
            for r in roads:
                if not self.world.RIGHT:
                    tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
                else:
                    tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
                self.lanes.append(tmp)
                # TODO: rank lanes by lane ranking [0,1,2], assume we only have one digit for ranking
        elif isinstance(world, world_cityflow.World):
            # 通过将roads中的每条路（四个方向上的路） 加上012 就成为了每条lane了（每个方向3条道）
            # 所以需要先知道知道交叉口的路，存在roads中，向上寻找roads是如何建立的
            for road in roads:
                from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
                self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])

            # 添加邻居lanes的信息
            lane_n = {}
            for inter_n in self.I.inter_neighbor_id:
                # 添加road
                r = []
                for road in world.roadnet["roads"]:
                    # self.all_roads.append(road["id"])
                    # i = 0
                    # road_l = self.get_road_length(road)
                    # for lane in road["lanes"]:
                    #     self.all_lanes.append(road["id"] + "_" + str(i))
                    #     i += 1
                    # iid = road["startIntersection"]
                    # if iid == inter_n:
                    #     road_n[iid].insert_road(road, True)
                    # iid = road["endIntersection"]
                    # if iid == inter_n:
                    #     road_n[iid].insert_road(road, False)
                    # 进车道
                    if road["endIntersection"] == inter_n:
                        r.append(road)
                l = []
                for road in r:
                    from_zero = (road["startIntersection"] == inter_n) if self.world.RIGHT else (road["endIntersection"] == inter_n)
                    l.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])
                lane_n[inter_n] = l
            self.lanes_n = lane_n
        # ---------------------------------------------------------------------------------------------------------------
        
        # elif isinstance(world, world_openengine.World):
        #     for r in roads:
        #         if self.world.RIGHT:
        #             tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(str(ob)[-1]), reverse=True)
        #         else:
        #             tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(str(ob)[-1]))
        #         self.lanes.append(tmp)
        else:
            raise Exception('NOT IMPLEMENTED YET')




        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns

        # calculate result dimensions
        size = sum(len(x) for x in self.lanes)
        # if fns[0] == 'pressure':
        #     size = 1
        if average == "road":
            size = len(roads)
        elif average == "all":
            size = 1
        self.ob_length = len(fns) * size
        if self.ob_length == 3:
            self.ob_length = 4

        self.average = average
        self.negative = negative

    # def getMapOfVehicles(area_length=600):


    def generate(self):
        '''
        generate
        Generate state or reward based on current simulation state.
        
        :param: None
        :return ret: state or reward
        '''
        results = [self.world.get_info(fn) for fn in self.fns]

        #need modification here

        ret = np.array([])
        for i in range(len(self.fns)):
            result = results[i]
            if self.fns[i] != 'map':
                # pressure returns result of each intersections, so return directly
                if self.I.id in result:
                    ret = np.append(ret, result[self.I.id])
                    continue
                fn_result = np.array([])

                for road_lanes in self.lanes:
                    road_result = []
                    for lane_id in road_lanes:
                        road_result.append(result[lane_id])
                    if self.average == "road" or self.average == "all":
                        road_result = np.mean(road_result)
                    else:
                        road_result = np.array(road_result)
                    fn_result = np.append(fn_result, road_result)

                if self.average == "all":
                    fn_result = np.mean(fn_result)
                ret = np.append(ret, fn_result)
            else:
                ret = [ret]
                ret.append(result)
        if self.negative:
            ret = ret * (-1)
        origin_ret = ret
        # if len(ret) == 3:
        #     ret_list = list(ret)
        #     ret_list.append(0)
        #     ret = np.array(ret_list)
        # if len(ret) == 2:
        #     ret_list = list(ret)
        #     ret_list.append(0)
        #     ret_list.append(0)
        #     ret = np.array(ret_list)
        return ret

    # ifI==1即是自己，==0是邻居
    # 或者不要ifI直接吧对应的lanes传过来
    # 本论文不需要求平均，把求平均的都注释掉了，直接求和就可
    def generate_ret(self, intersection_id, lanes):
        ret = np.array([])
        results = [self.world.get_info(fn) for fn in self.fns]
        for i in range(len(self.fns)):
            result = results[i]

            # pressure returns result of each intersections, so return directly
            if intersection_id in result:
                ret = np.append(ret, result[intersection_id])
                continue
            fn_result = np.array([])
            # TODO self.lanes要改成对应的intersection的lanes
            # road_lanes一个方向上的三条道
            for road_lanes in lanes:
                road_result = []
                # 一条道
                for lane_id in road_lanes:
                    road_result.append(result[lane_id])
                # if self.average == "road" or self.average == "all":
                #     # 将每个方向上的三个车道的等待车辆做平均
                #     road_result = np.mean(road_result)
                # else:
                #     road_result = np.array(road_result)
                road_result = np.array(road_result)
                fn_result = np.append(fn_result, road_result)

            # # 将四个方向上的等待车辆做平均，现在的fn_result是一个交叉口四个方向12条进车道做了平均，即一个交叉口的12个车道的平均等待车辆
            # if self.average == "all":
            #     fn_result = np.mean(fn_result)
            # ret存一个交叉口上每条车道的平均等待车辆
            ret = np.append(ret, fn_result)
        ret = (sum(ret)) * (-1)
        # ret = np.array([ret])
        return ret

    def generate_reward(self):
        '''
        generate
        根据当前模拟状态生成状态或奖励
        Generate state or reward based on current simulation state.

        :param: None
        :return ret: state or reward
        '''


        # need modification here

        # ret存一个交叉口上每条车道的平均等待车辆
        # ret = np.array([])
        ret = self.generate_ret(self.I.id, self.lanes)
        # ret_all中包含了自己和邻居的ret结果，自己：‘inter_self’=‘’ 邻居的：‘邻居的id’=‘ ’
        ret_all = {"inter_self": ret}
        # ret_n = np.array([])
        # ret_N = []

        ret_i = ret_all["inter_self"]
        # self.roadnet["intersections"]或许用中括号？
        # inter_id==0是追随者，==1是领导者
        if self.I.inter_id == 0:
            # TODO sum(miga*r_邻居)
            # TODO sum(miga_邻居)
            # TODO 外面套一层循环，ret中不止是自己的reward，加上邻居的reward（即通过这层循环将邻居的ret（车道等待车辆）也算出来）
            for inter in self.world.intersections:
                # 是邻居
                if inter.id in self.I.inter_precursor_idx:
                    inter_nid = inter.id
                    ret = self.generate_ret(inter_nid, self.lanes_n[inter_nid])
                    ret_all[inter_nid] = ret
            ret_j = 0
            ret_mega = 0
            for inter_n in self.world.intersections:
                if inter_n.id in self.I.inter_precursor_idx:
                    inter_n_id = inter_n.id
                    ret_j += inter_n.mega * ret_all[inter_n_id]
                    ret_mega += inter_n.mega
            # 完整公式
            ret_i = (1/(ret_mega+1)) * (ret_i+ret_j)

        # if self.negative:
        #     ret = ret * (-1)
        # origin_ret = ret
        # if len(ret) == 3:
        #     ret_list = list(ret)
        #     ret_list.append(0)
        #     ret = np.array(ret_list)
        # if len(ret) == 2:
        #     ret_list = list(ret)
        #     ret_list.append(0)
        #     ret_list.append(0)
        #     ret = np.array(ret_list)
        ret_i = np.array([ret_i])
        return ret_i

if __name__ == "__main__":
    from world.world_cityflow import World
    world = World("examples/configs.json", thread_num=1)
    laneVehicle = LaneVehicleGenerator(world, world.intersections[0], ["count"], False, "road")
    for _ in range(100):
        world.step()
    print(laneVehicle.generate())

