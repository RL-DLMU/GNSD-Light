import pickle
import numpy as np
import json
import os
import random
import sys
import yaml
import copy
import logging
import pandas as pd
from datetime import datetime
from keras.utils import to_categorical
from common.registry import Registry
import world


def get_road_dict(roadnet_dict, road_id):
    for item in roadnet_dict['roads']:
        if item['id'] == road_id:
            return item
    raise KeyError("environment and graph setting mapping error, no such road exists")

# flag_self是否加入自己都列表中, flag_even输出的维度是否统一，false代表有几个前驱，就几项，true代表统一为4或5（5是有自己，即flag_self=true）
def adjacency_index2matrix(adjacency_index, num, out_dim, flag_self, flag_even):
    # adjacency_index(the nearest K neighbors):[1,2,3]
    """
    if in 1*6 aterial and
        - the 0th intersection,then the adjacency_index should be [0,1,2,3]
        - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
        - the 2nd intersection, then adj [2,0,1,3]

    """
    # [batch,agents,neighbors]
    # out_dim = len(adjacency_index)
    if flag_self:
        for id,node_list in enumerate(adjacency_index):
            node_list.append(id)
    if flag_even:
        for id, node_list in enumerate(adjacency_index):
            if -1 in node_list:
                adjacency_index.remove(node_list)
    # adjacency_index = np.array(pd.DataFrame(adjacency_index))
    # 生成空的输出矩阵
    padd = np.zeros(out_dim, dtype=np.float32)
    out_matrix = []
    for i, classes in enumerate(adjacency_index):
        # -1 的 全为0
        one_hot = to_categorical(np.array(classes), num_classes=out_dim)
        # -1 的最后多的那项
        # one_hot = to_categorical(np.array(classes), num_classes=out_dim)
        # if flag_even:
        while one_hot.shape[0] < num:
            one_hot = np.vstack([one_hot, padd])
        # if i == 0:
        #     out_matrix = one_hot
        # else:
        #     out_matrix = np.stack([out_matrix, one_hot], axis=0)
        # out_matrix.append(one_hot)
        out_matrix = one_hot if i == 0 else np.concatenate([out_matrix, one_hot], axis=0)
        # n += len(classes)
    out_matrix = out_matrix.reshape(-1, num, out_dim)
    # adjacency_index_new = np.sort(adjacency_index, axis=-1)
    # l = to_categorical(adjacency_index_new, num_classes=num_agents)
    return out_matrix
def adjacency_concat2matrix(adjacency_index, num, out_dim):
    '''前n个是前驱，后n个是自己，输出前驱（前16）及自己（后16）的位置下标'''
    # adjacency_index(the nearest K neighbors):[1,2,3]
    """
    if in 1*6 aterial and
        - the 0th intersection,then the adjacency_index should be [0,1,2,3]
        - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
        - the 2nd intersection, then adj [2,0,1,3]

    """
    # [batch,agents,neighbors]
    lead_index = len(adjacency_index)
    for id,node_list in enumerate(adjacency_index):
        if -1 in node_list:
            # node_list.append(lead_index)
            node_list.remove(-1)
        node_list.append(lead_index)
        lead_index += 1
    # adjacency_index = np.array(pd.DataFrame(adjacency_index))
    # 生成空的输出矩阵
    padd = np.zeros(out_dim, dtype=np.float32)
    # n = 0
    out_matrix = []
    for i, classes in enumerate(adjacency_index):
        one_hot = to_categorical(np.array(classes), num_classes=out_dim)
        while one_hot.shape[0] < num:
            one_hot = np.vstack([one_hot, padd])
        out_matrix = one_hot if i == 0 else np.concatenate([out_matrix, one_hot], axis=0)
    out_matrix = out_matrix.reshape(-1, num, out_dim)
    return out_matrix
def build_index_intersection_map(roadnet_file, flow_path):
    """
    generate the map between identity ---> index ,index --->identity
    generate the map between int ---> roads,  roads ----> int
    generate the required adjacent matrix
    generate the degree vector of node (we only care the valid roads(not connect with virtual intersection), and intersections)
    return: map_dict, and adjacent matrix
    res = [net_node_dict_id2inter,net_node_dict_inter2id,net_edge_dict_id2edge,net_edge_dict_edge2id,
        node_degree_node,node_degree_edge,node_adjacent_node_matrix,node_adjacent_edge_matrix,
        edge_adjacent_node_matrix]
    """
    roadnet_dict = json.load(open(roadnet_file, "r"))
    virt = "virtual" # judge whether is virtual node, especially in convert_sumo file
    if "gt_virtual" in roadnet_dict["intersections"][0]:
        virt = "gt_virtual"
    valid_intersection_id = [node["id"] for node in roadnet_dict["intersections"] if not node[virt]]
    node_id2idx = {}
    node_idx2id = {}
    edge_id2idx = {}
    edge_idx2id = {}
    node_id2lead = {}
    node_degrees = []  # the num of adjacent nodes of node

    edge_list = []  # adjacent node of each node
    node_list = []  # adjacent edge of each node
    sparse_adj = []  # adjacent node of each edge
    invalid_roads = []
    node_leadid = []
    Heterogeneous_id = [] #异构结点
    lead_id = []

    cur_num = 0 #同构节点数量
    num = 0 # 节点总数
    # 点集合
    # build the map between identity and index of node
    for node_dict in roadnet_dict["intersections"]:
        if node_dict[virt]:
            for node in node_dict["roads"]:
                invalid_roads.append(node)
            continue
        cur_id = node_dict["id"]
        # all_node_idx2id[num] = cur_id
        # all_node_id2idx[cur_id] = num
        num += 1
        if len(node_dict['roads']) == 8 : # 同构
            node_idx2id[cur_num] = cur_id
            node_id2idx[cur_id] = cur_num
            cur_num += 1
        else: # 异构
            Heterogeneous_id.append(cur_id)
            # Heterogeneous_idx.append(all_node_id2idx[cur_id])
        # 选领导者
        # if cur_num == 5 or cur_num == 17 or cur_num == 30 or cur_num == 31 or cur_num == 32 or cur_num == 33 or cur_num == 34 or cur_num == 35 or cur_num == 36 or cur_num == 37 or cur_num == 38 or cur_num == 39 or cur_num == 41 or cur_num == 43 or cur_num == 45 or cur_num == 47:  #newyork16*3
        # if cur_num == 1 or cur_num == 3 or cur_num == 4 or cur_num == 7 or cur_num == 10 or cur_num == 12:  # jinan
        # if cur_num == 3 or cur_num == 8 or cur_num == 11:  # jinan
        # if cur_num == 1 or cur_num == 4 or cur_num == 16: # hangzhou
        # if cur_num == 25 or cur_num == 28: # syn
        if cur_num == 6: # syn
        # if cur_num == 7 or cur_num == 10 or cur_num == 11: # syn
        # if cur_num == 1 or cur_num == 2 or cur_num == 3 or cur_num == 4 or cur_num == 13 or cur_num == 14 or cur_num == 15 or cur_num == 16:  # hangzhou
            node_id2lead[cur_id] = 1
            node_leadid.append(1)
            lead_id.append(cur_num - 1)
        else:
            node_id2lead[cur_id] = 0
            node_leadid.append(0)
        # node_id2lead[cur_id] = random.randint(0, 1)
        # if node_id2lead[cur_id] == 1:
        #     lead_id.append(cur_num-1)
    # map between identity and index built done

    # sanity check of node number equals intersection numbers
    if cur_num + len(Heterogeneous_id) != len(valid_intersection_id):
        raise ValueError("environment and graph setting mapping error, node 1 to 1 mapping error")

    # 边集合
    # build the map between identity and index and built the adjacent matrix of edge
    cur_num = 0
    for edge_dict in roadnet_dict["roads"]:
        edge_id = edge_dict["id"]
        if edge_id in invalid_roads:
            continue
        else:
            edge_idx2id[cur_num] = edge_id
            edge_id2idx[edge_id] = cur_num
            cur_num += 1
            input_node_id = edge_dict['startIntersection']
            output_node_id = edge_dict['endIntersection']
            input_node_idx = node_id2idx[input_node_id]
            output_node_idx = node_id2idx[output_node_id]
            sparse_adj.append([input_node_idx, output_node_idx])



    # for edge_arr in sparse_adj:
    #     input_node_idx = edge_arr[0]
    #     output_node_idx = edge_arr[1]
    #     if output_node_idx in Heterogeneous_idx:
    #         while() # 找到异构节点连接的同构节点
    #         sparse_adj.remove(edge_arr)
    #
    #
    #     input_node_id = edge_dict['startIntersection']
    #     output_node_id = edge_dict['endIntersection']
    #     if input_node_id not in Heterogeneous_id and output_node_id not in Heterogeneous_id: # 同构
    #         edge_idx2id[cur_num] = edge_id
    #         edge_id2idx[edge_id] = cur_num
    #         cur_num += 1
    #         # input_node_id = edge_dict['startIntersection']
    #         # output_node_id = edge_dict['endIntersection']
    #         input_node_idx = node_id2idx[input_node_id]
    #         output_node_idx = node_id2idx[output_node_id]
    #         sparse_adj.append([input_node_idx, output_node_idx])
    #     elif output_node_id in Heterogeneous_nodes: # 需要处理的异构的边（不是所有异构的边都处理）
    #         input_node_idx = node_id2idx[input_node_id]
    #         for road in roadnet_dict["roads"]:
    #             if road["startIntersection"] == output_node_id:
    #                 if road["endIntersection"] not in Heterogeneous_nodes:
    #                     output_node_idx = node_id2idx[road["endIntersection"]]
    #                     sparse_adj.append([input_node_idx, output_node_idx])
    #                 else:
    #                     a = 1
            # node_id = node_idx2id[output_node_id]
            # for road_link in roadnet_dict["intersection"][node_id]["roadLinks"]:
            #     if road_link["startRoad"] == edge_dict:
            #         endroad = road_link["endRoad"] # 找到异构节点连接的其他路
            #         output_node_id = 1
    # build adjacent matrix for node (i.e the adjacent node of the node, and the
    # adjacent edge of the node)
    for node_dict in roadnet_dict["intersections"]:
        if node_dict[virt]:
            continue
        node_id = node_dict["id"]
        road_links = node_dict['roads']
        input_nodes = []  # should be node_degree
        input_edges = []  # needed, should be node_degree
        for road_link_id in road_links:
            road_link_dict = get_road_dict(roadnet_dict, road_link_id)
            if road_link_dict['endIntersection'] == node_id:
                if road_link_id in edge_id2idx.keys():
                    input_edge_idx = edge_id2idx[road_link_id]
                    input_edges.append(input_edge_idx)
                else:
                    continue
                start_node = road_link_dict['startIntersection']
                if start_node in node_id2idx.keys():
                    start_node_idx = node_id2idx[start_node]
                    input_nodes.append(start_node_idx)
        if len(input_nodes) != len(input_edges):
            raise ValueError(f"{node_id} : number of input node and edge not equals")
        node_degrees.append(len(input_nodes))
        # node_list：每个节点的每条边的源头是哪个节点，即第一项（0节点）内容是1（这条边来源于1节点）
        # edge_list：每个节点拥有的进车道的边的序号
        edge_list.append(input_edges)
        node_list.append(input_nodes)
    [l.sort() for l in node_list]
    # node_degrees是每个node的入度；sparse_adj存在了哪些边
    node_degrees = np.array(node_degrees)  # the num of adjacent nodes of node
    sparse_adj = np.array(sparse_adj)  # the valid num of adjacent edges of node
    nei_matrix = copy.deepcopy(node_list)
    Adj_matrix_onehot = adjacency_index2matrix(nei_matrix, 5, len(nei_matrix), True, False)
    Adj_matrix = np.sum(Adj_matrix_onehot, -2)
    nei_pos_matrix = generate_pos(nei_matrix, Adj_matrix)
    # pos_matrix = np.array(generate_pos(nei_matrix, Adj_matrix))
    # nei_pos_matrix = []
    # # pos_matrix = pos_matrix.repeat(len(pos_matrix)).reshape(len(pos_matrix),len(pos_matrix),len(pos_matrix))
    # for matrix in pos_matrix:
    #     nei_pos_matrix.append(np.diag(matrix)) #现在是矩阵在对角线上，就可以乘了
    # matrix = copy.deepcopy(pos_list)
    # nei_pos_matrix = adjacency_index2matrix(matrix, len(matrix[1]), 5, False, False)
    # 广度优先遍历
    node_BFS = []  # 存放最终的结果
    nodelayer_BFS = []
    node_precursor = [[-1] for i in range(len(node_list))]
    flag = [0]*len(node_idx2id)
    flag_p = [10000] * len(node_idx2id)
    # queue = np.array(lead_id)  # 存储当前层的节点，这个相当于队列，先进先出，后进后出
    queue = lead_id  # 存储当前层的节点，这个相当于队列，先进先出，后进后出
    level = 0
    # 标记遍历过的节点
    for id in queue:
        flag[id] = 1
        flag_p[id] = level
    while queue:  # 当当前层没有节点时，退出循环
        n = len(queue)  # 获取当前队列的长度，这个长度相当于 当前这一层的节点个数,以区分来自那一层
        temp = []  # 暂存,在while的循环中，别写在下面的for循环中！
        for i in range(n):  # 遍历当前层的节点数
            a = queue.pop(0)  # 推出去,将queue中存的上一层的节点全部pop出去
            temp.append(a)
            # 将下一层的节点（左右）添加到queue中，以进行下一次迭代（循环）
            a_n = node_list[a]
            flag_p[a] = level
            for b in a_n:
                if flag[b] == 0:
                    if flag_p[b] > flag_p[a]:
                        node_precursor[b][0] = a
                    queue.append(b)
                    flag[b] = 1
                else:
                    if flag_p[b] > flag_p[a]:
                        node_precursor[b].append(a)
        level += 1
        for i in temp:
            node_BFS.append(i)
        for id in queue:
            flag_p[id] = level
        nodelayer_BFS.append(temp)

    # 定义排序规则的比较函数

    # [l.sort() for l in node_precursor]
    [l.sort() for l in nodelayer_BFS]
    # teammates = generate_teammates(nodelayer_BFS)
    # matrix = copy.deepcopy(teammates)
    # teammates_matrix = adjacency_index2matrix(matrix, len(matrix), len(matrix), False, False)
    node_precursor_sort = []
    for id, array in enumerate(node_precursor):
        def sort_key(tuple_item):
            idx = tuple_item
            if idx == id - 1:
                return 0
            elif idx == id + 1:
                return 1
            elif idx < id:
                return 2
            else:
                return 3
        # a = [1,4,6,9]
        # a=sorted(a, key=sort_key)
        node_precursor_sort.append(sorted(array, key=sort_key))
    pre_matrix = copy.deepcopy(node_precursor_sort)
    precursor_matrix_onehot = adjacency_index2matrix(pre_matrix, 4, len(pre_matrix)+1, False, False)
    precursor_matrix = np.sum(precursor_matrix_onehot, -2)
    # pos_list = generate_pos(node_precursor_sort, precursor_matrix, nodelayer_BFS[0])
    # matrix = copy.deepcopy(pos_list)
    # pre_pos_matrix = adjacency_index2matrix(matrix, len(matrix[1]), 5, False, False)

    his_pos_matrix = generate_pos_his_vari(node_precursor_sort, precursor_matrix, nodelayer_BFS[0])
    # matrix = copy.deepcopy(his_pos_list)
    # his_pos_matrix = adjacency_index2matrix(matrix, len(matrix[1]), 4, False, False)
    result = {'node_idx2id': node_idx2id, 'node_id2idx': node_id2idx, 'node_id2lead': node_id2lead,
              'edge_idx2id': edge_idx2id, 'edge_id2idx': edge_id2idx, 'node_BFS': node_BFS, 'nodelayer_BFS': nodelayer_BFS, 'his_pos_matrix': his_pos_matrix,
              'node_degrees': node_degrees, 'sparse_adj': sparse_adj, 'node_precursor': node_precursor_sort, 'precursor_matrix': precursor_matrix, 'nei_pos_matrix': nei_pos_matrix,
              'node_list': node_list, 'Adj_matrix': Adj_matrix, 'edge_list': edge_list, 'node_leadid': node_leadid}
    return result
def generate_pos(matrix, precursor_matrix, leader=None):
    pos_list = (np.zeros_like(precursor_matrix)).tolist()
    for i, nodes in enumerate(matrix):
        for node in nodes:
            # if node - i == len(pos_list):  # 自己
            #     pos_list[i][node] = 1
            if node == -1:
                break
            elif node == i-1: # 左
                pos_list[i][node] = 3
            elif node == i+1: # 右
                pos_list[i][node] = 5
            elif node < i: #上
                pos_list[i][node] = 7
            elif node > i: # 下
                pos_list[i][node] = 11
            else: # 自己
                pos_list[i][node] = 13
    # if leader:
    #     pos_list = [value for i, value in enumerate(pos_list) if i not in leader]
    return pos_list

def generate_pos_his1(matrix, precursor_matrix, leader=None):
    pos_list = (np.zeros_like(precursor_matrix)).tolist()
    # padd = np.zeros(len(pos_list[0]), dtype=np.float32)
    # pos_list.append(padd)
    pos_list = np.tile(pos_list, (1, 5))
    n_agent = len(matrix)
    for i, nodes in enumerate(matrix):
        for node in nodes:
            # if node - i == len(pos_list):  # 自己
            #     pos_list[i][node] = 1
            if node == -1:
                break
            elif node == i-1: # 左
                pos_list[i][node] = 3
                pos_list[i][(n_agent+1)*1 + node] = 3
                pos_list[i][(n_agent+1)*2 + node] = 3
                pos_list[i][(n_agent+1)*3 + node] = 3
                pos_list[i][(n_agent+1)*4 + node] = 3
            elif node == i+1: # 右
                pos_list[i][node] = 5
                pos_list[i][(n_agent + 1) * 1 + node] = 5
                pos_list[i][(n_agent + 1) * 2 + node] = 5
                pos_list[i][(n_agent + 1) * 3 + node] = 5
                pos_list[i][(n_agent + 1) * 4 + node] = 5
            elif node < i: #上
                pos_list[i][node] = 7
                pos_list[i][(n_agent + 1) * 1 + node] = 7
                pos_list[i][(n_agent + 1) * 2 + node] = 7
                pos_list[i][(n_agent + 1) * 3 + node] = 7
                pos_list[i][(n_agent + 1) * 4 + node] = 7
            elif node > i: # 下
                pos_list[i][node] = 11
                pos_list[i][(n_agent + 1) * 1 + node] = 11
                pos_list[i][(n_agent + 1) * 2 + node] = 11
                pos_list[i][(n_agent + 1) * 3 + node] = 11
                pos_list[i][(n_agent + 1) * 4 + node] = 11
            else: # 自己
                pos_list[i][node] = 5
    return pos_list

def generate_pos_his_vari(matrix, precursor_matrix, leader=None): # -MF，将3 5 7 11 都改成1
    pos_list = (np.zeros_like(precursor_matrix)).tolist()
    # padd = np.zeros(len(pos_list[0]), dtype=np.float32)
    # pos_list.append(padd)
    his_length = Registry.mapping['trainer_mapping']['setting'].param['his_length']
    pos_list = np.tile(pos_list, (1, his_length))
    n_agent = len(matrix)
    for i, nodes in enumerate(matrix):
        for node in nodes:
            # if node - i == len(pos_list):  # 自己
            #     pos_list[i][node] = 1
            for j in range(his_length):
                if node == -1:
                    break
                elif node == i-1: # 左
                    pos_list[i][(n_agent + 1) * j + node] = 3
                elif node == i+1: # 右
                    pos_list[i][(n_agent + 1) * j + node] = 5
                elif node < i: #上
                    pos_list[i][(n_agent + 1) * j + node] = 7
                elif node > i: # 下
                    pos_list[i][(n_agent + 1) * j + node] = 11
                else: # 自己
                    pos_list[i][node] = 5
    return pos_list

def generate_pos_his(matrix, precursor_matrix, leader=None):
    pos_list = (np.zeros_like(precursor_matrix)).tolist()
    pos_list = np.tile(pos_list, (1, 4))
    n_agent = len(matrix)
    for i, nodes in enumerate(matrix):
        for node in nodes:
            # if node - i == len(pos_list):  # 自己
            #     pos_list[i][node] = 1
            if node == -1:
                break
            elif node == i-1: # 左
                pos_list[i][node] = 0
                pos_list[i][(n_agent+1)*1 + node] = 4
                pos_list[i][(n_agent+1)*2 + node] = 8
                pos_list[i][(n_agent+1)*3 + node] = 12
            elif node == i+1: # 右
                pos_list[i][node] = 1
                pos_list[i][(n_agent + 1) * 1 + node] = 5
                pos_list[i][(n_agent + 1) * 2 + node] = 9
                pos_list[i][(n_agent + 1) * 3 + node] = 13
            elif node < i: #上
                pos_list[i][node] = 2
                pos_list[i][(n_agent + 1) * 1 + node] = 6
                pos_list[i][(n_agent + 1) * 2 + node] = 10
                pos_list[i][(n_agent + 1) * 3 + node] = 14
            elif node > i: # 下
                pos_list[i][node] = 3
                pos_list[i][(n_agent + 1) * 1 + node] = 7
                pos_list[i][(n_agent + 1) * 2 + node] = 11
                pos_list[i][(n_agent + 1) * 3 + node] = 15
            else: # 自己
                pos_list[i][node] = 5
    # if leader:
    #     pos_list = [value for i, value in enumerate(pos_list) if i not in leader]
    return pos_list
def generate_teammates(input_array):
    teammates = [[0]*1]*16
    for i, array in enumerate(teammates):
        def find_index(arr, target):
            for x, sublist in enumerate(arr):
                if target in sublist:
                    return x
            return -1
        array = input_array[find_index(input_array, i)]
        teammates[i] = array
    # for i, array in enumerate(teammates):
    #     teammates[i].remove(i)

    return teammates

def analyse_vehicle_nums(file_path):
    replay_buffer = pickle.load(open(file_path, "rb"))
    observation = [i[0] for i in replay_buffer]
    observation = np.array(observation)
    observation = observation.reshape([-1])
    print("the mean of vehicle nums is ", observation.mean())
    print("the max of vehicle nums is ", observation.max())
    print("the min of vehicle nums is ", observation.min())
    print("the std of vehicle nums is ", observation.std())


def get_output_file_path(task, model, prefix):
    path = os.path.join('./data/output_data', task, model, prefix)
    return path


def load_config(path, previous_includes=[]):
    if path in previous_includes:
        raise ValueError(
            f"Cyclic configs include detected. {path} included in sequence {previous_includes}."
        )
    previous_includes = previous_includes + [path]

    direct_config = yaml.load(open(path, "r"), Loader=yaml.Loader)

    # Load configs from included files.
    if "includes" in direct_config:
        includes = direct_config.pop("includes")
    else:
        includes = []
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, '{}' provided".format(type(includes))
        )

    config = {}
    duplicates_warning = []
    duplicates_error = []

    # TODO: Need test duplication here
    for include in includes:
        include_config, inc_dup_warning, inc_dup_error = load_config(
            include, previous_includes
        )
        duplicates_warning += inc_dup_warning
        duplicates_error += inc_dup_error

        # Duplicates between includes causes an error
        config, merge_dup_error = merge_dicts(config, include_config)
        duplicates_error += merge_dup_error

    # Duplicates between included and main file causes warnings
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning += merge_dup_warning

    return config, duplicates_warning, duplicates_error


def merge_dicts(dict1: dict, dict2: dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py
    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.
    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates


def build_config(args):
    # configs file of specific agents is loaded from configs/agents/{agent_name}
    agent_name = os.path.join('./configs/agents', args.task, f'{args.agent}.yml')
    config, duplicates_warning, duplicates_error = load_config(agent_name)
    if len(duplicates_warning) > 0:
        logging.warning(
            f"Overwritten configs parameters from included configs "
            f"(non-included parameters take precedence): {duplicates_warning}"
        )
    if len(duplicates_error) > 0:
        raise ValueError(
            f"Conflicting (duplicate) parameters in simultaneously "
            f"included configs: {duplicates_error}"
        )
    args_dict = vars(args)
    for key in args_dict:
        config.update({key: args_dict[key]})  # short access for important param

    # add network(for FRAP and MPLight)
    cityflow_setting = json.load(open(config['path'], 'r'))
    config['traffic']['network'] = cityflow_setting['network'] if 'network' in cityflow_setting.keys() else None
    return config

