import torch.nn as nn
import pickledb
import pickle
from collections import defaultdict
import os
import numpy as np
import math
import gzip
import json
import socket
import requests

import torch
import torchvision.transforms as T

from PIL import Image
from PIL import ImageDraw
import pickledb

from ivlnce_baselines.common.mapping_module.mapper import EpisodesInfo, Observations, RobotCurrentState
from habitat.datasets.utils import VocabDict

class KeywordsMappingModule3:
    def __init__(self, split='train'):
        super().__init__()

        self.neighbour_distance_threshold = 3
        self.vp_highest_num = 3
        self.token_length = 20

        # self.detect_db_path = 'R2R_{}_enc_detection_owlvit-large-patch14.ivlnce.db'.format(split)
        # assert os.path.exists(self.detect_db_path)
        # self.setup_detection_result_db()

        self.viewpoint_position_file = './annotation/scan_viewpoint_to_r2rce_position.pkl'
        assert os.path.exists(self.viewpoint_position_file)
        self.setup_viewpoint_position()

        self.instruction_to_keywords_file = './annotation/r2r_ce_instruction_to_keywords.pkl'
        assert os.path.exists(self.instruction_to_keywords_file)
        self.setup_instruction_to_keywords()

        self.instruction_to_r2r_inst_id_file = './annotation/r2r_ce_instruction_to_r2r_inst_id.pkl'
        assert os.path.exists(self.instruction_to_r2r_inst_id_file)
        self.setup_instruction_to_r2r_inst_id()

        # self.connectivity = defaultdict(lambda: defaultdict(set))
        # self.detection_result = defaultdict(lambda: defaultdict(list))
        # self.recorded_viewpoint_with_instr_id = defaultdict(set)

        self.detection_result = defaultdict(lambda: defaultdict(list))
        self.scan_vp_to_visited_inst_id = defaultdict(set)

        # vocab -> token的映射
        self.annotation_processed_file = 'data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz'.format(split=split)
        self.setup_vocab_dict()
        
        # 注意，这里需要修改keep=("'s",)，因为habitat里面的默认keep写错了，没有写成tuple
        # self.instruction_vocab.tokenize_and_index("Leave the theater, and take a right. Stop next to the first dining room chair. ", keep=("'s",))

        # self.db_filename = 'real_time_detection.db'
        # self.setup_real_time_db()

        self.processor = None
        self.detector = None

        self.socket = None
        self.detector_port = 23456

        self.to_PIL = T.ToPILImage()

        self.viewpoints_rgbs = defaultdict(dict)

        print('Initialized keywords mapping module3')

    # def setup_real_time_db(self):
    #     self.real_time_detect_db = pickledb.load(self.db_filename, False)
    #     self.db_entry_num = len(self.real_time_detect_db.getall())
    #     # assert self.db_entry_num == 0
    #     print("Real-time Detections Saving to Database: {}".format(self.db_filename))
    #     print("Already {} entries in Database".format(self.db_entry_num))

    def setup_vocab_dict(self):
        self.keywords_token_cache = dict()
        with gzip.open(self.annotation_processed_file, "rt") as f:
            deserialized = json.loads(f.read())
            self.instruction_vocab = VocabDict(
                word_list=deserialized["instruction_vocab"]["word_list"]
            )
    
    # def setup_visibility(self):
    #     self.visibility = dict()
    #     for scan in self.scan_to_positions:
    #         self.visibility[scan] = torch.zeros(len(self.scan_to_positions[scan]), dtype=torch.bool).cuda()
        

    def setup_viewpoint_position(self):
        with open(self.viewpoint_position_file, 'rb') as f:
            # self.viewpoint_position = pickle.load(f)
            temp_file = pickle.load(f)
        self.scan_to_viewpoints = defaultdict(list)
        self.scan_to_positions = dict() # scan -> numpy array with positions
        # self.scan_viewpoints_to_index = defaultdict(dict)
        self.viewpoint_position = dict() # (scan, viewpoint) -> position
        # for k, v in self.viewpoint_position.items():
        #     self.scan_to_viewpoints[k[0]].append(k[1])
        #     self.scan_to_positions[k[0]].append(v)
        #     self.scan_viewpoints_to_index[k[0]][k[1]] = len(self.scan_to_viewpoints[k[0]]) - 1

        for k, v in temp_file.items():
            if k[0] not in self.scan_to_positions:
                self.scan_to_positions[k[0]] = torch.from_numpy(np.array([])).cuda()
        
        # for scan in self.scan_to_viewpoints.keys():
        #     self.scan_to_positions[scan] = torch.from_numpy(np.array(self.scan_to_positions[scan])).cuda()

        # print([len(v) for v in self.scan_to_viewpoints.values()])
    
    def setup_instruction_to_keywords(self):
        with open(self.instruction_to_keywords_file, 'rb') as f:
            self.instruction_to_keywords = pickle.load(f)

    def setup_instruction_to_r2r_inst_id(self):
        with open(self.instruction_to_r2r_inst_id_file, 'rb') as f:
            self.instruction_to_r2r_inst_id = pickle.load(f)

    # def setup_detection_result_db(self):
    #     self.pre_detect_result_db = pickledb.load(self.detect_db_path, False)

    def robot_state_to_viewpoint(self, episodes_info, robot_state, distance_threshold=0.25, create_viewpoint_threshold=1):
        min_distance_idxs = torch.zeros(len(episodes_info.env_names), dtype=torch.long).cuda()
        min_distance_viewpoints = list()
        for robot_idx, scan in enumerate(episodes_info.env_names):
            robot_position = robot_state.pose[robot_idx]
            viewpoints_positions = self.scan_to_positions[scan]
            if viewpoints_positions.shape[0] == 0:
                min_distance_viewpoints.append(-1)
                continue
            distance = torch.pow((robot_position - viewpoints_positions), 2)
            # ignore second dimension of distance
            distance[:, 1] = 0
            distance = torch.sum(distance, dim=1)
            assert len(distance.shape) == 1
            # min value and min index of distance
            min_value, min_distance_idxs[robot_idx] = torch.min(distance, dim=0)
            nearest_dist = torch.sqrt(min_value)
            if nearest_dist > distance_threshold and nearest_dist < create_viewpoint_threshold:
                min_distance_viewpoints.append(None)
            elif nearest_dist >= create_viewpoint_threshold:
                min_distance_viewpoints.append(-1)
            else:
                min_distance_viewpoints.append(self.scan_to_viewpoints[scan][min_distance_idxs[robot_idx]])
        return min_distance_viewpoints

    def update_memory(self, episode_info, observations, robot_state):
        # get nearest viewpoint
        viewpoints = self.robot_state_to_viewpoint(episode_info, robot_state)
        assert len(viewpoints) == len(episode_info.env_names)

        new_viewpoints = list()

        # update visibility
        for vp_idx, viewpoint in enumerate(viewpoints):
            if viewpoint is None:
                new_viewpoints.append(viewpoint)
                continue
            elif viewpoint == -1:
                scan = episode_info.env_names[vp_idx]
                current_pose = robot_state.pose[vp_idx]
                current_heading = robot_state.heading[vp_idx]
                viewpoint = len(self.scan_to_viewpoints[scan])
                self.scan_to_viewpoints[scan].append(viewpoint)
                self.scan_to_positions[scan] = torch.cat((self.scan_to_positions[scan], current_pose.unsqueeze(0)), dim=0)
                self.viewpoint_position[(scan, viewpoint)] = current_pose

                pano_rgbs = [observations['rgb_{}'.format(i)][vp_idx] for i in range(12)]
                # pano_rgbs = [observations['rgb_{}'.format(i)][vp_idx] for i in range(36)]
                pano_rgbs = [self.to_PIL(pano_rgb) for pano_rgb in pano_rgbs]

                current_instruction = observations['instruction_text'][vp_idx]
                keywords = self.instruction_to_keywords[current_instruction]
                inst_id = self.instruction_to_r2r_inst_id[current_instruction]

                assert len(pano_rgbs) == 12
                # assert len(pano_rgbs) == 36

                self.viewpoints_rgbs[scan][viewpoint] = pano_rgbs

                print('Detecting {}-th viewpoint for scan {} at position {}'.format(viewpoint, scan, current_pose))
                detection = self.detect_milestone_network_no_viewpoint_http(pano_rgbs=pano_rgbs, milestones=keywords, draw_boxes=False)

                for pano_idx in range(12):
                # for pano_idx in range(36):
                    _, boxes, scores, labels = detection[pano_idx]
                    for box, score, label in zip(boxes, scores, labels):
                        pano_heading = current_heading.item() + pano_idx * math.pi / 6
                        # 转换到-pi, pi区间内
                        pano_heading %= 2 * math.pi
                        pano_heading -= math.pi
                        assert pano_heading >= -math.pi and pano_heading <= math.pi
                        self.detection_result[scan][viewpoint].append((label, score, pano_heading, box))
                new_viewpoints.append(viewpoint)
                self.scan_vp_to_visited_inst_id[(scan, viewpoint)].add(inst_id)
            else:

                new_viewpoints.append(viewpoint)

                scan = episode_info.env_names[vp_idx]
                current_pose = robot_state.pose[vp_idx]
                current_heading = robot_state.heading[vp_idx]

                current_instruction = observations['instruction_text'][vp_idx]
                inst_id = self.instruction_to_r2r_inst_id[current_instruction]
                if inst_id in self.scan_vp_to_visited_inst_id[(scan, viewpoint)]:
                    continue
                else:
                    pano_rgbs = self.viewpoints_rgbs[scan][viewpoint]

                    keywords = self.instruction_to_keywords[current_instruction]

                    assert len(pano_rgbs) == 12
                    # assert len(pano_rgbs) == 36

                    print('Detecting {}-th viewpoint for scan {} at position {}'.format(viewpoint, scan, current_pose))
                    detection = self.detect_milestone_network_no_viewpoint_http(pano_rgbs=pano_rgbs, milestones=keywords, draw_boxes=False)

                    for pano_idx in range(12):
                    # for pano_idx in range(36):
                        _, boxes, scores, labels = detection[pano_idx]
                        for box, score, label in zip(boxes, scores, labels):
                            pano_heading = current_heading.item() + pano_idx * math.pi / 6
                            # 转换到-pi, pi区间内
                            pano_heading %= 2 * math.pi
                            pano_heading -= math.pi
                            assert pano_heading >= -math.pi and pano_heading <= math.pi
                            self.detection_result[scan][viewpoint].append((label, score, pano_heading, box))
                    self.scan_vp_to_visited_inst_id[(scan, viewpoint)].add(inst_id)                
                
        return new_viewpoints

    def dump_detection_result(self, scan, viewpoint, instr_id, detection):
        key = '{}%{}%{}'.format(scan, viewpoint, instr_id)
        self.real_time_detect_db.set(key, detection)
        self.db_entry_num += 1
        if self.db_entry_num % 100 == 0:
            print('Real Time Database with {} entries'.format(self.db_entry_num))
            self.real_time_detect_db.dump()


    def get_all_detection_results_with_mask(self, scan, neighbour_mask, inst_id):
        neighbour_viewpoints = [self.scan_to_viewpoints[scan][_] for _ in neighbour_mask.nonzero()]
        all_detection_results = list()
        for neighbor in neighbour_viewpoints:
            neighbour_detection_result = list()
            for ix in range(36):
                key = '{}%{}%{}%{}'.format(scan, neighbor, inst_id, ix)
                result = self.pre_detect_result_db.get(key)
                if not result:
                    continue
                neighbour_detection_result.append(result)
            all_detection_results.append((neighbor, neighbour_detection_result))
        return all_detection_results

    def get_all_detection_results_with_mask_and_keywords(self, scan, neighbour_mask, keywords):
        # all_detection_results = self.get_all_detection_results_with_mask(scan, neighbour_mask, inst_id)
        neighbour_viewpoints = [self.scan_to_viewpoints[scan][_] for _ in neighbour_mask.nonzero()]

        neighbours_detects = list()

        for neighbour in neighbour_viewpoints:
            vp_detection = self.detection_result[scan][neighbour]
            highest_score = dict()
            highest_index = dict()
            for index, (label, score, ix, box) in enumerate(vp_detection):
                if label not in highest_score or score > highest_score[label]:
                    highest_score[label] = score
                    highest_index[label] = index
            sorted_key = sorted(highest_score, key=highest_score.get, reverse=True)[:self.vp_highest_num]
            vp_new_detect = [vp_detection[highest_index[label]] for label in sorted_key]
            neighbours_detects.append((neighbour, vp_new_detect))

        return neighbours_detects


    def get_relative_orientation(self, robot_position, target_position, robot_heading):

        x, y, z = robot_position
        x_t, y_t, z_t = target_position
        x_offset = x - x_t
        z_offset = z - z_t
        target_heading = math.atan2(x_offset, z_offset)
        rel_heading = target_heading - robot_heading
        if rel_heading > math.pi:
            rel_heading -= 2 * math.pi
        elif rel_heading <= -math.pi:
            rel_heading += 2 * math.pi
        return rel_heading
        
    def angle_feature(self, heading, angle_feat_size=4):
        return np.array(
            [math.sin(heading), math.cos(heading)] * (angle_feat_size // 4),
            dtype=np.float32)

    def viewindex_to_rel_heading(self, ix, robot_heading):
        # viewindex 0/12/24 -> pi
        # viewindex 6/18/30 -> 0
        # viewindex 3/15/27 -> -pi/2
        # viewindex 9/21/33 -> pi/2
        while ix >= 12:
            ix -= 12
        heading = ix * math.pi / 6
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading <= -math.pi:
            heading += 2 * math.pi
        heading = heading + math.pi
        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading <= -math.pi:
            heading += 2 * math.pi
        rel_heading = heading - robot_heading
        if rel_heading > math.pi:
            rel_heading -= 2 * math.pi
        elif rel_heading <= -math.pi:
            rel_heading += 2 * math.pi
        return rel_heading
    
    def abs_heading_to_rel_heading(self, heading, robot_heading):
        rel_heading = heading - robot_heading
        if rel_heading > math.pi:
            rel_heading -= 2 * math.pi
        elif rel_heading <= -math.pi:
            rel_heading += 2 * math.pi
        return rel_heading

    def get_keywords_map(self, episodes_info, observations, robot_current_state, pose_nearest_viewpoints):

        batch_keywords = list()
        batch_heading_feats = list()

        neighbour_number = list()

        for robot_idx in range(len(episodes_info.env_names)):
            scan = episodes_info.env_names[robot_idx]
            robot_pose = robot_current_state.pose[robot_idx]
            robot_elevation = robot_current_state.elevation[robot_idx]
            robot_heading = robot_current_state.heading[robot_idx]
            instruction = observations['instruction_text'][robot_idx]
            nearest_viewpoint = pose_nearest_viewpoints[robot_idx]

            assert nearest_viewpoint != -1

            candidates_position = self.scan_to_positions[scan]
            distance = torch.pow(candidates_position - robot_pose, 2)
            assert distance.shape[1] == 3
            distance[:, 1] = 0
            distance = torch.sqrt(torch.sum(distance, dim=1))
            neighbour_mask = (distance < self.neighbour_distance_threshold)

            neighbour_number.append(neighbour_mask.sum())

            keywords = None

            detection_results = self.get_all_detection_results_with_mask_and_keywords(scan, neighbour_mask, keywords)

            keywords = list()
            positions = list()
            scores = list()
            rel_headings = list()
            rel_heading_feats = list()

            for neighbour, neighbour_detect in detection_results:
                if neighbour == nearest_viewpoint:
                    for label, score, ix, _ in neighbour_detect:
                        keywords.append(label)
                        positions.append(ix)
                        scores.append(score)
                        # rel_headings.append(self.viewindex_to_rel_heading(ix, robot_heading))
                        rel_headings.append(self.abs_heading_to_rel_heading(ix, robot_heading))
                        rel_heading_feats.append(self.angle_feature(rel_headings[-1]))
                else:
                    for label, score, ix, _ in neighbour_detect:
                        keywords.append(label)
                        positions.append(neighbour)
                        scores.append(score)
                        neighbour_position = self.viewpoint_position[(scan, neighbour)]
                        rel_headings.append(self.get_relative_orientation(robot_pose, neighbour_position, robot_heading))
                        rel_heading_feats.append(self.angle_feature(rel_headings[-1]))
            filtered_keywords_index = dict()
            filtered_keywords_scores = dict()
            for kw_idx, keyword in enumerate(keywords):
                if keyword not in filtered_keywords_index or filtered_keywords_scores[keyword] < scores[kw_idx]:
                    filtered_keywords_index[keyword] = kw_idx
                    filtered_keywords_scores[keyword] = scores[kw_idx]
            keywords = [keywords[v] for k,v in filtered_keywords_index.items()]
            scores = [scores[v] for k,v in filtered_keywords_index.items()]
            rel_headings = [rel_headings[v] for k,v in filtered_keywords_index.items()]
            rel_heading_feats = [rel_heading_feats[v] for k,v in filtered_keywords_index.items()]
            if len(rel_heading_feats) > 0:
                rel_heading_feats = np.stack(rel_heading_feats, axis=0)
            else:
                rel_heading_feats = np.zeros((0, 2), dtype=np.float32)
                assert len(keywords) == 0

            batch_keywords.append(keywords)
            batch_heading_feats.append(rel_heading_feats)

        # print('average neighbour_number: ', (sum(neighbour_number) / len(neighbour_number)).item())
        return batch_keywords, batch_heading_feats


    def tokenize_keywords(self, batch_keywords):
        tokenized_keywords = list()
        for keywords in batch_keywords:
            _ = list()
            for keyword in keywords:
                if keyword in self.keywords_token_cache:
                    _.append(self.keywords_token_cache[keyword])
                else:
                    tokens = self.instruction_vocab.tokenize_and_index(keyword, keep=("'s",))
                    tokens = np.pad(np.array(tokens), (0, self.token_length-len(tokens)), 'constant')
                    assert tokens.shape[0] == self.token_length
                    _.append(tokens)
                    self.keywords_token_cache[keyword] = tokens
            if len(_) > 0:
                _ = np.stack(_, axis=0)
            else:
                _ = np.zeros((0, self.token_length), dtype=np.int64)
            tokenized_keywords.append(_)
        return tokenized_keywords
    

    def clear_memory_by_scan(self, episodes_info):
        
        for finished_i in episodes_info.finished_indices():
            scan = episodes_info.env_names[finished_i]
            
            # self.visibility[scan] = torch.zeros(len(self.scan_to_positions[scan]), dtype=torch.bool).cuda()
            self.detection_result[scan].clear()
            self.scan_to_viewpoints[scan].clear()
            self.scan_to_positions[scan] = torch.zeros((0, 3), dtype=torch.float32).cuda()
            self.viewpoints_rgbs[scan].clear()
            
            keys_to_del = list()
            for k1, k2 in self.viewpoint_position:
                if k1 == scan:
                    # del self.viewpoint_position[(k1, k2)]
                    keys_to_del.append((k1, k2))
            for k in keys_to_del:
                del self.viewpoint_position[k]
            


    def __call__(
        self,
        episodes_info: EpisodesInfo,
        observations: Observations,
        robot_current_state: RobotCurrentState,
    ):
        self.clear_memory_by_scan(episodes_info)
        # 1. update memory
        pose_nearest_viewpoints = self.update_memory(episodes_info, observations, robot_current_state)
        
        # 2. get keywords map
        keywords, rel_heading_feats = self.get_keywords_map(episodes_info, observations, robot_current_state, pose_nearest_viewpoints)

        # tokenize
        keywords_tokens = self.tokenize_keywords(keywords)

        keywords_map = {
            'keywords': keywords_tokens,
            'rel_heading_feats': rel_heading_feats,
        }
        return keywords_map

    def detect_milestone_network(self, scan_id, viewpoint_id, milestones, draw_boxes=False):
        obj = (scan_id, viewpoint_id, milestones, draw_boxes)
        data = pickle.dumps(obj)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('127.0.0.1', self.detector_port))
        self.socket.send(data)
        print('Sent object:', obj)
        data = self.socket.recv(1024 * 1024)
        obj = pickle.loads(data)
        
        return obj

    def detect_milestone_network_no_viewpoint(self, pano_rgbs, milestones, draw_boxes=False):
        obj = (pano_rgbs, milestones, draw_boxes)
        data = pickle.dumps(obj)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('127.0.0.1', self.detector_port))
        self.socket.send(data)
        print('Sent object:', obj)
        data = self.socket.recv(1024 * 1024)
        obj = pickle.loads(data)
        
        return obj

    def detect_milestone_network_no_viewpoint_http(self, pano_rgbs, milestones, draw_boxes=False):
        # Pickle the object
        assert not draw_boxes
        obj = (pano_rgbs, milestones, draw_boxes)
        data = pickle.dumps(obj)
        # Send the object as a binary data to the server
        response = requests.post('http://127.0.0.1:5000', data=data)
        # Check the status code of the response
        assert response.status_code == 200

        obj = json.loads(response.text)['detection_results']
        
        return obj