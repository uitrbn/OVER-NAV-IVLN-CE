import torch.nn as nn
import pickledb
import pickle
from collections import defaultdict
import os
import numpy as np
import math
import gzip
import json
import time

import torch

from ivlnce_baselines.common.mapping_module.mapper import EpisodesInfo, Observations, RobotCurrentState
from habitat.datasets.utils import VocabDict

class KeywordsMappingModule5:
    def __init__(self, split='train'):
        super().__init__()

        # 重要参数
        self.neighbour_distance_threshold = 7
        self.vp_highest_num = 3
        self.token_length = 20

        self.detect_db_path = 'R2R_{}_enc_detection_owlvit-large-patch14.ivlnce.db'.format(split)
        assert os.path.exists(self.detect_db_path)
        self.setup_detection_result_db()

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

        self.setup_visibility()

        self.detection_result = defaultdict(lambda: defaultdict(dict))
        self.scan_vp_to_visited_inst_id = defaultdict(set)

        # vocab -> token的映射
        self.annotation_processed_file = 'data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz'.format(split=split)
        self.setup_vocab_dict()
        
        print('Initialized keywords mapping module')

    def setup_vocab_dict(self):
        self.keywords_token_cache = dict()
        with gzip.open(self.annotation_processed_file, "rt") as f:
            deserialized = json.loads(f.read())
            self.instruction_vocab = VocabDict(
                word_list=deserialized["instruction_vocab"]["word_list"]
            )
    
    def setup_visibility(self):
        self.visibility = dict()
        for scan in self.scan_to_positions:
            self.visibility[scan] = torch.zeros(len(self.scan_to_positions[scan]), dtype=torch.bool).cuda()
        

    def setup_viewpoint_position(self):
        with open(self.viewpoint_position_file, 'rb') as f:
            self.viewpoint_position = pickle.load(f)
        self.scan_to_viewpoints = defaultdict(list)
        self.scan_to_positions = defaultdict(list)
        self.scan_viewpoints_to_index = defaultdict(dict)
        for k, v in self.viewpoint_position.items():
            self.scan_to_viewpoints[k[0]].append(k[1])
            self.scan_to_positions[k[0]].append(v)
            self.scan_viewpoints_to_index[k[0]][k[1]] = len(self.scan_to_viewpoints[k[0]]) - 1
        
        for scan in self.scan_to_viewpoints.keys():
            self.scan_to_positions[scan] = torch.from_numpy(np.array(self.scan_to_positions[scan])).cuda()

        # print([len(v) for v in self.scan_to_viewpoints.values()])
    
    def setup_instruction_to_keywords(self):
        with open(self.instruction_to_keywords_file, 'rb') as f:
            self.instruction_to_keywords = pickle.load(f)

    def setup_instruction_to_r2r_inst_id(self):
        with open(self.instruction_to_r2r_inst_id_file, 'rb') as f:
            self.instruction_to_r2r_inst_id = pickle.load(f)

    def setup_detection_result_db(self):
        self.pre_detect_result_db = pickledb.load(self.detect_db_path, False)

    def prepare_viewpoints_db(self):
        pass

    def robot_state_to_viewpoint(self, episodes_info, robot_state, distance_threshold=0.25):
        min_distance_idxs = torch.zeros(len(episodes_info.env_names), dtype=torch.long).cuda()
        min_distance_viewpoints = list()
        for robot_idx, scan in enumerate(episodes_info.env_names):
            robot_position = robot_state.pose[robot_idx]
            viewpoints_positions = self.scan_to_positions[scan]
            distance = torch.pow((robot_position - viewpoints_positions), 2)
            # ignore second dimension of distance
            distance[:, 1] = 0
            distance = torch.sum(distance, dim=1)
            assert len(distance.shape) == 1
            # min value and min index of distance
            min_value, min_distance_idxs[robot_idx] = torch.min(distance, dim=0)
            if torch.sqrt(min_value) > distance_threshold:
                min_distance_viewpoints.append(None)
            else:
                min_distance_viewpoints.append(self.scan_to_viewpoints[scan][min_distance_idxs[robot_idx]])
        return min_distance_viewpoints

    def update_memory(self, episode_info, observations, robot_state):
        viewpoints = self.robot_state_to_viewpoint(episode_info, robot_state)
        assert len(viewpoints) == len(episode_info.env_names)

        for vp_idx, viewpoint in enumerate(viewpoints):
            if viewpoint is None:
                continue
            scan = episode_info.env_names[vp_idx]
            self.visibility[scan][self.scan_viewpoints_to_index[scan][viewpoint]] = True

            current_instruction = observations['instruction_text'][vp_idx]
            inst_id = self.instruction_to_r2r_inst_id[current_instruction]

            if inst_id in self.scan_vp_to_visited_inst_id[(scan, viewpoint)]:
                continue
            self.scan_vp_to_visited_inst_id[(scan, viewpoint)].add(inst_id)

            for ix in range(36):
                key = '{}%{}%{}%{}'.format(scan, viewpoint, inst_id, ix)
                result = self.pre_detect_result_db.get(key)
                if not result:
                    continue

                _, boxes, scores, labels = result
                for box, score, label in zip(boxes, scores, labels):
                    # self.detection_result[scan][viewpoint].append((label, score, ix, box))
                    if label not in self.detection_result[scan][viewpoint]:
                        self.detection_result[scan][viewpoint][label] = (label, score, ix, box)
                    elif score > self.detection_result[scan][viewpoint][label][1]:
                        self.detection_result[scan][viewpoint][label] = (label, score, ix, box)
        
        return viewpoints

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
        neighbour_viewpoints = [self.scan_to_viewpoints[scan][_] for _ in neighbour_mask.nonzero()]

        neighbours_detects = list()

        for neighbour in neighbour_viewpoints:
            vp_detection = list(self.detection_result[scan][neighbour].values())

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

    def distance_feature(self, distance, distance_feat_size=2):
        return np.array( [distance] * distance_feat_size,dtype=np.float32)
    
    def angle_feature_vec(self, headings, angle_feat_size=4):
        headings = torch.stack(headings)
        return torch.stack([torch.sin(headings), torch.cos(headings)], 1).cpu().numpy()
    
    def distance_feature_vec(self, distances, distance_feat_size=2):
        distances = torch.stack(distances)
        return torch.stack([distances] * distance_feat_size, 1).cpu().numpy()
    
    def viewindex_to_rel_heading(self, ix, robot_heading):
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

    def get_keywords_map(self, episodes_info, observations, robot_current_state, pose_nearest_viewpoints, return_dist_feats=False):

        batch_keywords = list()
        batch_heading_feats = list()
        batch_dist_feats = list()

        neighbour_number = list()

        for robot_idx in range(len(episodes_info.env_names)):
            scan = episodes_info.env_names[robot_idx]
            robot_pose = robot_current_state.pose[robot_idx]
            robot_elevation = robot_current_state.elevation[robot_idx]
            robot_heading = robot_current_state.heading[robot_idx]
            instruction = observations['instruction_text'][robot_idx]
            nearest_viewpoint = pose_nearest_viewpoints[robot_idx]

            candidates_position = self.scan_to_positions[scan]
            distance = torch.pow(candidates_position - robot_pose, 2)
            assert distance.shape[1] == 3
            distance[:, 1] = 0
            distance = torch.sqrt(torch.sum(distance, dim=1))
            neighbour_mask = (distance < self.neighbour_distance_threshold) & (self.visibility[scan])

            neighbour_number.append(neighbour_mask.sum())

            keywords = None

            detection_results = self.get_all_detection_results_with_mask_and_keywords(scan, neighbour_mask, keywords)

            keywords = list()
            positions = list()
            scores = list()
            rel_headings = list()
            rel_heading_feats = list()
            rel_distances = list()
            rel_distance_feats = list()

            for neighbour, neighbour_detect in detection_results:
                if neighbour == nearest_viewpoint:
                    for label, score, ix, _ in neighbour_detect:
                        keywords.append(label)
                        positions.append(ix)
                        scores.append(score)
                        rel_headings.append(self.viewindex_to_rel_heading(ix, robot_heading))
                        # rel_heading_feats.append(self.angle_feature(rel_headings[-1]))

                        rel_distances.append(torch.tensor(0).cuda())
                        # rel_distance_feats.append(self.distance_feature(rel_distances[-1]))
                else:
                    for label, score, ix, _ in neighbour_detect:
                        keywords.append(label)
                        positions.append(neighbour)
                        scores.append(score)
                        neighbour_position = self.viewpoint_position[(scan, neighbour)]
                        rel_headings.append(self.get_relative_orientation(robot_pose, neighbour_position, robot_heading))
                        # rel_heading_feats.append(self.angle_feature(rel_headings[-1]))

                        dist = torch.pow(torch.tensor(neighbour_position).cuda() - robot_pose, 2)
                        dist[1] = 0
                        dist = torch.sqrt(torch.sum(dist))
                        rel_distances.append(dist)
                        # rel_distance_feats.append(self.distance_feature(rel_distances[-1]))
            
            # print('Num of keywords before filtering: ', len(keywords))

            filtered_keywords_index = dict()
            filtered_keywords_scores = dict()
            for kw_idx, keyword in enumerate(keywords):
                if keyword not in filtered_keywords_index or filtered_keywords_scores[keyword] < scores[kw_idx]:
                    filtered_keywords_index[keyword] = kw_idx
                    filtered_keywords_scores[keyword] = scores[kw_idx]
            keywords = [keywords[v] for k,v in filtered_keywords_index.items()]
            scores = [scores[v] for k,v in filtered_keywords_index.items()]

            # heading related
            rel_headings = [rel_headings[v] for k,v in filtered_keywords_index.items()]
            # rel_heading_feats = [self.angle_feature(_) for _ in rel_headings]
            # rel_heading_feats = [rel_heading_feats[v] for k,v in filtered_keywords_index.items()]
            # if len(rel_heading_feats) > 0:
            #     rel_heading_feats = np.stack(rel_heading_feats, axis=0)
            # else:
            #     rel_heading_feats = np.zeros((0, 2), dtype=np.float32)
            #     assert len(keywords) == 0
            if len(rel_headings) > 0:
                rel_heading_feats = self.angle_feature_vec(rel_headings)
            else:
                rel_heading_feats = np.zeros((0, 2), dtype=np.float32)
            
            # distance related
            rel_distances = [rel_distances[v] for k,v in filtered_keywords_index.items()]
            # # rel_distance_feats = [rel_distance_feats[v] for k,v in filtered_keywords_index.items()]
            # rel_distance_feats = [self.distance_feature(_) for _ in rel_distances]
            # if len(rel_distance_feats) > 0:
            #     rel_distance_feats = np.stack(rel_distance_feats, axis=0)
            # else:
            #     rel_distance_feats = np.zeros((0, 2), dtype=np.float32)
            #     assert len(keywords) == 0
            if len(rel_distances) > 0:
                rel_distance_feats = self.distance_feature_vec(rel_distances)
            else:
                rel_distance_feats = np.zeros((0, 2), dtype=np.float32)

            try:
                assert len(keywords) == len(rel_heading_feats) == len(rel_distance_feats)
            except:
                print('\007')
                import pdb; pdb.set_trace()

            batch_keywords.append(keywords)
            batch_heading_feats.append(rel_heading_feats)
            batch_dist_feats.append(rel_distance_feats)

            # print('Num of keywords after filtering: ', len(keywords))

        # print('average neighbour_number: ', (sum(neighbour_number) / len(neighbour_number)).item())
        # print('neighbours num: {}'.format([n.item() for n in neighbour_number]))
        # print('keywords num: {}'.format([_.shape[0] for _ in batch_heading_feats]))
        if not return_dist_feats:
            return batch_keywords, batch_heading_feats
        else:
            return batch_keywords, batch_heading_feats, batch_dist_feats


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

    def __call__(
        self,
        episodes_info: EpisodesInfo,
        observations: Observations,
        robot_current_state: RobotCurrentState,
    ):
        # time0 = time.time()
        # 1. update memory
        pose_nearest_viewpoints = self.update_memory(episodes_info, observations, robot_current_state)

        # time1 = time.time()
        
        # 2. get keywords map
        # keywords, rel_heading_feats = self.get_keywords_map(episodes_info, observations, robot_current_state, pose_nearest_viewpoints)
        keywords, rel_heading_feats, rel_distance_feats = self.get_keywords_map(episodes_info, observations, robot_current_state, pose_nearest_viewpoints, return_dist_feats=True)

        # time2 = time.time()

        # tokenize
        keywords_tokens = self.tokenize_keywords(keywords)

        # time3 = time.time()

        # print('time1: {}, time2: {}, time3: {}'.format(time1 - time0, time2 - time1, time3 - time2))

        keywords_map = {
            'keywords': keywords_tokens,
            'rel_heading_feats': rel_heading_feats,
            'rel_distance_feats': rel_distance_feats,
        }
        return keywords_map