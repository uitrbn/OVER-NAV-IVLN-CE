import json
from collections import defaultdict

import lmdb
import msgpack_numpy
import numpy as np
import torch
import tqdm
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)

from ivlnce_baselines.common.env_utils import construct_envs
from ivlnce_baselines.common.utils import (
    add_batched_data_to_observations,
    batch_obs,
    extract_instruction_tokens,
)
from ivlnce_baselines.trainers.dagger_trainer import DaggerTrainer


@baseline_registry.register_trainer(name="iterative_collection_dagger")
class IterativeCollectionDaggerTrainer(DaggerTrainer):
    """Changes the _update_dataset() function to"""

    def add_map_to_observations(self, observations, batch, num_envs):
        """adds occupancy_map and semantic_map to observations from batch.
        Removes observation keys used to generate maps that are no longer
        necessary.
        """
        map_k_sum = int("occupancy_map" in batch) + int(
            "semantic_map" in batch
        )
        if map_k_sum == 1:
            raise RuntimeError(
                "either both map keys should exist in the batch or neither"
            )
        elif map_k_sum != 2:
            return observations

        for i in range(num_envs):
            for k in ["occupancy_map", "semantic_map"]:
                observations[i][k] = batch[k][i].cpu().numpy()
                observations[i][k] = batch[k][i].cpu().numpy()

            for k in [
                "semantic",
                "semantic12",
                "world_robot_pose",
                "world_robot_orientation",
                "env_name",
            ]:
                if k in observations[i]:
                    del observations[i][k]

        return observations
    
    def add_keywords_map_to_observations(self, observations, batch, num_envs):
        for i in range(num_envs):
            for k in ['keywords', 'rel_heading_feats', 'rel_distance_feats']:
                observations[i][k] = batch["keywords_map"][k][i]
            for k in ['instruction_text_for_keywords']:
                if k in observations[i]:
                    del observations[i][k]
                # if k in batch[i]:
                #     del observations[i][k]
                if k in batch:
                    del batch[k]
        return observations

    def save_episode_to_disk(self, episode, txn, lmdb_idx, expert_uuid):
        traj_obs = batch_obs(
            [step[0] for step in episode],
            device=torch.device("cpu"),
        )
        del traj_obs[expert_uuid]  # expert_uuid被删了
        for k, v in traj_obs.items():
            if k in ['keywords', 'rel_heading_feats', 'rel_distance_feats']:
                continue
            if k == 'instruction_text_for_keywords':
                continue
            traj_obs[k] = v.numpy()
            if self.config.IL.DAGGER.lmdb_fp16:
                traj_obs[k] = traj_obs[k].astype(np.float16)

        transposed_ep = [
            traj_obs,
            np.array([step[1] for step in episode], dtype=np.int64),
            np.array([step[2] for step in episode], dtype=np.int64),
        ]

        txn.put(
            str(lmdb_idx).encode(),
            msgpack_numpy.packb(transposed_ep, use_bin_type=True),
        )

    def masks_to_tensors(
        self,
        agent_episode_dones,
        sim_episode_dones,
        tour_dones,
        produce_actions,
    ):
        agent_episode_not_done_masks = torch.tensor(
            [[0] if done else [1] for done in agent_episode_dones],
            dtype=torch.uint8,
            device=self.device,
        )
        sim_episode_not_done_masks = torch.tensor(
            [[0] if done else [1] for done in sim_episode_dones],
            dtype=torch.uint8,
            device=self.device,
        )
        tour_not_done_masks = torch.tensor(
            [[0] if done else [1] for done in tour_dones],
            dtype=torch.uint8,
            device=self.device,
        )
        action_masks = torch.tensor(
            produce_actions,
            dtype=torch.uint8,
            device=self.device,
        )
        return (
            agent_episode_not_done_masks,
            sim_episode_not_done_masks,
            tour_not_done_masks,
            action_masks,
        )

    def batch_and_transform(self, observations, not_done_masks):
        """not_done_masks is used to reset maps. If tour_not_done_masks is
        used, then maps are reset only upon a new tour.
        """
        observations = extract_instruction_tokens(
            observations,
            self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
        )
        observations = add_batched_data_to_observations(
            observations, not_done_masks, "not_done_masks"
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        return batch, observations

    def _update_dataset(self, data_it: int, save_tour_idx_data: bool = False):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))

        expert_uuid = self.config.IL.DAGGER.expert_policy_sensor_uuid # shortest_path_sensor

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        ) # torch.Size([4, 2, 512])
        prev_actions = torch.zeros(
            envs.num_envs,
            1,
            device=self.device,
            dtype=torch.long,
        ) # torch.Size([4, 1])  
        agent_episode_not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )
        sim_episode_not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )
        tour_not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )
        action_masks = torch.ones(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        observations, _, _ = [list(x) for x in zip(*envs.reset())] # element keys: dict_keys(['rgb', 'depth', 'instruction', 'shortest_path_sensor', 'progress', 'world_robot_pose', 'world_robot_orientation', 'env_name'])

        batch, observations = self.batch_and_transform(
            observations, tour_not_done_masks
        )

        episodes = [[] for _ in range(envs.num_envs)]
        skips = [False for _ in range(envs.num_envs)]
        sim_episode_dones = [False for _ in range(envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        p = self.config.IL.DAGGER.p
        beta = 0.0 if p == 0.0 else p ** data_it

        ensure_unique_episodes = beta == 1.0

        def hook_builder(tgt_tensor):
            def hook(m, i, o):
                tgt_tensor.set_(o.cpu())

            return hook

        depth_features = None
        depth_hook = None
        if (
            not self.config.MODEL.DEPTH_ENCODER.trainable
            and self.config.MODEL.DEPTH_ENCODER.cnn_type
            == "VlnResnetDepthEncoder"
        ):
            # self.config.MODEL.DEPTH_ENCODER.trainable is false // self.config.MODEL.DEPTH_ENCODER.cnn_type is VlnResnetDepthEncoder
            depth_features = torch.zeros((1,), device="cpu")
            depth_hook = self.policy.net.depth_encoder.visual_encoder.register_forward_hook(
                hook_builder(depth_features)
            )

        rgb_features = None
        rgb_hook = None
        if not self.config.MODEL.RGB_ENCODER.trainable and hasattr(
            self.policy.net, "rgb_encoder"
        ):
            # self.config.MODEL.RGB_ENCODER.trainable is false // hasattr(self.policy.net, "rgb_encoder") is false
            rgb_features = torch.zeros((1,), device="cpu")
            rgb_hook = self.policy.net.rgb_encoder.cnn.register_forward_hook(
                hook_builder(rgb_features)
            )

        collected_eps = 0
        ep_ids_collected = None
        if ensure_unique_episodes:
            ep_ids_collected = {
                ep.episode_id for ep in envs.current_episodes()
            } # is a set

        with tqdm.tqdm(
            total=self.config.IL.DAGGER.update_size, dynamic_ncols=True
        ) as pbar, lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        ) as lmdb_env, torch.no_grad():
            start_id = lmdb_env.stat()["entries"] # number of data item in lmdb
            txn = lmdb_env.begin(write=True)

            tours_to_idxs = defaultdict(list) # dict from tour-id to list of imdb-idx, which indicates the episodes.
            if save_tour_idx_data: # false
                if start_id:
                    tours_to_idxs = defaultdict(
                        list, json.loads(txn.get(str(0).encode()).decode())
                    )
                else:
                    start_id += 1

            if 'PDB_DEBUG' in self.config and self.config.PDB_DEBUG:
                episodes_length_stats = []
                all_save_time = 0
                all_act_time = 0
                all_add_map_time = 0
                all_episode_update_time = 0
                all_step_time = 0
                all_batch_transform_time = 0

            while collected_eps < self.config.IL.DAGGER.update_size:
                if 'PDB_DEBUG' in self.config and self.config.PDB_DEBUG:
                    import time
                    time_start = time.time()
                current_episodes = None
                envs_to_pause = None
                if ensure_unique_episodes: # True
                    envs_to_pause = []
                    current_episodes = envs.current_episodes()
                    # sample of current episode
                    # VLNExtendedEpisode(episode_id='30204', scene_id='data/scene_datasets/mp3d/dhjEzFoUFzH/dhjEzFoUFzH.glb', start_position=[-4.386509895324707, 0.04884999990463257, -47.68450164794922], start_rotation=[-0.0, 0.7071067811865475, 0.0, 0.7071067811865476], info={'geodesic_distance': 30.170230865478516}, _shortest_path_cache=None, start_room=None, shortest_paths=None, goals=[NavigationGoal(position=[-4.654379844665527, -0.17570720613002777, -32.14699935913086], radius=3.0)], reference_path=[[-4.386509895324707, 0.04884999990463257, -47.68450164794922], [-4.608240127563477, 0.04884999990463257, -45.98040008544922], [-5.766359806060791, -0.06898114830255508, -42.70869827270508], [-5.857450008392334, -0.3511500358581543, -39.772499084472656], [-5.911399841308594, -0.3511500358581543, -37.53739929199219], [-5.8490400314331055, -0.3511500358581543, -35.826900482177734], [-4.654379844665527, -0.17570720613002777, -32.14699935913086]], instruction=InstructionData(instruction_text='walk forward and turn left . walk forward and stop at the column .', instruction_tokens=[2384, 915, 103, 2300, 1251, 9, 2384, 915, 103, 2104, 160, 2202, 501, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), trajectory_id='30204', tour_id='213')

                # when a sim episode is done, save it to disk.
                for i in range(envs.num_envs):
                    if not sim_episode_dones[i]: # only execute when sim episode is done
                        continue

                    if skips[i]:
                        episodes[i] = []
                        continue

                    if 'PDB_DEBUG' in self.config and self.config.PDB_DEBUG:
                        episodes_length_stats.append(len(episodes[i]))
                        print('Current episodes length stats: {}'.format(len(episodes[i])))
                        if len(episodes_length_stats) % 100 == 0:
                            from collections import Counter
                            print(Counter(episodes_length_stats))

                    lmdb_idx = start_id + collected_eps
                    self.save_episode_to_disk(
                        episodes[i], txn, lmdb_idx, expert_uuid
                    )
                    tour_id = str(episodes[i][0][3])
                    tours_to_idxs[tour_id].append(lmdb_idx)
                    collected_eps += 1  # noqa: SIM113
                    pbar.update()

                    txn.commit()
                    txn = lmdb_env.begin(write=True)

                    if ensure_unique_episodes:
                        if current_episodes[i].episode_id in ep_ids_collected:
                            envs_to_pause.append(i)
                        else:
                            ep_ids_collected.add(
                                current_episodes[i].episode_id
                            )

                    episodes[i] = []

                if 'PDB_DEBUG' in self.config and self.config.PDB_DEBUG:
                    time_after_save = time.time()

                if ensure_unique_episodes:
                    (
                        envs,
                        rnn_states,
                        agent_episode_not_done_masks,
                        sim_episode_not_done_masks,
                        tour_not_done_masks,
                        action_masks,
                        prev_actions,
                        batch,
                        _,
                    ) = self._pause_iterative_envs(
                        envs_to_pause,
                        envs,
                        rnn_states,
                        agent_episode_not_done_masks,
                        sim_episode_not_done_masks,
                        tour_not_done_masks,
                        action_masks,
                        prev_actions,
                        batch,
                    )
                    if envs.num_envs == 0:
                        break

                actions, rnn_states = self.policy.act_iterative(
                    batch,
                    rnn_states,
                    prev_actions,
                    agent_episode_not_done_masks,
                    sim_episode_not_done_masks,
                    tour_not_done_masks,
                    action_masks,
                    deterministic=False,
                )
                actions = torch.where(
                    torch.rand_like(actions, dtype=torch.float) < beta,
                    batch[expert_uuid].long(),
                    actions,
                )

                if 'PDB_DEBUG' in self.config and self.config.PDB_DEBUG:
                    time_after_act = time.time()

                if 'KEYWORDS_MAP' in self.config and self.config.KEYWORDS_MAP:
                    observations = self.add_keywords_map_to_observations(
                        observations, batch, envs.num_envs
                    )

                    if 'PDB_DEBUG' in self.config and self.config.PDB_DEBUG:
                       keyword_num_sum = sum([len(observation['keywords']) for observation in observations])
                       if collected_eps % 10 == 0:
                           print('Keyword num sum for current iteration: ', keyword_num_sum)

                observations = self.add_map_to_observations(
                    observations, batch, envs.num_envs
                )

                if 'PDB_DEBUG' in self.config and self.config.PDB_DEBUG:
                    time_after_add_map = time.time()

                # envs.current_episodes()似乎永远是4
                assert len(envs.current_episodes()) == self.config.NUM_ENVIRONMENTS
                for i, current_episode in enumerate(envs.current_episodes()):
                    # only add steps to lmdb if the agent is acting: skip oracle phases
                    if not action_masks[i]:
                        continue

                    if depth_features is not None:
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]

                    if rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]

                    if "rgb" in observations[i]:
                        del observations[i]["rgb"]

                    episodes[i].append(
                        (
                            observations[i], # dict_keys(['instruction', 'shortest_path_sensor', 'progress', 'not_done_masks', 'occupancy_map', 'semantic_map', 'depth_features'])
                            prev_actions[i].item(),
                            batch[expert_uuid][i].item(),
                            current_episode.tour_id,
                        )
                    ) 

                skips = batch[expert_uuid].long() == -1
                actions = torch.where(
                    skips, torch.zeros_like(actions), actions
                )
                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True)
                prev_actions.copy_(actions)
                
                outputs = envs.step([a[0].item() for a in actions]) 

                (
                    observations,
                    _,
                    agent_episode_dones,
                    sim_episode_dones,
                    tour_dones,
                    produce_actions,
                    _,
                ) = [list(x) for x in zip(*outputs)]

                if 'PDB_DEBUG' in self.config and self.config.PDB_DEBUG:
                    time_after_step = time.time()

                (
                    agent_episode_not_done_masks,
                    sim_episode_not_done_masks,
                    tour_not_done_masks,
                    action_masks,
                ) = self.masks_to_tensors(
                    agent_episode_dones,
                    sim_episode_dones,
                    tour_dones,
                    produce_actions,
                )

                batch, observations = self.batch_and_transform(
                    observations, tour_not_done_masks
                )

            if save_tour_idx_data:
                txn.put(
                    str(0).encode(),
                    msgpack_numpy.packb(
                        json.dumps(tours_to_idxs).encode(), use_bin_type=True
                    ),
                    overwrite=True,
                )
                txn.commit()

        envs.close()
        envs = None

        if depth_hook is not None:
            depth_hook.remove()
        if rgb_hook is not None:
            rgb_hook.remove()

        if save_tour_idx_data:
            return tours_to_idxs
        return None
