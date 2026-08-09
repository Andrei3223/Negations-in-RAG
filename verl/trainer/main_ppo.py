# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
import os
import functools

def _validate_data_sources(config):
    """Fail immediately if a dataset carries a data_source the reward can't score.

    Without this the mismatch only surfaces when the first reward is computed --
    which is *after* a full pre-training validation pass (val_before_train=true),
    roughly an hour of rollouts on a 1.3k-row val set, and it would hit training
    just the same. Reading a single parquet column costs about a second, so the
    check is essentially free and runs before any GPU work starts.
    """
    import pandas as pd

    known = set(config.reward_model.data_sources)
    problems = {}

    for split in ('train', 'val'):
        files = config.data.get(f'{split}_files')
        if not files:
            continue
        for path in ([files] if isinstance(files, str) else list(files)):
            try:
                counts = (pd.read_parquet(path, columns=['data_source'])['data_source']
                          .value_counts().to_dict())
            except Exception as exc:  # unreadable/remote path -- warn, don't block
                print(f"[preflight] WARNING: could not read data_source from {path}: {exc}")
                continue
            summary = ', '.join(f'{k}={v}' for k, v in sorted(counts.items()))
            print(f"[preflight] {split}: {path} -> {summary}")
            unknown = {k: v for k, v in counts.items() if k not in known}
            if unknown:
                problems[path] = unknown

    if problems:
        detail = '\n'.join(f'    {p}: {d}' for p, d in problems.items())
        raise ValueError(
            "These data_source values have no reward function:\n"
            f"{detail}\n"
            f"  configured reward_model.data_sources: {sorted(known)}\n"
            "  Add them to reward_model.data_sources in the config (they will use "
            "the same scoring function and weights as every other source), e.g.:\n"
            "    reward_model.data_sources='[nq,hotpotqa,nevir-with-answers]'"
        )
    print(f"[preflight] all data_source values are scoreable ({len(known)} configured)")


def _select_rm_score_fn(data_source, reward_cfg, use_answerability=True):
    """Build the scoring function for `data_source` from the reward_model config.

    Every value comes from config now; the defaults in ppo_trainer.yaml are the
    ones this function previously hardcoded, so behaviour is unchanged.
    """
    if data_source not in reward_cfg.data_sources:
        raise NotImplementedError(
            f"No reward function for data_source={data_source!r}. "
            f"Add it to reward_model.data_sources (currently: "
            f"{list(reward_cfg.data_sources)})."
        )

    if use_answerability:
        schedule = reward_cfg.schedule
        return functools.partial(
            qa_em.compute_score_em_plus_answerability_decomposition,
            method=reward_cfg.method,
            format_score=reward_cfg.format_score,
            score=reward_cfg.score,
            add_format=reward_cfg.add_format,
            structure_format_score=reward_cfg.structure_format_score,
            final_format_score=reward_cfg.final_format_score,
            retrieval_score=reward_cfg.retrieval_score,
            decomposition_bonus=reward_cfg.decomposition_bonus,
            # Previously dropped entirely -- these fell through to the qa_em
            # defaults and could not be tuned or logged.
            alpha=schedule.alpha,
            beta=schedule.beta,
            alpha_warmup_steps=schedule.alpha_warmup_steps,
            scheduler_type=schedule.scheduler_type,
            beta_decay_steps=schedule.beta_decay_steps,
            min_beta=schedule.min_beta,
            adaptive_beta=schedule.adaptive_beta,
            ema_decay=schedule.ema_decay,
            target_em=schedule.target_em,
            ema_slope_tol=schedule.ema_slope_tol,
            adaptive_rate=schedule.adaptive_rate,
            max_beta_step=schedule.max_beta_step,
            answerability_weight=reward_cfg.intermediate.answerability_weight,
            decomposition_weight=reward_cfg.intermediate.decomposition_weight,
            answerability_first_k=reward_cfg.answerability.first_k,
            judge_batch_size=reward_cfg.judge.batch_size,
            log_sample_rate=reward_cfg.log_sample_rate,
            decomp_coeff_identical_to_parent=reward_cfg.decomposition.coeff_identical_to_parent,
            decomp_coeff_coverage=reward_cfg.decomposition.coeff_coverage,
            decomp_coeff_disimilar=reward_cfg.decomposition.coeff_dissimilar,
        )

    # Validation path: plain EM, deliberately without format shaping, so val
    # numbers stay comparable across reward configurations.
    return functools.partial(
        qa_em.compute_score_em,
        method=reward_cfg.method,
        score=reward_cfg.score,
        add_format=reward_cfg.val_add_format,
        log_sample_rate=reward_cfg.log_sample_rate,
    )


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_cfg, format_score=0., use_answerability=True) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.use_answerability = use_answerability
        self.reward_cfg = reward_cfg

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        global_steps = int(data.meta_info.get('global_steps', 0))

        # all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source,
                                                   self.reward_cfg,
                                                   use_answerability=self.use_answerability)

            # Set sample ID for logging
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            sample_id = f"SAMPLE_R{rank}_I{i}"
            qa_em.set_sample_id(sample_id)
            
            if self.use_answerability:
                score = compute_score_fn(
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    format_score=self.format_score,
                    training_step=global_steps,
                )
            else:
                score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[{sample_id}] ========== REASONING TRACE ==========")
                print(f"[{sample_id}] {sequences_str}")
                print(f"[{sample_id}] ========== END TRACE ==========")
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        return reward_tensor


import ray
import hydra


def _resolve_resume_path(config):
    """Resolve trainer.resume_from_checkpoint and repoint model.path at it.

    Accepts null/'' (fresh run), 'auto' (newest checkpoint under
    trainer.default_local_dir), or an explicit global_step_* directory.

    Weights are resumed by loading them the normal way -- model.path is rewritten
    to the checkpoint directory before the workers are built -- while the
    optimizer, LR schedule, step counter and reward EMA are restored later by
    RayPPOTrainer._resume_from_checkpoint.
    """
    from omegaconf import open_dict

    resume = config.trainer.get('resume_from_checkpoint', None)
    if not resume:
        return

    if str(resume) == 'auto':
        base = config.trainer.default_local_dir
        pointer = os.path.join(base, 'latest_checkpoint.txt')
        resolved = None
        if os.path.exists(pointer):
            with open(pointer) as f:
                candidate = f.read().strip()
            if candidate and os.path.isdir(candidate):
                resolved = candidate
        if resolved is None:
            # Fall back to the highest-numbered checkpoint directory.
            actor_dir = os.path.join(base, 'actor')
            steps = []
            if os.path.isdir(actor_dir):
                for name in os.listdir(actor_dir):
                    if name.startswith('global_step_') and name[len('global_step_'):].isdigit():
                        steps.append(int(name[len('global_step_'):]))
            if steps:
                resolved = os.path.join(actor_dir, f'global_step_{max(steps)}')
        if resolved is None:
            print(f"[resume] resume_from_checkpoint='auto' but no checkpoint found under "
                  f"{base}; starting a fresh run.")
            with open_dict(config):
                config.trainer.resume_from_checkpoint = None
            return
        resume = resolved

    if not os.path.isdir(resume):
        raise FileNotFoundError(f'trainer.resume_from_checkpoint={resume!r} is not a directory')

    with open_dict(config):
        config.trainer.resume_from_checkpoint = resume
        config.actor_rollout_ref.model.path = resume
    print(f'[resume] loading weights from {resume} (model.path repointed); optimizer, LR '
          f'schedule and step counter restored from the same directory')


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        token = os.getenv("HF_TOKEN")
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN', 'HF_TOKEN': token}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer
    from verl.utils.reward_score import qa_em

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf

    ## ADDED
    # Must run before the trainer is built: without it qa_em._get_judge_fn()
    # falls back to a different (L6) model and warns.
    _judge_cfg = config.reward_model.judge
    qa_em.set_judge_fn(
        qa_em.init_mpnet_judge(
            model_name=_judge_cfg.model_name,
            device=_judge_cfg.device,
            threshold=_judge_cfg.threshold,
        )
    )



    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    _validate_data_sources(config)
    _resolve_resume_path(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_cfg = config.reward_model
    reward_fn = RewardManager(tokenizer=tokenizer,
                              num_examine=reward_cfg.num_examine,
                              reward_cfg=reward_cfg,
                              format_score=reward_cfg.format_score,
                              use_answerability=reward_cfg.use_answerability)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer,
                                  num_examine=reward_cfg.val_num_examine,
                                  reward_cfg=reward_cfg,
                                  format_score=reward_cfg.format_score,
                                  use_answerability=reward_cfg.val_use_answerability)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
