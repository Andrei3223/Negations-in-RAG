import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    # 'live' queries search_url; 'golden' / 'golden_hn' / 'golden_rn' / 'golden_hn_rn'
    # answer every <search> with the row's gold doc (+ hard and/or random negative)
    # passed to run_llm_loop as golden_docs.
    retriever_mode: str = 'live'
    # Golden modes only: a row with an empty gold_doc is searched live instead of
    # raising, so one batch can mix golden negation rows with live NQ/HotpotQA rows.
    # Off by default so a golden eval on a wrongly built parquet still fails loudly.
    live_fallback: bool = False
    # Copies of the gold doc shown in 'golden' mode. One copy makes SearchR1
    # (trained with topk=3) treat the block as unfinished and write fake docs.
    golden_copies: int = 2
    # Live retrieval: show each retrieved doc this many times (1 = normal). With
    # topk=1, 2 gives [d, d] -- avoids the single-passage failure golden_copies fixes.
    repeat_docs: int = 1

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        cut_attention_mask = new_attention_mask[:, -max_len:]

        # Position ids come from the mask AFTER the cut. Computing them first and
        # slicing left a truncated trajectory starting at a non-zero position; packed
        # flash-attention reads position_id == 0 as a sequence start, so such rows
        # either merged into their neighbour or, if every row in a micro-batch was
        # truncated, yielded zero sequences (job 347277). vLLM already treats the
        # truncated context as starting at 0, so this also makes them agree.
        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': self.tensor_fn.create_position_ids(cut_attention_mask),
            'attention_mask': cut_attention_mask
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    @staticmethod
    def _active_subset(rollings: DataProto, active_mask: torch.Tensor) -> DataProto:
        """Rows still generating, WITH the caller's meta_info.

        from_dict alone starts from an empty meta_info, which silently dropped
        _validate's do_sample=False: vllm_rollout falls back to do_sample=True, so
        every validation score until now came from T=1 sampling, not greedy decoding.
        Training's gen_batch carries no meta_info, so training is unaffected.
        """
        return DataProto.from_dict(
            {k: v[active_mask] for k, v in rollings.batch.items()},
            meta_info=dict(rollings.meta_info),
        )

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        # NOTE: do not set meta_info['recompute_log_prob'] = False here, tempting as it
        # is. The per-turn log-probs themselves are discarded, but generate_sequences
        # also writes micro_batch_size / max_token_len / use_dynamic_bsz / temperature
        # into its output meta_info, which flows through run_llm_loop into the final
        # batch. fsdp_workers.compute_log_prob (ray_trainer.fit) and update_actor read
        # those keys from there and never set them themselves -- job 347356 died with
        # KeyError: 'micro_batch_size' when this pass was skipped. The crash this pass
        # used to cause (job 347277) is fixed at its root in _update_rolling_state.
        # _validate's own recompute_log_prob=False (forwarded by _active_subset) is
        # fine: validation never calls compute_log_prob or update_actor.
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch, meta_info=dict(active_batch.meta_info))
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, golden_docs: List[Dict] = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop.

        golden_docs: in golden retriever modes, one {'gold', 'neg', 'order'} record
        per batch row, in batch order. Ignored in live mode.
        """

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Golden mode: format each row's observation once, not once per turn.
        golden_ctx = None
        if self.config.retriever_mode != 'live':
            batch_size = gen_batch.batch['input_ids'].shape[0]
            if golden_docs is None or len(golden_docs) != batch_size:
                raise ValueError(
                    f"retriever_mode={self.config.retriever_mode!r} needs one golden record per row; "
                    f"got {None if golden_docs is None else len(golden_docs)} for batch of {batch_size}. "
                    "Is data.val_files a *_docs.parquet build?")
            golden_ctx = [None if self.config.live_fallback and not (rec.get('gold') or '').strip()
                          else self._golden_passages(rec)
                          for rec in golden_docs]
            self.multi_subquery_searches = 0
            self.total_searches = 0

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = self._active_subset(rollings, active_mask)
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, golden_ctx=golden_ctx
            )

            # The only direct evidence that the oracle context reached the model.
            if golden_ctx is not None and step == 0:
                print(f"[golden:{self.config.retriever_mode}] row 0 response: {responses_str[0][-500:]!r}")
                print(f"[golden:{self.config.retriever_mode}] row 0 observation: {next_obs[0][:600]!r}")

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = self._active_subset(rollings, active_mask)
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        if golden_ctx is not None:
            # Golden mode answers every subquery with the same doc set, erasing the
            # decomposition signal; report how often that happened.
            print(f"GOLDEN_SEARCHES: {self.total_searches} total, "
                  f"{self.multi_subquery_searches} multi-subquery (collapsed to one doc set)")

        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

#     def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
#         """
#         Execute predictions across multiple environments.
#         NOTE: the function is the actual `step` function in the environment
#         NOTE penalty_for_invalid is not included in observation shown to the LLM
        
#         Args:
#             envs: List of environment instances
#             predictions: List of action predictions
#             pad_token: Token to use for padding
            
#         Returns:
#             List of observation strings
#         """
#         cur_actions, contents = self.postprocess_predictions(predictions)
#         next_obs, dones, valid_action, is_search = [], [], [], []
        
#         search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
#         if do_search:
#             search_results = self.batch_search(search_queries)
#             assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
#         else:
#             search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

#         for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
#             if not active:
#                 next_obs.append('')
#                 dones.append(1)
#                 valid_action.append(0)
#                 is_search.append(0)
#             else:
#                 if action == 'answer':
#                     next_obs.append('')
#                     dones.append(1)
#                     valid_action.append(1)
#                     is_search.append(0)
#                 elif action == 'search':
#                     next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
#                     dones.append(0)
#                     valid_action.append(1)
#                     is_search.append(1)
#                 else:
#                     next_obs.append(f'\nMy previous action is invalid. \
# If I want to search, I should put the query between <search> and </search>. \
# If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
#                     dones.append(0)
#                     valid_action.append(0)
#                     is_search.append(0)
            
#         assert len(search_results) == 0
            
#         return next_obs, dones, valid_action, is_search

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True, golden_ctx=None) -> List[str]:
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        use_golden = do_search and golden_ctx is not None

        def is_golden(i):
            # golden_ctx[i] is None only for live_fallback rows (no gold_doc).
            return use_golden and golden_ctx[i] is not None

        # Flatten subqueries of the live search rows for batch_search
        search_queries = []
        search_indices = []
        if do_search:
            for idx, (action, content) in enumerate(zip(cur_actions, contents)):
                if action == 'search' and not is_golden(idx):
                    for subquery in content:
                        search_queries.append(subquery)
                        search_indices.append(idx)

        if do_search:
            from collections import defaultdict
            grouped_results = defaultdict(list)
            if search_queries:
                for idx, result in zip(search_indices, self.batch_search(search_queries)):
                    grouped_results[idx].append(result)
            # Compacted, ascending row order, so the pop(0) consumption below and the
            # final assert hold. i is the original batch row: predictions stay
            # full-batch via _example_level_pad. Golden rows get their fixed doc set;
            # live rows join their subqueries' docs with ' ## '.
            search_results = [golden_ctx[i] if is_golden(i) else ' ## '.join(grouped_results[i])
                              for i in range(len(cur_actions)) if cur_actions[i] == 'search']
            if use_golden:
                golden_rows = [i for i in range(len(cur_actions)) if cur_actions[i] == 'search' and is_golden(i)]
                n_multi = sum(1 for i in golden_rows if len(contents[i]) > 1)
                self.total_searches = getattr(self, 'total_searches', 0) + len(golden_rows)
                self.multi_subquery_searches = getattr(self, 'multi_subquery_searches', 0) + n_multi
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        # search_results covers ALL search rows but is popped only for active ones;
        # an inactive search row would shift every later row onto the wrong result.
        assert not any(a == 'search' and not act for a, act in zip(cur_actions, active_mask)), \
            "inactive row produced a 'search' action; search_results would desync"

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
    If I want to search, I should put the query between <search> and </search>. \
    If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)

        assert len(search_results) == 0

        return next_obs, dones, valid_action, is_search

    # def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
    #     """
    #     Process (text-based) predictions from llm into actions and validity flags.
        
    #     Args:
    #         predictions: List of raw predictions
            
    #     Returns:
    #         Tuple of (actions list, validity flags list)
    #     """
    #     actions = []
    #     contents = []
                
    #     for prediction in predictions:
    #         if isinstance(prediction, str): # for llm output
    #             pattern = r'<(search|answer)>(.*?)</\1>'
    #             match = re.search(pattern, prediction, re.DOTALL)
    #             if match:
    #                 content = match.group(2).strip()  # Return only the content inside the tags
    #                 action = match.group(1)
    #             else:
    #                 content = ''
    #                 action = None
    #         else:
    #             raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
    #         actions.append(action)
    #         contents.append(content)
            
    #     return actions, contents

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[List[str]]]:
        """
        Process (text-based) predictions from llm into actions and lists of sub-queries.
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()
                    action = match.group(1)
                    # Split sub-queries by '##' if action is 'search'
                    if action == 'search':
                        subqueries = [q.strip() for q in content.split('##')]
                        content = subqueries
                    else:
                        content = [content]
                else:
                    content = []
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    # def batch_search(self, queries: List[str] = None) -> str:
    #     """
    #     Batchified search for queries.
    #     Args:
    #         queries: queries to call the search engine
    #     Returns:
    #         search results which is concatenated into a string
    #     """
    #     results = self._batch_search(queries)['result']
        
    #     return [self._passages2string(result) for result in results]

    def batch_search(self, queries: List[str] = None) -> List[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            List of search results, each as a string
        """
        results = self._batch_search(queries)['result']
        if self.config.repeat_docs < 1:
            raise ValueError(f"repeat_docs must be >= 1, got {self.config.repeat_docs}")
        if self.config.repeat_docs > 1:
            results = [[doc for doc in result for _ in range(self.config.repeat_docs)] for result in results]
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _golden_passages(self, rec: Dict) -> str:
        """Observation text for one row in golden mode, formatted exactly like live results."""
        gold, neg, order = rec.get('gold') or '', rec.get('neg') or '', rec.get('order') or 'gold,neg'
        rand = rec.get('rand') or ''
        if not gold.strip():
            raise ValueError(f"empty gold_doc in retriever_mode={self.config.retriever_mode!r}: {rec!r}")
        if self.config.retriever_mode == 'golden':
            if self.config.golden_copies < 1:
                raise ValueError(f"golden_copies must be >= 1, got {self.config.golden_copies}")
            docs = [gold] * self.config.golden_copies
        elif self.config.retriever_mode in ('golden_hn', 'golden_rn'):
            # Same slot for gold in both, so only the distractor differs between runs.
            col = 'hard_neg_doc' if self.config.retriever_mode == 'golden_hn' else 'rand_neg_doc'
            other = neg if self.config.retriever_mode == 'golden_hn' else rand
            if not other.strip():
                raise ValueError(f"retriever_mode={self.config.retriever_mode!r} but {col} is empty: {rec!r}")
            docs = [gold, other] if order == 'gold,neg' else [other, gold]
        elif self.config.retriever_mode == 'golden_hn_rn':
            # Three docs, matching the topk=3 the Search-R1 checkpoint was trained with:
            # gold, the pair's other half, and an unrelated doc, in the row's fixed order.
            order3 = rec.get('order3') or ''
            by_slot = {'gold': gold, 'neg': neg, 'rand': rand}
            slots = order3.split(',')
            if sorted(slots) != ['gold', 'neg', 'rand']:
                raise ValueError(f"retriever_mode='golden_hn_rn' needs doc_order3 (rebuild with "
                                 f"negations_golden_docs.py): {rec!r}")
            for col, text in (('hard_neg_doc', neg), ('rand_neg_doc', rand)):
                if not text.strip():
                    raise ValueError(f"retriever_mode='golden_hn_rn' but {col} is empty: {rec!r}")
            docs = [by_slot[k] for k in slots]
        else:
            raise ValueError(f"unknown retriever_mode {self.config.retriever_mode!r}")
        return self._passages2string([{'document': {'contents': c}} for c in docs])

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
