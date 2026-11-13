"""Fine-tune e5 on negation minimal pairs with the two-stage Plackett-Luce loss, with
optional MS MARCO replay (InfoNCE) mixed into every batch.

    python -m retriever_finetune.train [hydra overrides]

Config: retriever_finetune/conf/config.yaml. SLURM: retriever_finetune/train_e5.job.
"""
import json
import logging
import math
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import get_linear_schedule_with_warmup

from .data import (check_no_leakage, families_by_text, load_marco, load_pairs,
                   mine_negatives, train_batches)
from .loss import info_nce_loss, plackett_luce_loss
from .metrics import Evaluator, MarcoEvaluator
from .model import build_encoder
from subsearch.search.splade import flops

log = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)   # biases, LayerNorm
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]


def log_metrics(metrics, prefix, step):
    wandb.log({f'{prefix}/{k}': v for k, v in metrics.items()}, step=step)
    keys = ['all/pairwise_acc', 'all/inbatch_ndcg@10', 'all/pool_ndcg@10', 'all/pool_recall@2',
            'all/pool_recall@3',
            'nevir/pairwise_acc', 'nfs/pairwise_acc', 'marco/pool_ndcg@10', 'marco/pool_mrr@10',
            'marco/inbatch_ndcg@10', 'marco_drop', 'all/terms_per_query', 'all/terms_per_doc']
    log.info(f'[{prefix} step {step}] ' + '  '.join(
        f'{k}={metrics[k]:.4f}' for k in keys if k in metrics))


@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    tr = cfg.trainer
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if tr.device == 'auto' else tr.device
    out_dir = Path(tr.output_dir)
    log.info(f'device={device}  output_dir={out_dir}')

    train_pairs = load_pairs(cfg.data.train_files)
    val_pairs = load_pairs(cfg.data.val_files)
    test_pairs = load_pairs(cfg.data.test_files)
    check_no_leakage(train_pairs, val_pairs, test_pairs)
    for name, pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        by_source = {s: sum(p.source == s for p in pairs) for s in sorted({p.source for p in pairs})}
        log.info(f'{name}: {len(pairs)} pairs / {2 * len(pairs)} queries {by_source}')

    use_marco = cfg.marco.enabled
    marco_train = marco_eval = None
    if use_marco:
        marco_dir = Path(cfg.marco.data_dir)
        assert (marco_dir / 'train.parquet').exists(), \
            f'{marco_dir} not built: run python -m retriever_finetune.prepare_marco'
        marco_train = load_marco(marco_dir / 'train.parquet')
        marco_val = load_marco(marco_dir / 'val.parquet')
        log.info(f'marco: {len(marco_train)} train queries vs {2 * len(train_pairs)} negation '
                 f'queries ({len(marco_train) / (2 * len(train_pairs)):.2f}:1), '
                 f'{len(marco_val)} val queries')
        if len(marco_train) != 2 * len(train_pairs):
            log.warning('MARCO and negation query counts differ: batches are not 1:1')
        assert tr.eval_before_train, 'MARCO checkpoint selection needs the step-0 baseline'

    encoder = build_encoder(cfg.model, device)
    # Score temperature: e5 cosines are divided by loss.tau (0.01); SPLADE dot products
    # are already on a logit scale, so its model config sets tau: 1.0.
    tau = cfg.model.get('tau', cfg.loss.tau)
    flops_cfg = cfg.model.get('flops')   # SPLADE only: sparsity regulariser
    log.info(f'model.type={cfg.model.get("type", "e5")}  tau={tau}  flops={flops_cfg}')
    val_eval = Evaluator(val_pairs, cfg.eval)
    test_eval = Evaluator(test_pairs, cfg.eval) if tr.test_at_end or tr.eval_only else None
    custom_eval = None
    if tr.eval_only and cfg.eval.get('custom_corpus'):
        with open(cfg.eval.custom_corpus) as f:
            corpus = [json.loads(line)['contents'] for line in f]
        custom_eval = Evaluator(test_pairs, cfg.eval, corpus=corpus)
        log.info(f'test_custom: {2 * len(test_pairs)} test queries vs {len(corpus)} docs '
                 f'of {cfg.eval.custom_corpus}')
    if use_marco:
        marco_eval = MarcoEvaluator(marco_val, cfg.eval)

    def evaluate_val():
        metrics = val_eval.evaluate(encoder, tr.bf16)
        if marco_eval is not None:
            metrics.update(marco_eval.evaluate(encoder, tr.bf16))
        return metrics

    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.experiment_name,
               mode=cfg.wandb.mode, tags=list(cfg.wandb.tags),
               dir=HydraConfig.get().runtime.output_dir,
               config=OmegaConf.to_container(cfg, resolve=True))
    wandb.config.update({'n_train_pairs': len(train_pairs), 'n_val_pairs': len(val_pairs),
                         'n_test_pairs': len(test_pairs),
                         'n_marco_train': len(marco_train) if use_marco else 0,
                         'n_marco_val': len(marco_val) if use_marco else 0})
    results = {'experiment_name': cfg.experiment_name}

    if tr.eval_only:
        results['val'] = evaluate_val()
        log_metrics(results['val'], 'val', 0)
        results['test'] = test_eval.evaluate(encoder, tr.bf16)
        log_metrics(results['test'], 'test', 0)
        if custom_eval is not None:
            results['test_custom'] = custom_eval.evaluate(encoder, tr.bf16)
            log_metrics(results['test_custom'], 'test_custom', 0)
        _write_results(out_dir, cfg, results)
        wandb.finish()
        return

    # --- Negatives ------------------------------------------------------------------
    num_mined = cfg.negatives.num_mined
    mined = None
    if num_mined:
        t0 = time.time()
        mined = mine_negatives(train_pairs, encoder, cfg.negatives.mine_top_k,
                               cfg.eval.encode_batch_size, tr.bf16)
        assert num_mined <= cfg.negatives.mine_top_k, 'num_mined > mine_top_k'
        log.info(f'mined top-{cfg.negatives.mine_top_k} negatives for {len(mined)} queries '
                 f'in {time.time() - t0:.0f}s')
    text_families = families_by_text(train_pairs)

    # --- Optimiser ------------------------------------------------------------------
    steps_per_epoch = math.ceil(len(train_pairs) / tr.batch_pairs)
    total_steps = tr.epochs * steps_per_epoch
    if tr.max_steps > 0:
        total_steps = min(total_steps, tr.max_steps)
    optim = torch.optim.AdamW(param_groups(encoder, cfg.optim.weight_decay),
                              lr=cfg.optim.lr, betas=tuple(cfg.optim.betas))
    sched = get_linear_schedule_with_warmup(optim, int(cfg.optim.warmup_ratio * total_steps),
                                            total_steps)
    params = [p for p in encoder.parameters() if p.requires_grad]
    log.info(f'{steps_per_epoch} steps/epoch, {total_steps} steps, '
             f'{sum(p.numel() for p in params) / 1e6:.0f}M trainable params')

    # --- Best-checkpoint tracking (kept on CPU, written once at the end) --------------
    # With MARCO on, a step is eligible only if its MARCO metric is at most
    # marco_max_drop below step 0; step 0 itself always is, so the base model is the
    # fallback when every trained step forgets too much.
    best = {'key': None, 'step': None, 'val': None, 'state': None}
    marco_key = tr.marco_monitor.removeprefix('val/')
    marco_base = {}

    def consider(metrics, step):
        if use_marco:
            marco_base.setdefault('value', metrics[marco_key])
            metrics['marco_drop'] = marco_base['value'] - metrics[marco_key]   # logged as val/marco_drop
            if metrics['marco_drop'] > tr.marco_max_drop:
                log.info(f'step {step} not eligible: {marco_key} dropped '
                         f'{metrics["marco_drop"]:.4f} > {tr.marco_max_drop}')
                return
        key = (metrics[tr.monitor.removeprefix('val/')],
               metrics[tr.monitor_tiebreak.removeprefix('val/')])
        if best['key'] is None or key > best['key']:
            best.update(key=key, step=step, val=metrics,
                        state={k: v.detach().cpu().clone() for k, v in encoder.model.state_dict().items()})
            wandb.run.summary['best_step'] = step
            wandb.run.summary.update({f'best/val/{k}': v for k, v in metrics.items()})

    def validate(step):
        metrics = evaluate_val()
        consider(metrics, step)
        log_metrics(metrics, 'val', step)

    if tr.eval_before_train:
        validate(0)

    # --- Train ----------------------------------------------------------------------
    rng = random.Random(cfg.seed)
    step, last_eval = 0, 0
    encoder.train()
    for epoch in range(tr.epochs):
        for batch in train_batches(train_pairs, tr.batch_pairs, rng, mined, num_mined, text_families,
                                   marco_train, cfg.marco.get('hard_negatives', 0)):
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with encoder.autocast(tr.bf16):
                q = encoder.encode_queries(batch.queries)
                d = encoder.encode_passages(batch.docs)
            cos = q @ d.T   # cosine for e5, raw dot product for SPLADE
            pos, partner, mask = (batch.pos.to(device), batch.partner.to(device),
                                  batch.mask.to(device))
            scores, n = cos / tau, batch.n_neg   # rows [0, n) negation, [n, Q) MARCO
            loss_neg, stage1, stage2 = plackett_luce_loss(scores[:n], pos[:n], partner[:n],
                                                          mask[:n], cfg.loss.lam)
            loss = loss_neg
            if n < len(pos):
                loss_marco = info_nce_loss(scores[n:], pos[n:])
                loss = loss_neg + cfg.marco.loss_weight * loss_marco
            if flops_cfg is not None:
                ramp = min(1.0, step / max(1, flops_cfg.warmup_ratio * total_steps)) ** 2
                lam_q, lam_d = ramp * flops_cfg.lambda_q, ramp * flops_cfg.lambda_d
                flops_q, flops_d = flops(q), flops(d)
                loss = loss + lam_q * flops_q + lam_d * flops_d
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.optim.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1

            with torch.no_grad():
                rows = torch.arange(n, device=cos.device)
                margin = cos[rows, pos[:n]] - cos[rows, partner[:n]]
                pair_acc = (margin > 0).view(-1, 2).all(dim=1).float().mean()
            stats = {'train/loss': loss.item(), 'train/loss_neg': loss_neg.item(),
                     'train/loss_stage1': stage1.item(),
                     'train/loss_stage2': stage2.item(), 'train/pairwise_acc': pair_acc.item(),
                     'train/margin_own_minus_partner': margin.mean().item(),
                     'train/cos_own': cos[rows, pos[:n]].mean().item(),
                     'train/lr': sched.get_last_lr()[0], 'train/grad_norm': grad_norm.item(),
                     'train/epoch': epoch + 1, 'perf/n_candidates': len(batch.docs),
                     'perf/step_time': time.time() - t0}
            if flops_cfg is not None:
                stats.update({'train/flops_q': flops_q.item(), 'train/flops_d': flops_d.item(),
                              'train/lambda_q': lam_q, 'train/lambda_d': lam_d,
                              'train/terms_per_query': (q > 0).sum(1).float().mean().item(),
                              'train/terms_per_doc': (d > 0).sum(1).float().mean().item()})
            if n < len(pos):
                stats['train/loss_marco'] = loss_marco.item()
                stats['train/marco_acc@1'] = (scores[n:].argmax(1) == pos[n:]).float().mean().item()
                stats['perf/n_marco_queries'] = len(pos) - n
            if device == 'cuda':
                stats['perf/max_mem_gb'] = torch.cuda.max_memory_allocated() / 2**30
            wandb.log(stats, step=step)
            if step % 10 == 0 or step == 1:
                log.info(f'step {step}/{total_steps} loss={stats["train/loss"]:.4f} '
                         f'(s1={stats["train/loss_stage1"]:.4f} s2={stats["train/loss_stage2"]:.4f}'
                         + (f' marco={stats["train/loss_marco"]:.4f}' if 'train/loss_marco' in stats else '')
                         + (f' flops_q={stats["train/flops_q"]:.3f} flops_d={stats["train/flops_d"]:.3f}'
                            f' terms q/d={stats["train/terms_per_query"]:.0f}/{stats["train/terms_per_doc"]:.0f}'
                            if 'train/flops_q' in stats else '')
                         + f') train_pair_acc={stats["train/pairwise_acc"]:.3f} '
                         f'cands={len(batch.docs)} {stats["perf/step_time"]:.2f}s'
                         + (f' mem={stats["perf/max_mem_gb"]:.1f}GB' if device == 'cuda' else ''))

            if step % tr.eval_every == 0:
                validate(step)
                last_eval = step
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    if last_eval != step:
        validate(step)

    # --- Save, then test the best checkpoint -----------------------------------------
    results.update(best_step=best['step'], last_step=step, val_best=best['val'])
    if tr.save_checkpoints:
        encoder.save(out_dir / 'last')
    encoder.model.load_state_dict(best['state'])
    if tr.save_checkpoints:
        encoder.save(out_dir / 'best')
        log.info(f'saved best (step {best["step"]}) and last (step {step}) to {out_dir}')
    if tr.test_at_end:
        results['test_best'] = test_eval.evaluate(encoder, tr.bf16)
        log_metrics(results['test_best'], 'test', step)
        wandb.run.summary.update({f'best/test/{k}': v for k, v in results['test_best'].items()})
    _write_results(out_dir, cfg, results)
    wandb.finish()


def _write_results(out_dir, cfg, results):
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / 'config.yaml', resolve=True)
    (out_dir / 'metrics.json').write_text(json.dumps(results, indent=2))
    log.info(f'wrote {out_dir}/metrics.json')


if __name__ == '__main__':
    main()
