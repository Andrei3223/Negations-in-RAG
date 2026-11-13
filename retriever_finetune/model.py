"""Retriever encoders.

E5Encoder: the conventions of subsearch/search/retrieval_server.py, "query: " /
"passage: " prefixes, mean pooling over the attention mask, L2 norm (cosine scores).
SpladeEncoder: SPLADE sparse vectors, dot-product scores (subsearch/search/splade.py).
"""
import copy
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class E5Encoder(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path, use_fast=True)
        self.model = AutoModel.from_pretrained(cfg.name_or_path)
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        # Frozen copy of the starting weights for passages, so an existing index
        # stays valid. None = one shared encoder.
        self.passage_model = None
        if cfg.freeze_passage_encoder:
            self.passage_model = copy.deepcopy(self.model)
            self.passage_model.requires_grad_(False)
        self.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        if self.passage_model is not None:
            self.passage_model.eval()   # frozen: no dropout either
        return self

    def _embed(self, model, texts, prefix, max_length):
        batch = self.tokenizer([prefix + t for t in texts], max_length=max_length,
                               padding=True, truncation=True, return_tensors='pt').to(self.device)
        hidden = model(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1)
        emb = hidden.masked_fill(mask == 0, 0.0).sum(dim=1) / mask.sum(dim=1)
        return F.normalize(emb.float(), dim=-1)

    def encode_queries(self, texts):
        return self._embed(self.model, texts, self.cfg.query_prefix, self.cfg.query_max_length)

    def encode_passages(self, texts):
        model = self.passage_model if self.passage_model is not None else self.model
        return self._embed(model, texts, self.cfg.passage_prefix, self.cfg.passage_max_length)

    def autocast(self, bf16):
        if bf16 and self.device.type == 'cuda':
            return torch.autocast('cuda', dtype=torch.bfloat16)
        return nullcontext()

    @torch.no_grad()
    def embed_all(self, texts, is_query, batch_size, bf16):
        """Embeddings for any number of texts, without gradients; restores train mode."""
        was_training = self.training
        self.eval()
        encode = self.encode_queries if is_query else self.encode_passages
        out = []
        with self.autocast(bf16):
            for i in range(0, len(texts), batch_size):
                out.append(encode(texts[i:i + batch_size]))
        self.train(was_training)
        return torch.cat(out)

    def save(self, path):
        """HF format: retrieval_server.py --retriever_model <path> --retriever_name e5."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class SpladeEncoder(torch.nn.Module):
    """SPLADE (subsearch/search/splade.py): texts become vocabulary-sized sparse weight
    vectors, returned dense, so q @ d.T is the SPLADE dot-product score. No prefixes,
    no normalisation. One shared MLM model for queries and passages, as in SPLADE-v3.
    Trainable: train.py adds the FLOPS sparsity regulariser (model.flops)."""

    sparse = True   # evaluators log active terms per text

    def __init__(self, cfg, device):
        super().__init__()
        from transformers import AutoModelForMaskedLM
        assert not cfg.get('freeze_passage_encoder', False), 'SPLADE trains one shared encoder'
        self.cfg = cfg
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(cfg.name_or_path)
        if cfg.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.to(self.device)

    def _embed(self, texts, max_length):
        from subsearch.search.splade import splade_pool
        batch = self.tokenizer(list(texts), max_length=max_length, padding=True,
                               truncation=True, return_tensors='pt').to(self.device)
        return splade_pool(self.model(**batch).logits, batch['attention_mask'])

    def encode_queries(self, texts):
        return self._embed(texts, self.cfg.query_max_length)

    def encode_passages(self, texts):
        return self._embed(texts, self.cfg.passage_max_length)

    def autocast(self, bf16):
        if bf16 and self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
            return torch.autocast('cuda', dtype=torch.bfloat16)
        return nullcontext()

    embed_all = E5Encoder.embed_all

    def save(self, path):
        """HF format (safetensors): splade_index_builder.py --model <path>, then
        retrieval_server.py --retriever_name splade --retriever_model <path>."""
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)


def build_encoder(cfg, device):
    kind = cfg.get('type', 'e5')
    if kind == 'e5':
        return E5Encoder(cfg, device)
    if kind == 'splade':
        return SpladeEncoder(cfg, device)
    raise ValueError(f'unknown model.type {kind!r}')
