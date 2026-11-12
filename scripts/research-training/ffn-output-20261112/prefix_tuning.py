"""Bounded CPU tuning of native attention above an immutable prefix.

The cache stores only inputs to the final physical Transformer block. That
block's attention and normalization, and the ordinary output head, remain
trainable. Feed-forward weights stay frozen by default; an explicit experiment
can additionally train its native output projection. All weights export in the
existing native format.
"""
from collections import OrderedDict
from contextlib import contextmanager
from types import MethodType

import torch
from torch.nn import functional as F


class _PrefixCaptured(Exception):
    """Internal short circuit before the original final block executes."""


class PrefixFeatureCache:
    def __init__(self, model, *, max_bytes, max_entries=4096, train_ffn_output=False):
        if (not isinstance(max_bytes, int) or isinstance(max_bytes, bool)
                or not 0 <= max_bytes <= 512 * 1024 * 1024):
            raise ValueError('invalid bounded prefix cache size')
        if (not isinstance(max_entries, int) or isinstance(max_entries, bool)
                or not 0 <= max_entries <= 16384):
            raise ValueError('invalid bounded prefix cache entries')
        self.model = model
        self.train_ffn_output = train_ffn_output
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.invalidations = 0
        self.original = model.forward
        self.configuration = dict(model.config.to_dict())
        self.last_index = len(model.blocks) - 1
        self.last_block = model.blocks[-1]
        self.tail_prefix = f'blocks.{self.last_index}.'
        self.trainable_names = tuple(name for name, p in model.named_parameters()
                                     if p.requires_grad)
        self._capturing = False
        self._captured = None
        self.versions = self._prefix_versions()
        self.hook = self.last_block.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        if self._capturing:
            # The enclosing prefix call has grad disabled. Never retain an
            # autograd graph, labels, logits or trainable tail activations.
            self._captured = args[0].detach()
            raise _PrefixCaptured()

    def _is_prefix(self, name):
        return not name.startswith((self.tail_prefix, 'final_norm.', 'lm_head.'))

    def _prefix_versions(self):
        tensors = list(self.model.named_parameters()) + list(self.model.named_buffers())
        return tuple((name, id(value), value._version, tuple(value.shape),
                      value.dtype, value.device)
                     for name, value in tensors if self._is_prefix(name))

    def _check(self):
        if (self.model.config.to_dict() != self.configuration
                or float(self.model.config.dropout) != 0
                or int(getattr(self.model.config, 'recurrent_depth', 1)) != 1
                or len(self.model.blocks) != self.last_index + 1
                or self.model.blocks[-1] is not self.last_block):
            raise RuntimeError('prefix architecture changed after configuration')
        if torch.is_autocast_enabled('cpu'):
            raise RuntimeError('prefix cache requires its configured native precision')
        actual = tuple(name for name, p in self.model.named_parameters() if p.requires_grad)
        if actual != self.trainable_names:
            raise RuntimeError('prefix tuning trainable parameter scope changed')
        if any(p.device.type != 'cpu' for p in self.model.parameters()):
            raise RuntimeError('prefix cache is restricted to deterministic CPU tuning')
        versions = self._prefix_versions()
        if versions != self.versions:
            self.entries.clear()
            self.bytes = 0
            self.versions = versions
            self.invalidations += 1

    def _features(self, ids):
        key = (str(ids.dtype), tuple(ids.shape),
               ids.detach().contiguous().cpu().numpy().tobytes())
        if key in self.entries:
            self.hits += 1
            value = self.entries.pop(key)
            self.entries[key] = value
            return value
        self.misses += 1
        self._capturing = True
        self._captured = None
        try:
            # Let the native code compute positions, embeddings, attention masks
            # and every frozen block. Interrupt at exactly the final block's
            # entrance; no second implementation of those operations exists.
            with torch.no_grad():
                try:
                    self.original(ids)
                except _PrefixCaptured:
                    pass
        finally:
            self._capturing = False
        value = self._captured
        self._captured = None
        if value is None:
            raise RuntimeError('native prefix activation was not captured')
        size = value.numel() * value.element_size() + len(key[2])
        if size <= self.max_bytes and self.max_entries:
            while self.entries and (self.bytes + size > self.max_bytes
                                    or len(self.entries) >= self.max_entries):
                old_key, old_value = self.entries.popitem(last=False)
                self.bytes -= old_value.numel() * old_value.element_size() + len(old_key[2])
            self.entries[key] = value
            self.bytes += size
        return value

    def forward(self, ids, labels=None, *, past_key_values=None, use_cache=False):
        self._check()
        # Native KV generation retains every physical layer and its exact
        # positional offsets. Only full-sequence training uses prefix features.
        if past_key_values is not None or use_cache:
            return self.original(ids, labels, past_key_values=past_key_values,
                                 use_cache=use_cache)
        if ids.ndim != 2:
            raise ValueError('input_ids must have shape [batch, time]')
        if ids.device.type != 'cpu':
            raise ValueError('native prefix inputs must remain on CPU')
        if ids.shape[1] > self.model.config.context_length:
            raise ValueError('sequence exceeds native context_length')
        if labels is not None and labels.shape != ids.shape:
            raise ValueError('labels must match input_ids shape')
        features = self._features(ids)
        hidden, _ = self.last_block(features, past_key_value=None, use_cache=False)
        logits = self.model.lm_head(self.model.final_norm(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                                   labels[:, 1:].contiguous().view(-1), ignore_index=-100)
        return dict(logits=logits, loss=loss, past_key_values=None)

    def summary(self):
        return dict(hits=self.hits, misses=self.misses, bytes=self.bytes,
                    entries=len(self.entries), invalidations=self.invalidations,
                    max_bytes=self.max_bytes, max_entries=self.max_entries)


def configure_prefix_tuning(model, *, cache_bytes=256 * 1024 * 1024, cache_entries=4096,
                            train_ffn_output=False):
    """Train final attention/norms/head; optionally its native FFN output too.

    Call this before constructing Adam. Migrate from a prior optimizer by
    explicitly starting fresh moments once; subsequent resumes must reconstruct
    this exact scope before loading the ordinary native optimizer checkpoint.
    """
    if type(train_ffn_output) is not bool:
        raise ValueError('train_ffn_output must be a literal boolean')
    existing = getattr(model, '_airi_prefix_features', None)
    if existing is not None:
        if existing.train_ffn_output != train_ffn_output:
            raise ValueError('configured FFN plasticity differs from requested training scope')
        existing._check()
        return existing
    if (hasattr(model, '_airi_head_features')
            or hasattr(model, '_airi_head_projection')):
        raise ValueError('prefix tuning cannot reuse a head-only cache or projection')
    if (float(model.config.dropout) != 0
            or int(getattr(model.config, 'recurrent_depth', 1)) != 1
            or len(model.blocks) < 2
            or any(p.device.type != 'cpu' for p in model.parameters())):
        raise ValueError('prefix tuning requires deterministic, nonrecurrent CPU blocks')
    last = model.blocks[-1]
    if any(not hasattr(last, name) for name in ('attn', 'attn_norm', 'ff_norm', 'ff')):
        raise ValueError('unsupported native final block layout')
    if train_ffn_output and not isinstance(getattr(last.ff, 'down', None), torch.nn.Linear):
        raise ValueError('FFN-output tuning requires a native linear final down projection')
    # Detach output from the frozen input embeddings without consuming RNG.
    if model.lm_head.weight is model.token_embedding.weight:
        with torch.random.fork_rng(devices=[]):
            head = torch.nn.Linear(model.config.d_model, model.config.vocab_size, bias=False)
        head = head.to(dtype=model.lm_head.weight.dtype, device=model.lm_head.weight.device)
        with torch.no_grad():
            head.weight.copy_(model.lm_head.weight)
        model.lm_head = head
    model.config.tie_embeddings = False
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    for module in (last.attn, last.attn_norm, last.ff_norm, model.final_norm, model.lm_head):
        for p in module.parameters():
            p.requires_grad_(True)
    if train_ffn_output:
        for p in last.ff.down.parameters():
            p.requires_grad_(True)
    cache = PrefixFeatureCache(model, max_bytes=cache_bytes, max_entries=cache_entries,
                               train_ffn_output=train_ffn_output)
    model._airi_prefix_features = cache

    def forward(this, ids, labels=None, *, past_key_values=None, use_cache=False):
        active = getattr(this, '_airi_prefix_features', None)
        if active is None or active.model is not this:
            raise RuntimeError('prefix cache belongs to a different native model')
        return active.forward(ids, labels, past_key_values=past_key_values, use_cache=use_cache)

    model.forward = MethodType(forward, model)
    return cache


def prefix_summary(model):
    cache = getattr(model, '_airi_prefix_features', None)
    if cache is None:
        raise ValueError('native prefix tuning has not been configured')
    cache._check()
    return dict(mode=('cached_native_final_attention_ffn_output' if cache.train_ffn_output
                      else 'cached_native_final_attention'),
                frozen_prefix_blocks=cache.last_index,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_names=list(cache.trainable_names),
                feature_cache=cache.summary(),
                scope='training strategy only; independent quality gates still required')


@contextmanager
def native_evaluation(model):
    """Use ordinary native forward during temporary evaluation configuration.

    Original gate implementations may select their historical context budget.
    Such calls cannot reuse training activations. Restore the exact current
    forward even if evaluation raises; the caller remains responsible for
    restoring any temporary configuration in its own finally block.
    """
    cache = getattr(model, '_airi_prefix_features', None)
    if cache is None or cache.model is not model:
        raise ValueError('native evaluation requires this model\'s configured prefix cache')
    cache._check()
    if cache._capturing:
        raise RuntimeError('native evaluation cannot interrupt a prefix cache fill')
    current = model.forward
    model.forward = cache.original
    try:
        yield model
    finally:
        model.forward = current


# Short public spelling used by the runner and real-checkpoint comparison.
configure_prefix = configure_prefix_tuning
