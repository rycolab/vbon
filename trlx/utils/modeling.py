import functools
from typing import Dict, MutableMapping, Tuple, Union

import accelerate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
import logging

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def make_head(n_embd: int, out: int, dtype: type = torch.float32) -> nn.Sequential:
    """Returns a generic sequential MLP head."""
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2, dtype=dtype),
        nn.ReLU(),
        nn.Linear(n_embd * 2, out, dtype=dtype),
    )


def freeze_bottom_causal_layers(model: transformers.PreTrainedModel, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)

    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
        hidden_layers_to_freeze += [model.get_input_embeddings(), model.get_output_embeddings()]
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
        hidden_layers_to_freeze += [model.get_input_embeddings()]
        if model.config.tie_word_embeddings:
            hidden_layers_to_freeze += [model.get_output_embeddings()]
    else:
        hidden_layers_to_freeze = []

    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)


def freeze_bottom_seq2seq_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    if num_layers_unfrozen == -1:
        return
    shared_embed = model.shared
    decoder_embed = model.decoder.embed_tokens
    encoder_blocks = model.encoder.block
    encoder_norm_layer = model.encoder.final_layer_norm
    decoder_norm_layer = model.decoder.final_layer_norm
    decoder_blocks = model.decoder.block[:-num_layers_unfrozen]
    blocks_to_freeze = (
        list(encoder_blocks)
        + list(decoder_blocks)
        + [shared_embed]
        + [encoder_norm_layer]
        + [decoder_norm_layer]
        + [decoder_embed]
    )
    for block in blocks_to_freeze:
        block.requires_grad_(False)


def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args) -> object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs: Tuple[str]) -> Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")


def hf_get_decoder(model: nn.Module) -> nn.Module:
    """Returns the causal decoder backbone of the specified HuggingFace transformers
    model.
    NOTE: Different model configurations have different causal decoder attribute
    names.
        - transformer: (GPT2LMHeadModel, GPTJConfig)
        - model.decoder: (OPTConfig, BloomConfig)
        - gpt_neox: (GPTNeoXConfig)
    """
    decoder_attrs = ("transformer", "model.decoder", "gpt_neox", "decoder")
    return findattr(model, decoder_attrs)


def hf_get_decoder_final_norm(model: nn.Module) -> float:
    """Returns the final (layer) norm of the specified decoder.
    NOTE: Different model configurations have different final norm attribute names.
        - transformer.ln_f: (GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.final_layer_norm: (OPTForCausalLM)
        - gpt_neox.layers.final_layer_norm: (GPTNeoXForCausalLM)
    """
    norm_attrs = (
        "transformer.ln_f",
        "model.decoder.final_layer_norm",
        "model.norm",
        "decoder.final_layer_norm",
        "gpt_neox.final_layer_norm",
    )
    return findattr(model, norm_attrs)


def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
    """
    hidden_layers_attrs = (
        "h",
        "layers",
        "model.layers",
        "decoder.layers",
        "transformer.h",
        "transformer.blocks",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
    )
    return findattr(model, hidden_layers_attrs)


def hf_get_lm_head(model: nn.Module) -> nn.Module:
    """Returns the language modeling (lm) head of the specified HuggingFace
    transformers model.
    NOTE: Different model configurations have different `lm_head` attribute names.
        - lm_head: (GPT2LMHeadModel, BloomForCausalLM)
        - embed_out: (GPTNeoXForCausalLM)
    """
    return model.get_output_embeddings()


def hf_get_hidden_size(config: transformers.PretrainedConfig) -> int:
    """Returns the hidden layer dimensionality of the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different hidden size attribute names.
        - hidden_size: (OPTConfig, BloomConfig)
        - n_embd: (GPT2Config, GPTJConfig)
        - d_model: (PegasusConfig, XLNetConfig)
    """
    hidden_size_attrs = ("hidden_size", "n_embd", "d_model")
    return findattr(config, hidden_size_attrs)


def hf_get_num_hidden_layers(config: transformers.PretrainedConfig) -> int:
    """Returns the number of hidden layers in the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different number-of-layers attribute
    names.
        - num_hidden_layers: (GPTNeoXConfig, OPTConfig)
        - n_layer: (GPT2Config, GPTJConfig, BloomConfig)
    """
    num_hidden_layers_attrs = ("num_hidden_layers", "n_layer")
    return findattr(config, num_hidden_layers_attrs)


def get_global_statistics(xs: torch.Tensor, group=None) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    dist.all_reduce(sum_and_count, dist.ReduceOp.SUM, group=group)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean) ** 2)
    dist.all_reduce(sum_var, dist.ReduceOp.SUM, group=group)
    global_var = sum_var / count
    return global_mean, global_var, count


def whiten(xs: torch.Tensor, shift_mean=True, distributed=True, group=None) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs, group=group)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def calc_stepwise_kl(logits_p, logits_q, temperature=1.):
    logits_pt = logits_p / temperature + 1e-7
    logits_qt = logits_q / temperature + 1e-7

    fullp = logits_pt.log_softmax(-1) 
    fullq = logits_qt.log_softmax(-1)

    return torch.sum(torch.exp(fullp) * (fullp - fullq), -1)

def logprobs_of_labels(logits, labels, temperature=1.):
    """Log probabilities of the labels

    These are calculated from the logits."""
    logits_t = logits / temperature + 1e-7
    logprobs = F.log_softmax(logits_t, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)

@np.vectorize
def logsubexp(x, y):
    """
    Numerically stable computation of subtraction in log-space
    z = log(exp(x) - exp(y))
    """
    if x == y:
        return -np.inf
    elif x < y:
        return np.nan
    else:
        return x + log1mexp(y-x)


@np.vectorize
def log1mexp(x):
    """
    Numerically stable implementation of log(1-exp(x))

    Note: function is finite for x < 0.

    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x >= 0:
        return np.nan
    else:
        a = abs(x)
        if 0 < a <= 0.693:
            return np.log(-np.expm1(-a))
        else:
            return np.log1p(-np.exp(-a))


def samples_to_tokens(samples, tokenizer):
    outputs = tokenizer(samples).input_ids
    
    outputs = list(map(torch.LongTensor, outputs))
    maxsize = max(map(len, outputs))
    outputs = [
        F.pad(
            output,
            (0, maxsize - len(output)),
            value=tokenizer.pad_token_id,
        )
        for output in outputs
    ]
    if torch.cuda.is_available():
        sample_outputs = torch.vstack(outputs).to("cuda")
    else:
        sample_outputs = torch.vstack(outputs)
    return sample_outputs

def calc_forward_kl(bon_data, prompts, model, tokenizer, ref_model):
    samples = []
    for prompt in prompts:
        samples.append(bon_data[prompt]['responses'][0])

    tokens = samples_to_tokens(samples, tokenizer)
    attention_mask = tokens.not_equal(tokenizer.pad_token_id).long().to(tokens.device)
    logprobs = calc_logprobs(tokens, model, tokenizer)
    ref_logprobs = calc_logprobs(tokens, ref_model, tokenizer)

    log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
    # log_ratio = 0.5 * (log_ratio ** 2)
    log_ratio = ((torch.exp(log_ratio) - 1) - log_ratio)
    return log_ratio.sum(1).detach().cpu()

def calc_kl(samples, model, tokenizer, ref_model):
    tokens = samples_to_tokens(samples, tokenizer)
    attention_mask = tokens.not_equal(tokenizer.pad_token_id).long().to(tokens.device)
    logprobs = calc_logprobs(tokens, model, tokenizer)
    ref_logprobs = calc_logprobs(tokens, ref_model, tokenizer)

    log_ratio = (ref_logprobs - logprobs) * attention_mask[:, :-1]
    # log_ratio = 0.5 * (log_ratio ** 2)
    log_ratio = ((torch.exp(log_ratio) - 1) - log_ratio)
    return log_ratio.sum(1).detach().cpu()

def calc_log_ref_probs(samples, tokenizer, ref_model):
    ref_model.eval()
    with torch.no_grad():
        tokens = samples_to_tokens(samples, tokenizer)
        attention_mask = tokens.not_equal(tokenizer.pad_token_id).long().to(tokens.device)
        # logging.warning(f"input size in reward calc {tokens.shape}")
        ref_logprobs = calc_logprobs(tokens, ref_model, tokenizer) * attention_mask[:, :-1]
        return ref_logprobs.sum(1).detach().cpu().numpy()


def calc_logprobs(tokens, model, tokenizer):
    model.eval()
    with torch.no_grad():
        attention_mask = tokens.not_equal(tokenizer.pad_token_id).long().to(tokens.device)
        outputs = model(tokens, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits

        logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])
        return logprobs

def calc_win_rate(prompts, sentiments, bon_data):
        wins = []
        for prompt in prompts:
            ref_reward = bon_data[prompt]['logit_rewards'][0]
            if ref_reward < sentiments[prompts.index(prompt)]: 
                wins.append(1.0)
            else:
                wins.append(0.0)
        return wins


def flatten_dict(
    d: Union[dict, MutableMapping],
    parent_key: str = "",
    sep: str = "/",
) -> dict:
    # From: https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def gather_dict(obj: Dict, grad_state: accelerate.state.GradientState = None):
    """
    Gather and concatenates key-values from a dictionary, optionally
    trimming them if some of them were out of dataloader's padding
    """
    if not torch.distributed.is_initialized():
        return obj

    objs = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(objs, obj)

    acc, *objs = objs
    for obj in objs:
        for k in obj:
            acc[k].extend(obj[k])

    if grad_state:
        if grad_state.end_of_dataloader and grad_state.remainder > 0:
            for k in acc:
                acc[k] = acc[k][: grad_state.remainder]

    return acc


def get_tensor_stats(xs: torch.Tensor, mask: torch.Tensor, n: int):
    if xs.numel() == 0:
        return dict(mean=0, min=0, max=0, std=0)

    mean = (xs * mask).sum() / n
    return dict(
        mean=mean,
        min=torch.where(mask.bool(), xs, np.inf).min(),
        max=torch.where(mask.bool(), xs, -np.inf).max(),
        std=torch.sqrt(((xs - mean) * mask).pow(2).sum() / n),
    )


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """Updates running moments from batch's moments computed across ranks"""
        if dist.is_initialized():
            xs_mean, xs_var, xs_count = get_global_statistics(xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).sqrt()
        self.count = tot_count

        return xs_mean, (xs_var * xs_count / (xs_count - 1)).sqrt()
