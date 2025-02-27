import gc
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Any, Optional, Tuple

import deepspeed  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from jaxtyping import Float
from torch import Tensor
from transformers.modeling_outputs import ModelOutput

from trlx.data.ilql_types import ILQLBatch
from trlx.data.method_configs import MethodConfig, register_method
from trlx.models.modeling_base import PreTrainedModelWrapper
from trlx.utils.modeling import (
    flatten_dict,
    get_tensor_stats,
    hf_get_hidden_size,
    hf_get_lm_head,
    make_head,
)


def topk_mask(xs: torch.FloatTensor, k: int):
    if k > xs.shape[-1]:
        return xs
    mintop = torch.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return torch.where(xs < mintop, -np.inf * torch.ones_like(xs, dtype=xs.dtype), xs)


def batched_index_select(
    x: Float[Tensor, "batch seq_len hidden"],
    idxs: Float[Tensor, "batch index_len"],
    dim: int,
) -> Float[Tensor, "batch index_len hidden"]:
    """
    Gather vectors at idxs along dim from x
    """
    idxs = idxs.unsqueeze(-1).expand(idxs.shape[0], idxs.shape[1], x.shape[-1])
    return x.gather(dim=dim, index=idxs)


@dataclass
@register_method
class ILQLConfig(MethodConfig):
    """
    Configuration for ILQL method.

    :param tau: Parameter for expectile regression for the value function to q
    estimates, \in (0, 1), where tau=0.5 is equivalent to the mean square error
    and tau=1 is equivalent to taking a maximum over q estimates
    :type tau: float

    :param gamma: Discount factor
    :type gamma: float

    :param cql_scale: Scale for the CQL loss (conservative q-learning loss)
    :type cql_scale: float

    :param awac_scale: Scale for the AWAC loss (weighted cross-entropy loss)
    :type awac_scale: float

    :param alpha: Parameter for Polyak averaging of the target Q-head sync, \in (0, 1)
    :type alpha: float

    :param beta: Parameter for magnitude of weighting effect in the AWAC loss, \in (0, 1)
    :type beta: float

    :param steps_for_target_q_sync: Number of steps between target Q-head syncs
    :type steps_for_target_q_sync: int

    :param two_qs: Whether to use two Q-heads and taking minimum of separate estimates or using only one
    :type two_qs: bool

    :param gen_kwargs: Keyword arguments for the generation method
    :type gen_kwargs: dict
    """

    tau: float
    gamma: float
    cql_scale: float
    awac_scale: float
    alpha: float
    beta: float
    steps_for_target_q_sync: int
    two_qs: bool
    gen_kwargs: dict

    def loss(self, outputs, labels):
        logits, (qs, target_qs, vs) = outputs
        terminal_mask = labels.dones[:, :-1]
        n_nonterminal = max(1, terminal_mask.sum())
        # check type of labels
        if isinstance(labels, ILQLBatch):
            actions = labels.input_ids[:, 1:].gather(dim=1, index=labels.actions_ixs).unsqueeze(-1)
        else:
            actions = labels.decoder_input_ids[:, 1:].unsqueeze(-1)
        nactions = actions.shape[1]
        bsize, _, dsize = logits.shape

        Q = [q.gather(-1, actions).squeeze(-1) for q in qs]
        targetQs = [q.gather(-1, actions).squeeze(-1).detach() for q in target_qs]
        targetQ = reduce(torch.minimum, targetQs)

        # The loss_q assumes len(states) == len(rewards) + 1
        # values of current states
        V = vs[:, :-1, 0]
        # values of next states
        Vnext = vs[:, 1:, 0] * labels.dones[:, 1:].to(vs.dtype)
        # target to fit Q
        Q_ = labels.rewards + self.gamma * Vnext.detach()

        loss_qs = [((Qi - Q_) * terminal_mask).pow(2).sum() / n_nonterminal for Qi in Q]
        loss_q = sum(loss_qs)

        targetQ = targetQ.detach()

        loss_v = (
            (
                (targetQ >= V).int() * self.tau * (targetQ - V).pow(2)
                + (targetQ < V).int() * (1 - self.tau) * (targetQ - V).pow(2)
            )
            * terminal_mask
        ).sum() / n_nonterminal

        def cql_loss(q):
            loss = F.cross_entropy(q.reshape(-1, dsize), actions.reshape(-1), reduction="none")
            loss = loss.reshape(bsize, nactions) * terminal_mask
            loss = loss.sum() / n_nonterminal
            return loss

        loss_cql = sum(cql_loss(q) for q in qs)

        # select logits from continuations
        action_logits = batched_index_select(logits, labels.actions_ixs, dim=1)
        cross_entropy = F.cross_entropy(
            action_logits.reshape(-1, dsize),
            actions.reshape(-1),
            reduction="none",
        ).reshape(bsize, nactions)

        with torch.no_grad():
            awac_weight = torch.exp(self.beta * (targetQ - V))

        loss_awac = torch.sum(cross_entropy * awac_weight * terminal_mask) / n_nonterminal
        loss = loss_q + loss_v + self.cql_scale * loss_cql + self.awac_scale * loss_awac

        stats = dict(
            losses=dict(
                loss=loss.item(),
                loss_q=loss_q.item(),
                loss_v=loss_v.item(),
                loss_cql=loss_cql.item(),
                loss_awac=loss_awac.item(),
            ),
            values=get_tensor_stats(V, terminal_mask, n_nonterminal),
            qvalues={str(ix): get_tensor_stats(Q[ix], terminal_mask, n_nonterminal) for ix in range(len(Q))},
            awac_weight=get_tensor_stats(awac_weight, terminal_mask, n_nonterminal),
        )

        return loss, flatten_dict(stats)


class ILQLHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        two_qs: bool,
        alpha: float,
        dtype: type,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.two_qs = two_qs
        self.alpha = alpha
        self.v_head = make_head(self.hidden_size, 1, dtype)

        n_qs = 2 if self.two_qs else 1
        self.q_heads = nn.ModuleList(make_head(self.hidden_size, self.vocab_size, dtype) for _ in range(n_qs))
        self.target_q_heads = nn.ModuleList(deepcopy(q_head) for q_head in self.q_heads)

        for target_q_head in self.target_q_heads:
            target_q_head.requires_grad_(False)

    def forward(
        self,
        hs: Float[Tensor, "batch seq_len hidden"],
        states_ixs: Optional[Float[Tensor, "batch states_seq_len"]] = None,
        actions_ixs: Optional[Float[Tensor, "batch actions_seq_len"]] = None,
        **kwargs,
    ) -> Tuple[
        Tuple[Float[Tensor, "batch actions_seq_len hidden"]],
        Tuple[Float[Tensor, "batch actions_seq_len hidden"]],
        Float[Tensor, "batch states_seq_len hidden"],
    ]:
        if states_ixs is not None:
            states_hs = batched_index_select(hs, states_ixs, 1)
            actions_hs = batched_index_select(hs, actions_ixs, 1)
        else:
            states_hs = actions_hs = hs

        qs = tuple(q_head(actions_hs) for q_head in self.q_heads)
        target_qs = tuple(q_head(actions_hs) for q_head in self.target_q_heads)
        vs = self.v_head(states_hs)

        return qs, target_qs, vs

    def _sync_target_q_heads(self, alpha):
        for target_q_head, q_head in zip(self.target_q_heads, self.q_heads):
            for target_param, copy_param in zip(target_q_head.parameters(), q_head.parameters()):
                target_param.data.copy_((alpha * copy_param.data) + (1.0 - alpha) * target_param.data)

    def sync_target_q_heads(self):
        if os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "0") == "3":
            with deepspeed.zero.GatheredParameters(list(self.parameters()), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    self._sync_target_q_heads(self.alpha)
        else:
            self._sync_target_q_heads(self.alpha)


@dataclass
class CausalILQLOutput(ModelOutput):
    """
    Output of the causal model with ILQL heads.

    :param logits: Logits of the causal model.
    :type logits: torch.FloatTensor

    :param past_key_values: Tuple of past key values of the causal model.
    :type past_key_values: Tuple[Tuple[torch.FloatTensor]]

    :param hidden_states: Last hidden state of the causal model.
    :type hidden_states: Tuple[torch.FloatTensor]

    :param value: Value function estimation for each token in the input sequence.
    :type value: torch.FloatTensor

    :param qs: Q-function estimations for each token in the input sequence.
    :type qs: Tuple[torch.FloatTensor]

    :param target_qs: Q-function estimations from the target Q-head for each token in the input sequence.
    :type target_qs: Tuple[torch.FloatTensor]
    """

    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None
    qs: Optional[Tuple[torch.FloatTensor]] = None
    target_qs: Optional[Tuple[torch.FloatTensor]] = None


class AutoModelForCausalLMWithILQLHeads(PreTrainedModelWrapper):
    """An `AutoModel` class wrapper for `transformers` causal models with a language
    modeling head and ILQL heads.

    References:
        [1] Snell et al., "Offline RL for Natural Language Generation with Implicit Language Q Learning",
            https://arxiv.org/abs/2206.11871, 2022
    """

    _auto_model_parent_class = transformers.AutoModelForCausalLM
    _supported_modules = ["ilql_heads"]
    _supported_args = ["two_qs", "alpha", "peft_config"]

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        *,
        two_qs: bool = True,
        alpha: float = 0.99,
        peft_config=None,
    ):
        super().__init__(base_model, peft_config=peft_config)
        hidden_size = hf_get_hidden_size(self.base_model.config)
        vocab_size = self.base_model.config.vocab_size
        dtype = next(hf_get_lm_head(self.base_model).parameters()).dtype
        self.two_qs = two_qs
        self.alpha = alpha
        self.ilql_heads = ILQLHeads(hidden_size, vocab_size, self.two_qs, self.alpha, dtype=dtype)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        actions_ixs=None,
        states_ixs=None,
        return_dict=False,
        bypass_peft_prompt_adapter=False,
    ):
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        forward_kwargs["output_hidden_states"] = True

        if self.peft_type == "PREFIX_TUNING" and not bypass_peft_prompt_adapter:
            # Peft redefines past_key_values, remove it to avoid an exception.
            forward_kwargs.pop("past_key_values", None)

        if bypass_peft_prompt_adapter:
            outputs = self.base_model.base_model(**forward_kwargs)
        else:
            outputs = self.base_model(**forward_kwargs)
        qs, target_qs, vs = self.ilql_heads(outputs.hidden_states[-1], states_ixs=states_ixs, actions_ixs=actions_ixs)

        if return_dict:
            return CausalILQLOutput(outputs.logits, outputs.past_key_values, outputs.hidden_states, vs, qs, target_qs)

        return outputs.logits, qs, target_qs, vs, outputs.past_key_values

    def generate(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        beta=1,
        max_new_tokens=32,
        max_length=1024,
        temperature=1,
        top_k=20,
        logit_mask=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """
        Generates samples akin to hf's `.generate` but with custom logp preprocessing:
        changing token probabilities as to how advantageous they would be
        according to value functions estimations.
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.base_model.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.base_model.config.eos_token_id

        if attention_mask is None:
            attention_mask = input_ids.not_equal(pad_token_id)

        if position_ids is None:
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)

        samples = input_ids.clone()
        max_new_tokens = min(max_new_tokens, max_length - input_ids.shape[1])

        finished = torch.zeros(input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device)
        bypass_peft = False
        for token in range(max_new_tokens):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                bypass_peft_prompt_adapter=bypass_peft,
            )

            logits, _, target_qs, vs, past_key_values = out
            if self.two_qs:
                qs = torch.minimum(target_qs[0][:, -1, :], target_qs[1][:, -1, :])
            else:
                qs = target_qs[0][:, -1, :]

            logits = logits[:, -1, :]
            vs = vs[:, -1, :]

            if logit_mask is not None:
                mask = logit_mask[input_ids[:, -1].squeeze().to(logit_mask.device)]
                logits[torch.where(mask)] = -np.inf

            adv = qs - vs
            pi_beta = F.log_softmax(logits, -1)
            pi_top_k = topk_mask(pi_beta + beta * adv, top_k)

            if temperature == 0.0:
                input_ids = pi_top_k.argmax(dim=-1, keepdim=True)
            else:
                pi = F.softmax(pi_top_k / temperature, -1)
                input_ids = torch.multinomial(pi, num_samples=1)

            input_ids = (1 - finished) * input_ids + finished * eos_token_id
            finished = (input_ids == eos_token_id).long()

            samples = torch.hstack((samples, input_ids))
            attention_mask = torch.hstack((attention_mask, (input_ids != eos_token_id).long()))
            position_ids = (position_ids[:, -1] + 1).view(-1, 1)

            # Some peft models add a prefix to the prompt at each forward pass.
            # We need to bypass it so that it doesn't add multiple times the prefix.
            if self.peft_type and token == 0 and "LORA" not in self.peft_type:
                bypass_peft = True

                prefix_attention_mask = torch.ones(input_ids.shape[0], self.peft_config.num_virtual_tokens).to(
                    input_ids.device
                )
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            if os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "0") != "3" and torch.all(finished):
                break

        return samples

    def sync_target_q_heads(self):
        self.ilql_heads.sync_target_q_heads()

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the ilql heads
        to the state dictionary of the wrapped model by prepending the key with `ilql_heads.`.
        """
        state_dict = self.ilql_heads.state_dict(*args, **dict(prefix="ilql_heads.", **kwargs))
        if not heads_only:
            state_dict = {
                **state_dict,
                **self.base_model.state_dict(*args, **dict(prefix="" if self.peft_type else "base_model.", **kwargs)),
            }

        return state_dict

    def post_init(self, state_dict):
        """
        We add the state dictionary of the ilql heads to the state dictionary of the wrapped model
        by prepending the key with `ilql_heads.`.
        """
        super().post_init()
        strict = not self.peft_type and any(
            k.startswith("base_model.") or k.startswith("ilql_heads.") for k in state_dict
        )
        self.load_state_dict(state_dict, strict=strict)
        del state_dict
        gc.collect()


@dataclass
class Seq2SeqILQLOutput(ModelOutput):
    """
    Output of the seq2seq model with ILQL heads.

    :param logits: Logits of the seq2seq model.
    :type logits: torch.FloatTensor

    :param past_key_values: Tuple of past key values of the seq2seq model.
    :type past_key_values: Tuple[Tuple[torch.FloatTensor]]

    :param hidden_states: Last hidden state of the seq2seq model.
    :type hidden_states: Tuple[torch.FloatTensor]

    :param value: Value function estimation for each token in the input sequence.
    :type value: torch.FloatTensor

    :param qs: Q-function estimations for each token in the input sequence.
    :type qs: Tuple[torch.FloatTensor]

    :param target_qs: Q-function estimations from the target Q-head for each token in the input sequence.
    :type target_qs: Tuple[torch.FloatTensor]

    :param encoder_outputs: Tuple of encoder outputs of the seq2seq model.
    :type encoder_outputs: Tuple[Any]
    """

    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None
    qs: Optional[Tuple[torch.FloatTensor]] = None
    target_qs: Optional[Tuple[torch.FloatTensor]] = None
    encoder_outputs: Optional[Tuple[Any]] = None


class AutoModelForSeq2SeqLMWithILQLHeads(PreTrainedModelWrapper):
    """This is a wrapper around huggingface AutoModelForSeq2Seq with two additional scalar heads"""

    _auto_model_parent_class = transformers.AutoModelForSeq2SeqLM
    _supported_modules = ["ilql_heads"]
    _supported_args = ["two_qs", "alpha", "peft_config"]

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        *,
        two_qs: bool = True,
        alpha: float = 0.99,
        peft_config=None,
    ):
        super().__init__(base_model, peft_config=peft_config)
        hidden_size = hf_get_hidden_size(self.base_model.config)
        vocab_size = self.base_model.config.vocab_size
        dtype = next(hf_get_lm_head(self.base_model).parameters()).dtype
        self.two_qs = two_qs
        self.alpha = alpha
        self.ilql_heads = ILQLHeads(hidden_size, vocab_size, self.two_qs, self.alpha, dtype=dtype)

    def sync_target_q_heads(self):
        self.ilql_heads.sync_target_q_heads()

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the ilql heads
        to the state dictionary of the wrapped model by prepending the key with `ilql_heads.`.
        """
        state_dict = self.ilql_heads.state_dict(*args, **dict(prefix="ilql_heads.", **kwargs))
        if not heads_only:
            state_dict = {
                **state_dict,
                **self.base_model.state_dict(*args, **dict(prefix="" if self.peft_type else "base_model.", **kwargs)),
            }

        return state_dict

    def post_init(self, state_dict):
        """
        We add the state dictionary of the ilql heads to the state dictionary of the wrapped model
        by prepending the key with `ilql_heads.`.
        """
        super().post_init()
        strict = not self.peft_type and any(
            k.startswith("base_model.") or k.startswith("ilql_heads.") for k in state_dict
        )
        self.load_state_dict(state_dict, strict=strict)
        del state_dict
        gc.collect()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        decoder_input_ids=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        encoder_outputs=None,
        actions_ixs=None,
        states_ixs=None,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=False,
        bypass_peft_prompt_adapter=False,
    ):
        forward_kwargs = self.get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if self.peft_type == "PREFIX_TUNING" and not bypass_peft_prompt_adapter:
            # Peft redefines past_key_values, remove it to avoid an exception.
            forward_kwargs.pop("past_key_values", None)

        if bypass_peft_prompt_adapter:
            out = self.base_model.base_model(**forward_kwargs)
        else:
            out = self.base_model(**forward_kwargs)

        hs = out.decoder_hidden_states[-1]

        logits = self.base_model.lm_head(hs)
        qs, target_qs, vs = self.ilql_heads(hs, states_ixs=states_ixs, actions_ixs=actions_ixs)
        encoder_outputs = (out.encoder_last_hidden_state, out.encoder_hidden_states, out.encoder_attentions)

        if return_dict:
            return Seq2SeqILQLOutput(
                logits, out.past_key_values, out.decoder_hidden_states, vs, qs, target_qs, encoder_outputs
            )

        return logits, qs, target_qs, vs, out.past_key_values, encoder_outputs

    def generate(
        self,
        input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        decoder_input_ids=None,
        past_key_values=None,
        encoder_outputs=None,
        beta=1,
        max_new_tokens=32,
        max_length=1024,
        temperature=1,
        top_k=20,
        logit_mask=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """
        Generates samples akin to hf's `.generate` but with custom logp preprocessing:
        changing token probabilities as to how advantageous they would be
        according to value functions estimations.
        """

        if eos_token_id is None or pad_token_id is None:
            raise ValueError("eos_token_id and pad_token_id must be provided")

        if attention_mask is None:
            if decoder_attention_mask is not None:
                attention_mask = decoder_attention_mask
            else:
                attention_mask = input_ids.not_equal(pad_token_id)

        samples = input_ids.clone()
        max_new_tokens = min(max_new_tokens, max_length - input_ids.shape[1])
        if decoder_input_ids is None:
            decoder_input_ids = input_ids.new_zeros(input_ids.shape[0], 1)

        finished = torch.zeros(input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device)
        bypass_peft = False
        for token in range(max_new_tokens):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids[:, -1].unsqueeze(-1),
                past_key_values=past_key_values,
                encoder_outputs=encoder_outputs,
                bypass_peft_prompt_adapter=bypass_peft,
            )
            logits, _, target_qs, vs, past_key_values, encoder_outputs = out
            if self.two_qs:
                qs = torch.minimum(target_qs[0][:, -1, :], target_qs[1][:, -1, :])
            else:
                qs = target_qs[0][:, -1, :]

            logits = logits[:, -1, :]
            vs = vs[:, -1, :]
            adv = qs - vs
            pi_beta = F.log_softmax(logits, -1)
            pi_top_k = topk_mask(pi_beta + beta * adv, top_k)

            if temperature == 0.0:
                next_tokens = pi_top_k.argmax(dim=-1, keepdim=True)
            else:
                pi = F.softmax(pi_top_k / temperature, -1)
                next_tokens = torch.multinomial(pi, num_samples=1)
            next_tokens = (1 - finished) * next_tokens + finished * eos_token_id
            finished = (next_tokens == eos_token_id).long() | (next_tokens == pad_token_id).long()
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
            samples = decoder_input_ids

            # Some peft models add a prefix to the prompt at each forward pass.
            # We need to bypass it so that it doesn't add multiple times the prefix.
            if self.peft_type and token == 0 and "LORA" not in self.peft_type:
                bypass_peft = True

                prefix_attention_mask = torch.ones(input_ids.shape[0], self.peft_config.num_virtual_tokens).to(
                    input_ids.device
                )
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            if os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "0") != "3" and torch.all(finished):
                break

        return samples
