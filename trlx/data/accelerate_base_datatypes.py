from dataclasses import dataclass
from typing import Iterable

from jaxtyping import Float
from torch import Tensor


@dataclass
class PromptElement:
    """
    Dataclass for a single prompt, containing its string and tokenized form.

    :param text: The prompt text.
    :type text: str

    :param tokens: The prompt tokens. Should be a long tensor
    :type tokens: torch.Tensor
    """

    text: str
    tokens: Float[Tensor, "num_tokens"]


@dataclass
class PromptBatch:
    """
    Batched PromptElement

    :param text: An iterable of prompt texts.
    :type text: Iterable[str]

    :param tokens: A long tensor batch of prompt tokens.
    :type tokens: torch.Tensor
    """

    text: Iterable[str]
    tokens: Float[Tensor, "batch_size num_tokens"]


@dataclass
class AccelerateRLElement:
    """
    Dataclass for RL elements, containing output tokens and rewards for each token.

    :param tokens: The output tokens. Should be a long tensor
    :type tokens: torch.Tensor

    :param rewards: The rewards for each token. Should be a float tensor of same size as tokens.
    :type rewards: torch.Tensor
    """

    output_tokens: Float[Tensor, "output_size"]
    rewards: Float[Tensor, "output_size"]


@dataclass
class AccelerateRLBatchElement:
    """
    Batched accelerate RL element

    :param tokens: Batches of long tensors of output tokens.
    :type tokens: torch.Tensor

    :param rewards: Batches of float tensors of rewards for each output token.
    :type rewards: torch.Tensor
    """

    output_tokens: Float[Tensor, "batch_size output_size"]
    rewards: Float[Tensor, "batch_size output_size"]
