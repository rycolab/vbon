import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import argparse

from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from collections import defaultdict
import numpy as np
import logging
from scipy.optimize import curve_fit

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig
from trlx.utils.modeling import disable_dropout


def main(bon, bon_dict_path, models_path):
    temp = 0.7 + 1e-7
    config = TRLConfig(
        train=TrainConfig(
            seq_length=600,
            epochs=1,
            total_steps=100000,
            batch_size=4,
            checkpoint_interval=10000,
            eval_interval=200,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
        ),
        model=ModelConfig(
            model_path=f"{models_path}/EleutherAI_pythia-2.8b-deduped__sft__tldr",
            num_layers_unfrozen=8,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_path=f"{models_path}/EleutherAI_pythia-2.8b-deduped__sft__tldr",
            truncation_side="right",
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            kwargs={
                "lr": 3.0e-6,
                "betas": [0.9, 0.999],
                "eps": 1.0e-5,
                "weight_decay": 0.01,
            },
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing",
            kwargs={
                "T_max": 100000,
                "eta_min": 5.0e-6,
            },
        ),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=16,
            ppo_epochs=4,
            init_kl_coef=0.005,
            target=None,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.2,
            scale_reward=None,
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs={
                "temperature": temp,
                "max_new_tokens": 50,
                "top_k": 0,
                "top_p": 1.0,
                "do_sample": True,
            },
        ),
    )
    # Load the pre-trained reward model
    rw_tokenizer = AutoTokenizer.from_pretrained(
        f"{models_path}/EleutherAI_pythia-2.8b-deduped__reward__tldr")
    rw_tokenizer.add_special_tokens({
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "[PAD]",
        "unk_token": "<|endoftext|>"
    }
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    rw_model = AutoModelForSequenceClassification.from_pretrained(
        f"{models_path}/EleutherAI_pythia-2.8b-deduped__reward__tldr",
        quantization_config=bnb_config)
    disable_dropout(rw_model)

    rw_model.config.pad_token_id = rw_tokenizer.pad_token_id
    rw_model.eval()
    rw_device = "cuda"

    def get_scores(samples):
        encodings_dict = rw_tokenizer(
            samples,
            truncation=True,
            max_length=config.train.seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings_dict["input_ids"].to(rw_device)
        attn_masks = encodings_dict["attention_mask"].to(rw_device)

        with torch.no_grad():
            acc_outputs = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores = acc_outputs.logits[:, 0]

            contain_eos_token = torch.any(input_ids == tokenizer.eos_token_id, dim=-1)
            scores = torch.where(contain_eos_token, scores,
                                 torch.full_like(scores, -20))  # -20 bon, -1 ppo
        return scores

    def func(x, a, b):
        "Exponential function for curve fitting"
        return -a * np.exp(-b * x)

    def load_bon_dict(bon_path, do_curve_fit=False):
        bon_dict = defaultdict(lambda: defaultdict(list))
        with open(bon_path, "r") as f:
            for line in f:
                line_dict = json.loads(line)
                prompt = line_dict["prompt"].strip()
                rewards = line_dict["rewards"]
                bon_dict[prompt]["rewards"] = rewards
                sorted_rewards, order_idx = torch.sort(torch.tensor(rewards))
                bon_dict[prompt]["sorted_rewards"] = sorted_rewards

                p_emp = torch.ones(len(sorted_rewards) + 1)
                f = torch.cumsum(p_emp, dim=0)
                f = f / len(f)
                logf = torch.log(f)
                bon_dict[prompt]["logf"] = logf

                if do_curve_fit:
                    popt, _ = curve_fit(func, sorted_rewards, logf[:-1], maxfev=5000)
                    bon_dict[prompt]["popt"] = popt
        return bon_dict

    def calc_bon_reward(bon_dict, prompts, rewards, curve_fit=False):
        bon_rewards = []

        for idx, p in enumerate(prompts):
            if curve_fit:
                bon_rewards.append(
                    func(rewards[idx].detach().cpu().numpy(), *bon_dict[p]["popt"]))
            else:
                sorted_rewards = torch.tensor(bon_dict[p]["sorted_rewards"]).to(
                    rewards.device)
                logf = torch.tensor(bon_dict[p]["logf"]).to(rewards.device)
                reward_rank = torch.searchsorted(sorted_rewards, rewards[idx])
                bon_rewards.append(logf[reward_rank])
        return bon_rewards

    def get_prompt_dataset(prompts, max_length):
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline
        """
        formatted_prompts = []
        for i in tqdm(range(len(prompts))):
            tmp = tokenizer.decode(
                tokenizer(
                    prompts[i].split("TL;DR:")[0],
                    truncation=True,
                    max_length=max_length - 5,  # to make sure "TL;DR" dont get truncated
                    add_special_tokens=False,
                )["input_ids"],
                skip_special_tokens=True,
            ).strip()
            tmp = tmp + "\nTL;DR:"
            tmp = tokenizer.decode(
                tokenizer(tmp, truncation=True, max_length=max_length,
                          add_special_tokens=False)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            formatted_prompts.append(tmp)
        return formatted_prompts

    def reward_fn(samples: List[str], prompts: List[str], **kwargs):
        scores = get_scores(samples)
        if bon:
            bon_scores = calc_bon_reward(bon_dict, prompts, scores, curve_fit=False)
            return torch.stack(bon_scores, axis=0)
        return scores

    bon_dict = load_bon_dict(f"{bon_dict_path}/tldr-sftent-train.jsonl")
    train_prompts = list(set(bon_dict.keys()))

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.add_special_tokens({
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "[PAD]",
        "unk_token": "<|endoftext|>"
    }
    )
    tokenizer.padding_side = "right"
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    config.train.epochs = int(len(train_prompts) / config.method.num_rollouts)

    dataset = load_dataset("CarperAI/openai_summarize_tldr")

    val_set = [(sample["prompt"], sample["label"]) for sample in dataset["valid"]]
    val_posts, _ = zip(*val_set)
    val_prompts = get_prompt_dataset(val_posts, max_length_input)

    logging.warning(f"len train prompts === {len(train_prompts)}")
    trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:16],
        # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bon", action=argparse.BooleanOptionalAction)
    parser.add_argument("--bon_dict_path", type=str, help="path to read the F function from")
    parser.add_argument("--models_path", type=str, help="path to the reward and ref model")

    args = parser.parse_args()

    main(args.bon, args.bon_dict_path, args.models_path)
