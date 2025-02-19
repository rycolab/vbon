# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from math import comb
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import pipeline, AutoModelForCausalLM
from collections import defaultdict
import logging
import numpy as np
from scipy.optimize import curve_fit
import trlx
from trlx.data.default_configs import TRLConfig, default_ppo_config
from transformers import AutoTokenizer
import argparse
from trlx.utils import set_seed
from trlx.utils.modeling import calc_kl, calc_win_rate, calc_log_ref_probs


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return scores[1]["score"]


def get_positive_sentiment_prob(scores):
    "Extract probs associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def func(x, a, b, c):
    "Exponential function for curve fitting"
    return -a * np.exp(-b * x) - c


def load_bon_dict(bon_dict_path, tokenizer, splits=["train", "test"], do_curve_fit=False):
    data = defaultdict(lambda: defaultdict(list))
    for split in splits:
        with open(f"{bon_dict_path}/imdb-{split}-bon-vllm.jsonl", "r") as f:
            for line in f:
                line_dict = json.loads(line)
                prompt = line_dict["prompt"]

                # just a hack to be consistent with the trainer and its spurious ambiguity
                prompt_id = tokenizer(prompt)["input_ids"]
                prompt_str = tokenizer.decode(prompt_id)
                if len(data[prompt_str][
                           "responses"]) > 300:  # we converge to true F after ~300 samples
                    continue

                data[prompt_str]["responses"].append(line_dict["text"])
                data[prompt_str]["rewards"].append(line_dict["reward"])
                data[prompt_str]["logit_rewards"].append(line_dict["reward"])  # logit_reward

    data[prompt_str]["responses"] = data[prompt_str]["responses"]
    data[prompt_str]["rewards"] = data[prompt_str]["rewards"]
    data[prompt_str]["logit_rewards"] = data[prompt_str]["logit_rewards"]

    for _, v in data.items():
        sorted_rewards, order_idx = torch.sort(torch.tensor(v["logit_rewards"]))
        p_emp = torch.ones(len(sorted_rewards) + 1)
        f = torch.cumsum(p_emp, dim=0)
        f = f / len(f)
        logf = torch.log(f)

        v["sorted_rewards"] = sorted_rewards
        v["logf"] = logf
        v["f"] = f

        if do_curve_fit:
            popt, _ = curve_fit(func, sorted_rewards, logf[:-1], maxfev=5000)
            v["popt"] = popt
    return data


def main(bon, bound, do_curve_fit, bon_dict_path, hparams={}):
    logging.warning(torch.cuda.is_available())

    n = 1
    if bon and bound == "exact":
        n = int(1 / hparams["method"]["init_kl_coef"]) + 1

    # Merge sweep config with default config if given
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    set_seed(config.train.seed)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        truncation=True,
        batch_size=256,
        device=device,
    )

    sentiment_fn_top2 = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def calc_bon_reward(data, prompts, rewards, do_curve_fit=False):
        bon_rewards = []

        for idx, p in enumerate(prompts):
            if do_curve_fit:
                bon_rewards.append(func(rewards[idx], *data[p]["popt"]))
            else:
                sorted_rewards = data[p]["sorted_rewards"]
                logf = data[p]["logf"]
                reward_rank = np.searchsorted(sorted_rewards, rewards[idx])
                bon_rewards.append(logf[reward_rank])
        return bon_rewards

    def exact_bon_reward(data, prompts, rewards, samples, do_curve_fit):
        logfs = np.asarray(calc_bon_reward(data, prompts, rewards, do_curve_fit=do_curve_fit))
        logref = calc_log_ref_probs(samples, tokenizer, ref_model)

        results = -np.inf
        for i in range(1, int(n) + 1):
            term_i = np.log(float(comb(int(n), i))) + (n - i) * logfs + i * logref
            results = np.logaddexp(results, term_i)

        return (1. / n) * (results - logref)

    def reward_bon_builder(bound="l1", do_curve_fit=False):
        def reward_fn_bon(samples: List[str], prompts: List[str], **kwargs) -> List[float]:
            sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}
            sentiments = list(map(get_positive_score, sentiment_fn(samples, **sent_kwargs)))
            return calc_bon_reward(bon_data, prompts, sentiments, do_curve_fit=do_curve_fit)

        def reward_fn_bon_exact(samples: List[str], prompts: List[str], **kwargs) -> List[
            float]:
            sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}
            sentiments = list(map(get_positive_score, sentiment_fn(samples, **sent_kwargs)))
            return exact_bon_reward(bon_data, prompts, sentiments, samples,
                                    do_curve_fit=do_curve_fit)

        if bound == "exact":
            return reward_fn_bon_exact

        return reward_fn_bon

    def reward_fn_normal(samples: List[str], **kwargs):
        sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}
        sentiments = list(map(get_positive_score, sentiment_fn(samples, **sent_kwargs)))
        return sentiments

    def kl_ppo(samples, model):
        return calc_kl(samples, model, tokenizer, ref_model)

    def metric_fn(samples: List[str], prompts: List[str], **kwargs):
        sentiments = list(map(get_positive_sentiment_prob, sentiment_fn_top2(samples)))

        sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}
        sentiment_scores = list(map(get_positive_score, sentiment_fn(samples, **sent_kwargs)))
        win_rates = calc_win_rate(prompts, sentiment_scores, bon_data)
        return {"average_positive_prob": sentiments, "win_rate": win_rates}

    def load_prompts(split="train"):
        prompts = []
        with open(f"{bon_dict_path}/imdb-{split}-bon-vllm.jsonl", "r") as f:
            for line in f:
                line_dict = json.loads(line)
                prompts.append(line_dict["prompt"])
        return list(set(prompts))

    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|padding|>"
    train_prompts = load_prompts("train")
    test_prompts = load_prompts("test")

    if bon:
        bon_data = load_bon_dict(bon_dict_path, tokenizer, do_curve_fit=do_curve_fit)
        reward_fn = reward_bon_builder(bound, do_curve_fit)
    else:
        bon_data = load_bon_dict(bon_dict_path, tokenizer, splits=["test"])
        reward_fn = reward_fn_normal

    ref_model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
    if device != -1:
        ref_model = ref_model.to("cuda")

    trlx.train(
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        kl_fn=kl_ppo,
        prompts=train_prompts,
        eval_prompts=test_prompts,
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bon", action=argparse.BooleanOptionalAction)
    parser.add_argument("--curvefit", action=argparse.BooleanOptionalAction)
    parser.add_argument("--init_kl_coef", type=float, default=0.01,
                        help="KL coefficient for PPO training")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--bound", type=str, default="exact",
                        help="bound used for vbon optimization: either exact or l1")
    parser.add_argument("--bon_dict_path", type=str, help="path to read the F function from")
    args = parser.parse_args()

    main(args.bon, args.bound, args.curvefit, args.bon_dict_path,
         {"train": {"seed": args.seed},
          "method": {"bon_reward": args.bon,
                     "init_kl_coef": args.init_kl_coef}})
