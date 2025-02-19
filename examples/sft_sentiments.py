import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List

from datasets import load_dataset
from transformers import pipeline
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

import trlx
from trlx.data.default_configs import TRLConfig, default_sft_config
from operator import itemgetter
from trlx.utils.modeling import calc_kl, calc_win_rate, calc_forward_kl
from ppo_sentiments import load_bon_dict
import torch
import logging


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def load_prompts(bon_dict_path, split="test"):
    prompts = []
    with open(f"{bon_dict_path}/imdb-{split}-bon-vllm.jsonl", "r") as f:
        for line in f:
            line_dict = json.loads(line)
            prompts.append(line_dict["prompt"])
    return list(set(prompts))


def main(n, bon_dict_path, hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(default_sft_config().to_dict(), hparams)

    imdb = load_dataset("imdb", split="train+test")
    # Finetune on only positive reviews
    imdb = imdb.filter(lambda sample: sample["label"] == 1)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def metric_fn(samples: List[str], prompts: List[str], **kwargs) -> Dict[str, List[float]]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))

        sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}
        sentiment_scores = list(map(get_positive_score, sentiment_fn(samples, **sent_kwargs)))
        win_rates = calc_win_rate(prompts, sentiment_scores, bon_data)

        return {"average_positive_prob": sentiments, "win_rate": win_rates}

    def kl_sft(samples, model):
        return calc_kl(samples, model, tokenizer, ref_model)

    def kl_forwrad_sft(prompts, model):
        return calc_forward_kl(bon_data, prompts, model, tokenizer, ref_model)

    def process_data(n):
        texts = []
        data = defaultdict(lambda: defaultdict(list))
        with open(f"{bon_dict_path}/imdb-train-bon-vllm.jsonl", "r") as f:
            for line in f:
                line_dict = json.loads(line)
                prompt = line_dict["prompt"]
                data[prompt]["responses"].append(line_dict["text"])
        for prompt, items in data.items():
            responses = items["responses"][:n]
            sentiment_out = sentiment_fn(responses)
            sentiments = list(map(get_positive_score, sentiment_out))
            index, _ = max(enumerate(sentiments), key=itemgetter(1))
            texts.append(responses[index])

        return texts

    eval_prompts = load_prompts(load_prompts, "test")
    train_samples = process_data(n)

    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|padding|>"
    logging.info(f"Padding token id: {tokenizer.pad_token_id}")

    bon_data = load_bon_dict(bon_dict_path, tokenizer, splits=["test"])

    ref_model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb").to("cuda")

    trainer = trlx.train(
        samples=train_samples,
        eval_prompts=eval_prompts,
        metric_fn=metric_fn,
        kl_fn=kl_sft,
        kl_forward_fn=kl_forwrad_sft,
        config=config,
    )
    trainer.save_pretrained("reviews-sft")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--n", type=int, default=0, help="n in best of n")
    parser.add_argument("--bon_dict_path", type=str, help="path to read the F function from")
    args = parser.parse_args()
    main(args.n, args.bon_dict_path, {"train": {"seed": args.seed}, "method": {"n": args.n}})
