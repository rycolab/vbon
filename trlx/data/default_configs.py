from pathlib import Path

from trlx.models.modeling_ilql import ILQLConfig
from trlx.models.modeling_ppo import PPOConfig
from trlx.trainer.accelerate_sft_trainer import SFTConfig

from .configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)


def default_ppo_config():
    return TRLConfig(
        train=TrainConfig(
            seed=0,
            seq_length=1024,
            epochs=100,
            total_steps=10000,
            batch_size=32, #32
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
        ),
        model=ModelConfig(model_path="lvwerra/gpt2-imdb", num_layers_unfrozen=2), #
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"), #gpt2
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=3e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=3e-5)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=128, #128
            ppo_epochs=4,
            init_kl_coef=0.01, #0.001
            ent_coef=0.0,
            target=None,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            bon_reward=False,
            gen_kwargs=dict(
                max_new_tokens=40,
                top_k=0,
                top_p=1.0,
                do_sample=True,
                temperature=1.,
            ),
        ),
    )


def default_ilql_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=64,
            batch_size=128,
            epochs=100,
            total_steps=1000,
            checkpoint_interval=1000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AccelerateILQLTrainer",
        ),
        model=ModelConfig(model_path="gpt2", num_layers_unfrozen=-1),
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=5.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=5.0e-5)  # train.total_steps
        ),
        method=ILQLConfig(
            name="ilqlconfig",
            tau=0.7,
            gamma=0.99,
            cql_scale=0.1,
            awac_scale=1,
            alpha=0.001,
            beta=0,
            steps_for_target_q_sync=5,
            two_qs=True,
            gen_kwargs=dict(max_new_tokens=56, top_k=20, beta=1, temperature=1.0),
        ),
    )


def default_sft_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=1, #100
            total_steps=625,
            batch_size=8,
            seed=0,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AccelerateSFTTrainer",
        ),
        model=ModelConfig(model_path="lvwerra/gpt2-imdb", num_layers_unfrozen=-1),
        tokenizer=TokenizerConfig(tokenizer_path="lvwerra/gpt2-imdb", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=5.0e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6) #1.0e-4
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=1.0e-4)  # train.total_steps
        ),
        method=SFTConfig(
            name="sftconfig",
            n=1,
            gen_kwargs=dict(max_new_tokens=40, top_k=0, top_p=1.0, do_sample=True),
        ),
    )


def default_nemo_20b_config():
    """Load nemo-megatron-20b model and trainer config"""
    # Import here to not require nemo as a dependency
    from omegaconf import OmegaConf

    here = Path(__file__).parent
    return OmegaConf.load(here.parent.parent / "configs" / "nemo_configs" / "megatron_20b.yaml")


def default_nemo_2b_config():
    """Load nemo-megatron-1.3b model and trainer config"""
    # Import here to not require nemo as a dependency
    from omegaconf import OmegaConf

    here = Path(__file__).parent
    return OmegaConf.load(here.parent.parent / "configs" / "nemo_configs" / "megatron_2b.yaml")


def default_nemo_1_3b_config():
    """Load nemo-megatron-1.3b model and trainer config"""
    # Import here to not require nemo as a dependency
    from omegaconf import OmegaConf

    here = Path(__file__).parent
    return OmegaConf.load(here.parent.parent / "configs" / "nemo_configs" / "megatron_1.3b.yaml")
