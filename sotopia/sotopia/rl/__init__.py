"""Reinforcement learning module for intervention training.

This module provides utilities for decomposing episode-level rewards into turn-level
rewards for the intervened agent only, using stance change as a local implicit signal.
"""

from .reward_decomposition import (
    build_turn_level_rewards_for_episode,
    compute_decomposed_reward_vector,
    compute_final_reward,
    compute_stance_signal_for_turn,
    extract_intervened_agent_turns,
    RewardDecompositionConfig,
    TurnLevelRewardInfo,
)
from .negotiation_pipeline import (
    NegotiationRLPipelineConfig,
    compute_final_reward as compute_negotiation_final_reward,
    compute_svi_reward_components,
    compute_triplet_stance_signal,
    compute_utility_reward_components,
    decompose_final_reward,
    extract_post_intervention_A_turns,
    finalize_episode_with_stance,
    finalize_episodes_with_stance_batch,
    rl_train_step,
    rollout_dialogue,
    run_negotiation_rl_training,
    run_episode_and_collect_rewards,
)

try:
    from .local_qwen_trainer import LocalQwenPolicyTrainer, LocalQwenTrainerConfig
except Exception:  # pragma: no cover
    LocalQwenPolicyTrainer = None  # type: ignore[assignment]
    LocalQwenTrainerConfig = None  # type: ignore[assignment]

__all__ = [
    "build_turn_level_rewards_for_episode",
    "compute_decomposed_reward_vector",
    "compute_final_reward",
    "compute_stance_signal_for_turn",
    "extract_intervened_agent_turns",
    "RewardDecompositionConfig",
    "TurnLevelRewardInfo",
    "rollout_dialogue",
    "NegotiationRLPipelineConfig",
    "compute_svi_reward_components",
    "compute_utility_reward_components",
    "compute_negotiation_final_reward",
    "extract_post_intervention_A_turns",
    "compute_triplet_stance_signal",
    "decompose_final_reward",
    "finalize_episode_with_stance",
    "finalize_episodes_with_stance_batch",
    "rl_train_step",
    "run_negotiation_rl_training",
    "run_episode_and_collect_rewards",
    "LocalQwenPolicyTrainer",
    "LocalQwenTrainerConfig",
]
