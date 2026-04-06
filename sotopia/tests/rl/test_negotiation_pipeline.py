from __future__ import annotations

from sotopia.rl.negotiation_pipeline import (
    NegotiationRLPipelineConfig,
    compute_final_reward,
    decompose_final_reward,
    extract_post_intervention_A_turns,
)


def test_extract_post_intervention_a_turns() -> None:
    dialogue = [
        {"turn_index": 1, "speaker": "mturk_agent_1", "text": "a1"},
        {"turn_index": 2, "speaker": "mturk_agent_2", "text": "b1"},
        {"turn_index": 3, "speaker": "mturk_agent_1", "text": "a2"},
        {"turn_index": 4, "speaker": "mturk_agent_2", "text": "b2"},
        {"turn_index": 5, "speaker": "mturk_agent_1", "text": "a3"},
    ]

    turns = extract_post_intervention_A_turns(
        dialogue=dialogue,
        current_a_id="mturk_agent_1",
        intervention_turn_index=2,
    )

    assert [turn["turn_index"] for turn in turns] == [3, 5]


def test_decompose_final_reward_sum_preserved() -> None:
    config = NegotiationRLPipelineConfig()
    final_reward = 0.73
    stance_signals = [
        {"turn_index": 3, "turn_id": "t3", "text": "x", "stance_signal_i": 0.2},
        {"turn_index": 5, "turn_id": "t5", "text": "y", "stance_signal_i": 0.8},
    ]

    rewards = decompose_final_reward(
        final_reward=final_reward,
        stance_signals=stance_signals,
        config=config,
    )

    total = sum(item["decomposed_reward"] for item in rewards)
    assert abs(total - final_reward) < 1e-8
    assert rewards[0]["weight_i"] < rewards[1]["weight_i"]


def test_compute_final_reward_weights() -> None:
    config = NegotiationRLPipelineConfig(
        reward_weights={
            "self_svi_A": 0.25,
            "other_svi_B_to_A": 0.25,
            "utility_A": 0.25,
            "joint_utility": 0.25,
        }
    )
    reward = compute_final_reward(
        self_svi_A=0.8,
        other_svi_B_to_A=0.6,
        utility_A_norm=0.5,
        joint_utility_norm=0.4,
        config=config,
    )
    assert reward == 0.575


def test_decompose_uniform_fallback_when_zero_signal() -> None:
    config = NegotiationRLPipelineConfig(alpha=0.0, beta=1.0)
    rewards = decompose_final_reward(
        final_reward=1.2,
        stance_signals=[
            {"turn_index": 3, "turn_id": "t3", "text": "x", "stance_signal_i": 0.0},
            {"turn_index": 5, "turn_id": "t5", "text": "y", "stance_signal_i": 0.0},
        ],
        config=config,
    )
    assert rewards[0]["weight_i"] == 0.5
    assert rewards[1]["weight_i"] == 0.5
