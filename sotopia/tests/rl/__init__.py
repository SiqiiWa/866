"""Tests for RL reward decomposition module.

Tests verify:
1. Reward sum preservation (decomposed rewards sum to final reward)
2. Uniform fallback when no signal
3. Higher signal -> higher reward allocation
4. Min/max reward handling
5. Turn extraction (intervened-only)
6. Edge cases (no turns, empty signals, etc.)
"""

import pytest
from sotopia.rl.reward_decomposition import (
    RewardDecompositionConfig,
    TurnInfo,
    TurnLevelRewardInfo,
    compute_final_reward,
    extract_intervened_agent_turns,
    compute_stance_signal_for_turn,
    decompose_final_reward_over_turns,
    build_turn_level_rewards_for_episode,
)


class TestComputeFinalReward:
    """Test final reward computation."""
    
    def test_basic_reward_computation(self):
        """Test basic weighted reward computation."""
        config = RewardDecompositionConfig(
            reward_weights={"svi": 1.0, "self": 1.0, "joint": 1.0}
        )
        reward = compute_final_reward(
            svi_score=10.0,
            agent_self_utility=5.0,
            joint_utility=3.0,
            config=config,
        )
        expected = 10.0 + 5.0 + 3.0
        assert reward == pytest.approx(expected)
    
    def test_weighted_reward_computation(self):
        """Test reward with non-uniform weights."""
        config = RewardDecompositionConfig(
            reward_weights={"svi": 2.0, "self": 0.5, "joint": 0.0}
        )
        reward = compute_final_reward(
            svi_score=10.0,
            agent_self_utility=4.0,
            joint_utility=100.0,
            config=config,
        )
        expected = 2.0 * 10.0 + 0.5 * 4.0 + 0.0 * 100.0
        assert reward == pytest.approx(expected)
    
    def test_zero_reward(self):
        """Test zero reward case."""
        config = RewardDecompositionConfig()
        reward = compute_final_reward(0.0, 0.0, 0.0, config)
        assert reward == 0.0
    
    def test_negative_reward(self):
        """Test negative reward case."""
        config = RewardDecompositionConfig()
        reward = compute_final_reward(-5.0, -3.0, -2.0, config)
        expected = -5.0 - 3.0 - 2.0
        assert reward == pytest.approx(expected)


class TestExtractIntervenedAgentTurns:
    """Test extraction of intervened agent turns."""
    
    def test_extract_single_turn(self):
        """Test extracting a single turn."""
        messages = [
            [("Environment", "Agent1", "You start"), ("Environment", "Agent2", "You start")],
            [("Agent1", "Environment", "I'll start"), ("Agent2", "Environment", "OK")],
        ]
        turns = extract_intervened_agent_turns(messages, "Agent1")
        assert len(turns) == 1
        assert turns[0].turn_index == 1
        assert turns[0].speaker_agent_id == "Agent1"
        assert turns[0].text == "I'll start"
    
    def test_extract_multiple_turns(self):
        """Test extracting multiple turns."""
        messages = [
            [("Environment", "Agent1", "You start"), ("Environment", "Agent2", "You start")],
            [("Agent1", "Environment", "First turn"), ("Agent2", "Environment", "OK")],
            [("Agent1", "Environment", "Second turn"), ("Agent2", "Environment", "Got it")],
            [("Agent1", "Environment", "Third turn")],
        ]
        turns = extract_intervened_agent_turns(messages, "Agent1")
        assert len(turns) == 3
        assert turns[0].turn_index == 1
        assert turns[1].turn_index == 2
        assert turns[2].turn_index == 3
    
    def test_extract_no_turns(self):
        """Test when intervened agent has no turns."""
        messages = [
            [("Environment", "Agent1", "Start"), ("Environment", "Agent2", "Start")],
            [("Agent2", "Environment", "Agent2 only")],
        ]
        turns = extract_intervened_agent_turns(messages, "Agent1")
        assert len(turns) == 0
    
    def test_extract_correct_agent_only(self):
        """Test that only the specified agent's turns are extracted."""
        messages = [
            [("Environment", "Agent1", "Start"), ("Environment", "Agent2", "Start")],
            [("Agent1", "Environment", "Agent1 turn"), ("Agent2", "Environment", "Agent2 turn")],
        ]
        turns = extract_intervened_agent_turns(messages, "Agent1")
        assert len(turns) == 1
        assert turns[0].text == "Agent1 turn"


class TestComputeStanceSignal:
    """Test stance signal computation."""
    
    def test_no_adjacent_utterances(self):
        """Test when no adjacent utterances exist."""
        turn = TurnInfo(turn_index=0, speaker_agent_id="Agent1", text="start")
        adjacent = {"prev_utterance": None, "next_utterance": None}
        signal = compute_stance_signal_for_turn(turn, adjacent)
        assert signal == pytest.approx(0.0)
    
    def test_stance_change_detection(self):
        """Test detecting stance change."""
        turn = TurnInfo(turn_index=1, speaker_agent_id="Agent1", text="let's cooperate")
        # Other agent goes from proself (-1) to prosocial (1) = change of 2
        adjacent = {
            "prev_utterance": ("proself", "I want everything"),
            "next_utterance": ("prosocial", "OK let's share"),
        }
        signal = compute_stance_signal_for_turn(turn, adjacent)
        expected_change = abs(1.0 - (-1.0))
        assert signal == pytest.approx(expected_change)
    
    def test_no_stance_change(self):
        """Test when no stance change occurs."""
        turn = TurnInfo(turn_index=1, speaker_agent_id="Agent1", text="ok")
        adjacent = {
            "prev_utterance": ("prosocial", "Let's be fair"),
            "next_utterance": ("prosocial", "I agree"),
        }
        signal = compute_stance_signal_for_turn(turn, adjacent)
        assert signal == pytest.approx(0.0)
    
    def test_action_bonus(self):
        """Test action bonus for influence/offer actions."""
        turn = TurnInfo(turn_index=1, speaker_agent_id="Agent1", text="offer")
        adjacent = {
            "prev_utterance": ("neutral", "What should we do?"),
            "next_utterance": ("prosocial", "I'll give you everything!"),
        }
        annotations = {
            2: {"stance": 1.0, "action_type": "offer"}
        }
        base_bonus = 0.5
        signal = compute_stance_signal_for_turn(
            turn, adjacent, annotations, action_bonus=base_bonus
        )
        # stance_delta = |1 - 0| = 1, + 0.5 bonus for offer
        expected = 1.0 + 0.5
        assert signal == pytest.approx(expected)


class TestDecomposeReward:
    """Test final reward decomposition."""
    
    def test_sum_preservation(self):
        """Critical test: decomposed rewards sum to final reward."""
        final_reward = 10.0
        turns = [
            TurnInfo(0, "Agent1", "text1"),
            TurnInfo(1, "Agent1", "text2"),
            TurnInfo(2, "Agent1", "text3"),
        ]
        stance_signals = {0: 1.0, 1: 2.0, 2: 1.0}
        config = RewardDecompositionConfig()
        
        decomposed = decompose_final_reward_over_turns(
            final_reward, turns, stance_signals, config
        )
        
        computed_sum = sum(decomposed.values())
        assert computed_sum == pytest.approx(final_reward, rel=1e-5)
    
    def test_uniform_fallback(self):
        """Test fallback to uniform when all signals are zero."""
        final_reward = 9.0
        turns = [
            TurnInfo(0, "Agent1", "text1"),
            TurnInfo(1, "Agent1", "text2"),
            TurnInfo(2, "Agent1", "text3"),
        ]
        stance_signals = {0: 0.0, 1: 0.0, 2: 0.0}
        config = RewardDecompositionConfig(
            decomposition={"fallback_uniform_if_no_signal": True}
        )
        
        decomposed = decompose_final_reward_over_turns(
            final_reward, turns, stance_signals, config
        )
        
        # Each turn should get equal reward
        expected_per_turn = final_reward / 3.0
        for turn_idx in decomposed:
            assert decomposed[turn_idx] == pytest.approx(expected_per_turn)
    
    def test_higher_signal_higher_reward(self):
        """Test that higher stance signal -> higher reward allocation."""
        final_reward = 10.0
        turns = [
            TurnInfo(0, "Agent1", "low_impact"),
            TurnInfo(1, "Agent1", "high_impact"),
        ]
        stance_signals = {0: 1.0, 1: 3.0}  # Turn 1 has 3x the signal
        config = RewardDecompositionConfig(
            decomposition_mode="stance_weighted"
        )
        
        decomposed = decompose_final_reward_over_turns(
            final_reward, turns, stance_signals, config
        )
        
        # Turn 1 should get ~75% of reward (3 out of 4 normalized weight)
        # Turn 0 should get ~25% of reward (1 out of 4 normalized weight)
        assert decomposed[1] > decomposed[0]
        assert decomposed[1] == pytest.approx(7.5, rel=0.01)
        assert decomposed[0] == pytest.approx(2.5, rel=0.01)
    
    def test_negative_final_reward(self):
        """Test decomposition with negative reward."""
        final_reward = -8.0
        turns = [
            TurnInfo(0, "Agent1", "text1"),
            TurnInfo(1, "Agent1", "text2"),
        ]
        stance_signals = {0: 1.0, 1: 1.0}
        config = RewardDecompositionConfig()
        
        decomposed = decompose_final_reward_over_turns(
            final_reward, turns, stance_signals, config
        )
        
        computed_sum = sum(decomposed.values())
        assert computed_sum == pytest.approx(final_reward)
        assert all(r < 0 for r in decomposed.values())
    
    def test_empty_turns_list(self):
        """Test with no turns."""
        final_reward = 10.0
        turns = []
        stance_signals = {}
        config = RewardDecompositionConfig()
        
        decomposed = decompose_final_reward_over_turns(
            final_reward, turns, stance_signals, config
        )
        
        assert len(decomposed) == 0
    
    def test_hybrid_mode(self):
        """Test hybrid decomposition mode."""
        final_reward = 10.0
        turns = [
            TurnInfo(0, "Agent1", "text1"),
            TurnInfo(1, "Agent1", "text2"),
        ]
        stance_signals = {0: 0.5, 1: 1.5}
        
        # Pure uniform (lambda=0)
        config_uniform = RewardDecompositionConfig(
            decomposition_mode="hybrid",
            hybrid_lambda=0.0,
        )
        decomposed_uniform = decompose_final_reward_over_turns(
            final_reward, turns, stance_signals, config_uniform
        )
        
        # Pure stance-weighted (lambda=1)
        config_stance = RewardDecompositionConfig(
            decomposition_mode="hybrid",
            hybrid_lambda=1.0,
        )
        decomposed_stance = decompose_final_reward_over_turns(
            final_reward, turns, stance_signals, config_stance
        )
        
        # Uniform should give equal rewards
        assert decomposed_uniform[0] == pytest.approx(5.0)
        assert decomposed_uniform[1] == pytest.approx(5.0)
        
        # Stance-weighted should favor turn 1 (higher signal)
        assert decomposed_stance[1] > decomposed_stance[0]


class TestBuildTurnLevelRewards:
    """Test the main orchestration function."""
    
    def test_basic_orchestration(self):
        """Test end-to-end call."""
        final_reward = 10.0
        messages = [
            [("Environment", "Agent1", "Start"), ("Environment", "Agent2", "Start")],
            [("Agent1", "Environment", "Move 1"), ("Agent2", "Environment", "Move 1")],
            [("Agent1", "Environment", "Move 2"), ("Agent2", "Environment", "Move 2")],
        ]
        agents = ["Agent1", "Agent2"]
        
        turn_rewards, log_dict = build_turn_level_rewards_for_episode(
            final_reward=final_reward,
            intervened_agent_id="Agent1",
            other_agent_id="Agent2",
            messages=messages,
            agents=agents,
        )
        
        assert len(turn_rewards) == 2
        assert log_dict["intervened_agent_id"] == "Agent1"
        assert log_dict["final_reward"] == final_reward
        
        # Sum verification
        computed_sum = sum(info.final_decomposed_reward for info in turn_rewards)
        assert computed_sum == pytest.approx(final_reward)
    
    def test_no_intervened_turns(self):
        """Test when intervened agent has no turns."""
        final_reward = 10.0
        messages = [
            [("Environment", "Agent1", "Start"), ("Environment", "Agent2", "Start")],
            [("Agent2", "Environment", "Only Agent2 moves")],
        ]
        agents = ["Agent1", "Agent2"]
        
        turn_rewards, log_dict = build_turn_level_rewards_for_episode(
            final_reward=final_reward,
            intervened_agent_id="Agent1",
            other_agent_id="Agent2",
            messages=messages,
            agents=agents,
        )
        
        assert len(turn_rewards) == 0
        assert "error" in log_dict


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""
    
    def test_realistic_negotiation_episode(self):
        """Test with a realistic negotiation episode structure."""
        final_reward = 15.5
        messages = [
            [("Environment", "Negotiator", "Negotiation starting"), ("Environment", "CounterpartyGent", "Negotiation starting")],
            [("Negotiator", "Environment", "I want item X"), ("CounterpartyGent", "Environment", "I also want X")],
            [("Negotiator", "Environment", "Let's compromise"), ("CounterpartyGent", "Environment", "You can have X, I take Y")],
            [("Negotiator", "Environment", "Fair deal!"), ("CounterpartyGent", "Environment", "Great!")],
        ]
        agents = ["Negotiator", "CounterpartyGent"]
        
        # Suppose Negotiator made 3 moves with different importance
        stance_annotations = {
            1: {"stance": -1.0, "action_type": "non-strategic"},  # Counterparty was proself
            2: {"stance": 1.0, "action_type": "offer"},           # Counterparty became prosocial (big change!)
            3: {"stance": 1.0, "action_type": "info"},            # Settled
        }
        
        config = RewardDecompositionConfig(
            decomposition_mode="stance_weighted",
            decomposition={
                "alpha_base": 1.0,
                "beta_stance": 2.0,
                "action_bonus": 0.3,
                "fallback_uniform_if_no_signal": True,
            }
        )
        
        turn_rewards, log_dict = build_turn_level_rewards_for_episode(
            final_reward=final_reward,
            intervened_agent_id="Negotiator",
            other_agent_id="CounterpartyGent",
            messages=messages,
            agents=agents,
            stance_annotations=stance_annotations,
            config=config,
            verbose=True,
        )
        
        assert len(turn_rewards) == 3
        computed_sum = sum(info.final_decomposed_reward for info in turn_rewards)
        assert computed_sum == pytest.approx(final_reward)
        
        # Turn 2 should get more reward (bigger stance change from -1 to 1 = 2.0 magnitude)
        # plus it has an "offer" action
        rewards_by_turn = {info.turn_index: info.final_decomposed_reward for info in turn_rewards}
        assert rewards_by_turn[2] > rewards_by_turn[1]
        assert rewards_by_turn[2] > rewards_by_turn[3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
