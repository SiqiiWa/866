"""Reward decomposition for intervention training.

This module implements turn-level reward decomposition for RL training where only the
intervened agent's moves are optimized. Rewards are decomposed using stance change as
a local implicit signal, inspired by the GELI paper (arXiv:2403.11330).

Key concept:
- Decompose a single episode-level final reward R_final into per-turn rewards 
  for only the intervened agent's turns
- Use stance change of the other agents as an "importance weight" for each turn
- Ensures sum of decomposed rewards equals the final reward
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class RewardDecompositionConfig:
    """Configuration for reward decomposition.
    
    Attributes:
        reward_weights: Weights for different reward components
            - svi: weight on SVI score
            - self: weight on individual utility
            - joint: weight on joint utility
        decomposition: Settings for turn-level decomposition
            - alpha_base: base weight per turn (default 1.0)
            - beta_stance: multiplier for stance signal importance (default 1.0)
            - action_bonus: extra weight for high-impact action types (default 0.5)
            - fallback_uniform_if_no_signal: if True, use uniform when no signal (default True)
            - normalize_to_final_reward: always normalize to sum to final (default True)
        decomposition_mode: One of ["uniform", "stance_weighted", "hybrid"]
            - uniform: all turns get equal reward
            - stance_weighted: weight by stance signal
            - hybrid: blend between uniform and stance-weighted (default)
        hybrid_lambda: blending parameter for hybrid mode, 0 <= lambda <= 1
            - 0 = fully uniform
            - 1 = fully stance-weighted
            - default 0.8
    """
    
    # Reward weight config
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "svi": 1.0,
        "self": 1.0,
        "joint": 1.0,
    })
    
    # Decomposition config
    decomposition: Dict[str, Any] = field(default_factory=lambda: {
        "alpha_base": 1.0,
        "beta_stance": 1.0,
        "action_bonus": 0.5,
        "fallback_uniform_if_no_signal": True,
        "normalize_to_final_reward": True,
    })
    
    decomposition_mode: str = "hybrid"
    hybrid_lambda: float = 0.8
    
    def __post_init__(self):
        """Validate configuration."""
        if self.decomposition_mode not in ("uniform", "stance_weighted", "hybrid"):
            raise ValueError(f"Unknown decomposition_mode: {self.decomposition_mode}")
        if not (0.0 <= self.hybrid_lambda <= 1.0):
            raise ValueError(f"hybrid_lambda must be in [0, 1], got {self.hybrid_lambda}")


@dataclass
class TurnInfo:
    """Information about a single turn in an episode.
    
    Attributes:
        turn_index: Turn number in the episode
        speaker_agent_id: ID of the agent who spoke in this turn
        text: The utterance text
        action_type: Type of action (if available from annotation)
    """
    turn_index: int
    speaker_agent_id: str
    text: Optional[str] = None
    action_type: Optional[str] = None


@dataclass
class TurnLevelRewardInfo:
    """Per-turn reward information for training.
    
    Attributes:
        turn_index: Turn number
        agent_id: ID of the intervened agent
        text: Utterance text (for logging)
        final_decomposed_reward: The reward assigned to this turn
        stance_signal: The stance-change signal strength (for logging)
        decomposition_weight: The normalized weight used (for logging)
    """
    turn_index: int
    agent_id: str
    text: Optional[str] = None
    final_decomposed_reward: float = 0.0
    stance_signal: float = 0.0
    decomposition_weight: float = 0.0


def compute_final_reward(
    svi_score: float,
    agent_self_utility: float,
    joint_utility: float,
    config: RewardDecompositionConfig,
) -> float:
    """Compute the episode-level final reward for the intervened agent.
    
    Args:
        svi_score: SVI score that other agent(s) give to interventioned agent
        agent_self_utility: Individual utility score of the intervened agent
        joint_utility: Joint utility score of both agents
        config: Reward decomposition configuration
        
    Returns:
        The final scalar reward for the episode.
    """
    w_svi = config.reward_weights.get("svi", 1.0)
    w_self = config.reward_weights.get("self", 1.0)
    w_joint = config.reward_weights.get("joint", 1.0)
    
    final_reward = (
        w_svi * svi_score +
        w_self * agent_self_utility +
        w_joint * joint_utility
    )
    return final_reward


def extract_intervened_agent_turns(
    messages: List[List[Tuple[str, str, str]]],
    intervened_agent_id: str,
) -> List[TurnInfo]:
    """Extract turns where the intervened agent spoke.
    
    Args:
        messages: Episode messages organized by turn. Each turn is a list of
                 (sender, receiver, text) tuples. Typically turn[0] is environment
                 messages to agent 1, turn[1] is environment messages to agent 2,
                 and turn[2+] are agent utterances.
        intervened_agent_id: ID of the agent being trained
        
    Returns:
        List of TurnInfo objects for turns where intervened_agent_id spoke.
    """
    intervened_turns = []
    
    for turn_idx, turn_messages in enumerate(messages):
        # Turn messages are (sender, receiver, text) tuples
        # We look for messages where sender is the intervened agent 
        # and receiver is the Environment (meaning it's an agent action)
        for sender, receiver, text in turn_messages:
            if sender == intervened_agent_id and receiver == "Environment":
                # This is an agent action/utterance
                intervened_turns.append(
                    TurnInfo(
                        turn_index=turn_idx,
                        speaker_agent_id=intervened_agent_id,
                        text=text,
                        action_type=None,  # Will be filled in if stance annotation available
                    )
                )
    
    return intervened_turns


def compute_stance_signal_for_turn(
    agent_turn: TurnInfo,
    adjacent_other_agent_utterances: Dict[str, Optional[Tuple[str, str]]],
    stance_annotations: Optional[Dict[int, Dict[str, Any]]] = None,
    action_bonus: float = 0.5,
) -> float:
    """Compute stance-change signal strength for a single agent turn.
    
    The stance signal measures how much the other agent's stance changed (in magnitude)
    as a result of this utterance. Turns with larger stance changes are more "influential"
    and should receive more reward mass.
    
    Args:
        agent_turn: Information about the agent's turn
        adjacent_other_agent_utterances: Dict with keys "prev_utterance" and "next_utterance",
                                        each value is (stance_label, text) or None
        stance_annotations: Optional dict mapping turn indices to stance annotation dicts.
                          Each annotation should have keys: "stance" (str), "action_type" (str).
                          Stance values: "prosocial" (+1), "neutral" (0), "proself" (-1).
        action_bonus: Extra weight for high-impact action types like "influence" or "offer".
        
    Returns:
        A scalar stance signal (non-negative). 0 means no detected stance change.
    """
    prev_utterance = adjacent_other_agent_utterances.get("prev_utterance")
    next_utterance = adjacent_other_agent_utterances.get("next_utterance")
    
    # Helper to convert stance label to scalar
    def stance_to_scalar(stance_label: Optional[str]) -> float:
        if stance_label == "prosocial":
            return 1.0
        elif stance_label == "proself":
            return -1.0
        else:  # "neutral" or None
            return 0.0
    
    # Extract stance values
    prev_stance = stance_to_scalar(prev_utterance[0] if prev_utterance else None)
    next_stance = stance_to_scalar(next_utterance[0] if next_utterance else None)
    
    # Compute stance-change magnitude
    stance_delta = abs(next_stance - prev_stance)
    
    # Check if next utterance is high-impact type
    action_bonus_term = 0.0
    if next_utterance and stance_annotations:
        # Try to find stance annotation for the next turn
        # This is a best-effort lookup; we don't require it
        if isinstance(next_utterance, tuple) and len(next_utterance) > 2:
            next_turn_idx = next_utterance[2]  # Optional turn index
            if isinstance(next_turn_idx, int) and next_turn_idx in stance_annotations:
                annotation = stance_annotations[next_turn_idx]
                action_type = annotation.get("action_type", "")
                if action_type in ("influence", "offer"):
                    action_bonus_term = action_bonus
    
    signal = stance_delta + action_bonus_term
    return signal


def decompose_final_reward_over_turns(
    final_reward: float,
    intervened_turns: List[TurnInfo],
    stance_signals: Dict[int, float],
    config: RewardDecompositionConfig,
) -> Dict[int, float]:
    """Decompose episode-level reward across intervened agent turns.
    
    Uses stance signal as relative importance weight. Ensures the sum of decomposed
    rewards equals the final reward (or preserves sign if negative).
    
    Args:
        final_reward: The episode-level reward to decompose
        intervened_turns: List of turns where the intervened agent spoke
        stance_signals: Dict mapping turn_index -> stance signal strength (non-negative)
        config: Reward decomposition configuration
        
    Returns:
        Dict mapping turn_index -> decomposed reward for that turn.
    """
    if not intervened_turns:
        return {}
    
    # Start with raw weights based on decomposition
    raw_weights = []
    
    alpha_base = config.decomposition.get("alpha_base", 1.0)
    beta_stance = config.decomposition.get("beta_stance", 1.0)
    fallback_uniform = config.decomposition.get("fallback_uniform_if_no_signal", True)
    
    # Calculate raw weights for each turn
    for turn in intervened_turns:
        signal = stance_signals.get(turn.turn_index, 0.0)
        raw = alpha_base + beta_stance * signal
        # Clamp to avoid negative weights
        raw = max(raw, 1e-8)
        raw_weights.append(raw)
    
    # Check if we should fall back to uniform
    total_weight = sum(raw_weights)
    all_equal = all(abs(w - raw_weights[0]) < 1e-12 for w in raw_weights)
    
    if fallback_uniform and (total_weight <= 0 or all_equal):
        # Fall back to uniform if weights are degenerate
        raw_weights = [1.0] * len(intervened_turns)
        total_weight = len(intervened_turns)
    
    # Apply decomposition mode (uniform, stance_weighted, or hybrid)
    if config.decomposition_mode == "uniform":
        final_weights = [1.0 / len(intervened_turns)] * len(intervened_turns)
    elif config.decomposition_mode == "stance_weighted":
        final_weights = [w / total_weight for w in raw_weights]
    else:  # hybrid
        uniform_weights = [1.0 / len(intervened_turns)] * len(intervened_turns)
        stance_weights = [w / total_weight for w in raw_weights]
        lam = config.hybrid_lambda
        final_weights = [
            (1.0 - lam) * u + lam * s
            for u, s in zip(uniform_weights, stance_weights)
        ]
    
    # Decompose reward
    decomposed = {}
    for turn, weight in zip(intervened_turns, final_weights):
        decomposed[turn.turn_index] = final_reward * weight
    
    # Add back other turns with 0
    for turn in intervened_turns:
        if turn.turn_index not in decomposed:
            decomposed[turn.turn_index] = 0.0
    
    return decomposed


def build_turn_level_rewards_for_episode(
    final_reward: float,
    intervened_agent_id: str,
    other_agent_id: str,
    messages: List[List[Tuple[str, str, str]]],
    agents: List[str],
    stance_annotations: Optional[Dict[int, Dict[str, Any]]] = None,
    config: Optional[RewardDecompositionConfig] = None,
    verbose: bool = False,
) -> Tuple[List[TurnLevelRewardInfo], Dict[str, Any]]:
    """Main entry point: build turn-level rewards for an episode.
    
    Orchestrates the full decomposition pipeline:
    1. Extract intervened agent's turns
    2. Compute stance signals for each turn
    3. Decompose final reward
    4. Return annotated turn-level reward info for RL training
    
    Args:
        final_reward: The episode-level final reward to decompose
        intervened_agent_id: ID of the agent being trained
        other_agent_id: ID of the other agent (for stance tracking)
        messages: Episode messages organized by turn
        agents: List of all agent IDs in the episode
        stance_annotations: Optional dict of turn indices to stance annotation data
        config: Reward decomposition config (uses default if None)
        verbose: If True, print logging information
        
    Returns:
        A tuple of:
        - List of TurnLevelRewardInfo objects (one per intervened agent turn)
        - A logging dict with decomposition details for monitoring/debugging
    """
    if config is None:
        config = RewardDecompositionConfig()
    
    # Extract interventened agent's turns
    intervened_turns = extract_intervened_agent_turns(
        messages=messages,
        intervened_agent_id=intervened_agent_id,
    )
    
    if not intervened_turns:
        if verbose:
            print(f"No turns found for intervened agent {intervened_agent_id}")
        return [], {"error": "no_intervened_turns", "intervened_agent_id": intervened_agent_id}
    
    # For each intervened turn, find neighboring other-agent utterances and compute stance signal
    stance_signals = {}
    
    for turn in intervened_turns:
        # Find other agent's utterances around this turn
        prev_other_utt = None
        next_other_utt = None
        
        # Look backwards for most recent other agent utterance
        for i in range(turn.turn_index - 1, -1, -1):
            for sender, receiver, text in messages[i]:
                if sender == other_agent_id and receiver == "Environment":
                    prev_other_utt = (
                        stance_annotations.get(i, {}).get("stance") if stance_annotations else None,
                        text,
                        i,
                    )
                    break
            if prev_other_utt:
                break
        
        # Look forwards for first next other agent utterance
        for i in range(turn.turn_index + 1, len(messages)):
            for sender, receiver, text in messages[i]:
                if sender == other_agent_id and receiver == "Environment":
                    next_other_utt = (
                        stance_annotations.get(i, {}).get("stance") if stance_annotations else None,
                        text,
                        i,
                    )
                    break
            if next_other_utt:
                break
        
        # Compute stance signal
        adjacent_utts = {
            "prev_utterance": (prev_other_utt[0], prev_other_utt[1]) if prev_other_utt else None,
            "next_utterance": (next_other_utt[0], next_other_utt[1]) if next_other_utt else None,
        }
        
        signal = compute_stance_signal_for_turn(
            agent_turn=turn,
            adjacent_other_agent_utterances=adjacent_utts,
            stance_annotations=stance_annotations,
            action_bonus=config.decomposition.get("action_bonus", 0.5),
        )
        stance_signals[turn.turn_index] = signal
    
    # Decompose final reward
    decomposed_rewards = decompose_final_reward_over_turns(
        final_reward=final_reward,
        intervened_turns=intervened_turns,
        stance_signals=stance_signals,
        config=config,
    )
    
    # Build TurnLevelRewardInfo list
    turn_rewards_info = []
    for turn in intervened_turns:
        decomposed_reward = decomposed_rewards.get(turn.turn_index, 0.0)
        # Compute normalized weight for logging
        total_reward = sum(decomposed_rewards.values())
        if abs(total_reward) > 1e-12:
            norm_weight = decomposed_reward / total_reward
        else:
            norm_weight = 0.0
        
        info = TurnLevelRewardInfo(
            turn_index=turn.turn_index,
            agent_id=intervened_agent_id,
            text=turn.text,
            final_decomposed_reward=decomposed_reward,
            stance_signal=stance_signals.get(turn.turn_index, 0.0),
            decomposition_weight=norm_weight,
        )
        turn_rewards_info.append(info)
    
    # Build logging dict
    logging_dict = {
        "intervened_agent_id": intervened_agent_id,
        "other_agent_id": other_agent_id,
        "final_reward": final_reward,
        "num_intervened_turns": len(intervened_turns),
        "stance_signals": stance_signals,
        "decomposed_rewards": decomposed_rewards,
        "decomposition_mode": config.decomposition_mode,
        "reward_weights": config.reward_weights,
        "decomposition_params": config.decomposition,
    }
    
    # Verification: sum should equal final reward (within floating point tolerance)
    computed_sum = sum(decomposed_rewards.values())
    if abs(computed_sum - final_reward) > 1e-5:
        logging_dict["warning"] = f"Sum mismatch: expected {final_reward}, got {computed_sum}"
    
    if verbose:
        print(f"Decomposed {final_reward} across {len(intervened_turns)} turns")
        print(f"Stance signals: {stance_signals}")
        print(f"Decomposed rewards: {decomposed_rewards}")
        if "warning" in logging_dict:
            print(f"Warning: {logging_dict['warning']}")
    
    return turn_rewards_info, logging_dict


def compute_decomposed_reward_vector(
    turn_rewards_info: List[TurnLevelRewardInfo],
) -> List[float]:
    """Extract the decomposed reward values as a simple vector.
    
    Args:
        turn_rewards_info: List of TurnLevelRewardInfo from build_turn_level_rewards_for_episode
        
    Returns:
        Ordered list of decomposed rewards for each turn.
    """
    return [info.final_decomposed_reward for info in turn_rewards_info]
