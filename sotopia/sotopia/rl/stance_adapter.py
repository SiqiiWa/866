"""Stance adapter module for RL reward decomposition.

This module adapts the stance annotation logic from Baseline_2/stance_prompt.py
into a reusable, module-friendly API for the RL reward decomposition pipeline.

Instead of requiring full LLM calls for every turn (which is expensive), this module
provides utilities to:
1. Parse existing stance annotations from stored JSON schemas
2. Convert stance labels to numeric values for reward decomposition
3. Cache and look up stance information efficiently
"""

from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path


# Stance label to numeric mapping
STANCE_TO_VALUE = {
    "prosocial": 1.0,
    "neutral": 0.0,
    "proself": -1.0,
}


def stance_label_to_value(stance: str) -> float:
    """Convert stance label to numeric value.
    
    Args:
        stance: One of "prosocial", "neutral", "proself"
        
    Returns:
        Numeric value: +1.0 for prosocial, 0.0 for neutral, -1.0 for proself, 0.0 for unknown
    """
    return STANCE_TO_VALUE.get(stance, 0.0)


def value_to_stance_label(value: float) -> str:
    """Convert numeric value back to stance label (for display/logging).
    
    Args:
        value: Numeric stance value
        
    Returns:
        Stance label string
    """
    if value > 0.5:
        return "prosocial"
    elif value < -0.5:
        return "proself"
    else:
        return "neutral"


def parse_stance_annotation_dict(annotation_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a stance annotation dict from the stance_prompt.py format.
    
    Expected format (from judge_stance):
    {
        "reasoning": str,
        "stance": "prosocial" | "neutral" | "proself",
        "action_type": "info" | "influence" | "offer" | "non-strategic",
        "xml_output": str,
    }
    
    Args:
        annotation_dict: Raw annotation dict from stance annotation LLM call
        
    Returns:
        Normalized dict with keys:
        - "stance": numeric value (-1, 0, or 1)
        - "action_type": str
        - "reasoning": str (original)
    """
    stance_label = annotation_dict.get("stance", "neutral")
    stance_value = stance_label_to_value(stance_label)
    
    return {
        "stance": stance_value,
        "stance_label": stance_label,
        "action_type": annotation_dict.get("action_type", "non-strategic"),
        "reasoning": annotation_dict.get("reasoning", ""),
    }


def load_stance_annotations_from_json(
    json_path: str,
    perspective: Optional[str] = None,
) -> Dict[int, Dict[str, Any]]:
    """Load stance annotations from a JSON file created by stance_prompt.py.
    
    Expected file structure (from stance_prompt.py output):
    {
        "results": [
            {
                "dialogue_index": int,
                "perspective": str,
                "annotations": [
                    {
                        "turn_index": int,
                        "speaker": str,
                        "text": str,
                        "stance": str,
                        "action_type": str,
                        ...
                    },
                    ...
                ]
            },
            ...
        ]
    }
    
    Args:
        json_path: Path to the JSON file
        perspective: Optional perspective filter (e.g., "mturk_agent_1")
        
    Returns:
        Dict mapping turn_index -> normalized annotation dict for easy lookup
    """
    annotations_by_turn = {}
    
    if not Path(json_path).exists():
        return annotations_by_turn
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could not load JSON from {json_path}: {e}")
        return annotations_by_turn
    
    results = data.get("results", [])
    
    for result in results:
        # Skip if perspective filter is set and doesn't match
        if perspective and result.get("perspective") != perspective:
            continue
        
        annotations = result.get("annotations", [])
        for annotation in annotations:
            turn_idx = annotation.get("turn_index", -1)
            if turn_idx >= 0:
                parsed = parse_stance_annotation_dict(annotation)
                annotations_by_turn[turn_idx] = parsed
    
    return annotations_by_turn


def create_stance_annotation_lookup_from_episode_annotations(
    annotations: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Create a turn-index lookup table from a list of turn annotations.
    
    This is useful when you already have annotations (e.g., from a prior pass)
    and want to use them for reward decomposition.
    
    Args:
        annotations: List of annotation dicts, each with at least "turn_index" and "stance"
        
    Returns:
        Dict mapping turn_index -> normalized annotation dict
    """
    lookup = {}
    for annotation in annotations:
        turn_idx = annotation.get("turn_index", -1)
        if turn_idx >= 0:
            parsed = parse_stance_annotation_dict(annotation)
            lookup[turn_idx] = parsed
    return lookup


def dialogue_turns_to_annotation_prompt_format(
    messages: List[List[Tuple[str, str, str]]],
    agent_id: str,
) -> str:
    """Convert episode messages to the dialogue format expected by stance_prompt.py.
    
    This is primarily for reference/debugging. The actual stance annotation happens
    outside this module using stance_prompt.py directly.
    
    Args:
        messages: Episode messages organized by turn
        agent_id: The perspective agent (will be labeled "YOU" in output)
        
    Returns:
        Dialogue string in the "YOU: ...\nTHEM: ..." format
    """
    lines = []
    for turn_messages in messages:
        for sender, receiver, text in turn_messages:
            if sender in ("Environment",):
                # Skip environment messages
                continue
            
            # Determine perspective
            if sender == agent_id:
                label = "YOU"
            else:
                label = "THEM"
            
            # Format as dialogue line
            lines.append(f"{label}: {text}")
    
    return "\n".join(lines)


def get_action_bonus_for_stance_signal(
    action_type: str,
    base_bonus: float = 0.5,
) -> float:
    """Compute action-type bonus for stance signal strength.
    
    High-impact action types get an extra bonus to their stance signal,
    making turns with "influence" or "offer" actions more likely to receive
    larger reward allocations.
    
    Args:
        action_type: One of "info", "influence", "offer", "non-strategic"
        base_bonus: Base bonus amount for high-impact actions
        
    Returns:
        Bonus amount to add to stance signal
    """
    if action_type in ("influence", "offer"):
        return base_bonus
    return 0.0


def validate_stance_annotations_consistency(
    annotations_by_turn: Dict[int, Dict[str, Any]],
    expected_num_turns: int,
) -> Tuple[bool, List[str]]:
    """Validate consistency of stance annotations for an episode.
    
    Checks:
    - All turns in expected range have annotations (or report missing)
    - Stance values are in valid range (-1 to 1)
    - Action types are recognized
    
    Args:
        annotations_by_turn: Turn-indexed annotations
        expected_num_turns: Expected number of turns
        
    Returns:
        Tuple of (is_valid: bool, issues: list of issue strings)
    """
    issues = []
    
    # Check for missing turns
    found_turn_indices = set(annotations_by_turn.keys())
    expected_turn_indices = set(range(expected_num_turns))
    missing = expected_turn_indices - found_turn_indices
    if missing:
        issues.append(f"Missing annotations for turns: {sorted(missing)}")
    
    # Check each annotation
    for turn_idx, annotation in annotations_by_turn.items():
        stance_val = annotation.get("stance", 0.0)
        if not isinstance(stance_val, (int, float)) or not (-1.0 <= stance_val <= 1.0):
            issues.append(f"Turn {turn_idx}: Invalid stance value {stance_val}")
        
        action_type = annotation.get("action_type", "non-strategic")
        if action_type not in ("info", "influence", "offer", "non-strategic"):
            issues.append(f"Turn {turn_idx}: Unknown action_type {action_type}")
    
    is_valid = len(issues) == 0
    return is_valid, issues
