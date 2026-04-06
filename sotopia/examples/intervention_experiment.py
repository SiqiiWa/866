#!/usr/bin/env python3
"""
Intervention Experiment Script

This script runs intervention experiments on negotiation dialogues.
For each dialogue in important_turns.json, it intervenes on the top 3 important turns
for each agent (6 interventions per dialogue), generates new utterances, and evaluates the outcomes with three rewards:
1. SVI (Subjective Value Index) - LLM-based satisfaction scores
2. Utility - Preference-based utility scores
3. Joint Utility - Combined utility scores

Usage:
    python intervention_experiment.py --num_dialogues 5  # Test with 5 dialogues
    python intervention_experiment.py --num_dialogues 100  # Full run
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, List, Tuple, Any
from pathlib import Path
import pandas as pd
import argparse

import gin
from pydantic import validate_call
import rich

from sotopia.agents import Agents, LLMAgent
from sotopia.database import EpisodeLog, EnvironmentProfile, AgentProfile, RelationshipType
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator
from sotopia.generation_utils import agenerate, StrOutputParser
from sotopia.messages import AgentAction, Message, Observation, SimpleMessage
from sotopia.samplers import BaseSampler, EnvAgentCombo
from sotopia.server import arun_one_episode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("/home/swang4/866/data/baseline_2")
IMPORTANT_TURNS_FILE = DATA_DIR / "important_turns.json"
STANCE_LABELS_FILE = DATA_DIR / "stance_labels.json"
RESULTS_FILE = Path("/home/swang4/866/intervention_results.json")

# Model configuration
MODEL_NAME = "custom/openai/Qwen3-8B@http://localhost:8000/v1"
os.environ["CUSTOM_API_KEY"] = "EMPTY"

class InterventionExperiment:
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        """Load all necessary data."""
        self.important_turns = self.load_important_turns()
        self.stance_labels = self.load_stance_labels()
        self.dialogue_data = self.load_dialogue_data()
        self.results = []

    def load_important_turns(self) -> Dict:
        """Load important turns data."""
        with open(IMPORTANT_TURNS_FILE, 'r') as f:
            return json.load(f)

    def load_stance_labels(self) -> Dict:
        """Load stance labels data."""
        with open(STANCE_LABELS_FILE, 'r') as f:
            return json.load(f)

    def load_dialogue_data(self) -> pd.DataFrame:
        """Load dialogue data from parquet."""
        return pd.read_parquet("/home/swang4/866/data/train-00000-of-00001.parquet")

    def get_participant_info(self, dialogue_index: int) -> Dict:
        """Get participant info for a specific dialogue."""
        return self.dialogue_data.iloc[dialogue_index]['participant_info']
        """Load dialogue data from parquet file."""
        parquet_file = Path("/home/swang4/866/data/train-00000-of-00001.parquet")
        return pd.read_parquet(parquet_file)

    def load_important_turns(self) -> Dict:
        """Load important turns data."""
        with open(IMPORTANT_TURNS_FILE, 'r') as f:
            return json.load(f)

    def load_stance_labels(self) -> Dict:
        """Load stance labels data."""
        with open(STANCE_LABELS_FILE, 'r') as f:
            return json.load(f)

    def get_important_turns(self, dialogue_index: int, agent_perspective: str, top_k: int = 3) -> List[Dict]:
        """Get top-k important turns for a specific dialogue and agent."""
        for result in self.important_turns["results"]:
            if (
                result["dialogue_index"] == dialogue_index
                and result["perspective"] == f"mturk_agent_{agent_perspective}"
            ):
                selected_turns = result["analysis"].get("selected_turns", [])
                sorted_turns = sorted(
                    selected_turns,
                    key=lambda turn: turn.get("importance_rank", 999),
                )
                return sorted_turns[:top_k]
        return []

    def get_agent_preferences(self, dialogue_index: int, agent_perspective: str) -> Dict:
        """Get agent preferences for a specific dialogue and agent."""
        for result in self.stance_labels["results"]:
            if (result["dialogue_index"] == dialogue_index and
                result["perspective"] == f"mturk_agent_{agent_perspective}"):
                return json.loads(result["own_preferences"])
        return None

    def get_dialogue_history(self, dialogue_index: int) -> List[Dict]:
        """Get the full dialogue history for a specific dialogue."""
        dialogue_row = self.dialogue_data.iloc[dialogue_index]
        chat_logs = dialogue_row['chat_logs']
        
        # Convert to a more usable format
        dialogue_history = []
        for i, turn in enumerate(chat_logs):
            dialogue_history.append({
                'turn_index': i,
                'speaker': turn['id'],
                'text': turn['text'],
                'task_data': turn['task_data']
            })
        
        return dialogue_history

    def create_intervention_prompt(self, turn_data: Dict, agent_perspective: str) -> str:
        """Create the intervention prompt for the model."""
        return f"""
You are participating in a negotiation. This is a critical turn where you need to improve your response.

Current situation:
- Your original response: "{turn_data['text']}"
- Analysis: {turn_data['why_important']}
- Improvement suggestion: {turn_data['better_direction']}

Please generate a new, improved utterance that follows the better direction.
The new utterance should be more cooperative and lead to better negotiation outcomes.

Respond with only the new utterance text, nothing else.
"""

    def extract_final_allocation(
        self,
        dialogue_history: List[Dict],
        fallback_history: List[Dict] | None = None,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Extract final allocation from dialogue history."""
        def parse_from_history(history: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]] | None:
            for turn in reversed(history):
                text = turn.get("text", "") or ""
                if "submit-deal" in text.lower() or "submit deal" in text.lower():
                    task_data = turn.get("task_data", {})
                    if "issue2youget" in task_data and "issue2theyget" in task_data:
                        agent2_allocation = {}
                        agent1_allocation = {}
                        for item in ["Firewood", "Water", "Food"]:
                            agent2_allocation[item] = int(task_data["issue2youget"].get(item, "0") or "0")
                            agent1_allocation[item] = int(task_data["issue2theyget"].get(item, "0") or "0")
                        return agent1_allocation, agent2_allocation
            return None

        allocation = parse_from_history(dialogue_history)
        if allocation:
            return allocation

        if fallback_history is not None:
            allocation = parse_from_history(fallback_history)
            if allocation:
                return allocation

        logger.warning("No Submit-Deal found in dialogue")
        return {"Firewood": 0, "Water": 0, "Food": 0}, {"Firewood": 0, "Water": 0, "Food": 0}

    async def continue_dialogue_from_intervention(
        self,
        dialogue_index: int,
        original_history: List[Dict],
        important_turn_index: int,
        new_utterance: str,
        max_turns: int = 30,
    ) -> List[Dict]:
        """Continue the dialogue after intervention using LLM agents in a sotopia env."""
        from sotopia.agents import Agents, LLMAgent
        from sotopia.messages import AgentAction

        participant_info = self.get_participant_info(dialogue_index)
        agent1_info = participant_info.get("mturk_agent_1", {})
        agent2_info = participant_info.get("mturk_agent_2", {})

        # Minimal agent profiles
        agent1_profile = AgentProfile(
            first_name="Agent1",
            last_name="",
            age=agent1_info.get("demographics", {}).get("age", 30),
            occupation=agent1_info.get("demographics", {}).get("occupation", ""),
            gender=agent1_info.get("demographics", {}).get("gender", ""),
            gender_pronoun=agent1_info.get("demographics", {}).get("gender_pronoun", ""),
            public_info="",
            big_five="",
            personality_and_values="",
            decision_making_style="",
            secret="",
        )
        agent2_profile = AgentProfile(
            first_name="Agent2",
            last_name="",
            age=agent2_info.get("demographics", {}).get("age", 25),
            occupation=agent2_info.get("demographics", {}).get("occupation", ""),
            gender=agent2_info.get("demographics", {}).get("gender", ""),
            gender_pronoun=agent2_info.get("demographics", {}).get("gender_pronoun", ""),
            public_info="",
            big_five="",
            personality_and_values="",
            decision_making_style="",
            secret="",
        )

        env_profile = EnvironmentProfile(
            codename=f"intervention-{dialogue_index}",
            source="intervention_experiment",
            scenario="Resource negotiation over firewood, water, food. If you reach a deal, say 'Submit-Deal' and specify the allocations in the format: issue2youget: Firewood=X, Water=Y, Food=Z; issue2theyget: Firewood=A, Water=B, Food=C",
            agent_goals=[
                "Negotiate effectively for your prioritized resources. If you reach a deal, say 'Submit-Deal' with the allocations.",
                "Negotiate effectively for your prioritized resources. If you reach a deal, say 'Submit-Deal' with the allocations.",
            ],
            relationship=RelationshipType.stranger,
        )

        env = ParallelSotopiaEnv(
            env_profile=env_profile,
            action_order="round-robin",
            evaluators=[],
            model_name=MODEL_NAME,
            available_action_types={"none", "speak", "non-verbal communication", "action", "leave", "submit-deal"}
        )

        agents = Agents(
            {
                "mturk_agent_1": LLMAgent(agent_name="mturk_agent_1", agent_profile=agent1_profile, model_name=MODEL_NAME),
                "mturk_agent_2": LLMAgent(agent_name="mturk_agent_2", agent_profile=agent2_profile, model_name=MODEL_NAME),
            }
        )

        obs = env.reset(agents=agents)

        for idx, agent_name in enumerate(env.agents):
            agents[agent_name].goal = env_profile.agent_goals[idx]

        continued_history: List[Dict] = []

        def parse_allocations_from_text(text: str) -> Dict:
            """Parse allocations from submit-deal text."""
            import re
            task_data = {"data": "", "issue2youget": {}, "issue2theyget": {}}
            # Look for patterns like issue2youget: Firewood=2, Water=1, Food=0; issue2theyget: Firewood=1, Water=2, Food=3
            youget_match = re.search(r'issue2youget:\s*(.+?)(?:;\s*issue2theyget|$)', text, re.IGNORECASE)
            theyget_match = re.search(r'issue2theyget:\s*(.+?)(?:$|;)', text, re.IGNORECASE)
            if youget_match:
                for item in youget_match.group(1).split(','):
                    item = item.strip()
                    if '=' in item:
                        key, val = item.split('=', 1)
                        key = key.strip().title()
                        val = val.strip()
                        if key in ["Firewood", "Water", "Food"]:
                            task_data["issue2youget"][key] = val
            if theyget_match:
                for item in theyget_match.group(1).split(','):
                    item = item.strip()
                    if '=' in item:
                        key, val = item.split('=', 1)
                        key = key.strip().title()
                        val = val.strip()
                        if key in ["Firewood", "Water", "Food"]:
                            task_data["issue2theyget"][key] = val
            return task_data

        def is_repetitive(history: List[Dict], threshold: int = 3) -> bool:
            """Check if the last few turns are repetitive."""
            if len(history) < threshold:
                return False
            recent_texts = [turn.get("text", "").lower().strip() for turn in history[-threshold:]]
            # Check if all recent texts are the same or very similar
            first_text = recent_texts[0]
            return all(text == first_text or text in first_text or first_text in text for text in recent_texts)

        # Replay through intervention point
        for turn_index, turn in enumerate(original_history):
            if turn_index > important_turn_index:
                break

            speaker = turn["speaker"]
            text_to_use = new_utterance if turn_index == important_turn_index else turn["text"]

            act = AgentAction(action_type="speak", argument=text_to_use, to=[])
            actions = {
                agent_name: (act if agent_name == speaker else AgentAction(action_type="none", argument="", to=[]))
                for agent_name in env.agents
            }

            obs, _, terminated, _, _ = env.step(actions)
            for agent_name in env.agents:
                agents[agent_name].recv_message("Environment", obs[agent_name])

            continued_history.append(
                {
                    "turn_index": turn_index,
                    "speaker": speaker,
                    "text": text_to_use,
                    "task_data": turn.get("task_data", {}),
                }
            )

            if all(terminated.values()):
                return continued_history

        # Continue with model-generated turns
        extra_turn = 0
        while not all(terminated.values()) and extra_turn < max_turns:
            action_requests = {
                agent_name: await agents[agent_name].aact(obs[agent_name])
                for agent_name in env.agents
            }

            for agent_name, action_obj in action_requests.items():
                if action_obj.action_type != "none":
                    text = action_obj.argument
                    task_data = {}
                    if action_obj.action_type == "submit-deal":
                        # Parse allocations from text
                        task_data = parse_allocations_from_text(text)
                    continued_history.append(
                        {
                            "turn_index": len(continued_history),
                            "speaker": agent_name,
                            "text": text,
                            "task_data": task_data,
                        }
                    )

            if is_repetitive(continued_history) or any(action_obj.action_type in ["leave", "submit-deal"] for action_obj in action_requests.values()):
                return continued_history

            obs, _, terminated, _, _ = env.step(action_requests)
            for agent_name in env.agents:
                agents[agent_name].recv_message("Environment", obs[agent_name])

            extra_turn += 1

            if all(terminated.values()):
                return continued_history

        return continued_history

    def calculate_utility(self, preferences: Dict[str, str], allocation: Dict[str, int]) -> float:
        """Calculate normalized utility score based on preferences and allocation."""
        # preferences: {"Low": "Water", "Medium": "Food", "High": "Firewood"}
        # allocation: {"Water": count, "Food": count, "Firewood": count}

        priority_scores = {"Low": 3, "Medium": 4, "High": 5}

        total_utility = 0
        for priority, item in preferences.items():
            count = allocation.get(item, 0)
            total_utility += count * priority_scores[priority]

        # Max possible utility: 3 Water (3pts) + 3 Food (4pts) + 3 Firewood (5pts) = 9+12+15=36
        max_utility = 36
        normalized_utility = total_utility / max_utility if max_utility > 0 else 0

        return normalized_utility

    async def evaluate_svi(self, dialogue_index: int, dialogue_history: List[Dict], agent_perspective: str) -> Tuple[float, Dict[str, int]]:
        """Use machine_SVI.py logic to evaluate Subjective Value Index (SVI) for an agent."""
        try:
            import sys
            sys.path.append('/home/swang4/866/Baseline_2')
            from machine_SVI import build_prompt, query_model, SVI_14

            participant_info = self.get_participant_info(dialogue_index)
            pov_agent_id = f"mturk_agent_{agent_perspective}"
            example = {
                "participant_info": participant_info,
                "chat_logs": dialogue_history
            }

            prompt = build_prompt(example, pov_agent_id)
            pred = query_model(prompt)

            svi_scores = pred.get("svi_scores", {})
            total_score = 0
            count = 0
            processed_scores = {}
            for q in SVI_14:
                qid = q["qid"]
                if qid in svi_scores:
                    raw_score = svi_scores[qid].get("score", 4)
                    if isinstance(raw_score, str):
                        import re
                        m = re.search(r"\d+", raw_score)
                        if m:
                            raw_score = int(m.group())
                        else:
                            raw_score = 4
                    score = max(1, min(7, int(raw_score)))
                    processed_scores[qid] = score
                    total_score += score
                    count += 1

            if count == 0:
                return 5.0, {}

            avg_score = total_score / count
            normalized_score = (avg_score - 1) * (10 / 6)
            return min(max(normalized_score, 0), 10), processed_scores

        except Exception as e:
            logger.warning(f"Failed to evaluate SVI with machine_SVI logic: {e}")
            return 5.0, {}

    def calculate_joint_utility(self, utility1: float, utility2: float) -> float:
        """Calculate joint utility as sqrt(utility1 * utility2)."""
        import math
        return math.sqrt(utility1 * utility2)

    def format_dialogue_as_you_them(self, dialogue_history: List[Dict], pov_agent_id: str) -> str:
        """Format dialogue as YOU/THEM from the perspective of pov_agent_id."""
        lines = []
        for turn in dialogue_history:
            speaker = turn.get("speaker", turn.get("id", ""))
            text = turn.get("text", "")
            if not text:
                continue
            prefix = "YOU" if str(speaker) == str(pov_agent_id) else "THEM"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)

    async def run_intervention_experiment(self, dialogue_index: int, intervene_agent: str, important_turn: Dict):
        """Run intervention experiment for a specific dialogue, agent, and important turn."""
        logger.info(
            f"Running intervention for dialogue {dialogue_index}, agent {intervene_agent}, "
            f"importance_rank={important_turn.get('importance_rank')}, turn_index={important_turn.get('turn_index')}"
        )

        # Get preferences
        preferences = self.get_agent_preferences(dialogue_index, intervene_agent)
        if not preferences:
            logger.warning(f"No preferences found for dialogue {dialogue_index}, agent {intervene_agent}")
            return

        # Get dialogue history up to the important turn
        dialogue_history = self.get_dialogue_history(dialogue_index)
        history_up_to_turn = [turn for turn in dialogue_history if turn['turn_index'] < important_turn['turn_index']]

        # Create intervention prompt
        intervention_prompt = self.create_intervention_prompt(important_turn, intervene_agent)

        # Generate new utterance
        try:
            response = await agenerate(
                model_name=MODEL_NAME,
                template=intervention_prompt,
                input_values={},
                output_parser=StrOutputParser(),
                temperature=0.7,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            new_utterance = str(response).strip()
        except Exception as e:
            logger.error(f"Failed to generate new utterance: {e}")
            return

        # Calculate utilities from original preferences (will reassign after continuation)
        agent1_prefs = self.get_agent_preferences(dialogue_index, "1")
        agent2_prefs = self.get_agent_preferences(dialogue_index, "2")

        utility1 = 0.0
        utility2 = 0.0

        # Create intervened dialogue history (replace important turn text with new utterance)
        intervened_history = []
        for turn in dialogue_history:
            if turn["turn_index"] == important_turn["turn_index"]:
                modified_turn = dict(turn)
                modified_turn["text"] = new_utterance
                modified_turn["intervened"] = True
                intervened_history.append(modified_turn)
            else:
                intervened_history.append(turn)

        # Continue simulation from intervention and get completed dialogue
        completed_history = await self.continue_dialogue_from_intervention(
            dialogue_index,
            dialogue_history,
            important_turn["turn_index"],
            new_utterance,
        )

        # If no continuation produced, fall back to intervened history
        if not completed_history:
            completed_history = intervened_history

        # Extract final allocation from completed history, fallback to original when needed
        agent1_allocation, agent2_allocation = self.extract_final_allocation(
            completed_history, fallback_history=dialogue_history
        )

        # Recompute utilities from completed history allocation
        utility1 = self.calculate_utility(agent1_prefs, agent1_allocation)
        utility2 = self.calculate_utility(agent2_prefs, agent2_allocation)

        # Evaluate SVI scores using machine_SVI prompt logic on completed history
        svi1_avg, svi1_scores = await self.evaluate_svi(dialogue_index, completed_history, "1")
        svi2_avg, svi2_scores = await self.evaluate_svi(dialogue_index, completed_history, "2")

        # Calculate joint utility
        joint_utility = self.calculate_joint_utility(utility1, utility2)

        # Determine if submit-deal occurred in completed history
        def check_ended_by_agreement(history: List[Dict]) -> bool:
            for turn in history:
                text = turn.get("text", "").lower()
                if ("submit-deal" in text or "submit deal" in text or 
                    "deal struck" in text or "agreement reached" in text or 
                    "we have a deal" in text or "deal made" in text):
                    return True
            return False

        ended_by_submit_deal = check_ended_by_agreement(completed_history)

        # Store results
        result = {
            "dialogue_index": dialogue_index,
            "intervene_agent": intervene_agent,
            "important_turn_rank": important_turn.get("importance_rank"),
            "important_turn_index": important_turn.get("turn_index"),
            "important_turn": important_turn,
            "new_utterance": new_utterance,
            "preferences": preferences,
            "original_history": dialogue_history,
            "intervened_history": intervened_history,
            "completed_history": completed_history,
            "ended_by_submit_deal": ended_by_submit_deal,
            "outcome": {
                "agent1_allocation": agent1_allocation,
                "agent2_allocation": agent2_allocation
            },
            "svi_scores": {
                "agent1": svi1_scores,
                "agent2": svi2_scores
            },
            "rewards": {
                "svi_agent1": svi1_avg,
                "svi_agent2": svi2_avg,
                "utility_agent1": utility1,
                "utility_agent2": utility2,
                "joint_utility": joint_utility
            }
        }

        self.results.append(result)
        logger.info(f"Completed intervention for dialogue {dialogue_index}, agent {intervene_agent}")

    async def run_all_experiments(self, num_dialogues=100):
        """Run experiments for specified number of dialogues (6 interventions per dialogue)."""
        total_dialogues = min(num_dialogues, len(self.dialogue_data))
        for dialogue_index in range(total_dialogues):
            for agent in ["1", "2"]:
                important_turns = self.get_important_turns(dialogue_index, agent, top_k=3)
                if not important_turns:
                    logger.warning(f"No important turns found for dialogue {dialogue_index}, agent {agent}")
                    continue

                for turn in important_turns:
                    await self.run_intervention_experiment(dialogue_index, agent, turn)
                    # Save results after each intervention
                    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(self.results, f, ensure_ascii=False, indent=2)
                    logger.info(
                        f"Saved {len(self.results)} results to {RESULTS_FILE} "
                        f"after dialogue {dialogue_index}, agent {agent}, "
                        f"rank {turn.get('importance_rank')}"
                    )

        logger.info(f"Completed all experiments. Total results: {len(self.results)}")
async def main():
    parser = argparse.ArgumentParser(description='Run intervention experiments on negotiation dialogues.')
    parser.add_argument('--num_dialogues', type=int, default=100, help='Number of dialogues to process (default: 100)')
    args = parser.parse_args()

    experiment = InterventionExperiment()
    await experiment.run_all_experiments(args.num_dialogues)

if __name__ == "__main__":
    asyncio.run(main())