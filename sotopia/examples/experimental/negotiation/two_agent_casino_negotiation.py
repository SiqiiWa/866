"""
Continue a two-agent camping negotiation from a selected important turn.

This script:
1. Loads one dialogue from the structured casino negotiation data.
2. Loads the first important turn for a chosen perspective from the important-turns file.
3. Replays the dialogue prefix up to that turn as fixed history.
4. Adds improvement instructions only to the chosen agent.
5. Continues the negotiation with two LLM agents.
6. Writes the full conversation (prefix + simulated continuation) to a JSON file.


uv run python examples/experimental/negotiation/run_batch_continuations.py \
    --base-url "http://127.0.0.1:8000/v1" \
    --model "openai/Qwen3-8B" \
    --api-key "dummy" \
  --perspective both \
  --count 100

uv run python examples/experimental/negotiation/two_agent_casino_negotiation.py \
    --base-url "http://127.0.0.1:8000/v1" \
    --model "openai/Qwen3-8B" \
    --api-key "dummy" \
    --perspective "mturk_agent_1"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

# Prefer Redis unless the user already set a backend explicitly.
os.environ.setdefault("SOTOPIA_STORAGE_BACKEND", "redis")
os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6379")

# Make the repo importable even if the current Python environment did not
# install the local package into site-packages.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from litellm import acompletion

from sotopia.agents import LLMAgent
from sotopia.database.persistent_profile import (
    AgentProfile,
    EnvironmentProfile,
    RelationshipType,
)
from sotopia.envs import ParallelSotopiaEnv
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator
from sotopia.generation_utils.generate import DEFAULT_TEMPERATURE
from sotopia.generation_utils.output_parsers import PydanticOutputParser
from sotopia.messages.message_classes import AgentAction, Observation


DEFAULT_DIALOGUE_PATH = Path(
    "/home/swang4/866/data/casino_125_174_structured.json"
)
DEFAULT_IMPORTANT_TURNS_PATH = Path(
    "/home/swang4/866/data/baseline_2/important_turns_125-174.json"
)

POINTS_BY_PRIORITY = {"High": 5, "Medium": 4, "Low": 3}
PACKAGE_TYPES = ("Water", "Food", "Firewood")
PACKAGE_COUNT_PER_TYPE = 3

CONTINUATION_TEMPLATE = """
Imagine you are {agent} in a two-person camping-supplies negotiation.

The conversation history below already happened. Do not rewrite, summarize, or alter those earlier turns.
Continue naturally from the next turn only.

Visible context includes:
- the shared negotiation setting,
- your demographic and personality profile,
- your private value ranking and reasons,
- the conversation history so far.

Important rules:
- All negotiation messages are public. Always return `"to": []`.
- Negotiate only over these extra supplies: 3 Water, 3 Food, and 3 Firewood.
- Use integer package counts only. Do not invent extra items or extra packages.
- Usually use `"action_type": "speak"`.
- Make exactly one concrete move per turn: one offer, one counteroffer, one clarification, one agreement confirmation, or one leave action.
- Keep each spoken reply concise, ideally under 80 words.
- Do not output markdown fences or any extra explanation outside the JSON object.
- If a full allocation is agreed, clearly restate the final split.
- After the final split is clearly confirmed, on your next turn you should prefer `"action_type": "leave"` instead of repeating the same deal again.

Your private negotiation objective:
{goal}

Additional private instruction for this continuation:
{additional_instructions}

Conversation history:
{history}

You are now taking Turn #{turn_number}. Available action types:
{action_list}

Return exactly one JSON object that follows this schema:
{format_instructions}
"""

REPAIR_PROMPT_TEMPLATE = """
Your previous response was not a valid JSON object matching the required schema.

Schema:
{format_instructions}

Previous response:
{bad_response}

Error:
{error}

Return only one corrected JSON object. Do not include markdown fences or any extra text.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue one negotiation from an important-turn intervention point."
    )
    parser.add_argument(
        "--dialogue-path",
        type=Path,
        default=DEFAULT_DIALOGUE_PATH,
        help="Path to the structured dialogue JSON.",
    )
    parser.add_argument(
        "--important-turns-path",
        type=Path,
        default=DEFAULT_IMPORTANT_TURNS_PATH,
        help="Path to the important-turn JSON.",
    )
    parser.add_argument(
        "--dialogue-index",
        type=int,
        default=0,
        help="Dialogue index to load.",
    )
    parser.add_argument(
        "--perspective",
        type=str,
        required=True,
        choices=["mturk_agent_1", "mturk_agent_2"],
        help="Which agent receives the intervention instructions.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="LiteLLM proxy base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("LITELLM_PROXY_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("CUSTOM_API_KEY")
        or "",
        help="Proxy API key. Defaults to LITELLM_PROXY_API_KEY, OPENAI_API_KEY, or CUSTOM_API_KEY.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model name exposed by the proxy.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum total number of turns in the final conversation.",
    )
    parser.add_argument(
        "--max-stale-turn",
        type=int,
        default=2,
        help="Terminate if agents do nothing for too many consecutive turns.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=768,
        help="Max completion tokens for each generated turn.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
        help="Reasoning effort for GPT-5 style models.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output path. If omitted, a file is written next to this script.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_dialogue_record(dialogue_path: Path, dialogue_index: int) -> dict[str, Any]:
    dataset = load_json(dialogue_path)
    if not isinstance(dataset, list):
        raise ValueError(f"Expected a list in {dialogue_path}")
    for item in dataset:
        if item.get("dialogue_index") == dialogue_index:
            return item
    raise ValueError(f"dialogue_index={dialogue_index} not found in {dialogue_path}")


def load_first_important_turn(
    important_turns_path: Path,
    dialogue_index: int,
    perspective: str,
) -> dict[str, Any]:
    payload = load_json(important_turns_path)
    results = payload.get("results", [])
    for item in results:
        if (
            item.get("dialogue_index") == dialogue_index
            and item.get("perspective") == perspective
        ):
            selected_turns = item.get("analysis", {}).get("selected_turns", [])
            if not selected_turns:
                raise ValueError(
                    f"No selected_turns found for dialogue_index={dialogue_index}, perspective={perspective}"
                )
            return selected_turns[0]
    raise ValueError(
        f"No important turn found for dialogue_index={dialogue_index}, perspective={perspective}"
    )


def format_big_five(big_five: dict[str, float]) -> str:
    ordered_traits = [
        ("extraversion", "extraversion"),
        ("agreeableness", "agreeableness"),
        ("conscientiousness", "conscientiousness"),
        ("emotional-stability", "emotional stability"),
        ("openness-to-experiences", "openness to experiences"),
    ]
    return ", ".join(
        f"{label} {big_five[key]:.1f}/7" for key, label in ordered_traits
    )


def infer_decision_style(agent_payload: dict[str, Any]) -> str:
    big_five = agent_payload["personality"]["big-five"]
    agreeableness = big_five["agreeableness"]
    conscientiousness = big_five["conscientiousness"]
    openness = big_five["openness-to-experiences"]

    traits: list[str] = []
    traits.append(
        "self-interested"
        if agent_payload["personality"]["svo"] == "proself"
        else "other-regarding"
    )
    traits.append("cooperative" if agreeableness >= 5.5 else "assertive")
    traits.append("methodical" if conscientiousness >= 5.5 else "flexible")
    traits.append("creative" if openness >= 6.0 else "practical")
    return ", ".join(traits)


def map_gender(raw_gender: str) -> str:
    normalized = raw_gender.strip().lower()
    if normalized == "male":
        return "Man"
    if normalized == "female":
        return "Woman"
    return "Nonbinary"


def map_pronoun(raw_gender: str) -> str:
    normalized = raw_gender.strip().lower()
    if normalized == "male":
        return "he/him"
    if normalized == "female":
        return "she/her"
    return "they/them"


def format_inventory() -> str:
    return ", ".join(f"{PACKAGE_COUNT_PER_TYPE} {item}" for item in PACKAGE_TYPES)


def build_private_preference_summary(agent_payload: dict[str, Any]) -> str:
    lines = [
        "You privately know your scoring rule for the extra camping supplies.",
        f"Each issue has exactly {PACKAGE_COUNT_PER_TYPE} packages available in total.",
        "Your per-package value is:",
    ]
    for priority in ("High", "Medium", "Low"):
        issue = agent_payload["value2issue"][priority]
        reason = agent_payload["value2reason"][priority]
        points = POINTS_BY_PRIORITY[priority]
        lines.append(
            f"- {issue}: {priority} priority, worth {points} points per package to you. Reason: {reason}"
        )
    lines.append(
        "Your objective is to maximize your own point total while still reaching a plausible agreement."
    )
    return "\n".join(lines)


def build_goal(agent_id: str, agent_payload: dict[str, Any]) -> str:
    preference_lines = []
    for priority in ("High", "Medium", "Low"):
        issue = agent_payload["value2issue"][priority]
        reason = agent_payload["value2reason"][priority]
        points = POINTS_BY_PRIORITY[priority]
        preference_lines.append(
            f"{priority} priority: {issue} ({points} points each). Reason: {reason}"
        )

    return "\n".join(
        [
            f"You are {agent_id}.",
            "Your negotiation task is to divide 9 extra camping-supply packages with the other camper.",
            f"There are exactly {format_inventory()} packages.",
            "You already have basic supplies, so this negotiation is only about the extra packages.",
            "Try to reach a full allocation across all 9 packages.",
            "Use your personal needs and reasons to persuade the other side.",
            "Stay consistent with your demographic and personality profile.",
            "Your private preferences are:",
            *preference_lines,
            "If both sides agree on a complete split, restate the final allocation clearly and then leave on a later turn.",
        ]
    )


def build_agent_profile(agent_id: str, agent_payload: dict[str, Any]) -> AgentProfile:
    demographics = agent_payload["demographics"]
    personality = agent_payload["personality"]
    big_five = personality["big-five"]

    public_info = (
        f"{agent_id} is on a camping trip. "
        f"Demographics: {demographics['age']} years old, {demographics['gender']}, "
        f"{demographics['ethnicity']}, education level: {demographics['education']}."
    )
    personality_and_values = (
        f"SVO: {personality['svo']}. "
        f"Big Five profile: {format_big_five(big_five)}. "
        f"Likely negotiation style: {infer_decision_style(agent_payload)}."
    )

    return AgentProfile(
        first_name=agent_id,
        last_name="",
        age=demographics["age"],
        occupation="Camper",
        gender=map_gender(demographics["gender"]),
        gender_pronoun=map_pronoun(demographics["gender"]),
        public_info=public_info,
        big_five=format_big_five(big_five),
        moral_values=["preparedness", "fairness", "group welfare"],
        personality_and_values=personality_and_values,
        decision_making_style=infer_decision_style(agent_payload),
        secret=build_private_preference_summary(agent_payload),
        tag="important_turn_continuation",
    )


def build_environment_profile(record: dict[str, Any]) -> EnvironmentProfile:
    agent_1_payload = record["agents"]["mturk_agent_1"]
    agent_2_payload = record["agents"]["mturk_agent_2"]

    scenario = (
        "Two campers are dividing extra supplies before continuing a camping trip. "
        f"Besides their basic supplies, they must negotiate over exactly 9 extra packages: {format_inventory()}. "
        "Both negotiators know the total inventory and know "
        "that their own High, Medium, and Low priorities are worth 5, 4, and 3 points per package, respectively. "
        "They should use their personal needs and character traits to justify offers, explore tradeoffs, "
        "and try to reach a final allocation that assigns all packages."
    )

    return EnvironmentProfile(
        scenario=scenario,
        relationship=RelationshipType.stranger,
        agent_goals=[
            build_goal("mturk_agent_1", agent_1_payload),
            build_goal("mturk_agent_2", agent_2_payload),
        ],
        tag="important_turn_continuation",
    )


def build_proxy_model_name(model: str, base_url: str) -> str:
    return f"custom/{model}@{base_url.rstrip('/')}"


def split_custom_model(model_name: str) -> tuple[str, str | None]:
    if model_name.startswith("custom/"):
        model_part, base_url = model_name.split("@", 1)
        # Use custom_openai/ so litellm forwards the model name unchanged to the
        # vLLM server (which registers it as e.g. "openai/Qwen3-8B").
        return model_part.replace("custom/", "custom_openai/"), base_url
    return model_name, None


def other_agent(agent_name: str) -> str:
    return "mturk_agent_2" if agent_name == "mturk_agent_1" else "mturk_agent_1"


def build_intervention_instructions(selected_turn: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are continuing from a known intervention point.",
            f"Why this point matters: {selected_turn['why_important']}",
            f"Why the original move was improvable or worth amplifying: {selected_turn['why_improvable_or_worth_amplifying']}",
            f"Better direction to follow now: {selected_turn['better_direction']}",
            "Apply this guidance from this point onward while still pursuing your own negotiation objective.",
        ]
    )


def find_matched_text_turn(
    dialogue: list[dict[str, Any]], selected_turn: dict[str, Any]
) -> dict[str, Any]:
    selected_text = str(selected_turn["text"]).strip()
    matches = [
        turn for turn in dialogue if str(turn.get("text", "")).strip() == selected_text
    ]
    if not matches:
        raise ValueError(
            "Could not match selected important-turn text in the original dialogue. "
            f"text={selected_text!r}"
        )
    if len(matches) > 1:
        print(
            "Warning: selected important-turn text matched multiple original turns; "
            f"using the first match at turn_index={matches[0]['turn_index']}"
        )
    return matches[0]


def make_blank_task_data() -> dict[str, Any]:
    return {
        "data": "",
        "issue2youget": {"Firewood": "", "Water": "", "Food": ""},
        "issue2theyget": {"Firewood": "", "Water": "", "Food": ""},
    }


def turn_to_action(turn: dict[str, Any]) -> AgentAction:
    text = turn["text"]
    if text == "Accept-Deal":
        return AgentAction(action_type="speak", argument="Accept-Deal", to=[])
    if text == "Submit-Deal":
        return AgentAction(action_type="speak", argument="Submit-Deal", to=[])
    return AgentAction(action_type="speak", argument=text, to=[])


def action_to_dialogue_text(action: AgentAction) -> str:
    if action.action_type == "speak":
        return action.argument
    if action.action_type == "leave":
        return action.argument or "Leave conversation."
    if action.action_type == "non-verbal communication":
        return f"[non-verbal communication] {action.argument}"
    if action.action_type == "action":
        return f"[action] {action.argument}"
    return ""


def extract_first_json_object(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for index, character in enumerate(text[start:], start=start):
        if in_string:
            if escape_next:
                escape_next = False
                continue
            if character == "\\":
                escape_next = True
                continue
            if character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
            continue
        if character == "{":
            depth += 1
            continue
        if character == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1].strip()
                try:
                    loaded = json.loads(candidate)
                    if isinstance(loaded, dict):
                        return candidate
                except json.JSONDecodeError:
                    return None
    return None


def build_output_turn(
    dialogue_name: str,
    turn_index: int,
    speaker: str,
    action: AgentAction,
) -> dict[str, Any]:
    return {
        "text": action_to_dialogue_text(action),
        "task_data": make_blank_task_data(),
        "id": speaker,
        "turn_index": turn_index,
        "turn_id": f"{dialogue_name}_turn_{turn_index:02d}",
        "speaker": speaker,
        "listener": other_agent(speaker),
        "generated": True,
        "action_type": action.action_type,
        "to": action.to,
    }


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def is_final_deal_confirmation(action: AgentAction) -> bool:
    if action.action_type not in {"speak", "leave"}:
        return False
    text = _normalize_text(action.argument)
    confirmation_markers = [
        "final split",
        "final allocation",
        "allocation confirmed",
        "confirmed final allocation",
        "deal confirmed",
        "agreed",
        "deal.",
        "deal?",
        "submit-deal",
        "accept-deal",
    ]
    return any(marker in text for marker in confirmation_markers)


def default_output_path(
    dialogue_name: str, dialogue_index: int, perspective: str
) -> Path:
    filename = (
        f"{dialogue_name}_dialogue_{dialogue_index:05d}_{perspective}_continued.json"
    )
    return REPO_ROOT / "examples" / "experimental" / "negotiation" / filename


class ContinuationLLMAgent(LLMAgent):
    def __init__(
        self,
        *args: Any,
        additional_instructions: str = "",
        max_completion_tokens: int = 768,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.additional_instructions = additional_instructions
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="", to=[])

        parser = PydanticOutputParser(pydantic_object=AgentAction)
        history = "\n".join(
            message.to_natural_language() for _, message in self.inbox
        )
        prompt = CONTINUATION_TEMPLATE.format(
            agent=self.agent_name,
            goal=self.goal,
            additional_instructions=self.additional_instructions or "No extra intervention instruction for you.",
            history=history,
            turn_number=obs.turn_number + 1,
            action_list=" ".join(obs.available_actions),
            format_instructions=parser.get_format_instructions(),
        )

        provider_model, base_url = split_custom_model(self.model_name)
        api_key = os.environ.get("CUSTOM_API_KEY")
        if not api_key:
            raise ValueError("CUSTOM_API_KEY is not set.")

        validation_context = None
        if self.script_background is not None:
            validation_context = {
                "agent_names": self.script_background.agent_names,
                "available_action_types": obs.available_actions,
                "sender": self.agent_name,
            }

        completion_kwargs: dict[str, Any] = {
            "model": provider_model,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": DEFAULT_TEMPERATURE,
            "max_completion_tokens": self.max_completion_tokens,
            "drop_params": True,
        }
        extra_body: dict[str, Any] = {}
        if "gpt-5" in provider_model:
            extra_body["reasoning_effort"] = self.reasoning_effort
        if "qwen" in provider_model.lower():
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            completion_kwargs["response_format"] = {"type": "json_object"}
        if extra_body:
            completion_kwargs["extra_body"] = extra_body

        print(
            f"[agent={self.agent_name}] requesting action for turn={obs.turn_number + 1} "
            f"available_actions={obs.available_actions}",
            flush=True,
        )

        model_candidates = [provider_model]

        repair_prompt = prompt
        for attempt in range(2):
            last_exception: Exception | None = None
            response = None
            for candidate_model in model_candidates:
                try:
                    response = await asyncio.wait_for(
                        acompletion(
                            **{**completion_kwargs, "model": candidate_model},
                            messages=[{"role": "user", "content": repair_prompt}],
                        ),
                        timeout=90,
                    )
                    break
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"Timed out waiting for LiteLLM proxy response for {self.agent_name} "
                        f"on turn {obs.turn_number + 1} after 90s."
                    ) from exc
                except Exception as exc:
                    last_exception = exc
                    break

            if response is None:
                assert last_exception is not None
                raise last_exception

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Model returned empty content.")

            print(
                f"[agent={self.agent_name}] received response on turn={obs.turn_number + 1}: "
                f"{content[:200]!r}",
                flush=True,
            )

            parsed_action: AgentAction | None = None
            parse_error: Exception | None = None
            parse_candidates = [content]
            extracted_json = extract_first_json_object(content)
            if extracted_json and extracted_json != content:
                parse_candidates.append(extracted_json)

            for parse_candidate in parse_candidates:
                try:
                    parsed_action = parser.parse(parse_candidate, context=validation_context)
                    break
                except Exception as exc:
                    parse_error = exc

            if parsed_action is not None:
                print(
                    f"[agent={self.agent_name}] parsed action turn={obs.turn_number + 1}: "
                    f"{parsed_action.model_dump()}",
                    flush=True,
                )
                return parsed_action

            if attempt == 1:
                raise RuntimeError(
                    f"Failed to parse agent action from proxy response: {parse_error}\nRaw response: {content}"
                ) from parse_error

            repair_prompt = REPAIR_PROMPT_TEMPLATE.format(
                format_instructions=parser.get_format_instructions(),
                bad_response=content,
                error=str(parse_error),
            )

        raise RuntimeError("Unreachable continuation state.")


async def bootstrap_prefix(
    env: ParallelSotopiaEnv,
    agent_map: dict[str, ContinuationLLMAgent],
    prefix_turns: list[dict[str, Any]],
) -> dict[str, Observation]:
    observations = env.reset(agents=agent_map, omniscient=False)
    for agent in agent_map.values():
        agent.reset()
        agent.script_background = env.background

    for agent_name, observation in observations.items():
        agent_map[agent_name].recv_message("Environment", observation)

    for index, agent_name in enumerate(env.agents):
        agent_map[agent_name].goal = env.profile.agent_goals[index]

    for turn in prefix_turns:
        speaker = turn["speaker"]
        actions = {
            name: AgentAction(action_type="none", argument="", to=[])
            for name in env.agents
        }
        actions[speaker] = turn_to_action(turn)
        observations, _, _, _, _ = await env.astep(actions)
        for agent_name, observation in observations.items():
            agent_map[agent_name].recv_message("Environment", observation)

    return observations


async def run_continuation(
    env: ParallelSotopiaEnv,
    agents: list[ContinuationLLMAgent],
    prefix_turns: list[dict[str, Any]],
    start_turn_index: int,
    dialogue_name: str,
) -> list[dict[str, Any]]:
    agent_map = {agent.agent_name: agent for agent in agents}
    observations = await bootstrap_prefix(
        env=env,
        agent_map=agent_map,
        prefix_turns=prefix_turns,
    )

    generated_turns: list[dict[str, Any]] = []
    next_turn_index = start_turn_index
    done = False
    force_leave_agents: set[str] = set()
    last_confirmation_speaker: str | None = None

    while not done:
        actions_list = await asyncio.gather(
            *[agent_map[agent_name].aact(observations[agent_name]) for agent_name in env.agents]
        )
        actions = {
            agent_name: action for agent_name, action in zip(env.agents, actions_list)
        }

        active_agents = [
            env.agents[idx] for idx, allowed in enumerate(env.action_mask) if allowed
        ]
        for agent_name in active_agents:
            if agent_name in force_leave_agents:
                actions[agent_name] = AgentAction(
                    action_type="leave",
                    argument="The deal has already been confirmed. I am leaving now.",
                    to=[],
                )
                force_leave_agents.discard(agent_name)

        repeated_final_deal = False
        for agent_name in env.agents:
            action = actions[agent_name]
            if action.action_type != "none":
                generated_turns.append(
                    build_output_turn(
                        dialogue_name=dialogue_name,
                        turn_index=next_turn_index,
                        speaker=agent_name,
                        action=action,
                    )
                )
                next_turn_index += 1
                if is_final_deal_confirmation(action):
                    if (
                        last_confirmation_speaker is not None
                        and last_confirmation_speaker != agent_name
                    ):
                        repeated_final_deal = True
                    last_confirmation_speaker = agent_name
                    force_leave_agents.update(
                        other_name
                        for other_name in env.agents
                        if other_name != agent_name
                    )
                else:
                    last_confirmation_speaker = None

        observations, _, terminated, _, _ = await env.astep(actions)
        for agent_name, observation in observations.items():
            agent_map[agent_name].recv_message("Environment", observation)

        done = all(terminated.values()) or repeated_final_deal

    return generated_turns


def write_output(
    output_path: Path,
    record: dict[str, Any],
    selected_turn: dict[str, Any],
    perspective: str,
    prefix_turns: list[dict[str, Any]],
    generated_turns: list[dict[str, Any]],
    annotation_turn_index: int,
    matched_text_turn_index: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dialogue_index": record["dialogue_index"],
        "dialogue_name": record["dialogue_name"],
        "perspective": perspective,
        "selected_important_turn": selected_turn,
        "annotation_turn_index": annotation_turn_index,
        "matched_text_turn_index": matched_text_turn_index,
        "preserved_through_turn_index": matched_text_turn_index,
        "prefix_turn_count": len(prefix_turns),
        "continued_from_turn_index": len(prefix_turns) + 1,
        "agents": record["agents"],
        "dialogue": prefix_turns + generated_turns,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


async def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError(
            "Missing proxy API key. Pass --api-key or set LITELLM_PROXY_API_KEY."
        )

    os.environ["CUSTOM_API_KEY"] = args.api_key

    record = load_dialogue_record(args.dialogue_path, args.dialogue_index)
    selected_turn = load_first_important_turn(
        args.important_turns_path, args.dialogue_index, args.perspective
    )
    annotation_turn_index = int(selected_turn["turn_index"])
    matched_turn = find_matched_text_turn(record["dialogue"], selected_turn)
    matched_text_turn_index = int(matched_turn["turn_index"])
    prefix_turns = deepcopy(
        [
            turn
            for turn in record["dialogue"]
            if int(turn["turn_index"]) <= matched_text_turn_index
        ]
    )
    continued_from_turn_index = matched_text_turn_index + 1

    model_name = build_proxy_model_name(args.model, args.base_url)
    output_path = args.output_file or default_output_path(
        record["dialogue_name"], args.dialogue_index, args.perspective
    )

    print(f"Using storage_backend={os.environ.get('SOTOPIA_STORAGE_BACKEND')}")
    print(f"Using REDIS_OM_URL={os.environ.get('REDIS_OM_URL')}")
    print(f"Using base_url={args.base_url}")
    print(f"Using proxy model={model_name}")
    print(f"Using max_completion_tokens={args.max_completion_tokens}")
    print(f"Using reasoning_effort={args.reasoning_effort}")
    print(f"Using max_turns={args.max_turns}")
    print(f"Using max_stale_turn={args.max_stale_turn}")
    print(f"Perspective to change={args.perspective}")
    print(f"Selected important turn annotation index={annotation_turn_index}")
    print(f"Matched important-turn text to original turn_index={matched_text_turn_index}")
    if annotation_turn_index != matched_text_turn_index:
        print(
            "Warning: important_turns annotation index does not match the original "
            "turn matched by text; preserving through the matched text turn."
        )
    print(f"Continuing simulation from turn_index={continued_from_turn_index}")
    print(f"Output file={output_path}")

    profile_1 = build_agent_profile("mturk_agent_1", record["agents"]["mturk_agent_1"])
    profile_2 = build_agent_profile("mturk_agent_2", record["agents"]["mturk_agent_2"])
    env_profile = build_environment_profile(record)

    env = ParallelSotopiaEnv(
        env_profile=env_profile,
        model_name=model_name,
        action_order="round-robin",
        evaluators=[
            RuleBasedTerminatedEvaluator(
                max_turn_number=args.max_turns, max_stale_turn=args.max_stale_turn
            )
        ],
        terminal_evaluators=[],
    )

    intervention_text = build_intervention_instructions(selected_turn)
    agents = [
        ContinuationLLMAgent(
            agent_profile=profile_1,
            model_name=model_name,
            additional_instructions=(
                intervention_text if args.perspective == "mturk_agent_1" else ""
            ),
            max_completion_tokens=args.max_completion_tokens,
            reasoning_effort=args.reasoning_effort,
        ),
        ContinuationLLMAgent(
            agent_profile=profile_2,
            model_name=model_name,
            additional_instructions=(
                intervention_text if args.perspective == "mturk_agent_2" else ""
            ),
            max_completion_tokens=args.max_completion_tokens,
            reasoning_effort=args.reasoning_effort,
        ),
    ]

    generated_turns = await run_continuation(
        env=env,
        agents=agents,
        prefix_turns=prefix_turns,
        start_turn_index=continued_from_turn_index,
        dialogue_name=record["dialogue_name"],
    )

    write_output(
        output_path=output_path,
        record=record,
        selected_turn=selected_turn,
        perspective=args.perspective,
        prefix_turns=prefix_turns,
        generated_turns=generated_turns,
        annotation_turn_index=annotation_turn_index,
        matched_text_turn_index=matched_text_turn_index,
    )

    print(f"Wrote continued dialogue to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
