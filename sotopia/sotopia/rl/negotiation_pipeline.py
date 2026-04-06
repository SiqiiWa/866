from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import json_repair
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from .local_qwen_trainer import LocalQwenPolicyTrainer, LocalQwenTrainerConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTINUATION_SCRIPT = (
    REPO_ROOT / "examples" / "experimental" / "negotiation" / "two_agent_casino_negotiation.py"
)
DEFAULT_DIALOGUE_PATH = Path("/home/swang4/866/data/casino_first100_structured.json")
DEFAULT_IMPORTANT_TURNS_PATH = Path("/home/swang4/866/data/baseline_2/important_turns.json")
DEFAULT_STANCE_PROMPT_PATH = Path("/home/swang4/866/Baseline_2/stance_prompt.py")
DEFAULT_STANCE_LABELS_PATH = Path("/home/swang4/866/data/baseline_2/stance_labels_0-099.json")


def _default_api_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LITELLM_PROXY_API_KEY")
        or os.environ.get("CUSTOM_API_KEY")
        or "dummy"
    )


def _default_rollout_api_key() -> str:
    return os.environ.get("CUSTOM_API_KEY") or "dummy"


def _is_model_not_found_error(error: Exception) -> bool:
    return "does not exist" in str(error) or "NotFoundError" in str(error)


def _raise_gateway_model_error(
    *,
    stage: str,
    model_name: str,
    base_url: str,
    original_error: Exception,
) -> None:
    raise RuntimeError(
        f"{stage} failed because model={model_name!r} is not available at base_url={base_url!r}. "
        "The request reached the gateway, but the gateway rejected that model name. "
        "For CMU LiteLLM gateway, pass the exact deployed evaluator/stance alias with "
        "`--evaluator-model` and `--stance-model`. Also note this pipeline now defaults "
        "to `OPENAI_API_KEY` if `--api-key` is omitted."
    ) from original_error


@dataclass
class NegotiationRLPipelineConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = field(default_factory=_default_api_key)
    rollout_base_url: str | None = None
    rollout_api_key: str | None = None
    eval_base_url: str | None = None
    eval_api_key: str | None = None
    rollout_model: str = "openai/Qwen3-8B"
    evaluator_model: str = "gpt-5-mini"
    stance_model: str = "gpt-5-mini"
    dialogue_path: Path = DEFAULT_DIALOGUE_PATH
    important_turns_path: Path = DEFAULT_IMPORTANT_TURNS_PATH
    stance_prompt_path: Path = DEFAULT_STANCE_PROMPT_PATH
    stance_labels_path: Path = DEFAULT_STANCE_LABELS_PATH
    start_index: int = 0
    count: int = 100
    perspectives: tuple[str, ...] = ("mturk_agent_1", "mturk_agent_2")
    num_rollouts: int = 3
    max_turns: int = 20
    max_completion_tokens: int = 768
    reasoning_effort: str = "low"
    svi_max_concurrency: int = 4
    utility_max_concurrency: int = 4
    continuation_max_concurrency: int = 2
    stance_max_concurrency: int = 8
    utility_A_max: float = 36.0
    joint_utility_max: float = 72.0
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "self_svi_A": 0.25,
            "other_svi_B_to_A": 0.25,
            "utility_A": 0.25,
            "joint_utility": 0.25,
        }
    )
    alpha: float = 0.0
    beta: float = 1.0
    lambda_local: float = 0.5
    fallback_uniform_if_no_signal: bool = True
    train_batch_update_size: int = 16
    checkpoint_dialogue_counts: tuple[int, ...] = (25, 50, 75, 100)
    output_parent: Path = REPO_ROOT / "examples" / "experimental" / "negotiation"
    run_name: str | None = None
    online_policy_rollout: bool = True
    online_max_new_tokens: int = 80
    online_do_sample: bool = False
    online_temperature: float = 0.0
    online_top_p: float = 1.0
    online_repair_attempts: int = 2
    online_batch_size: int = 4
    online_batch_wait_ms: int = 15
    online_disable_thinking: bool = True
    online_speak_max_new_tokens: int | None = None
    online_non_speak_max_new_tokens: int = 48
    dialogues_per_update: int = 1
    checkpoint_every_dialogues: int = 0
    checkpoint_every_updates: int = 0
    retry_failed_updates: int = 1
    skip_failed_update_batches: bool = True

    def output_dir(self) -> Path:
        run_name = self.run_name or f"rl_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.output_parent / run_name

    def resolved_rollout_base_url(self) -> str:
        return self.rollout_base_url or self.base_url

    def resolved_rollout_api_key(self) -> str:
        return self.rollout_api_key or self.api_key or _default_rollout_api_key()

    def resolved_eval_base_url(self) -> str:
        return self.eval_base_url or self.base_url

    def resolved_eval_api_key(self) -> str:
        return self.eval_api_key or self.api_key or _default_api_key()


def _import_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _other_agent(agent_id: str) -> str:
    return "mturk_agent_2" if agent_id == "mturk_agent_1" else "mturk_agent_1"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_negotiation_module() -> Any:
    return _import_module(
        "two_agent_casino_negotiation",
        CONTINUATION_SCRIPT,
    )


def _sanitize_action_json(
    payload: dict[str, Any],
    *,
    agent_name: str,
    allowed_agent_names: list[str] | None,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    # Ignore schema-shaped objects leaked from prompt formatting.
    if (
        "action_type" not in payload
        and "argument" not in payload
        and "content" not in payload
        and isinstance(payload.get("properties"), dict)
    ):
        return None

    action_type = payload.get("action_type", "speak")
    if not isinstance(action_type, str) or not action_type.strip():
        action_type = "speak"

    argument = payload.get("argument")
    if not isinstance(argument, str) or not argument.strip():
        content_value = payload.get("content")
        if isinstance(content_value, str):
            argument = content_value
        else:
            argument = ""

    cleaned = {
        "action_type": action_type.strip(),
        "argument": argument.strip(),
    }
    # Negotiation continuation uses public messages only, so normalize any
    # generated recipient list back to [] for safety and parser stability.
    cleaned["to"] = []
    return cleaned


def _strip_thinking_blocks(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if cleaned:
        return cleaned
    if "<think>" in text.lower() and "</think>" not in text.lower():
        return ""
    return text.strip()


def _recover_action_json_candidate(
    candidate: str,
    *,
    agent_name: str,
    allowed_agent_names: list[str] | None,
) -> dict[str, Any] | None:
    try:
        repaired = json_repair.loads(candidate)
        if isinstance(repaired, dict):
            sanitized = _sanitize_action_json(
                repaired,
                agent_name=agent_name,
                allowed_agent_names=allowed_agent_names,
            )
            if sanitized is not None:
                return sanitized
    except Exception:
        pass

    action_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', candidate, flags=re.DOTALL)
    argument_match = re.search(r'"argument"\s*:\s*"([\s\S]*)', candidate, flags=re.DOTALL)
    content_match = re.search(r'"content"\s*:\s*"([\s\S]*)', candidate, flags=re.DOTALL)
    if action_match is None and argument_match is None and content_match is None:
        return None

    action_type = action_match.group(1).strip() if action_match is not None else "speak"
    argument = ""
    value_match = argument_match or content_match
    if value_match is not None:
        argument = value_match.group(1)
        argument = re.split(r'"\s*,\s*"to"\s*:', argument, maxsplit=1)[0]
        argument = re.split(r'"\s*,\s*"(?:argument|content|action_type)"\s*:', argument, maxsplit=1)[0]
        argument = re.split(r'"\s*}\s*$', argument, maxsplit=1)[0]
        argument = argument.rstrip('",} \n\t')
        argument = argument.replace('\\"', '"').replace("\\n", "\n").strip()

    return _sanitize_action_json(
        {"action_type": action_type or "speak", "argument": argument, "to": []},
        agent_name=agent_name,
        allowed_agent_names=allowed_agent_names,
    )


class OnlineGenerationBatcher:
    def __init__(
        self,
        *,
        trainer: LocalQwenPolicyTrainer,
        config: NegotiationRLPipelineConfig,
    ) -> None:
        self.trainer = trainer
        self.config = config
        self._pending: list[dict[str, Any]] = []
        self._flush_handle: asyncio.Handle | None = None
        self._lock = asyncio.Lock()

    async def generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        request = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "future": future,
        }
        async with self._lock:
            self._pending.append(request)
            if len(self._pending) >= max(1, self.config.online_batch_size):
                if self._flush_handle is not None:
                    self._flush_handle.cancel()
                    self._flush_handle = None
                loop.create_task(self._flush())
            elif self._flush_handle is None:
                self._flush_handle = loop.call_later(
                    max(0.0, self.config.online_batch_wait_ms / 1000.0),
                    lambda: loop.create_task(self._flush()),
                )
        return await future

    async def _flush(self) -> None:
        async with self._lock:
            if not self._pending:
                return
            batch = self._pending[:]
            self._pending.clear()
            if self._flush_handle is not None:
                self._flush_handle.cancel()
                self._flush_handle = None

        try:
            outputs = self._generate_batch(batch)
            for item, text in zip(batch, outputs):
                item["future"].set_result(text)
        except Exception as error:
            for item in batch:
                if not item["future"].done():
                    item["future"].set_exception(error)

    def _render_messages(self, messages: list[dict[str, str]]) -> str:
        rendered_messages = messages
        if self.config.online_disable_thinking and rendered_messages:
            rendered_messages = [dict(message) for message in rendered_messages]
            last = dict(rendered_messages[-1])
            last["content"] = "/no_think\n" + str(last.get("content", ""))
            rendered_messages[-1] = last
        if hasattr(self.trainer.tokenizer, "apply_chat_template"):
            try:
                rendered = self.trainer.tokenizer.apply_chat_template(
                    rendered_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = self.trainer.tokenizer.apply_chat_template(
                    rendered_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            if isinstance(rendered, str):
                return rendered
        return "\n\n".join(
            f"{message.get('role', 'user').upper()}: {message.get('content', '')}"
            for message in rendered_messages
        )

    def _generate_batch(self, batch: list[dict[str, Any]]) -> list[str]:
        prompts = [self._render_messages(item["messages"]) for item in batch]
        encoded = self.trainer.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.trainer.config.max_prompt_length,
        )
        encoded = {key: value.to(self.trainer.model.device) for key, value in encoded.items()}

        generation_kwargs: dict[str, Any] = dict(
            encoded,
            max_new_tokens=min(
                max(int(item["max_new_tokens"]) for item in batch),
                self.trainer.config.max_completion_length,
            ),
            do_sample=self.config.online_do_sample,
            use_cache=True,
            pad_token_id=self.trainer.tokenizer.pad_token_id,
            eos_token_id=self.trainer.tokenizer.eos_token_id,
        )
        if self.config.online_do_sample:
            generation_kwargs["temperature"] = self.config.online_temperature
            generation_kwargs["top_p"] = self.config.online_top_p

        was_training = self.trainer.model.training
        self.trainer.model.eval()
        with self.trainer.torch.no_grad():
            output_ids = self.trainer.model.generate(**generation_kwargs)
        if was_training:
            self.trainer.model.train()

        outputs: list[str] = []
        attention_mask = encoded["attention_mask"]
        for idx in range(len(batch)):
            prompt_len = int(attention_mask[idx].sum().item())
            generated_ids = output_ids[idx][prompt_len:]
            outputs.append(
                self.trainer.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ).strip()
            )
        print(
            "[online-batch] "
            f"batch_size={len(batch)} "
            f"max_new_tokens={generation_kwargs['max_new_tokens']}",
            flush=True,
        )
        return outputs


@lru_cache(maxsize=4)
def _load_stance_label_cache(cache_path: str) -> dict[tuple[int, str, int], dict[str, Any]]:
    path = Path(cache_path)
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", []) if isinstance(payload, dict) else []
    cache: dict[tuple[int, str, int], dict[str, Any]] = {}
    for item in results:
        dialogue_index = int(item.get("dialogue_index", -1))
        perspective = str(item.get("perspective", "")).strip()
        if dialogue_index < 0 or not perspective:
            continue
        for annotation in item.get("annotations", []):
            turn_index = int(annotation.get("turn_index", -1))
            if turn_index < 0:
                continue
            cache[(dialogue_index, perspective, turn_index)] = {
                "stance": str(annotation.get("stance", "neutral")).lower().strip(),
                "text": str(annotation.get("text", "")),
            }
    return cache


def _lookup_cached_stance_label(
    *,
    config: NegotiationRLPipelineConfig,
    dialogue_index: int,
    perspective: str,
    turn_index: int,
    text: str,
) -> tuple[str, float] | None:
    cache = _load_stance_label_cache(str(config.stance_labels_path.resolve()))
    cached = cache.get((int(dialogue_index), str(perspective), int(turn_index)))
    if not cached:
        return None

    cached_text = str(cached.get("text", "")).strip()
    if cached_text and text.strip() and cached_text != text.strip():
        return None

    label = str(cached.get("stance", "neutral")).lower().strip()
    score_map = {"proself": -1.0, "neutral": 0.0, "prosocial": 1.0}
    return label, score_map.get(label, 0.0)


def _normalize_svi(score_1_to_7: float) -> float:
    return _clip01((score_1_to_7 - 1.0) / 6.0)


def _mean_svi_norm(prediction: dict[str, Any]) -> float:
    svi_scores = prediction.get("svi_scores", {})
    values = [_normalize_svi(_safe_float(item.get("score", 1.0), 1.0)) for item in svi_scores.values()]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _build_continued_output_file(
    output_dir: Path,
    dialogue_index: int,
    current_a_id: str,
    rollout_index: int,
) -> Path:
    return output_dir / (
        f"dialogue_{dialogue_index:05d}_{current_a_id}_rollout_{rollout_index:02d}_continued.json"
    )


def rollout_dialogue(
    dialogue_index: int,
    current_a_id: str,
    rollout_index: int,
    config: NegotiationRLPipelineConfig,
    output_dir: Path,
    trainer: LocalQwenPolicyTrainer | None = None,
    runner: asyncio.Runner | None = None,
    batcher: OnlineGenerationBatcher | None = None,
) -> Path:
    if trainer is not None and config.online_policy_rollout:
        return rollout_dialogue_online(
            dialogue_index=dialogue_index,
            current_a_id=current_a_id,
            rollout_index=rollout_index,
            config=config,
            output_dir=output_dir,
            trainer=trainer,
            runner=runner,
            batcher=batcher,
        )

    output_file = _build_continued_output_file(
        output_dir=output_dir,
        dialogue_index=dialogue_index,
        current_a_id=current_a_id,
        rollout_index=rollout_index,
    )
    cmd = [
        sys.executable,
        str(CONTINUATION_SCRIPT),
        "--dialogue-path",
        str(config.dialogue_path),
        "--important-turns-path",
        str(config.important_turns_path),
        "--dialogue-index",
        str(dialogue_index),
        "--perspective",
        current_a_id,
        "--base-url",
        config.resolved_rollout_base_url(),
        "--model",
        config.rollout_model,
        "--api-key",
        config.resolved_rollout_api_key(),
        "--max-turns",
        str(config.max_turns),
        "--max-completion-tokens",
        str(config.max_completion_tokens),
        "--reasoning-effort",
        config.reasoning_effort,
        "--output-file",
        str(output_file),
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Continuation failed "
            f"dialogue_index={dialogue_index} perspective={current_a_id} rollout={rollout_index}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return output_file


def rollout_dialogues_batch(
    tasks: list[dict[str, Any]],
    config: NegotiationRLPipelineConfig,
    output_dir: Path,
    trainer: LocalQwenPolicyTrainer | None = None,
    runner: asyncio.Runner | None = None,
    batcher: OnlineGenerationBatcher | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if trainer is not None and config.online_policy_rollout:
        if runner is None:
            raise RuntimeError("Online rollout batch requires a shared asyncio.Runner.")

        async def _run_online_tasks() -> list[dict[str, Any]]:
            semaphore = asyncio.Semaphore(max(1, config.continuation_max_concurrency))

            async def _run_one(task: dict[str, Any]) -> dict[str, Any]:
                async with semaphore:
                    path = await _rollout_dialogue_online_async(
                        dialogue_index=int(task["dialogue_index"]),
                        current_a_id=str(task["current_A_id"]),
                        rollout_index=int(task["rollout_index"]),
                        config=config,
                        output_dir=output_dir,
                        trainer=trainer,
                        batcher=batcher,
                    )
                    return {**task, "continued_dialogue_path": path}

            return await asyncio.gather(*[_run_one(task) for task in tasks])

        results = runner.run(_run_online_tasks())
        results.sort(
            key=lambda item: (
                int(item["dialogue_index"]),
                str(item["current_A_id"]),
                int(item["rollout_index"]),
            )
        )
        return results

    def _worker(task: dict[str, Any]) -> dict[str, Any]:
        path = rollout_dialogue(
            dialogue_index=int(task["dialogue_index"]),
            current_a_id=str(task["current_A_id"]),
            rollout_index=int(task["rollout_index"]),
            config=config,
            output_dir=output_dir,
            trainer=trainer,
            runner=runner,
        )
        return {**task, "continued_dialogue_path": path}

    with ThreadPoolExecutor(max_workers=max(1, config.continuation_max_concurrency)) as executor:
        future_map = {executor.submit(_worker, task): task for task in tasks}
        for future in as_completed(future_map):
            results.append(future.result())

    results.sort(
        key=lambda item: (
            int(item["dialogue_index"]),
            str(item["current_A_id"]),
            int(item["rollout_index"]),
        )
    )
    return results


def _build_online_policy_agent(
    *,
    neg_mod: Any,
    trainer: LocalQwenPolicyTrainer,
    config: NegotiationRLPipelineConfig,
    batcher: OnlineGenerationBatcher,
    agent_profile: Any,
    agent_name: str,
    model_name: str,
    additional_instructions: str,
    max_completion_tokens: int,
    reasoning_effort: str,
) -> Any:
    class OnlinePolicyContinuationAgent(neg_mod.ContinuationLLMAgent):
        async def aact(self, obs: Any) -> Any:
            self.recv_message("Environment", obs)

            if len(obs.available_actions) == 1 and "none" in obs.available_actions:
                return neg_mod.AgentAction(action_type="none", argument="", to=[])

            parser = neg_mod.PydanticOutputParser(pydantic_object=neg_mod.AgentAction)
            history = "\n".join(message.to_natural_language() for _, message in self.inbox)
            prompt = neg_mod.CONTINUATION_TEMPLATE.format(
                agent=self.agent_name,
                goal=self.goal,
                additional_instructions=self.additional_instructions
                or "No extra intervention instruction for you.",
                history=history,
                turn_number=obs.turn_number + 1,
                action_list=" ".join(obs.available_actions),
                format_instructions=parser.get_format_instructions(),
            )

            validation_context = None
            if self.script_background is not None:
                validation_context = {
                    "agent_names": self.script_background.agent_names,
                    "available_action_types": obs.available_actions,
                    "sender": self.agent_name,
                }

            print(
                f"[online-agent={self.agent_name}] requesting action for turn={obs.turn_number + 1} "
                f"available_actions={obs.available_actions}",
                flush=True,
            )

            per_turn_max_new_tokens = config.online_max_new_tokens
            available_actions = set(str(action) for action in obs.available_actions)
            if "speak" in available_actions:
                if config.online_speak_max_new_tokens is not None:
                    per_turn_max_new_tokens = config.online_speak_max_new_tokens
            else:
                per_turn_max_new_tokens = min(
                    per_turn_max_new_tokens,
                    config.online_non_speak_max_new_tokens,
                )

            json_only_suffix = (
                "\n\nReturn exactly one JSON object and nothing else."
                "\nDo not include analysis, reasoning, markdown, code fences, or <think> tags."
                "\nIf unsure, still output a single valid JSON object matching the schema."
            )
            repair_prompt = prompt + json_only_suffix
            for attempt in range(max(1, config.online_repair_attempts)):
                if hasattr(trainer.tokenizer, "apply_chat_template"):
                    chat_messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a negotiation agent. "
                                "Return exactly one JSON object and nothing else. "
                                "Never output chain-of-thought, analysis, markdown, or <think> tags."
                            ),
                        },
                        {"role": "user", "content": repair_prompt},
                    ]
                    raw_content = await batcher.generate(
                        messages=chat_messages,
                        max_new_tokens=min(
                            max_completion_tokens,
                            per_turn_max_new_tokens,
                            trainer.config.max_completion_length,
                        ),
                    )
                else:
                    raw_content = await batcher.generate(
                        messages=[{"role": "user", "content": repair_prompt}],
                        max_new_tokens=min(
                            max_completion_tokens,
                            per_turn_max_new_tokens,
                            trainer.config.max_completion_length,
                        ),
                    )
                content = _strip_thinking_blocks(raw_content)
                if not content:
                    if attempt == max(1, config.online_repair_attempts) - 1:
                        raise ValueError("Online policy model returned empty content after stripping thinking blocks.")
                    repair_prompt = (
                        prompt
                        + json_only_suffix
                        + "\n\nYour previous response only contained reasoning or invalid text."
                        "\nReturn a single valid JSON object now."
                    )
                    continue

                print(
                    f"[online-agent={self.agent_name}] raw response on turn={obs.turn_number + 1}: "
                    f"{content[:200]!r}",
                    flush=True,
                )

                parsed_action = None
                parse_error: Exception | None = None
                parse_candidates: list[str] = []
                for candidate in (content, raw_content):
                    candidate = _strip_thinking_blocks(candidate)
                    if candidate and candidate not in parse_candidates:
                        parse_candidates.append(candidate)
                extracted_json = neg_mod.extract_first_json_object(content)
                if extracted_json and extracted_json != content:
                    parse_candidates.append(extracted_json)
                extracted_from_raw = neg_mod.extract_first_json_object(raw_content)
                if extracted_from_raw and extracted_from_raw not in parse_candidates:
                    parse_candidates.append(extracted_from_raw)

                for parse_candidate in parse_candidates:
                    try:
                        candidate_for_parse = parse_candidate
                        maybe_json_dict: dict[str, Any] | None = None
                        try:
                            parsed_json = json.loads(parse_candidate)
                            if isinstance(parsed_json, dict):
                                maybe_json_dict = _sanitize_action_json(
                                    parsed_json,
                                    agent_name=self.agent_name,
                                    allowed_agent_names=(
                                        validation_context.get("agent_names")
                                        if validation_context is not None
                                        else None
                                    ),
                                )
                                if maybe_json_dict is not None:
                                    original_to = parsed_json.get("to", [])
                                    if maybe_json_dict.get("to", []) != original_to:
                                        print(
                                            f"[online-agent={self.agent_name}] sanitized invalid to="
                                            f"{original_to!r} -> {maybe_json_dict.get('to', [])!r}",
                                            flush=True,
                                        )
                                    candidate_for_parse = json.dumps(maybe_json_dict, ensure_ascii=False)
                        except Exception:
                            maybe_json_dict = _recover_action_json_candidate(
                                parse_candidate,
                                agent_name=self.agent_name,
                                allowed_agent_names=(
                                    validation_context.get("agent_names")
                                    if validation_context is not None
                                    else None
                                ),
                            )
                            if maybe_json_dict is not None:
                                print(
                                    f"[online-agent={self.agent_name}] recovered partial JSON "
                                    f"for turn={obs.turn_number + 1}",
                                    flush=True,
                                )
                                candidate_for_parse = json.dumps(maybe_json_dict, ensure_ascii=False)

                        if maybe_json_dict is not None:
                            parsed_action = parser.pydantic_object.model_validate(
                                maybe_json_dict,
                                context=validation_context,
                            )
                        else:
                            parsed_action = parser.parse(
                                candidate_for_parse,
                                context=validation_context,
                            )
                        break
                    except Exception as exc:
                        parse_error = exc

                if parsed_action is not None:
                    print(
                        f"[online-agent={self.agent_name}] parsed action turn={obs.turn_number + 1}: "
                        f"{parsed_action.model_dump()}",
                        flush=True,
                    )
                    return parsed_action

                if attempt == max(1, config.online_repair_attempts) - 1:
                    raise RuntimeError(
                        "Failed to parse online policy action. "
                        f"parse_error={parse_error} raw_response={content}"
                    ) from parse_error

                repair_prompt = neg_mod.REPAIR_PROMPT_TEMPLATE.format(
                    format_instructions=parser.get_format_instructions(),
                    bad_response=content,
                    error=str(parse_error),
                ) + json_only_suffix

            raise RuntimeError("Unreachable online policy continuation state.")

    return OnlinePolicyContinuationAgent(
        agent_profile=agent_profile,
        agent_name=agent_name,
        model_name=model_name,
        additional_instructions=additional_instructions,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
    )


async def _rollout_dialogue_online_async(
    dialogue_index: int,
    current_a_id: str,
    rollout_index: int,
    config: NegotiationRLPipelineConfig,
    output_dir: Path,
    trainer: LocalQwenPolicyTrainer,
    batcher: OnlineGenerationBatcher | None,
) -> Path:
    neg_mod = _load_negotiation_module()
    os.environ["CUSTOM_API_KEY"] = config.resolved_rollout_api_key()

    record = neg_mod.load_dialogue_record(config.dialogue_path, dialogue_index)
    selected_turn = neg_mod.load_first_important_turn(
        config.important_turns_path,
        dialogue_index,
        current_a_id,
    )
    annotation_turn_index = int(selected_turn["turn_index"])
    matched_turn = neg_mod.find_matched_text_turn(record["dialogue"], selected_turn)
    matched_text_turn_index = int(matched_turn["turn_index"])
    prefix_turns = deepcopy(
        [
            turn
            for turn in record["dialogue"]
            if int(turn["turn_index"]) <= matched_text_turn_index
        ]
    )
    continued_from_turn_index = matched_text_turn_index + 1
    output_path = _build_continued_output_file(
        output_dir=output_dir,
        dialogue_index=dialogue_index,
        current_a_id=current_a_id,
        rollout_index=rollout_index,
    )

    proxy_model_name = neg_mod.build_proxy_model_name(
        config.rollout_model,
        config.resolved_rollout_base_url(),
    )
    profile_1 = neg_mod.build_agent_profile("mturk_agent_1", record["agents"]["mturk_agent_1"])
    profile_2 = neg_mod.build_agent_profile("mturk_agent_2", record["agents"]["mturk_agent_2"])
    env_profile = neg_mod.build_environment_profile(record)
    env = neg_mod.ParallelSotopiaEnv(
        env_profile=env_profile,
        model_name=proxy_model_name,
        action_order="round-robin",
        evaluators=[
            neg_mod.RuleBasedTerminatedEvaluator(
                max_turn_number=config.max_turns,
                max_stale_turn=2,
            )
        ],
        terminal_evaluators=[],
    )

    intervention_text = neg_mod.build_intervention_instructions(selected_turn)
    agent_1 = (
        _build_online_policy_agent(
            neg_mod=neg_mod,
            trainer=trainer,
            config=config,
            batcher=batcher or OnlineGenerationBatcher(trainer=trainer, config=config),
            agent_profile=profile_1,
            agent_name="mturk_agent_1",
            model_name=proxy_model_name,
            additional_instructions=intervention_text if current_a_id == "mturk_agent_1" else "",
            max_completion_tokens=config.max_completion_tokens,
            reasoning_effort=config.reasoning_effort,
        )
        if current_a_id == "mturk_agent_1"
        else neg_mod.ContinuationLLMAgent(
            agent_profile=profile_1,
            model_name=proxy_model_name,
            additional_instructions="",
            max_completion_tokens=config.max_completion_tokens,
            reasoning_effort=config.reasoning_effort,
        )
    )
    agent_2 = (
        _build_online_policy_agent(
            neg_mod=neg_mod,
            trainer=trainer,
            config=config,
            batcher=batcher or OnlineGenerationBatcher(trainer=trainer, config=config),
            agent_profile=profile_2,
            agent_name="mturk_agent_2",
            model_name=proxy_model_name,
            additional_instructions=intervention_text if current_a_id == "mturk_agent_2" else "",
            max_completion_tokens=config.max_completion_tokens,
            reasoning_effort=config.reasoning_effort,
        )
        if current_a_id == "mturk_agent_2"
        else neg_mod.ContinuationLLMAgent(
            agent_profile=profile_2,
            model_name=proxy_model_name,
            additional_instructions="",
            max_completion_tokens=config.max_completion_tokens,
            reasoning_effort=config.reasoning_effort,
        )
    )

    print(
        f"[online-rollout] dialogue_index={dialogue_index} current_A={current_a_id} "
        f"continued_from_turn_index={continued_from_turn_index}",
        flush=True,
    )
    generated_turns = await neg_mod.run_continuation(
        env=env,
        agents=[agent_1, agent_2],
        prefix_turns=prefix_turns,
        start_turn_index=continued_from_turn_index,
        dialogue_name=record["dialogue_name"],
    )
    neg_mod.write_output(
        output_path=output_path,
        record=record,
        selected_turn=selected_turn,
        perspective=current_a_id,
        prefix_turns=prefix_turns,
        generated_turns=generated_turns,
        annotation_turn_index=annotation_turn_index,
        matched_text_turn_index=matched_text_turn_index,
    )
    print(f"[online-rollout] wrote continued dialogue to {output_path}", flush=True)
    return output_path


def rollout_dialogue_online(
    dialogue_index: int,
    current_a_id: str,
    rollout_index: int,
    config: NegotiationRLPipelineConfig,
    output_dir: Path,
    trainer: LocalQwenPolicyTrainer,
    runner: asyncio.Runner | None,
    batcher: OnlineGenerationBatcher | None,
) -> Path:
    if runner is None:
        raise RuntimeError("Online rollout requires a shared asyncio.Runner.")
    return runner.run(
        _rollout_dialogue_online_async(
            dialogue_index=dialogue_index,
            current_a_id=current_a_id,
            rollout_index=rollout_index,
            config=config,
            output_dir=output_dir,
            trainer=trainer,
            batcher=batcher,
        )
    )


def compute_svi_reward_components(
    continued_dialogue_paths: list[Path],
    config: NegotiationRLPipelineConfig,
) -> dict[str, dict[str, float]]:
    svi_mod = _import_module(
        "eval_batch_svi_addBatch",
        REPO_ROOT / "examples" / "experimental" / "negotiation" / "eval_batch_svi_addBatch.py",
    )

    tasks: list[tuple[Path, dict[str, Any], str]] = []
    for path in continued_dialogue_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        for pov in ("mturk_agent_1", "mturk_agent_2"):
            tasks.append((path, record, pov))

    raw_rows: list[dict[str, Any]] = []

    def _eval_task(path: Path, record: dict[str, Any], pov_agent_id: str) -> dict[str, Any]:
        try:
            return svi_mod.evaluate_pov_task(
                base_url=config.resolved_eval_base_url(),
                api_key=config.resolved_eval_api_key(),
                model=config.evaluator_model,
                record=record,
                source_file=path,
                pov_agent_id=pov_agent_id,
                reasoning_effort=config.reasoning_effort,
                max_completion_tokens=None,
                sleep_between_calls=0.0,
            )
        except Exception as error:
            if _is_model_not_found_error(error):
                _raise_gateway_model_error(
                    stage="SVI evaluation",
                    model_name=config.evaluator_model,
                    base_url=config.resolved_eval_base_url(),
                    original_error=error,
                )
            return svi_mod.build_failed_result(record, path, pov_agent_id, error)

    with ThreadPoolExecutor(max_workers=max(1, config.svi_max_concurrency)) as executor:
        futures = [executor.submit(_eval_task, path, record, pov) for path, record, pov in tasks]
        for future in as_completed(futures):
            raw_rows.append(future.result())

    by_file_and_pov: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw_rows:
        by_file_and_pov[(str(row["source_file"]), str(row["pov_agent_id"]))] = row

    components: dict[str, dict[str, float]] = {}
    for path in continued_dialogue_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        current_a = str(record["perspective"])
        current_b = _other_agent(current_a)

        self_row = by_file_and_pov.get((str(path), current_a), {})
        other_row = by_file_and_pov.get((str(path), current_b), {})

        self_norm = 0.0
        other_norm = 0.0
        if self_row.get("status") == "ok":
            self_norm = _mean_svi_norm(self_row.get("prediction", {}))
        if other_row.get("status") == "ok":
            other_norm = _mean_svi_norm(other_row.get("prediction", {}))

        components[str(path)] = {
            "self_svi_A": self_norm,
            "other_svi_B_to_A": other_norm,
        }

    return components


def compute_utility_reward_components(
    continued_dialogue_paths: list[Path],
    config: NegotiationRLPipelineConfig,
) -> dict[str, dict[str, float]]:
    util_mod = _import_module(
        "eval_batch_utilities",
        REPO_ROOT / "examples" / "experimental" / "negotiation" / "eval_batch_utilities.py",
    )

    def _eval_file(path: Path) -> tuple[str, dict[str, Any]]:
        record = json.loads(path.read_text(encoding="utf-8"))
        client = util_mod.make_client(
            config.resolved_eval_base_url(),
            config.resolved_eval_api_key(),
        )
        try:
            extraction = util_mod.call_extractor_with_retry(
                client=client,
                model=config.evaluator_model,
                record=record,
                max_completion_tokens=None,
                reasoning_effort=config.reasoning_effort,
            )
            row = util_mod.evaluate_record(record, extraction, path)
        except Exception as error:
            if _is_model_not_found_error(error):
                _raise_gateway_model_error(
                    stage="Utility evaluation",
                    model_name=config.evaluator_model,
                    base_url=config.resolved_eval_base_url(),
                    original_error=error,
                )
            row = util_mod.build_failed_result(record, path, error)
        return str(path), row

    rows: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, config.utility_max_concurrency)) as executor:
        futures = [executor.submit(_eval_file, path) for path in continued_dialogue_paths]
        for future in as_completed(futures):
            key, row = future.result()
            rows[key] = row

    components: dict[str, dict[str, float]] = {}
    for path in continued_dialogue_paths:
        row = rows[str(path)]
        current_a = str(row.get("perspective") or json.loads(path.read_text(encoding="utf-8"))["perspective"])
        current_b = _other_agent(current_a)

        utility_a = _safe_float(row.get("utilities", {}).get(current_a, {}).get("total_utility", 0.0))
        utility_b = _safe_float(row.get("utilities", {}).get(current_b, {}).get("total_utility", 0.0))
        joint = utility_a + utility_b

        components[str(path)] = {
            "utility_A": utility_a,
            "utility_B": utility_b,
            "joint_utility": joint,
            "utility_A_norm": _clip01(utility_a / config.utility_A_max),
            "joint_utility_norm": _clip01(joint / config.joint_utility_max),
        }

    return components


def compute_final_reward(
    self_svi_A: float,
    other_svi_B_to_A: float,
    utility_A_norm: float,
    joint_utility_norm: float,
    config: NegotiationRLPipelineConfig,
) -> float:
    w = config.reward_weights
    return (
        _safe_float(w.get("self_svi_A", 0.0)) * self_svi_A
        + _safe_float(w.get("other_svi_B_to_A", 0.0)) * other_svi_B_to_A
        + _safe_float(w.get("utility_A", 0.0)) * utility_A_norm
        + _safe_float(w.get("joint_utility", 0.0)) * joint_utility_norm
    )


def extract_post_intervention_A_turns(
    dialogue: list[dict[str, Any]],
    current_a_id: str,
    intervention_turn_index: int,
) -> list[dict[str, Any]]:
    turns = []
    for turn in dialogue:
        if int(turn.get("turn_index", -1)) <= intervention_turn_index:
            continue
        if str(turn.get("speaker") or turn.get("id")) != current_a_id:
            continue
        turns.append(turn)
    turns.sort(key=lambda item: int(item.get("turn_index", 0)))
    return turns


def _find_prev_next_b_turns(
    dialogue: list[dict[str, Any]],
    current_a_turn_index: int,
    current_b_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    prev_b = None
    next_b = None
    for turn in dialogue:
        turn_index = int(turn.get("turn_index", -1))
        speaker = str(turn.get("speaker") or turn.get("id"))
        if speaker != current_b_id:
            continue
        if turn_index < current_a_turn_index:
            if prev_b is None or turn_index > int(prev_b.get("turn_index", -1)):
                prev_b = turn
        if turn_index > current_a_turn_index:
            if next_b is None or turn_index < int(next_b.get("turn_index", 10**9)):
                next_b = turn
    return prev_b, next_b


def _load_stance_module(config: NegotiationRLPipelineConfig) -> Any:
    stance_mod = _import_module("stance_prompt", config.stance_prompt_path)
    import openai

    stance_mod.MODEL = config.stance_model
    stance_mod.client = openai.OpenAI(
        api_key=config.resolved_eval_api_key(),
        base_url=config.resolved_eval_base_url(),
    )
    return stance_mod


def _judge_single_turn_stance(
    stance_mod: Any,
    text: str,
    speaker_id: str,
    current_a_id: str,
    own_preferences: str,
) -> tuple[str, float]:
    speaker_token = "YOU" if speaker_id == current_a_id else "THEM"
    dialogue_text = f"{speaker_token}: {text}".strip()
    pref = stance_mod.call_with_retry(
        stance_mod.infer_opponent_preference,
        current_dialogue=dialogue_text,
        last_preference_inference=None,
    )
    stance = stance_mod.call_with_retry(
        stance_mod.judge_stance,
        current_dialogue=dialogue_text,
        last_turn_context="No previous annotated turns.",
        own_preferences=own_preferences,
        inferred_opponent_preference=pref,
        current_speaker=speaker_token,
    )
    label = str(stance.get("stance", "neutral")).lower().strip()
    score_map = {"proself": -1.0, "neutral": 0.0, "prosocial": 1.0}
    return label, score_map.get(label, 0.0)


def _fallback_neutral_stance_on_error(
    *,
    error: Exception,
    stage: str,
    speaker_id: str,
    turn_index: int,
    dialogue_index: int,
) -> tuple[str, float]:
    print(
        "[stance] "
        f"fallbacking to neutral due to error at stage={stage} "
        f"dialogue_index={dialogue_index} turn_index={turn_index} speaker={speaker_id} "
        f"error={error!r}",
        flush=True,
    )
    return "neutral", 0.0


def compute_triplet_stance_signal(
    dialogue: list[dict[str, Any]],
    current_a_turn: dict[str, Any],
    current_a_id: str,
    current_b_id: str,
    own_preferences: str,
    config: NegotiationRLPipelineConfig,
    dialogue_index: int,
    intervention_turn_index: int,
    stance_mod: Any | None = None,
) -> dict[str, Any]:
    stance_mod = stance_mod or _load_stance_module(config)

    turn_index = int(current_a_turn["turn_index"])
    prev_b, next_b = _find_prev_next_b_turns(dialogue, turn_index, current_b_id)

    b1_label, b1 = ("neutral", 0.0)
    if prev_b is not None:
        prev_b_text = str(prev_b.get("text", ""))
        cached = None
        if int(prev_b.get("turn_index", -1)) <= intervention_turn_index:
            cached = _lookup_cached_stance_label(
                config=config,
                dialogue_index=dialogue_index,
                perspective=current_a_id,
                turn_index=int(prev_b.get("turn_index", -1)),
                text=prev_b_text,
            )
        if cached is not None:
            b1_label, b1 = cached
        else:
            try:
                b1_label, b1 = _judge_single_turn_stance(
                    stance_mod=stance_mod,
                    text=prev_b_text,
                    speaker_id=current_b_id,
                    current_a_id=current_a_id,
                    own_preferences=own_preferences,
                )
            except Exception as error:
                if _is_model_not_found_error(error):
                    _raise_gateway_model_error(
                        stage="Stance evaluation",
                        model_name=config.stance_model,
                        base_url=config.resolved_eval_base_url(),
                        original_error=error,
                    )
                b1_label, b1 = _fallback_neutral_stance_on_error(
                    error=error,
                    stage="prev_b",
                    speaker_id=current_b_id,
                    turn_index=int(prev_b.get("turn_index", -1)),
                    dialogue_index=dialogue_index,
                )

    try:
        a_label, a = _judge_single_turn_stance(
            stance_mod=stance_mod,
            text=str(current_a_turn.get("text", "")),
            speaker_id=current_a_id,
            current_a_id=current_a_id,
            own_preferences=own_preferences,
        )
    except Exception as error:
        if _is_model_not_found_error(error):
            _raise_gateway_model_error(
                stage="Stance evaluation",
                model_name=config.stance_model,
                base_url=config.resolved_eval_base_url(),
                original_error=error,
            )
        a_label, a = _fallback_neutral_stance_on_error(
            error=error,
            stage="current_a",
            speaker_id=current_a_id,
            turn_index=turn_index,
            dialogue_index=dialogue_index,
        )

    b2_label, b2 = ("neutral", 0.0)
    if next_b is not None:
        next_b_text = str(next_b.get("text", ""))
        cached = None
        if int(next_b.get("turn_index", -1)) <= intervention_turn_index:
            cached = _lookup_cached_stance_label(
                config=config,
                dialogue_index=dialogue_index,
                perspective=current_a_id,
                turn_index=int(next_b.get("turn_index", -1)),
                text=next_b_text,
            )
        if cached is not None:
            b2_label, b2 = cached
        else:
            try:
                b2_label, b2 = _judge_single_turn_stance(
                    stance_mod=stance_mod,
                    text=next_b_text,
                    speaker_id=current_b_id,
                    current_a_id=current_a_id,
                    own_preferences=own_preferences,
                )
            except Exception as error:
                if _is_model_not_found_error(error):
                    _raise_gateway_model_error(
                        stage="Stance evaluation",
                        model_name=config.stance_model,
                        base_url=config.resolved_eval_base_url(),
                        original_error=error,
                    )
                b2_label, b2 = _fallback_neutral_stance_on_error(
                    error=error,
                    stage="next_b",
                    speaker_id=current_b_id,
                    turn_index=int(next_b.get("turn_index", -1)),
                    dialogue_index=dialogue_index,
                )

    raw_stance_score = 2.0 * a - b1 + b2
    stance_signal = _clip01((raw_stance_score + 4.0) / 8.0)

    return {
        "turn_index": turn_index,
        "turn_id": str(current_a_turn.get("turn_id", f"turn_{turn_index:02d}")),
        "text": str(current_a_turn.get("text", "")),
        "b1": b1,
        "a": a,
        "b2": b2,
        "b1_label": b1_label,
        "a_label": a_label,
        "b2_label": b2_label,
        "b1_source": "cached" if prev_b is not None and int(prev_b.get("turn_index", -1)) <= intervention_turn_index and _lookup_cached_stance_label(
            config=config,
            dialogue_index=dialogue_index,
            perspective=current_a_id,
            turn_index=int(prev_b.get("turn_index", -1)),
            text=str(prev_b.get("text", "")),
        ) is not None else ("fallback-neutral" if prev_b is None else "model"),
        "a_source": "model",
        "b2_source": "cached" if next_b is not None and int(next_b.get("turn_index", -1)) <= intervention_turn_index and _lookup_cached_stance_label(
            config=config,
            dialogue_index=dialogue_index,
            perspective=current_a_id,
            turn_index=int(next_b.get("turn_index", -1)),
            text=str(next_b.get("text", "")),
        ) is not None else ("fallback-neutral" if next_b is None else "model"),
        "raw_stance_score": raw_stance_score,
        "stance_signal_i": stance_signal,
    }


def _format_own_preferences(agent_profile: dict[str, Any]) -> str:
    value2issue = agent_profile.get("value2issue", {})
    return json.dumps(value2issue, ensure_ascii=False)


def _compute_triplet_stance_batch(
    dialogue: list[dict[str, Any]],
    current_a_turns: list[dict[str, Any]],
    current_a_id: str,
    current_b_id: str,
    own_preferences: str,
    dialogue_index: int,
    intervention_turn_index: int,
    config: NegotiationRLPipelineConfig,
) -> list[dict[str, Any]]:
    stance_mod = _load_stance_module(config)
    batch_start = time.perf_counter()
    print(
        "[stance] "
        f"dialogue_index={dialogue_index} A={current_a_id} B={current_b_id} "
        f"a_turns={len(current_a_turns)} max_workers={max(1, config.stance_max_concurrency)}",
        flush=True,
    )

    def _worker(turn: dict[str, Any]) -> dict[str, Any]:
        return compute_triplet_stance_signal(
            dialogue=dialogue,
            current_a_turn=turn,
            current_a_id=current_a_id,
            current_b_id=current_b_id,
            own_preferences=own_preferences,
            dialogue_index=dialogue_index,
            intervention_turn_index=intervention_turn_index,
            config=config,
            stance_mod=stance_mod,
        )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, config.stance_max_concurrency)) as executor:
        future_map = {executor.submit(_worker, turn): turn for turn in current_a_turns}
        completed = 0
        for future in as_completed(future_map):
            turn = future_map[future]
            result = future.result()
            results.append(result)
            completed += 1
            print(
                "[stance] "
                f"dialogue_index={dialogue_index} A={current_a_id} "
                f"completed={completed}/{len(current_a_turns)} "
                f"turn_index={int(turn.get('turn_index', -1))} "
                f"elapsed={time.perf_counter() - batch_start:.2f}s",
                flush=True,
            )

    results.sort(key=lambda item: int(item["turn_index"]))
    cache_hits = sum(
        1
        for item in results
        for source_key in ("b1_source", "a_source", "b2_source")
        if item.get(source_key) == "cached"
    )
    model_calls = sum(
        1
        for item in results
        for source_key in ("b1_source", "a_source", "b2_source")
        if item.get(source_key) == "model"
    )
    print(
        "[stance] "
        f"dialogue_index={dialogue_index} A={current_a_id} finished "
        f"elapsed={time.perf_counter() - batch_start:.2f}s "
        f"cached={cache_hits} model_calls={model_calls}",
        flush=True,
    )
    return results


def decompose_final_reward(
    final_reward: float,
    stance_signals: list[dict[str, Any]],
    config: NegotiationRLPipelineConfig,
) -> list[dict[str, Any]]:
    if not stance_signals:
        return []

    k = len(stance_signals)
    raw_values: list[float] = []
    for item in stance_signals:
        raw_i = config.alpha + config.beta * _safe_float(item.get("stance_signal_i", 0.0))
        raw_values.append(max(0.0, raw_i))

    sum_raw = sum(raw_values)
    if sum_raw <= 0.0:
        if not config.fallback_uniform_if_no_signal:
            weights = [0.0 for _ in stance_signals]
        else:
            weights = [1.0 / k for _ in stance_signals]
    else:
        local_weights = [value / sum_raw for value in raw_values]
        uniform_weights = [1.0 / k for _ in stance_signals]
        lam = _clip01(config.lambda_local)
        weights = [
            (1.0 - lam) * uniform + lam * local
            for uniform, local in zip(uniform_weights, local_weights)
        ]

    decomposed: list[dict[str, Any]] = []
    running_sum = 0.0
    for idx, (item, raw_i, weight_i) in enumerate(zip(stance_signals, raw_values, weights)):
        reward_i = final_reward * weight_i
        if idx == len(stance_signals) - 1:
            reward_i = final_reward - running_sum
        running_sum += reward_i
        decomposed.append(
            {
                **item,
                "raw_i": raw_i,
                "weight_i": weight_i,
                "decomposed_reward": reward_i,
            }
        )
    return decomposed


def _build_turn_prompt(
    dialogue: list[dict[str, Any]],
    current_a_id: str,
    target_turn_index: int,
) -> str:
    lines = [
        "You are continuing a negotiation as agent A.",
        "History so far:",
    ]
    for turn in sorted(dialogue, key=lambda t: int(t.get("turn_index", 0))):
        turn_index = int(turn.get("turn_index", 0))
        if turn_index >= target_turn_index:
            break
        speaker = str(turn.get("speaker") or turn.get("id"))
        speaker_token = "YOU" if speaker == current_a_id else "THEM"
        lines.append(f"Turn {turn_index} | {speaker_token}: {str(turn.get('text', ''))}")
    lines.append("Now generate your next negotiation utterance as YOU.")
    return "\n".join(lines)


def _intervention_turn_index(record: dict[str, Any]) -> int:
    for key in ("preserved_through_turn_index", "matched_text_turn_index", "annotation_turn_index"):
        if key in record:
            return int(record[key])
    selected = record.get("selected_important_turn", {})
    if "turn_index" in selected:
        return int(selected["turn_index"])
    return -1


def run_episode_and_collect_rewards(
    episode: dict[str, Any],
    svi_components_by_path: dict[str, dict[str, float]],
    utility_components_by_path: dict[str, dict[str, float]],
    config: NegotiationRLPipelineConfig,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_start = time.perf_counter()
    path = Path(episode["continued_dialogue_path"])
    record = json.loads(path.read_text(encoding="utf-8"))

    current_a = str(episode["current_A_id"])
    current_b = str(episode["current_B_id"])

    svi = svi_components_by_path.get(str(path), {"self_svi_A": 0.0, "other_svi_B_to_A": 0.0})
    util = utility_components_by_path.get(
        str(path),
        {
            "utility_A": 0.0,
            "utility_B": 0.0,
            "joint_utility": 0.0,
            "utility_A_norm": 0.0,
            "joint_utility_norm": 0.0,
        },
    )

    final_reward = compute_final_reward(
        self_svi_A=_safe_float(svi["self_svi_A"]),
        other_svi_B_to_A=_safe_float(svi["other_svi_B_to_A"]),
        utility_A_norm=_safe_float(util["utility_A_norm"]),
        joint_utility_norm=_safe_float(util["joint_utility_norm"]),
        config=config,
    )

    dialogue = record["dialogue"]
    intervention_idx = _intervention_turn_index(record)
    extract_start = time.perf_counter()
    a_turns = extract_post_intervention_A_turns(
        dialogue=dialogue,
        current_a_id=current_a,
        intervention_turn_index=intervention_idx,
    )
    print(
        "[reward] "
        f"dialogue_index={episode['dialogue_index']} A={current_a} "
        f"intervention_turn_index={intervention_idx} "
        f"post_intervention_a_turns={len(a_turns)} "
        f"extract_elapsed={time.perf_counter() - extract_start:.2f}s",
        flush=True,
    )

    own_preferences = _format_own_preferences(record["agents"][current_a])
    stance_start = time.perf_counter()
    stance_signals = _compute_triplet_stance_batch(
        dialogue=dialogue,
        current_a_turns=a_turns,
        current_a_id=current_a,
        current_b_id=current_b,
        own_preferences=own_preferences,
        dialogue_index=int(episode["dialogue_index"]),
        intervention_turn_index=intervention_idx,
        config=config,
    )
    print(
        "[reward] "
        f"dialogue_index={episode['dialogue_index']} A={current_a} "
        f"stance_elapsed={time.perf_counter() - stance_start:.2f}s",
        flush=True,
    )
    decompose_start = time.perf_counter()
    turn_level = decompose_final_reward(
        final_reward=final_reward,
        stance_signals=stance_signals,
        config=config,
    )
    stance_cache_hits = sum(
        1
        for item in stance_signals
        for source_key in ("b1_source", "a_source", "b2_source")
        if item.get(source_key) == "cached"
    )
    stance_model_calls = sum(
        1
        for item in stance_signals
        for source_key in ("b1_source", "a_source", "b2_source")
        if item.get(source_key) == "model"
    )
    stance_fallback_neutral = sum(
        1
        for item in stance_signals
        for source_key in ("b1_source", "a_source", "b2_source")
        if item.get(source_key) == "fallback-neutral"
    )

    reward_summary = {
        "dialogue_index": int(episode["dialogue_index"]),
        "current_A_id": current_a,
        "current_B_id": current_b,
        "continued_dialogue_path": _display_path(path),
        "self_svi_A": _safe_float(svi["self_svi_A"]),
        "other_svi_B_to_A": _safe_float(svi["other_svi_B_to_A"]),
        "utility_A": _safe_float(util["utility_A"]),
        "utility_B": _safe_float(util["utility_B"]),
        "joint_utility": _safe_float(util["joint_utility"]),
        "normalized_reward_components": {
            "self_svi_A": _safe_float(svi["self_svi_A"]),
            "other_svi_B_to_A": _safe_float(svi["other_svi_B_to_A"]),
            "utility_A": _safe_float(util["utility_A_norm"]),
            "joint_utility": _safe_float(util["joint_utility_norm"]),
        },
        "final_reward": final_reward,
        "stance_usage": {
            "cached_labels": stance_cache_hits,
            "model_calls": stance_model_calls,
            "fallback_neutral": stance_fallback_neutral,
        },
        "post_intervention_A_turn_ids": [item["turn_id"] for item in turn_level],
        "turn_level": turn_level,
    }

    summary_path = output_dir / (
        f"dialogue_{int(episode['dialogue_index']):05d}_{current_a}_rollout_{int(episode['rollout_index']):02d}_reward_summary.json"
    )
    summary_path.write_text(json.dumps(reward_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[reward] "
        f"dialogue_index={episode['dialogue_index']} A={current_a} "
        f"decompose_and_write_elapsed={time.perf_counter() - decompose_start:.2f}s",
        flush=True,
    )
    print(
        "[episode] "
        f"dialogue_index={episode['dialogue_index']} A={current_a} B={current_b} "
        f"rollout={episode['rollout_index']} final_reward={final_reward:.4f} "
        f"a_turns={len(turn_level)} stance_cached={stance_cache_hits} "
        f"stance_model_calls={stance_model_calls} "
        f"stance_fallback_neutral={stance_fallback_neutral} "
        f"summary={_display_path(summary_path)}",
        flush=True,
    )

    sample_start = time.perf_counter()
    train_samples = [
        {
            "prompt": _build_turn_prompt(
                dialogue=dialogue,
                current_a_id=current_a,
                target_turn_index=int(turn["turn_index"]),
            ),
            "completion": str(turn["text"]),
            "reward": _safe_float(turn["decomposed_reward"]),
            "group_id": f"dialogue_{int(episode['dialogue_index']):05d}_{current_a}",
            "metadata": {
                "dialogue_index": int(episode["dialogue_index"]),
                "current_A_id": current_a,
                "current_B_id": current_b,
                "turn_id": str(turn["turn_id"]),
                "turn_index": int(turn["turn_index"]),
                "rollout_index": int(episode["rollout_index"]),
                "reward_summary_path": _display_path(summary_path),
            },
        }
        for turn in turn_level
    ]
    print(
        "[reward] "
        f"dialogue_index={episode['dialogue_index']} A={current_a} "
        f"train_sample_build_elapsed={time.perf_counter() - sample_start:.2f}s "
        f"episode_finalize_elapsed={time.perf_counter() - episode_start:.2f}s",
        flush=True,
    )

    return reward_summary, train_samples


def finalize_episode_with_stance(
    episode: dict[str, Any],
    svi_components_by_path: dict[str, dict[str, float]],
    utility_components_by_path: dict[str, dict[str, float]],
    config: NegotiationRLPipelineConfig,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return run_episode_and_collect_rewards(
        episode=episode,
        svi_components_by_path=svi_components_by_path,
        utility_components_by_path=utility_components_by_path,
        config=config,
        output_dir=output_dir,
    )


def finalize_episodes_with_stance_batch(
    episodes: list[dict[str, Any]],
    svi_components_by_path: dict[str, dict[str, float]],
    utility_components_by_path: dict[str, dict[str, float]],
    config: NegotiationRLPipelineConfig,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    train_samples: list[dict[str, Any]] = []
    for episode in episodes:
        summary, samples = finalize_episode_with_stance(
            episode=episode,
            svi_components_by_path=svi_components_by_path,
            utility_components_by_path=utility_components_by_path,
            config=config,
            output_dir=output_dir,
        )
        summaries.append(summary)
        train_samples.extend(samples)
    return summaries, train_samples


def rl_train_step(
    trainer: LocalQwenPolicyTrainer,
    train_samples: list[dict[str, Any]],
) -> dict[str, float]:
    return trainer.rl_train_step(train_samples)


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _maybe_save_update_checkpoint(
    *,
    trainer: LocalQwenPolicyTrainer,
    config: NegotiationRLPipelineConfig,
    output_dir: Path,
    update_step: int,
) -> None:
    if config.checkpoint_every_updates <= 0:
        return
    if update_step % max(1, config.checkpoint_every_updates) != 0:
        return
    checkpoint_dir = output_dir / "checkpoints" / f"update_{update_step:05d}"
    print(f"[checkpoint] saving update checkpoint to {checkpoint_dir}", flush=True)
    trainer.save(checkpoint_dir)


def _maybe_save_dialogue_checkpoint(
    *,
    trainer: LocalQwenPolicyTrainer,
    config: NegotiationRLPipelineConfig,
    output_dir: Path,
    dialogues_done: int,
) -> None:
    if config.checkpoint_every_dialogues <= 0:
        return
    if dialogues_done % max(1, config.checkpoint_every_dialogues) != 0:
        return
    checkpoint_dir = output_dir / "checkpoints" / f"dialogue_{dialogues_done:05d}"
    print(f"[checkpoint] saving dialogue checkpoint to {checkpoint_dir}", flush=True)
    trainer.save(checkpoint_dir)


def _run_update_batch_with_retry(
    *,
    trainer: LocalQwenPolicyTrainer,
    batch: list[dict[str, Any]],
    config: NegotiationRLPipelineConfig,
    output_dir: Path,
    metrics_path: Path,
    dialogue_index: int,
    total_updates: int,
) -> tuple[int, bool]:
    snapshot = trainer.snapshot_train_state()
    update_attempt = 0
    update_success = False
    last_error: Exception | None = None
    while update_attempt <= max(0, config.retry_failed_updates):
        update_attempt += 1
        train_start_time = time.perf_counter()
        try:
            train_metrics = rl_train_step(trainer, batch)
            total_updates += 1
            print(
                f"[train] update_step={total_updates} "
                f"attempt={update_attempt} "
                f"elapsed={_format_seconds(time.perf_counter() - train_start_time)} "
                f"metrics={json.dumps(train_metrics, ensure_ascii=False)}",
                flush=True,
            )
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "update_step": total_updates,
                            "dialogue_index": dialogue_index,
                            "num_samples": len(batch),
                            "attempt": update_attempt,
                            **train_metrics,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            _maybe_save_update_checkpoint(
                trainer=trainer,
                config=config,
                output_dir=output_dir,
                update_step=total_updates,
            )
            update_success = True
            break
        except Exception as error:
            last_error = error
            trainer.restore_train_state(snapshot)
            print(
                f"[train] update_step={total_updates + 1} failed on attempt={update_attempt} "
                f"error={error!r}",
                flush=True,
            )
            if update_attempt <= max(0, config.retry_failed_updates):
                print(
                    f"[train] rolled back to pre-update snapshot; retrying batch once more "
                    f"(attempt {update_attempt + 1}/{config.retry_failed_updates + 1})",
                    flush=True,
                )

    if not update_success:
        if config.skip_failed_update_batches:
            print(
                f"[train] skipping failed batch after rollback attempts "
                f"num_samples={len(batch)} error={last_error!r}",
                flush=True,
            )
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "update_step": total_updates,
                            "dialogue_index": dialogue_index,
                            "num_samples": len(batch),
                            "status": "skipped_failed_batch",
                            "error": repr(last_error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        else:
            raise RuntimeError(
                f"Update batch failed after {config.retry_failed_updates + 1} attempt(s)."
            ) from last_error

    return total_updates, update_success


def run_negotiation_rl_training(
    config: NegotiationRLPipelineConfig,
    trainer_config: LocalQwenTrainerConfig,
) -> dict[str, Any]:
    pipeline_start_time = time.perf_counter()
    output_dir = config.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] output_dir={output_dir}", flush=True)
    print(
        "[pipeline] checkpoint_policy "
        f"dialogues={config.checkpoint_every_dialogues} "
        f"updates={config.checkpoint_every_updates} "
        f"legacy_dialogue_markers={list(config.checkpoint_dialogue_counts)} "
        f"retry_failed_updates={config.retry_failed_updates} "
        f"skip_failed_update_batches={config.skip_failed_update_batches} "
        f"checkpoints_dir={checkpoints_dir}",
        flush=True,
    )

    trainer_config.output_dir = output_dir / "adapter"
    print(
        "[pipeline] initializing trainer "
        f"model_name_or_path={trainer_config.model_name_or_path} "
        f"trainer_type={trainer_config.trainer_type} "
        f"online_policy_rollout={config.online_policy_rollout} "
        f"online_max_new_tokens={config.online_max_new_tokens} "
        f"online_do_sample={config.online_do_sample} "
        f"online_repair_attempts={config.online_repair_attempts}",
        flush=True,
    )
    trainer = LocalQwenPolicyTrainer(trainer_config)
    online_batcher = OnlineGenerationBatcher(trainer=trainer, config=config)

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "pipeline": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()},
                "trainer": {k: str(v) if isinstance(v, Path) else v for k, v in trainer_config.__dict__.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics_path = output_dir / "training_metrics.jsonl"
    episode_log_path = output_dir / "episode_summaries.jsonl"

    pending_samples: list[dict[str, Any]] = []
    total_updates = 0
    dialogues_done = 0
    all_summaries = 0
    dialogues_since_update_gate = 0
    try:
        with asyncio.Runner() as runner:
            for dialogue_index in range(config.start_index, config.start_index + config.count):
                dialogue_start_time = time.perf_counter()
                print(f"[pipeline] dialogue_index={dialogue_index} preparing episode tasks", flush=True)
                tasks: list[dict[str, Any]] = []
                for current_a in config.perspectives:
                    current_b = _other_agent(current_a)
                    for rollout_index in range(config.num_rollouts):
                        tasks.append(
                            {
                                "dialogue_index": dialogue_index,
                                "current_A_id": current_a,
                                "current_B_id": current_b,
                                "rollout_index": rollout_index,
                            }
                        )

                print(f"[pipeline] dialogue_index={dialogue_index} rollout_count={len(tasks)}", flush=True)
                rollout_start_time = time.perf_counter()
                episodes = rollout_dialogues_batch(
                    tasks=tasks,
                    config=config,
                    output_dir=output_dir,
                    trainer=trainer,
                    runner=runner,
                    batcher=online_batcher,
                )
                continued_paths = [Path(item["continued_dialogue_path"]) for item in episodes]
                print(
                    f"[pipeline] dialogue_index={dialogue_index} completed rollouts={len(continued_paths)} "
                    f"elapsed={_format_seconds(time.perf_counter() - rollout_start_time)}",
                    flush=True,
                )

                print(f"[pipeline] dialogue_index={dialogue_index} running svi evaluation", flush=True)
                svi_start_time = time.perf_counter()
                svi_components = compute_svi_reward_components(continued_paths, config)
                print(
                    f"[pipeline] dialogue_index={dialogue_index} svi done "
                    f"elapsed={_format_seconds(time.perf_counter() - svi_start_time)}",
                    flush=True,
                )
                print(f"[pipeline] dialogue_index={dialogue_index} running utility evaluation", flush=True)
                utility_start_time = time.perf_counter()
                utility_components = compute_utility_reward_components(continued_paths, config)
                print(
                    f"[pipeline] dialogue_index={dialogue_index} utility done "
                    f"elapsed={_format_seconds(time.perf_counter() - utility_start_time)}",
                    flush=True,
                )

                print(
                    f"[pipeline] dialogue_index={dialogue_index} computing stance, rewards, and train samples",
                    flush=True,
                )
                finalize_start_time = time.perf_counter()
                summaries, samples = finalize_episodes_with_stance_batch(
                    episodes=episodes,
                    svi_components_by_path=svi_components,
                    utility_components_by_path=utility_components,
                    config=config,
                    output_dir=output_dir,
                )
                print(
                    f"[pipeline] dialogue_index={dialogue_index} summaries={len(summaries)} train_samples={len(samples)} "
                    f"elapsed={_format_seconds(time.perf_counter() - finalize_start_time)}",
                    flush=True,
                )

                with episode_log_path.open("a", encoding="utf-8") as file:
                    for summary in summaries:
                        file.write(json.dumps(summary, ensure_ascii=False) + "\n")
                all_summaries += len(summaries)

                pending_samples.extend(samples)
                dialogues_since_update_gate += 1

                should_train_now = dialogues_since_update_gate >= max(1, config.dialogues_per_update)
                if not should_train_now:
                    print(
                        f"[train] deferring updates until {config.dialogues_per_update} dialogue(s) are collected "
                        f"pending_samples={len(pending_samples)} dialogues_collected_since_gate={dialogues_since_update_gate}",
                        flush=True,
                    )

                while should_train_now and len(pending_samples) >= config.train_batch_update_size:
                    batch = pending_samples[: config.train_batch_update_size]
                    pending_samples = pending_samples[config.train_batch_update_size :]
                    print(
                        f"[train] update_step={total_updates + 1} batch_size={len(batch)} "
                        f"pending_after_pop={len(pending_samples)}",
                        flush=True,
                    )
                    total_updates, _ = _run_update_batch_with_retry(
                        trainer=trainer,
                        batch=batch,
                        config=config,
                        output_dir=output_dir,
                        metrics_path=metrics_path,
                        dialogue_index=dialogue_index,
                        total_updates=total_updates,
                    )

                if should_train_now:
                    dialogues_since_update_gate = 0

                dialogues_done += 1
                dialogue_elapsed = time.perf_counter() - dialogue_start_time
                print(
                    f"[pipeline] dialogue_index={dialogue_index} finished "
                    f"elapsed={_format_seconds(dialogue_elapsed)} "
                    f"avg_per_dialogue_so_far={_format_seconds((time.perf_counter() - pipeline_start_time) / max(dialogues_done, 1))}",
                    flush=True,
                )
                _maybe_save_dialogue_checkpoint(
                    trainer=trainer,
                    config=config,
                    output_dir=output_dir,
                    dialogues_done=dialogues_done,
                )
                if dialogues_done in set(config.checkpoint_dialogue_counts):
                    checkpoint_dir = output_dir / "checkpoints" / f"dialogue_{dialogues_done:05d}"
                    print(f"[checkpoint] saving checkpoint to {checkpoint_dir}", flush=True)
                    trainer.save(checkpoint_dir)
    except Exception:
        emergency_dir = output_dir / "adapter_emergency"
        print(f"[checkpoint] saving emergency adapter to {emergency_dir}", flush=True)
        trainer.save(emergency_dir)
        raise

    if pending_samples:
        print(
            f"[train] final update_step={total_updates + 1} batch_size={len(pending_samples)}",
            flush=True,
        )
        total_updates, _ = _run_update_batch_with_retry(
            trainer=trainer,
            batch=pending_samples,
            config=config,
            output_dir=output_dir,
            metrics_path=metrics_path,
            dialogue_index=config.start_index + config.count - 1,
            total_updates=total_updates,
        )

    final_adapter_dir = trainer.save(output_dir / "adapter")
    print(f"[pipeline] saved final adapter to {final_adapter_dir}", flush=True)
    total_elapsed = time.perf_counter() - pipeline_start_time
    print(
        f"[pipeline] total_elapsed={_format_seconds(total_elapsed)} "
        f"avg_seconds_per_dialogue={_format_seconds(total_elapsed / max(dialogues_done, 1))}",
        flush=True,
    )
    run_summary = {
        "output_dir": str(output_dir),
        "adapter_dir": str(final_adapter_dir),
        "num_dialogues": dialogues_done,
        "num_episode_summaries": all_summaries,
        "num_updates": total_updates,
        "metrics_log": str(metrics_path),
        "episode_log": str(episode_log_path),
        "elapsed_seconds": total_elapsed,
        "avg_seconds_per_dialogue": total_elapsed / max(dialogues_done, 1),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_summary
