from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sotopia.rl.local_qwen_trainer import LocalQwenTrainerConfig
from sotopia.rl.negotiation_pipeline import (
    NegotiationRLPipelineConfig,
    run_negotiation_rl_training,
)


def _env_api_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LITELLM_PROXY_API_KEY")
        or os.environ.get("CUSTOM_API_KEY")
        or ""
    )


def _is_local_url(url: str | None) -> bool:
    if not url:
        return False
    hostname = urlparse(url).hostname or ""
    return hostname in {"127.0.0.1", "localhost", "::1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal-intrusion RL orchestration for local Qwen3-8B on negotiation data."
    )
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
    )
    parser.add_argument("--rollout-base-url", type=str, default=None)
    parser.add_argument(
        "--rollout-api-key",
        type=str,
        default=None,
    )
    parser.add_argument("--eval-base-url", type=str, default=None)
    parser.add_argument(
        "--eval-api-key",
        type=str,
        default=None,
    )
    parser.add_argument("--rollout-model", type=str, default="openai/Qwen3-8B")
    parser.add_argument("--evaluator-model", type=str, default="gpt-5-mini")
    parser.add_argument("--stance-model", type=str, default="gpt-5-mini")

    parser.add_argument("--dialogue-path", type=Path, default=Path("/home/swang4/866/data/casino_first100_structured.json"))
    parser.add_argument("--important-turns-path", type=Path, default=Path("/home/swang4/866/data/baseline_2/important_turns.json"))
    parser.add_argument("--stance-prompt-path", type=Path, default=Path("/home/swang4/866/Baseline_2/stance_prompt.py"))
    parser.add_argument("--stance-labels-path", type=Path, default=Path("/home/swang4/866/data/baseline_2/stance_labels_0-099.json"))

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-completion-tokens", type=int, default=768)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--online-max-new-tokens", type=int, default=80)
    parser.add_argument("--online-speak-max-new-tokens", type=int, default=None)
    parser.add_argument("--online-non-speak-max-new-tokens", type=int, default=48)
    parser.add_argument("--online-do-sample", action="store_true", default=False)
    parser.add_argument("--online-temperature", type=float, default=0.0)
    parser.add_argument("--online-top-p", type=float, default=1.0)
    parser.add_argument("--online-repair-attempts", type=int, default=2)
    parser.add_argument("--online-batch-size", type=int, default=4)
    parser.add_argument("--online-batch-wait-ms", type=int, default=15)
    parser.add_argument("--online-disable-thinking", action="store_true", default=True)
    parser.add_argument("--allow-online-thinking", dest="online_disable_thinking", action="store_false")

    parser.add_argument("--continuation-max-concurrency", type=int, default=2)
    parser.add_argument("--svi-max-concurrency", type=int, default=4)
    parser.add_argument("--utility-max-concurrency", type=int, default=4)
    parser.add_argument("--stance-max-concurrency", type=int, default=8)

    parser.add_argument("--train-batch-update-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda-local", type=float, default=0.5)
    parser.add_argument("--dialogues-per-update", type=int, default=1)
    parser.add_argument("--checkpoint-every-dialogues", type=int, default=0)
    parser.add_argument("--checkpoint-every-updates", type=int, default=0)
    parser.add_argument("--retry-failed-updates", type=int, default=1)
    parser.add_argument("--skip-failed-update-batches", action="store_true", default=True)
    parser.add_argument("--no-skip-failed-update-batches", dest="skip_failed_update_batches", action="store_false")
    parser.add_argument("--output-parent", type=Path, default=Path("examples/experimental/negotiation"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--resume-checkpoint-path", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=160)

    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-use-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")

    parser.add_argument("--trainer-type", type=str, default="grpo", choices=["grpo", "weighted_sft"])
    parser.add_argument("--grpo-beta-kl", type=float, default=0.02)
    parser.add_argument("--grpo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--grpo-adv-eps", type=float, default=1e-6)
    parser.add_argument("--allow-cpu-training", action="store_true", default=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared_api_key = args.api_key or _env_api_key() or "dummy"
    rollout_base_url = args.rollout_base_url or args.base_url
    eval_base_url = args.eval_base_url or args.base_url

    if args.rollout_api_key is not None:
        rollout_api_key = args.rollout_api_key
    elif _is_local_url(rollout_base_url):
        rollout_api_key = "dummy"
    else:
        rollout_api_key = shared_api_key

    if args.eval_api_key is not None:
        eval_api_key = args.eval_api_key
    else:
        eval_api_key = shared_api_key

    pipeline_config = NegotiationRLPipelineConfig(
        base_url=args.base_url,
        api_key=shared_api_key,
        rollout_base_url=rollout_base_url,
        rollout_api_key=rollout_api_key,
        eval_base_url=eval_base_url,
        eval_api_key=eval_api_key,
        rollout_model=args.rollout_model,
        evaluator_model=args.evaluator_model,
        stance_model=args.stance_model,
        dialogue_path=args.dialogue_path,
        important_turns_path=args.important_turns_path,
        stance_prompt_path=args.stance_prompt_path,
        stance_labels_path=args.stance_labels_path,
        start_index=args.start_index,
        count=args.count,
        num_rollouts=args.num_rollouts,
        max_turns=args.max_turns,
        max_completion_tokens=args.max_completion_tokens,
        reasoning_effort=args.reasoning_effort,
        online_max_new_tokens=args.online_max_new_tokens,
        online_speak_max_new_tokens=args.online_speak_max_new_tokens,
        online_non_speak_max_new_tokens=args.online_non_speak_max_new_tokens,
        online_do_sample=args.online_do_sample,
        online_temperature=args.online_temperature,
        online_top_p=args.online_top_p,
        online_repair_attempts=args.online_repair_attempts,
        online_batch_size=args.online_batch_size,
        online_batch_wait_ms=args.online_batch_wait_ms,
        online_disable_thinking=args.online_disable_thinking,
        continuation_max_concurrency=args.continuation_max_concurrency,
        svi_max_concurrency=args.svi_max_concurrency,
        utility_max_concurrency=args.utility_max_concurrency,
        stance_max_concurrency=args.stance_max_concurrency,
        alpha=args.alpha,
        beta=args.beta,
        lambda_local=args.lambda_local,
        train_batch_update_size=args.train_batch_update_size,
        dialogues_per_update=args.dialogues_per_update,
        checkpoint_every_dialogues=args.checkpoint_every_dialogues,
        checkpoint_every_updates=args.checkpoint_every_updates,
        retry_failed_updates=args.retry_failed_updates,
        skip_failed_update_batches=args.skip_failed_update_batches,
        output_parent=args.output_parent,
        run_name=args.run_name,
    )

    print(f"Resolved rollout_base_url={rollout_base_url}")
    print(f"Resolved eval_base_url={eval_base_url}")
    print(f"Resolved rollout_api_key={'dummy' if rollout_api_key == 'dummy' else 'set'}")
    print(f"Resolved eval_api_key={'set' if eval_api_key else 'missing'}")

    trainer_config = LocalQwenTrainerConfig(
        model_name_or_path=args.model_name_or_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        trainer_type=args.trainer_type,
        grpo_beta_kl=args.grpo_beta_kl,
        grpo_clip_epsilon=args.grpo_clip_epsilon,
        grpo_adv_eps=args.grpo_adv_eps,
        allow_cpu_training=args.allow_cpu_training,
    )

    summary = run_negotiation_rl_training(
        config=pipeline_config,
        trainer_config=trainer_config,
    )

    print("RL pipeline finished")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
