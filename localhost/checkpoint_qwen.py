from __future__ import annotations

import argparse
import os
import subprocess


DEFAULT_CHECKPOINT = (
    "/data/user_data/swang4/sotopia_runs/"
    "rl_batch_sleep_resume_from_remote_scratch_u14/checkpoints/update_00111"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a trained Qwen checkpoint through vLLM."
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--served-model-name", default="openai/Qwen3-8B-u111")
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lora-name", default="u111")
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        "vllm",
        "serve",
        args.base_model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--enforce-eager",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enable-lora",
        "--lora-modules",
        f"{args.lora_name}={args.checkpoint_path}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print("Launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
