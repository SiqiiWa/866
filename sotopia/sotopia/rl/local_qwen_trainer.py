from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time
import copy


@dataclass
class LocalQwenTrainerConfig:
    model_name_or_path: str = "Qwen/Qwen3-8B"
    output_dir: Path = Path("examples/experimental/negotiation/rl_outputs")
    resume_checkpoint_path: Path | None = None
    trust_remote_code: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 2048
    max_completion_length: int = 160
    gradient_checkpointing: bool = True
    bf16: bool = True
    trainer_type: str = "grpo"
    grpo_beta_kl: float = 0.02
    grpo_clip_epsilon: float = 0.2
    grpo_adv_eps: float = 1e-6
    allow_cpu_training: bool = False


class LocalQwenPolicyTrainer:
    def __init__(self, config: LocalQwenTrainerConfig) -> None:
        self.config = config

        try:
            import torch
            import torch.nn.functional as F
        except Exception as error:  # pragma: no cover
            raise RuntimeError(
                "torch is required for local RL training. Install with `uv pip install torch`."
            ) from error

        self.torch = torch
        self.F = F

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as error:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for local RL training. Install with `uv pip install transformers torch peft`."
            ) from error

        self.has_cuda = self.torch.cuda.is_available()
        dtype = self.torch.bfloat16 if (config.bf16 and self.has_cuda) else self.torch.float32
        print(
            f"[trainer] torch.cuda.is_available()={self.has_cuda} "
            f"dtype={dtype}",
            flush=True,
        )
        if not self.has_cuda:
            print(
                "[trainer] CUDA is unavailable, so local RL updates will run on CPU. "
                "For Qwen3-8B this can be extremely slow and may look stuck.",
                flush=True,
            )
            if not config.allow_cpu_training:
                raise RuntimeError(
                    "CUDA is unavailable for local RL training. "
                    f"Installed torch={self.torch.__version__} was built with CUDA {self.torch.version.cuda}, "
                    "but this node cannot initialize that CUDA runtime. "
                    "Most likely the PyTorch wheel is newer than the node's NVIDIA driver. "
                    "Install a torch build compatible with this cluster driver, or rerun with "
                    "`allow_cpu_training=True` only if you intentionally want very slow CPU training."
                )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                use_fast=True,
                trust_remote_code=config.trust_remote_code,
            )
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=config.trust_remote_code,
            )
            if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
                # Qwen checkpoints often ship sampling defaults in generation_config.
                # For deterministic online rollout we clear those here so generate()
                # does not warn about unused sampling flags on every turn.
                self.model.generation_config.do_sample = False
                for attr in ("temperature", "top_p", "top_k", "typical_p"):
                    if hasattr(self.model.generation_config, attr):
                        try:
                            setattr(self.model.generation_config, attr, None)
                        except Exception:
                            pass
        except ValueError as error:  # pragma: no cover
            raise RuntimeError(
                "Failed to load the local model. For Qwen3 checkpoints, try upgrading transformers "
                "and ensure remote code loading is enabled. Example: `uv pip install -U transformers`."
            ) from error

        if config.use_lora:
            try:
                from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            except Exception as error:  # pragma: no cover
                raise RuntimeError(
                    "peft is required when use_lora=True. Install with `uv pip install peft`."
                ) from error

            if config.resume_checkpoint_path is not None:
                resume_path = Path(config.resume_checkpoint_path)
                if not resume_path.exists():
                    raise RuntimeError(f"Resume checkpoint path does not exist: {resume_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    str(resume_path),
                    is_trainable=True,
                )
                print(f"[trainer] resumed LoRA adapter from {resume_path}", flush=True)
            else:
                lora_config = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                )
                self.model = get_peft_model(self.model, lora_config)
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

            def _make_inputs_require_grad(_module: Any, _inputs: Any, output: Any) -> Any:
                if hasattr(output, "requires_grad") and not output.requires_grad:
                    output.requires_grad_(True)
                return output

            if hasattr(self.model, "get_input_embeddings"):
                embedding_module = self.model.get_input_embeddings()
                if embedding_module is not None:
                    embedding_module.register_forward_hook(_make_inputs_require_grad)

        if config.gradient_checkpointing:
            if hasattr(self.model, "config") and self.model.config is not None:
                self.model.config.use_cache = False
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()

        self.model.train()
        self.optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        if config.resume_checkpoint_path is not None:
            optimizer_path = Path(config.resume_checkpoint_path) / "optimizer.pt"
            if optimizer_path.exists():
                try:
                    optimizer_state = self.torch.load(optimizer_path, map_location="cpu")
                    self.optimizer.load_state_dict(optimizer_state)
                    print(f"[trainer] resumed optimizer state from {optimizer_path}", flush=True)
                except Exception as error:
                    print(
                        f"[trainer] failed to load optimizer state from {optimizer_path}; "
                        f"resuming weights only. error={error!r}",
                        flush=True,
                    )
            else:
                print(
                    f"[trainer] no optimizer state found at {optimizer_path}; resuming weights only",
                    flush=True,
                )
        try:
            trainable_params = sum(
                param.numel() for param in self.model.parameters() if param.requires_grad
            )
            total_params = sum(param.numel() for param in self.model.parameters())
            print(
                f"[trainer] model_device={self.model.device} "
                f"trainable_params={trainable_params} total_params={total_params}",
                flush=True,
            )
        except Exception:
            print(f"[trainer] model_device={self.model.device}", flush=True)

        self.reference_model = None
        if config.trainer_type == "grpo" and not config.use_lora:
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=config.trust_remote_code,
            )
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad_(False)

    def _encode_example(self, prompt: str, completion: str) -> dict[str, Any]:
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_prompt_length,
        )
        completion_ids = self.tokenizer.encode(
            completion,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_completion_length,
        )

        if not completion_ids:
            completion_ids = [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": self.torch.tensor([input_ids], dtype=self.torch.long, device=self.model.device),
            "labels": self.torch.tensor([labels], dtype=self.torch.long, device=self.model.device),
            "attention_mask": self.torch.tensor([attention_mask], dtype=self.torch.long, device=self.model.device),
        }

    def rl_train_step(self, train_samples: list[dict[str, Any]]) -> dict[str, float]:
        if not train_samples:
            return {"loss": 0.0, "avg_reward": 0.0, "num_samples": 0.0}

        if self.config.trainer_type == "grpo":
            return self._grpo_train_step(train_samples)

        total_loss = 0.0
        total_reward = 0.0
        step_count = 0

        self.optimizer.zero_grad(set_to_none=True)
        for index, sample in enumerate(train_samples, start=1):
            encoded = self._encode_example(sample["prompt"], sample["completion"])
            outputs = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = encoded["labels"][:, 1:].contiguous()

            token_loss = self.F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(shift_labels)

            valid_mask = (shift_labels != -100).float()
            denom = valid_mask.sum().clamp_min(1.0)
            nll = (token_loss * valid_mask).sum() / denom

            reward = float(sample["reward"])
            loss = reward * nll
            (loss / self.config.gradient_accumulation_steps).backward()

            total_loss += float(loss.detach().cpu())
            total_reward += reward
            step_count += 1

            if (
                index % self.config.gradient_accumulation_steps == 0
                or index == len(train_samples)
            ):
                self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": total_loss / max(step_count, 1),
            "avg_reward": total_reward / max(step_count, 1),
            "num_samples": float(step_count),
        }

    def _token_logprob_mean(self, model: Any, encoded: dict[str, Any]) -> Any:
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = encoded["labels"][:, 1:].contiguous()

        log_probs = self.F.log_softmax(shift_logits, dim=-1)
        gather_labels = shift_labels.clone()
        gather_labels[gather_labels == -100] = 0
        selected = log_probs.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)

        valid_mask = (shift_labels != -100).float()
        denom = valid_mask.sum().clamp_min(1.0)
        return (selected * valid_mask).sum() / denom

    def _group_normalized_advantages(self, train_samples: list[dict[str, Any]]) -> list[float]:
        grouped: dict[str, list[int]] = {}
        for idx, sample in enumerate(train_samples):
            group_id = str(sample.get("group_id", "global"))
            grouped.setdefault(group_id, []).append(idx)

        rewards = [float(sample["reward"]) for sample in train_samples]
        advantages = [0.0] * len(train_samples)
        for _, indices in grouped.items():
            values = self.torch.tensor([rewards[i] for i in indices], dtype=self.torch.float32)
            mean = values.mean()
            std = values.std(unbiased=False)
            normalized = (values - mean) / (std + self.config.grpo_adv_eps)
            for i, adv in zip(indices, normalized.tolist()):
                advantages[i] = float(adv)
        return advantages

    def _get_reference_logprob(self, encoded: dict[str, Any]) -> Any:
        with self.torch.no_grad():
            if self.reference_model is not None:
                return self._token_logprob_mean(self.reference_model, encoded)
            if self.config.use_lora and hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    return self._token_logprob_mean(self.model, encoded)
            return self._token_logprob_mean(self.model, encoded)

    def _grpo_train_step(self, train_samples: list[dict[str, Any]]) -> dict[str, float]:
        step_start_time = time.perf_counter()
        print(
            f"[trainer] starting grpo train step num_samples={len(train_samples)} "
            f"grad_accum={self.config.gradient_accumulation_steps}",
            flush=True,
        )
        advantages = self._group_normalized_advantages(train_samples)

        encode_start_time = time.perf_counter()
        encoded_batch = [
            self._encode_example(sample["prompt"], sample["completion"]) for sample in train_samples
        ]
        print(
            f"[trainer] encoded batch elapsed={time.perf_counter() - encode_start_time:.2f}s",
            flush=True,
        )

        old_logprobs = []
        ref_logprobs = []
        ref_start_time = time.perf_counter()
        with self.torch.no_grad():
            for encoded in encoded_batch:
                old_logprobs.append(self._token_logprob_mean(self.model, encoded))
                ref_logprobs.append(self._get_reference_logprob(encoded))
        print(
            f"[trainer] collected old/ref logprobs elapsed={time.perf_counter() - ref_start_time:.2f}s",
            flush=True,
        )

        total_loss = 0.0
        total_reward = 0.0
        total_kl = 0.0

        self.optimizer.zero_grad(set_to_none=True)
        for index, (sample, encoded) in enumerate(zip(train_samples, encoded_batch), start=1):
            sample_start_time = time.perf_counter()
            new_logprob = self._token_logprob_mean(self.model, encoded)
            old_logprob = old_logprobs[index - 1]
            ref_logprob = ref_logprobs[index - 1]

            advantage = self.torch.tensor(advantages[index - 1], device=self.model.device)
            ratio = self.torch.exp(new_logprob - old_logprob)
            clipped_ratio = self.torch.clamp(
                ratio,
                1.0 - self.config.grpo_clip_epsilon,
                1.0 + self.config.grpo_clip_epsilon,
            )
            policy_obj = self.torch.minimum(ratio * advantage, clipped_ratio * advantage)
            policy_loss = -policy_obj

            kl_penalty = (new_logprob - ref_logprob) ** 2
            loss = policy_loss + self.config.grpo_beta_kl * kl_penalty
            (loss / self.config.gradient_accumulation_steps).backward()

            total_loss += float(loss.detach().cpu())
            total_reward += float(sample["reward"])
            total_kl += float(kl_penalty.detach().cpu())

            if (
                index % self.config.gradient_accumulation_steps == 0
                or index == len(train_samples)
            ):
                self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            print(
                f"[trainer] sample={index}/{len(train_samples)} "
                f"reward={float(sample['reward']):.4f} "
                f"elapsed={time.perf_counter() - sample_start_time:.2f}s",
                flush=True,
            )

        denom = max(len(train_samples), 1)
        print(
            f"[trainer] finished grpo train step elapsed={time.perf_counter() - step_start_time:.2f}s",
            flush=True,
        )
        return {
            "loss": total_loss / denom,
            "avg_reward": total_reward / denom,
            "avg_kl": total_kl / denom,
            "num_samples": float(len(train_samples)),
        }

    def save(self, output_dir: Path | None = None) -> Path:
        save_dir = output_dir or self.config.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        self.torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pt")
        return save_dir

    def snapshot_train_state(self) -> dict[str, Any]:
        trainable_state = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        return {
            "trainable_state": trainable_state,
            "optimizer_state": optimizer_state,
        }

    def restore_train_state(self, snapshot: dict[str, Any]) -> None:
        trainable_state = snapshot["trainable_state"]
        param_map = dict(self.model.named_parameters())
        with self.torch.no_grad():
            for name, value in trainable_state.items():
                if name in param_map:
                    param_map[name].copy_(value.to(param_map[name].device))
        self.optimizer.load_state_dict(snapshot["optimizer_state"])
