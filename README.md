# 866

This repository contains a working research and experimentation workspace for negotiation modeling built around Sotopia and a set of custom baseline and evaluation scripts.

The project combines:

- `Baseline_1/`: an initial baseline pipeline and prompts
- `Baseline_2/`: stance labeling, important-turn selection, and SVI-style evaluation scripts
- `data/`: structured negotiation data and generated analysis artifacts
- `localhost/`: local model utilities and helper scripts
- `11866_temp/`: temporary or exploratory negotiation pipeline scripts
- `sotopia/`: a local copy of the Sotopia framework used as the main experimental environment

## What This Repo Is For

The repository is designed for running and analyzing negotiation experiments, especially:

- multi-agent casino-style negotiation rollouts
- stance annotation and turn-level analysis
- utility and SVI-oriented evaluation
- local or proxy-backed model execution
- rapid iteration on prompts, pipelines, and experiment settings

## Environment Setup

Several scripts in this repository require API credentials through environment variables rather than hard-coded keys.

At minimum, export:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Depending on which pipeline you run, some scripts may also read:

```bash
export LITELLM_PROXY_API_KEY="your_proxy_key_here"
export CUSTOM_API_KEY="your_custom_key_here"
```

## Repository Notes

- API keys have been migrated to environment-variable based configuration.
- Large local artifacts such as virtual environments, checkpoints, and backup git metadata are excluded through `.gitignore`.
- The `sotopia/` directory is included here as part of the full experiment workspace rather than as a standalone nested Git repository.

## Quick Start

1. Clone the repository.
2. Set the required environment variables.
3. Create or activate your Python environment.
4. Run the script or experiment pipeline you need from `Baseline_1/`, `Baseline_2/`, `localhost/`, or `sotopia/examples/experimental/negotiation/`.

## Structure

```text
866/
├── 11866_temp/
├── Baseline_1/
├── Baseline_2/
├── data/
├── localhost/
└── sotopia/
```

## Disclaimer

This repository is an active research workspace. Some folders contain intermediate outputs, experiment-specific scripts, and one-off analysis files that are useful for reproducibility but may not represent polished library interfaces.
