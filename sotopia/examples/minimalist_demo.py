# This demo servers as a minimal example of how to use the sotopia library.

# 1. Import the sotopia library
# 1.1. Import the `run_async_server` function: In sotopia, we use Python Async
#     API to optimize the throughput.
import asyncio
import logging
import os

# 1.2. Import the `UniformSampler` class: In sotopia, we use samplers to sample
#     the social tasks.
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server
from rich.logging import RichHandler

# 2. Run the server

# 2.1. Configure the logging
FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)

# 2.2. Set environment variables for local vLLM server
os.environ["CUSTOM_API_KEY"] = "EMPTY"  # vLLM doesn't require API key

# 2.3. Run the simulation with epilog saving enabled
# The dialogue results will be saved to the database with tag="qwen_test"
asyncio.run(
    run_async_server(
        model_dict={
            "env": "custom/openai/Qwen3-8B@http://localhost:8000/v1",
            "agent1": "custom/openai/Qwen3-8B@http://localhost:8000/v1",
            "agent2": "custom/openai/Qwen3-8B@http://localhost:8000/v1",
            "evaluator": "custom/openai/Qwen3-8B@http://localhost:8000/v1",
        },
        sampler=UniformSampler(),
        push_to_db=True,  # Enable saving to database
        tag="qwen_test",  # Tag for organizing episodes
    )
)
