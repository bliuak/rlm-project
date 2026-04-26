import os

from datasets import load_dataset
from dotenv import load_dotenv
from rlm import RLM

load_dotenv()

backend = os.getenv("RLM_BACKEND", "openai")
if backend == "openrouter":
    default_model_name = "openai/gpt-4o-mini"
else:
    default_model_name = "gpt-5.4-mini"
model_name = (
    os.getenv("RLM_MODEL_NAME")
    or os.getenv("OPENROUTER_MODEL")
    or default_model_name
)

required_api_key = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}.get(backend)

if required_api_key and not os.getenv(required_api_key):
    raise RuntimeError(
        f"{required_api_key} is required for RLM_BACKEND={backend!r}. "
        "Add it to your .env file."
    )

logic_longcot = load_dataset("LongHorizonReasoning/longcot", "logic")
medium_logic = logic_longcot["medium"]
question = medium_logic[2]

print(question["prompt"])
print("answer: ", question["answer"])

rlm = RLM(
    backend=backend,
    backend_kwargs={"model_name": model_name},
    max_depth=2,
    verbose=True,  # For printing to console with rich, disabled by default.
)
print(rlm.completion(question["prompt"]).response)
