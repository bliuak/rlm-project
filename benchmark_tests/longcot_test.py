from rlm import RLM 
from dotenv import load_dotenv


load_dotenv()

rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5.4-mini"},
    verbose=True,  # For printing to console with rich, disabled by default.
    max_depth=2
)

print(rlm.completion("What model are you?").response)