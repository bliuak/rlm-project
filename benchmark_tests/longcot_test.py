from rlm import RLM 
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()


chess_longcot = load_dataset("LongHorizonReasoning/longcot", "chess")
logic_longcot = load_dataset("LongHorizonReasoning/longcot", "logic")

medium_logic = logic_longcot["medium"] 

easy = chess_longcot["easy"]
medium = chess_longcot["medium"]
hard = chess_longcot["hard"]


question = medium_logic[2]
print(question["prompt"])
print("answer: ", question["answer"])
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5.4-mini"},
    max_depth=2,
    verbose=True,  # For printing to console with rich, disabled by default.
)
print(rlm.completion(question["prompt"]).response)