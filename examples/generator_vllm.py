import torch
from vllm import SamplingParams
from rankify.dataset.dataset import Document, Question, Answer, Context
from rankify.generator.generator import Generator

# Define question and answer
question = Question("What is the capital of France?")
answers=Answer("")
contexts = [
    Context(id=1, title="France", text="The capital of France is Paris.", score=0.9),
    Context(id=2, title="Germany", text="Berlin is the capital of Germany.", score=0.5)
]

# Construct document
doc = Document(question=question, answers=answers, contexts=contexts)

# Define sampling parameters for vllm
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=32, n=1, stop=["###", "</s>", "\n\n", "."])# stop=["\n"])

# Initialize Generator (e.g., Meta Llama)
generator = Generator(method="basic-rag", model_name='meta-llama/Meta-Llama-3.1-8B-Instruct', backend="vllm", dtype="float16",  max_model_len=2048)

# Generate answer
generated_answers = generator.generate([doc],sampling_params=sampling_params)

# Print the generated answers
print(generated_answers) 
#output = generated_answers[0][0]
#print(output.prompt.strip())
#print(output.outputs[0].text.strip())  # Output: ["Paris"]
