import os
import time
import csv
import torch
from pathlib import Path

from rankify.generator.generator import Generator
from rankify.dataset.dataset import Dataset
from rankify.metrics.metrics import Metrics
from rankify.n_retreivers.retriever import Retriever

# -------------------- Config --------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Files
summary_path = "benchmark_time_cost.txt"     # human-readable log
csv_path = "nq100_time_cost.csv"             # machine-readable time table
header_written = Path(summary_path).exists()
csv_header_written = Path(csv_path).exists()

# Dataset: only Natural Questions
DATASETS = ["nq-test"]

# RAG methods
RAG_METHODS = [
    "basic-rag",
    "chain-of-thought-rag",
    "fid",
    "in-context-ralm",
    "zero-shot",
    "self-consistency-rag",
    "react-rag"
]

# Model(s)
MODELS = [
    {
        "model_name": 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        "backend": "huggingface",
        "torch_dtype": torch.float16,
        "stop_at_period": True,
    },
]

FID_MODEL_NAME = "nq_reader_base"
FID_BACKEND = "fid"
N_DOCS = 1

# Use 100 questions from nq-test
N_QUESTIONS = 100

# Optional generation kwargs (kept for reference)
generation_kwargs = dict(
    temperature=0.7,
    top_p=0.95,
    max_new_tokens=32,
    num_return_sequences=1,
)

results = {}

# Prepare CSV header
if not csv_header_written:
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "model", "dataset", "method",
            "q_count",
            "init_time_s",
            "generation_time_s",
            "total_time_s",
            "avg_per_query_s",
            "throughput_qps"
        ])
    csv_header_written = True

for model_cfg in MODELS:
    print("\n" + "#" * 120)
    print(f"Evaluating model: {model_cfg['model_name']}")
    results[model_cfg['model_name']] = {}

    for dataset_name in DATASETS:
        print("=" * 120)
        print(f"Evaluating dataset: {dataset_name}")
        dataset = Dataset('bm25', dataset_name, N_DOCS)

        # Downloading / preparing the dataset is *not* timed
        documents = dataset.download(force_download=False)
        if N_QUESTIONS is not None:
            documents = documents[:N_QUESTIONS]

        q_count = len(documents)
        metrics = Metrics(documents)
        results[model_cfg['model_name']][dataset_name] = {}

        for rag_method in RAG_METHODS:
            print("-" * 80)
            print(f"Testing RAG method: {rag_method}")

            # -------------------- Initialization timing --------------------
            init_start = time.time()

            try:
                if rag_method == "fid":
                    generator = Generator(
                        method="fid",
                        model_name=FID_MODEL_NAME,
                        backend=FID_BACKEND
                    )
                elif rag_method == "react-rag":
                    retriever = Retriever(method="bm25", n_docs=N_DOCS, index_type="wiki")
                    generator = Generator(
                        method="react-rag",
                        retriever=retriever,
                        **model_cfg
                    )
                else:
                    generator = Generator(
                        method=rag_method,
                        **model_cfg
                    )
            except Exception as e:
                print(f"Init error for {rag_method} on {dataset_name}: {e}")
                # If init fails, record and continue
                init_time = time.time() - init_start
                generation_time = 0.0
                total_time = init_time
                generated_answers = [""] * q_count
            else:
                init_time = time.time() - init_start

                # -------------------- Generation timing --------------------
                gen_start = time.time()
                try:
                    # Batch generation (fastest, consistent with your current setup)
                    generated_answers = generator.generate(documents)
                except Exception as e:
                    print(f"Error with method {rag_method} on dataset {dataset_name}: {e}")
                    generated_answers = [""] * q_count
                generation_time = time.time() - gen_start
                total_time = init_time + generation_time

            # -------------------- Metrics & logging --------------------
            generation_metrics = metrics.calculate_generation_metrics(generated_answers)
            results[model_cfg['model_name']][dataset_name][rag_method] = {
                "metrics": generation_metrics,
                "init_time_s": round(init_time, 2),
                "generation_time_s": round(generation_time, 2),
                "total_time_s": round(total_time, 2),
                "avg_per_query_s": round(generation_time / max(q_count, 1), 3),
                "throughput_qps": round(max(q_count, 1) / generation_time, 3) if generation_time > 0 else 0.0,
            }

            # Append to TXT summary
            with open(summary_path, "a", encoding="utf-8") as f:
                if not header_written:
                    f.write("Benchmark Time-Cost Results (NQ-100):\n")
                    f.write("Times exclude dataset download; initialization includes model/method setup.\n")
                    header_written = True
                f.write(f"\nModel: {model_cfg['model_name']}\n")
                f.write(f"  Dataset: {dataset_name}\n")
                f.write(
                    f"    {rag_method}: metrics={generation_metrics}, "
                    f"init={init_time:.2f}s, gen={generation_time:.2f}s, "
                    f"total={total_time:.2f}s, avg/qp={generation_time/max(q_count,1):.3f}s, "
                    f"throughput={ (max(q_count,1)/generation_time) if generation_time>0 else 0.0 :.3f} qps\n"
                )

            # Append to CSV table
            with open(csv_path, "a", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow([
                    model_cfg['model_name'],
                    dataset_name,
                    rag_method,
                    q_count,
                    round(init_time, 3),
                    round(generation_time, 3),
                    round(total_time, 3),
                    round(generation_time / max(q_count, 1), 4),
                    round((max(q_count, 1) / generation_time) if generation_time > 0 else 0.0, 4)
                ])

            print(
                f"Done {rag_method}: init {init_time:.2f}s | "
                f"gen {generation_time:.2f}s | total {total_time:.2f}s | "
                f"avg {generation_time/max(q_count,1):.3f}s | "
                f"{ (max(q_count,1)/generation_time) if generation_time>0 else 0.0 :.3f} qps"
            )
            print("=" * 120)

        # Optionally save intermediate dataset cache
        dataset.save_dataset(f"{dataset_name}-bm25-eval.json", save_text=True)

print(f"\nTime-cost TXT summary: {summary_path}")
print(f"Time-cost CSV table : {csv_path}")
