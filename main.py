import argparse
import hashlib
import importlib
import random
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from openai import OpenAI
from tqdm import trange


@dataclass
class Metrics:
    time_ns: float
    rel_error: float
    age: int = 0


@dataclass
class Candidate:
    src: str
    metrics: Metrics | None = None

    def key(self) -> str:
        return hashlib.sha256(self.src.encode()).hexdigest()[:12]

    def __eq__(self, other):
        if not isinstance(other, Candidate):
            return NotImplemented

        return self.src == other.src

    def __hash__(self):
        return hash(self.src)


BENCH_SIZES = [128, 256]
DTYPE = np.float32


def compile(candidate: Candidate) -> str:
    path = Path(tempfile.mkdtemp()) / f"{candidate.key()}.py"
    path.write_text(candidate.src)

    return str(path)


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("matmul_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, "matmul"):
        raise AttributeError("Candidate has no matmul() function")

    return mod


def relative_error(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.max(np.abs(ref - test)) / (np.max(np.abs(ref)) + 1e-8))


def evaluate(candidate: Candidate, repeats: int = 3) -> Metrics:
    try:
        path = compile(candidate)
        mod = load_module(path)

        times = []
        errors = []
        for n in BENCH_SIZES:
            a = np.random.rand(n, n).astype(DTYPE)
            b = np.random.rand(n, n).astype(DTYPE)
            ref = a @ b

            best = float("inf")
            for _ in range(repeats):
                try:
                    start = time.perf_counter_ns()
                    out = mod.matmul(a, b)
                    best = min(best, time.perf_counter_ns() - start)
                except Exception:
                    return Metrics(time_ns=float("inf"), rel_error=float("inf"))

            times.append(best)
            errors.append(relative_error(ref, out))

        return Metrics(time_ns=float(np.mean(times)), rel_error=float(np.max(errors)))
    except Exception:
        return Metrics(time_ns=float("inf"), rel_error=float("inf"))


def dominates(a: Metrics, b: Metrics) -> bool:
    return (a.time_ns <= b.time_ns and a.rel_error <= b.rel_error) and (
        a.time_ns < b.time_ns or a.rel_error < b.rel_error
    )


def pareto_front(pop: List[Candidate]) -> List[Candidate]:
    front = []
    for c in pop:
        if c.metrics.time_ns == float("inf") or c.metrics.rel_error == float("inf"):
            continue

        if any(
            dominates(other.metrics, c.metrics)
            for other in pop
            if other is not c
            and other.metrics is not None
            and other.metrics.time_ns != float("inf")
            and other.metrics.rel_error != float("inf")
        ):
            continue

        front.append(c)

    return front


def crowding_distance(front: List[Candidate]) -> List[Tuple[Candidate, float]]:
    if len(front) <= 2:
        return [(c, float("inf")) for c in front]

    dist = {c: 0.0 for c in front}

    for attr in ("time_ns", "rel_error"):
        front.sort(key=lambda c: getattr(c.metrics, attr))
        dist[front[0]] = dist[front[-1]] = float("inf")

        lo = getattr(front[0].metrics, attr)
        hi = getattr(front[-1].metrics, attr)

        for i in range(1, len(front) - 1):
            prev = getattr(front[i - 1].metrics, attr)
            nxt = getattr(front[i + 1].metrics, attr)

            if hi - lo > 0:
                dist[front[i]] += (nxt - prev) / (hi - lo)

    return [(c, dist[c]) for c in front]


NAIVE_IMPL = textwrap.dedent(
    """
    import numpy as np
    
    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = a.shape[0]
        c = np.zeros((n, n), dtype=a.dtype)
        for i in range(n):
            for k in range(n):
                aik = a[i, k]
                for j in range(n):
                    c[i, j] += aik * b[k, j]
        return c
    """
)

client = OpenAI()

MUTATION_STRATEGIES = {
    "LOOP_MANIPULATION": [
        "Reorder the loops (e.g., from ijk to ikj, jik, jki, kij, or kji). Ensure all three loops (i, j, k) for matrix dimensions are present and iterate up to n (or corresponding dimension). Do not use np.matmul, np.dot, or @.",
        "Attempt to unroll the innermost loop by a factor of 2. Adjust loop bounds and handle any remaining elements if a dimension 'n' is not a multiple of 2. Do not use np.matmul, np.dot, or @.",
        "Attempt to unroll one of the non-innermost loops (e.g., middle loop) by a factor of 2. Adjust loop bounds and handle any remaining elements. Do not use np.matmul, np.dot, or @.",
        "Introduce loop tiling (blocking) for two outer loops. Use a specific block size (e.g., choose 16 or 32). The implementation must correctly handle matrix sizes that are not multiples of the block size. Do not use np.matmul, np.dot, or @.",
    ],
    "NUMPY_UTILIZATION": [
        "If manual loops are present for the core multiplication, replace the innermost loop with a NumPy vector operation (e.g., element-wise multiplication and sum) but do not use np.dot, np.matmul, or @.",
        "Vectorize the computation of one full row of the output matrix c at a time using NumPy operations and broadcasting, aiming to eliminate explicit Python loops for that row's calculation, but do not use np.matmul, np.dot, or @.",
    ],
    "DATA_ACCESS_PATTERNS": [
        "Explicitly transpose matrix b (i.e., b_T = b.T) before the main computation loops. Then, adjust the loop access for b_T (e.g., using b_T[j, k] instead of b[k, j]) to potentially improve performance by altering memory access patterns. Do not use np.matmul, np.dot, or @.",
        "If three nested loops (e.g., i,j,k) are present, identify a value that is constant within the innermost loop (e.g., a[i,k] if loop order is i,j,k with k as the middle loop and j as innermost). Load this value into a temporary variable right before the innermost loop begins to potentially reduce redundant memory accesses. Do not use np.matmul, np.dot, or @.",
    ],
    "ALTERNATIVE_NUMPY_OPS": [
        "If np.zeros is used for initialization of the result matrix c, try using np.empty instead. Ensure the rest of the logic correctly fills all elements of c (note: np.zeros is generally safer for accumulation, so this is exploratory). Do not use np.matmul, np.dot, or @.",
        "Explore using np.multiply and np.sum along an axis if a loop can be refactored into such a pattern, as an alternative to direct dot products or einsum for a part of the operation. Do not use np.matmul, np.dot, or @.",
    ],
    "ALGORITHMIC": [
        "Rewrite the matrix multiplication using Strassen’s algorithm. If matrix size is not a power of 2, pad or fall back to naive for small matrices. Do not use np.matmul, np.dot, or @.",
        "Implement matrix multiplication with a recursive divide-and-conquer (block splitting) approach. Do not use np.matmul, np.dot, or @.",
        "Split the matrices into blocks (16x16 or 32x32) and perform block-wise multiplication for cache efficiency. Handle edge cases for sizes not divisible by the block size. Do not use np.matmul, np.dot, or @.",
        "Use a naive loop-based multiplication for small matrices and a more advanced manual implementation for large matrices (e.g., tiling, unrolling). Do not use np.matmul, np.dot, or @.",
    ],
    "PERFORMANCE_TUNING": [
        "Transpose matrix b before multiplication and access elements in a cache-friendly way. Do not use np.matmul, np.dot, or @.",
        "Fuse or split nested loops in the multiplication; e.g., combine inner two loops, or split a loop into halves. Do not use np.matmul, np.dot, or @.",
        "Change the order in which the result matrix is accumulated—accumulate by rows first, then columns, or vice versa. Do not use np.matmul, np.dot, or @.",
        "Use temporary buffers to store intermediate sums or products before writing to the output. Do not use np.matmul, np.dot, or @.",
        "Add explicit manual blocking and memory prefetching in the core computation. Do not use np.matmul, np.dot, or @.",
    ],
    "DATA_REPRESENTATION": [
        "Assume one or both matrices are stored in column-major order, and adapt all indexing accordingly. Do not use np.matmul, np.dot, or @.",
        "If any input is sparse, use efficient sparse multiplication for those cases. Do not use np.matmul, np.dot, or @.",
        "Add a fast path for diagonal or banded matrices. Do not use np.matmul, np.dot, or @.",
    ],
    "NOVELTY_WEIRDNESS": [
        "Add small, random perturbations to the result matrix after multiplication (testing robustness). Do not use np.matmul, np.dot, or @.",
        "Combine elements from two different multiplication strategies, e.g., outer loop from naive, inner logic from blocked. Do not use np.matmul, np.dot, or @.",
        "Rewrite the function to use an intentionally inefficient approach, e.g., repeated addition instead of multiplication. Do not use np.matmul, np.dot, or @.",
        "Implement the multiplication in float16, float32, and float64; compare their speed and accuracy. Do not use np.matmul, np.dot, or @.",
        "Rewrite the function to use only Python lists and loops, without using any NumPy functions or methods. Do not use np.matmul, np.dot, or @.",
    ],
    "HYBRID_META": [
        "Apply two of the above strategies at once; for example, transpose matrix b and apply blocking, or combine recursive divide-and-conquer with switching numeric precision. Do not use np.matmul, np.dot, or @.",
        "Make the function call itself recursively for submatrices. Do not use np.matmul, np.dot, or @.",
    ],
}


def select_mutation_strategy() -> str:
    all_strategies = []

    for category_strategies in MUTATION_STRATEGIES.values():
        all_strategies.extend(category_strategies)

    return random.choice(all_strategies)


def llm_mutate(src: str) -> str:
    strategy = select_mutation_strategy()

    system_prompt = (
        "You are an expert Python programmer specializing in NumPy and high-performance "
        "scientific computing. Your task is to meticulously mutate the provided matrix "
        "multiplication code (`matmul` function) according to a specific strategy. "
        "The function signature MUST be `def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray`. "
        "The returned code must be a complete Python script, containing ONLY the necessary "
        "import statements (e.g., `import numpy as np`) and the `matmul` function. "
        "Absolutely NO explanations, NO comments outside the code, and NO markdown formatting "
        "(e.g., ```python ... ```) should be present in your output. "
        "The mutation must be a meaningful alteration of the computational approach, "
        "loop structure, or NumPy usage, based on the given strategy. "
        "Do not just rename variables or add/remove purely cosmetic comments."
    )

    user_prompt = (
        f"The current Python code for matrix multiplication is:\n\n"
        f"```python\n{src.strip()}\n```\n\n"
        f"Your specific task is to apply the following mutation strategy: **{strategy}**\n\n"
        f"Please provide the complete, mutated Python code for the `matmul` function, "
        f"including necessary imports. Remember, only the raw Python code, nothing else."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            top_p=0.8,
            temperature=0.7,
        )
        mutated_code = response.choices[0].message.content.strip()

        if mutated_code.startswith("```python"):
            mutated_code = mutated_code[len("```python") :].strip()
        if mutated_code.endswith("```"):
            mutated_code = mutated_code[: -len("```")].strip()

        if "def matmul" not in mutated_code or "import numpy as np" not in mutated_code:
            print(
                f"\nWarning: LLM mutation might be malformed or incomplete for strategy: {strategy}."
            )
            print(f"LLM Output:\n{mutated_code}\nReturning original source.\n")
            return src

        return mutated_code
    except Exception as e:
        print(f"Error during LLM mutation with strategy '{strategy}': {e}")

        return src


def evolve(pop_size: int, generations: int, immigrants: int = 2):
    population: List[Candidate] = [Candidate(NAIVE_IMPL)]

    while len(population) < pop_size:
        population.append(Candidate(llm_mutate(NAIVE_IMPL)))

    for c in list(population):
        if c.metrics is None:
            c.metrics = evaluate(c)

    for gen in trange(generations, desc="evolving"):
        for c in population:
            if c.metrics:
                c.metrics.age += 1

        front = pareto_front(population)

        selected = [c for c, _ in sorted(crowding_distance(front), key=lambda x: -x[1])]
        selected = selected[: pop_size // 2]

        children = [
            Candidate(llm_mutate(p.src)) for p in selected if p.metrics is not None
        ]

        if population:
            children += [
                Candidate(llm_mutate(random.choice(population).src))
                for _ in range(immigrants)
            ]
        else:
            children.append(Candidate(llm_mutate(NAIVE_IMPL)))

        for c in list(children):
            if c.metrics is None:
                c.metrics = evaluate(c)

        population = selected + children

        population = [
            c
            for c in population
            if c.metrics is not None
            and c.metrics.time_ns != float("inf")
            and c.metrics.rel_error != float("inf")
        ]
        population.sort(key=lambda c: (c.metrics.time_ns, c.metrics.rel_error))

        population = population[:pop_size]

        if population:
            best = population[0]
            if gen % 5 == 0:
                print(
                    f"Gen {gen}: best time {best.metrics.time_ns} ns, err {best.metrics.rel_error}"
                )
        else:
            print(f"Gen {gen}: No valid candidates in population.")

            population.append(Candidate(NAIVE_IMPL))
            population[0].metrics = evaluate(population[0])

    return pareto_front(population)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=20)
    parser.add_argument("--pop", type=int, default=50)
    args = parser.parse_args()

    final_pareto_front = evolve(pop_size=args.pop, generations=args.gen)
    print("\n--- Final Pareto Front Solutions ---")

    if not final_pareto_front:
        print("No valid solutions found in the Pareto front.")
        return

    final_pareto_front.sort(key=lambda c: (c.metrics.time_ns, c.metrics.rel_error))

    print("\nMetrics (time_ns, rel_error, key):")
    for c in final_pareto_front:
        m = c.metrics
        print(f"{m.time_ns}\t{m.rel_error}\t{c.key()}")

    print("\n--- Source Code for Pareto Front Solutions ---")
    for i, c in enumerate(final_pareto_front):
        print(
            f"\nSolution {i + 1} (Key: {c.key()} | Time: {c.metrics.time_ns} ns | Error: {c.metrics.rel_error}):"
        )
        print("```python")
        print(c.src)
        print("```")
        print("-" * 50)

    fastest_candidate = None
    for c in final_pareto_front:
        if c.metrics.rel_error < 1e-6:
            if (
                fastest_candidate is None
                or c.metrics.time_ns < fastest_candidate.metrics.time_ns
            ):
                fastest_candidate = c

    if fastest_candidate:
        print("\n--- Best Candidate (Fastest with Acceptable Error) ---")
        print(f"Key: {fastest_candidate.key()}")
        print(f"Time: {fastest_candidate.metrics.time_ns} ns")
        print(f"Error: {fastest_candidate.metrics.rel_error}")
        print("\nSource Code:")
        print("```python")
        print(fastest_candidate.src)
        print("```")
    else:
        print("\nNo candidate found with acceptable error in the Pareto front.")


if __name__ == "__main__":
    main()
