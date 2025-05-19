# Evolve

An open-source LLM-driven genetic programming framework that (kinda) recreates DeepMind's AlphaEvolve. It uses GPT-4.1 to evolve better algorithms, starting from naive baselines—currently focused on matrix multiplication.

No `np.dot`, no `@`. Just raw Python loops, strategic mutations, and heartbreak.

## Features

- 🧬 **Genetic Programming Core**  
  Tournament selection + mutation-only evolution + Pareto optimization (speed + accuracy).

- 🧠 **LLM-based Mutation Engine**  
  Uses GPT-4.1 to mutate Python code using meaningful, pre-curated strategies.

- 🧪 **Multi-Objective Evaluation**  
  Each candidate is scored by:

  - Runtime performance (in nanoseconds)
  - Accuracy (relative error vs NumPy reference)

- 🔥 **Mutation Strategies**  
  Over 40 real strategies like loop unrolling, loop tiling, Strassen’s algo, blocking, etc. No cosmetic garbage.

- ❌ **No Cheating Allowed**  
  Enforces no use of `np.dot`, `np.matmul`, or `@`.

- 🧱 **Modular + Hackable**  
  Clean and extensible. Add your own strategies, objectives, domains.

---

## Installation

```bash
git clone https://github.com/think-a-tron/evolve.git
cd evolve
uv sync
```

## Requirements:

- Python 3.10+
- `openai`, `numpy`

Set your OpenAI API key in your environment:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Usage

Basic run:

```bash
uv run main.py --gen 20 --pop 50
```

- `--gen`: number of generations to evolve
- `--pop`: initial population size

At the end, you'll get:

- Final Pareto front (best tradeoffs of speed and accuracy)
- Source code of each front-runner
- The best candidate that’s both **fast** and **accurate**

---

## How It Works

1. **Seed**: Starts with a naive matrix multiplication function.
2. **Mutation**: LLM mutates it using a randomly chosen strategy from a curated list.
3. **Evaluation**:

   - Runs the code on matrices of size 128x128 and 256x256.
   - Measures runtime and computes error vs NumPy.

4. **Selection**:

   - Builds Pareto front (minimizing both runtime and error).
   - Uses NSGA-style crowding distance to preserve diversity.

5. **Evolution**:

   - Surviving candidates are mutated again.
   - Occasional random immigrants added.

6. **Repeat**.

---

## Example Output

```
Gen 15: best time 2340000 ns, err 1.2e-6

--- Final Pareto Front Solutions ---

Solution 1 (Key: 2d6885552d05 | Time: 2340000 ns | Error: 1.2e-6):

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...

```

---

## Notes

- Sometimes GPT mutates nothing useful. It’s an LLM, not a genius.
- Reward hacking is real. Without constraints, it will try to “optimize” by doing nothing.
- You can extend this to other domains—FFT, convolution, sorting—if you're brave enough.

---

## License

MIT. Use, modify, break, evolve.

---

## Contributing

Fork it, mess with it, roast it. PRs welcome if they don't contain `a @ b`.

---

## Credits

Inspired by DeepMind’s [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
Built by someone who ran out of API credits multiple times.
