# RAG Failure Modes — Taxonomy and Empirical Validation

Code, configuration, and error-analysis materials for the paper
**"Why RAG Fails: A Taxonomy of Failure Modes in Retrieval-Augmented Generation"**
by Dawit Zeleke Admassu.

The paper proposes a taxonomy of five failure modes in retrieval-augmented
generation (RAG): Retrieval Blindness, Retrieval Noise, Evidence Ignorance,
Evidence Over-Reliance, and Latency-Induced Failure. It adds a preliminary
empirical validation of the first three modes using two generator models
(Claude Sonnet 4.6 and Llama 3.3 70B Instruct), one shared retrieval
pipeline, and controlled clean vs. stress conditions drawn from the public
[RGB](https://github.com/chen700564/RGB) and
[Lost in the Middle](https://github.com/nelson-liu/lost-in-the-middle) benchmarks.

## Repository contents

| Path | What it contains |
|---|---|
| `paper/` | Paper PDF, LaTeX source, and bibliography |
| `rag_eval/` | Evaluation pipeline: dataset loaders, embedding retrieval, condition builders, model adapters, runner, metrics, and tests |
| `rag_eval/config/` | Frozen experiment settings and the exact prompt templates |
| `rag_eval/data/dataset_manifest.json` | Dataset provenance: source repos, commit hashes, file hashes, and the 90 sampled item IDs |
| `rag_eval/error_analysis/` | Labeling rubric, human audit protocol, and the AI-assisted and human-audited error-analysis samples |
| `rag_eval/logs/` | Dry-run verification output (mock generator, no API calls) and the no-context baseline results |

See [`rag_eval/README.md`](rag_eval/README.md) for full setup, design decisions,
and run instructions.

## Experimental design (summary)

- **Failure modes tested:** Retrieval Blindness, Retrieval Noise, Evidence Ignorance (30 items each, 90 total, seed 42)
- **Retrieval:** `sentence-transformers/all-MiniLM-L6-v2` embeddings with an exact cosine-similarity flat index, top-5
- **Generators:** Claude Sonnet 4.6 and Llama 3.3 70B Instruct via OpenRouter, temperature 0, one identical prompt template
- **Metrics:** token F1, exact match, retrieval recall@5, precision@5, nDCG@5, gold rank, context-ablation sensitivity, tokens, latency, and cost

## Quick start (no API keys, no cost)

```bash
cd rag_eval
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover tests
python -m runner.run_generation --dry-run
python -m eval.run_eval --dry-run
```

The dry run uses the real 90 sampled items and real embedding retrieval, but
a mock generator, so it makes no paid API calls. The raw benchmark files are
not redistributed here; download them from the source repositories at the
commits listed in `rag_eval/data/dataset_manifest.json`.

## Running against real models

Copy `rag_eval/.env.example` to `rag_eval/.env`, add your own API keys, and run
`python -m runner.run_generation` followed by `python -m eval.run_eval`.
Real runs make paid API calls. Never commit your `.env` file.

## Citation

```bibtex
@misc{admassu2026ragfailures,
  author = {Admassu, Dawit Zeleke},
  title  = {Why RAG Fails: A Taxonomy of Failure Modes in Retrieval-Augmented Generation},
  year   = {2026}
}
```

## License

Code is released under the [MIT License](LICENSE). The RGB and Lost in the
Middle benchmarks remain under their original licenses.
