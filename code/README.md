# EvOrderBench: evaluation code

Code for *EvOrderBench: When Do Sampled Frames Establish Event Order?*

| File | Purpose |
|---|---|
| `evaluate.py` | Scores predictions with the metrics of the paper's main results table. |
| `run_openai_api.py` | Runs a model through an OpenAI-compatible endpoint (hosted API or a local server such as vLLM). |
| `extract_answer.py` | Maps a model output to `A`, `B`, or `UNCLEAR`. |
| `predictions/predictions.csv` | Answers of the 19 configurations evaluated in the paper to all 4,632 queries. |
| `expected_results.txt` | Output of `python evaluate.py --ci` on those predictions. |

The scripts need Python 3.8 or later and no packages; `evaluate.py --ci` additionally needs numpy.

## Reproduce the main results table

```
python evaluate.py          # reads ../datasets and predictions/predictions.csv
python evaluate.py --ci     # adds 95% cluster-bootstrap intervals
```

## Evaluate a model

```
export OPENAI_API_KEY=...   # omit for a local server without authentication
python run_openai_api.py --model MODEL --base-url http://localhost:8000/v1 --out outputs/MODEL
python evaluate.py --predictions outputs/MODEL/predictions.csv
```

For a model that reasons before answering, add `--thinking` and raise `--max-tokens` (the paper uses 4,096).
The reasoning may come inside `<think>...</think>` or in a separate `reasoning_content` field (vLLM with
`--reasoning-parser`); only the text after it is scored. Omit `--thinking` for hosted APIs that do not return
their reasoning.
`--limit 16` runs the first 16 queries as a quick test. A run that is interrupted resumes from
`outputs/MODEL/outputs.jsonl`.

To use another inference stack, read `../datasets/queries.jsonl`: give the model the images in `frames`, in
the listed order, followed by the text in `prompt`; pass the output through `extract_answer.extract`; and
write a CSV with the columns `query_id` and your model's name.

## Metrics

| Metric | Definition |
|---|---|
| Overall | Accuracy on all 4,632 queries. |
| Ord. | Accuracy on queries whose reference answer is an order. |
| Insuf. | Accuracy on queries whose reference answer is `UNCLEAR`. |
| U0, U1, U2 | Insuf. on inputs in which zero, one, or both target actions are evidenced. |
| Pair_HL | `H` ordered correctly and `UNCLEAR` on `L`, under the same answer-letter assignment. |
| Pair_KP | The same for `K` and `P`. |
| Family | All eight queries of a family correct. |

Confidence intervals resample the 558 recording and participant groups (`group` in `families.json`) 5,000 times.
