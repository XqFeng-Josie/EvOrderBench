# EvOrderBench

Code for *EvOrderBench: When Do Sampled Frames Establish Event Order?*

EvOrderBench tests whether a vision-language model states the order of two actions only when the frames it
receives establish that order. Each of its 579 event families fixes one episode and one question and varies only
the frames: some inputs establish the order, others miss an action (`U0`, `U1`) or show both actions but not
their order (`U2`), and the correct answer for those is `UNCLEAR`.

[Project page](https://xqfeng-josie.github.io/EvOrderBench/) ·
[Dataset on Hugging Face](https://huggingface.co/datasets/xiaoqinfeng/EvOrderBench)

## Setup

The scripts need Python 3.8 or later and no packages (`evaluate.py --ci` additionally needs numpy).
Download the dataset (about 4 GB, mostly frames) into `datasets/`:

```
pip install -U huggingface_hub
hf download xiaoqinfeng/EvOrderBench --repo-type dataset --local-dir datasets
```

## Reproduce the main results

```
python code/evaluate.py          # main results table of the paper, from code/predictions/predictions.csv
python code/evaluate.py --ci     # with 95% cluster-bootstrap intervals; matches code/expected_results.txt
```

## Evaluate a model

```
python code/run_openai_api.py --model MODEL --base-url http://localhost:8000/v1 --out outputs/MODEL
python code/evaluate.py --predictions outputs/MODEL/predictions.csv
```

`run_openai_api.py` works with any OpenAI-compatible endpoint, hosted or local (for example vLLM). See
[`code/README.md`](code/README.md) for the options, the answer-extraction rule, and the metrics, and
[`datasets/README.md`](https://huggingface.co/datasets/xiaoqinfeng/EvOrderBench) for the data format.

## Layout

| Path | Contents |
|---|---|
| `code/evaluate.py` | Scores predictions with the metrics of the main results table. |
| `code/run_openai_api.py` | Runs a model on all 4,632 queries through an OpenAI-compatible endpoint. |
| `code/extract_answer.py` | Maps a model output to `A`, `B`, or `UNCLEAR`. |
| `code/predictions/predictions.csv` | Answers of the 19 configurations evaluated in the paper. |
| `datasets/` | The benchmark, downloaded from Hugging Face (not part of this repository). |

## License

The code is released under the MIT License. The EvOrderBench annotations (questions, frame selections, and
labels) are released under CC BY 4.0; the frames keep the licenses of their source datasets (Perception Test and
HO-Cap: CC BY 4.0; CaptainCook4D: Apache 2.0).

## Citation

```
@misc{feng2026evorderbench,
  title  = {EvOrderBench: When Do Sampled Frames Establish Event Order?},
  author = {Xiaoqin Feng and Yuqi Li and Xinyuan Li and Neil Rodrigues and Xianglong Wang and Kuiye Ding and Yingli Tian},
  year   = {2026}
}
```
