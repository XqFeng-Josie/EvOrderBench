"""Run a model on EvOrderBench through an OpenAI-compatible chat-completions endpoint.

Works with hosted APIs and with local servers that expose the same interface (for example vLLM):

    export OPENAI_API_KEY=...                # omit for a local server without authentication
    python run_openai_api.py --model my-model --base-url http://localhost:8000/v1 \
        --data ../datasets --out outputs/my-model

Each query sends the frames of one input, in chronological order, followed by the prompt. Raw outputs are
appended to <out>/outputs.jsonl, so an interrupted run resumes where it stopped. When all queries are done,
<out>/predictions.csv holds the extracted labels and can be scored with

    python evaluate.py --data ../datasets --predictions outputs/my-model/predictions.csv

Only the Python standard library is needed.
"""
import argparse
import base64
import csv
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from extract_answer import extract


def image_part(path):
    mime = mimetypes.guess_type(str(path))[0] or 'image/png'
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{data}'}}


def ask(args, data, query):
    content = [image_part(data / f) for f in query['frames']] + [{'type': 'text', 'text': query['prompt']}]
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': content}], args.token_field: args.max_tokens}
    if args.temperature is not None:
        body['temperature'] = args.temperature
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('OPENAI_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
    request = urllib.request.Request(args.base_url.rstrip('/') + '/chat/completions', json.dumps(body).encode(), headers)
    for attempt in range(args.retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                choice = json.load(response)['choices'][0]
            message = choice['message']
            output = message.get('content') or ''
            # Servers that parse reasoning (e.g. vLLM with --reasoning-parser) return it in a separate field;
            # restore the <think> block so that extract() reads only the text after it.
            reasoning = message.get('reasoning_content') or message.get('reasoning')
            if isinstance(reasoning, str) and reasoning and '</think>' not in output:
                output = f'<think>{reasoning}</think>\n{output}'
            return {'query_id': query['query_id'], 'output': output,
                    'truncated': choice.get('finish_reason') == 'length'}
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
            if attempt == args.retries:
                raise RuntimeError(f"{query['query_id']}: {error}") from error
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--base-url', default='https://api.openai.com/v1')
    ap.add_argument('--data', default=str(Path(__file__).resolve().parents[1] / 'datasets'), help='dataset directory')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--name', help='column name in predictions.csv (default: the model name)')
    ap.add_argument('--max-tokens', type=int, default=64, help='use a few thousand for models that reason before answering')
    ap.add_argument('--token-field', default='max_tokens', help="name of the output-limit field, e.g. max_completion_tokens")
    ap.add_argument('--temperature', type=float, default=0.0, help='pass a negative value to leave the server default')
    ap.add_argument('--thinking', action='store_true',
                    help='the model reasons before answering and returns the reasoning, either inside <think>...</think> '
                         'or in a separate reasoning field; omit for APIs that do not return their reasoning')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--retries', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=600)
    ap.add_argument('--limit', type=int, help='only the first N queries (for a quick test)')
    args = ap.parse_args()
    if args.temperature is not None and args.temperature < 0:
        args.temperature = None

    data = Path(args.data)
    with open(data / 'queries.jsonl') as fh:
        queries = [json.loads(line) for line in fh][:args.limit]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    journal = out / 'outputs.jsonl'
    done = {}
    if journal.exists():
        with open(journal) as fh:
            done = {r['query_id']: r for r in map(json.loads, fh)}
    todo = [q for q in queries if q['query_id'] not in done]
    print(f'{len(done)} of {len(queries)} queries already answered; running {len(todo)}')
    with open(journal, 'a') as fh, ThreadPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(lambda q: ask(args, data, q), todo), 1):
            fh.write(json.dumps(row) + '\n'); fh.flush()
            done[row['query_id']] = row
            if i % 50 == 0 or i == len(todo):
                print(f'{i}/{len(todo)}')

    name = args.name or args.model
    invalid = 0
    with open(out / 'predictions.csv', 'w', newline='') as fh:
        writer = csv.writer(fh); writer.writerow(['query_id', name])
        for q in queries:
            row = done[q['query_id']]
            label = extract(row['output'], thinking=args.thinking, truncated=row['truncated'])
            invalid += label is None
            writer.writerow([q['query_id'], label or 'INVALID'])
    print(f"wrote {out / 'predictions.csv'} ({invalid} outputs without a label)")


if __name__ == '__main__':
    main()
