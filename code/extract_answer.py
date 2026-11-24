"""Map a model output to an answer label: A, B, UNCLEAR, or None when no label can be read.

The rule does not look at the reference answer and is the same for every model:

1. For a model that reasons before answering, only the text after the last </think> counts. An output that
   reached the token limit, or a reasoning output without a closing </think>, has no answer.
2. That text, or else its last non-empty line, must consist of exactly one label. The label may be wrapped as
   \\boxed{X}, **X**, "X", (X) or X., follow an answer phrase ("Answer: X", "Therefore, the answer is X."),
   or be one of the phrases "A before B", "A precedes B", "B before A", "B precedes A".

Nothing else is accepted: no label picked out of an explanation, and no output that names two labels.
"""
import re

LABELS = ('A', 'B', 'UNCLEAR')
ALIASES = {'A before B': 'A', 'A precedes B': 'A', 'B before A': 'B', 'B precedes A': 'B'}
_ALT = '|'.join(sorted(LABELS, key=len, reverse=True))
_PREFIX = r'(?:(?i:therefore|thus|so|hence),?\s+)?(?:(?i:the)\s+)?(?:(?i:final)\s+)?(?i:answer)(?:\s+(?i:is))?\s*[:：]?\s*'
_PATTERNS = [rf'\\boxed\{{({_ALT})\}}\.?', rf'\*\*({_ALT})\*\*\.?', rf'["\'`]({_ALT})["\'`]\.?', rf'\(({_ALT})\)\.?', rf'({_ALT})\.',
             _PREFIX + rf'(?:\\boxed\{{({_ALT})\}}|\*\*({_ALT})\*\*|({_ALT}))\.?']


def _label(text):
    if text in LABELS:
        return text
    if text in ALIASES:
        return ALIASES[text]
    for pattern in _PATTERNS:
        m = re.fullmatch(pattern, text)
        if m:
            return next(g for g in m.groups() if g)
    return None


def extract(output, thinking=False, truncated=False):
    """`thinking`: the model was run in a reasoning mode; `truncated`: generation stopped at the token limit."""
    if truncated:
        return None
    text = (output or '').strip()
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1].strip()
    elif thinking or '<think>' in text:
        return None
    label = _label(text)
    if label:
        return label
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _label(lines[-1]) if len(lines) > 1 else None


if __name__ == '__main__':
    cases = [('A', 'A'), ('UNCLEAR.', 'UNCLEAR'), ('**B**', 'B'), ('\\boxed{A}', 'A'), ('Answer: UNCLEAR', 'UNCLEAR'),
             ('The frames show the cup first.\nTherefore, the answer is A.', 'A'), ('A before B', 'A'), ('A or B', None),
             ('AA', None), ('<think>it is A</think>\nB', 'B'), ('<think>it is A', None), ('I think A happened first', None)]
    for text, want in cases:
        assert extract(text) == want, (text, extract(text), want)
    print('ok')
