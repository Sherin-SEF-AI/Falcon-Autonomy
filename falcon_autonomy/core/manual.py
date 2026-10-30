"""The built-in manual: the docs folder's Markdown files (bundled with the app), split into sections and searchable
offline (standard library only).

load_docs() is what Help shows; sections() cuts every file at its headings (#, ##, ###), keeping each section's
heading trail and an anchor GitHub-style (lower case, spaces to hyphens, punctuation dropped); search() ranks sections
by Okapi BM25 (k1 = 1.5, b = 0.75) over word tokens of the heading (counted three times) and the body. No stemming, no
synonyms: a query in the manual's own words finds the most.
"""
from __future__ import annotations

import math
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIRS = (os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'docs'),      # the source tree and the bundle
             os.path.join(os.path.dirname(_HERE), 'docs'))
_WORD = re.compile(r'[a-z0-9][a-z0-9_\-]*')
K1, B = 1.5, 0.75
_CACHE: dict = {}


def docs_dir() -> str:
    return next((d for d in DOCS_DIRS if os.path.isdir(d)), '')


def load_docs() -> list:
    """[(title, filename, markdown)] in the manual's order (index.md first, then by name)."""
    d = docs_dir()
    out = []
    if not d:
        return out
    for fn in sorted(os.listdir(d), key=lambda f: (f != 'index.md', f)):
        if not fn.endswith('.md'):
            continue
        with open(os.path.join(d, fn), encoding='utf-8') as fh:
            text = fh.read()
        m = re.search(r'^#\s+(.+)$', text, re.M)
        out.append((m.group(1).strip() if m else fn, fn, text))
    return out


def anchor(heading: str) -> str:
    a = re.sub(r'[^\w\- ]', '', heading.strip().lower())
    return a.replace(' ', '-')


def tokens(text: str) -> list:
    return _WORD.findall(text.lower())


def sections() -> list:
    """Every section of the manual: {'file', 'title' (the file's), 'heading', 'trail' (the headings above it and its
    own), 'anchor', 'text'}. Fenced code stays in its section; a heading inside a code fence is not a heading."""
    d = docs_dir()
    key = (d, tuple(sorted((f, os.path.getmtime(os.path.join(d, f))) for f in os.listdir(d) if f.endswith('.md')))
           if d else ())
    if _CACHE.get('key') == key:
        return _CACHE['sections']
    out = []
    for title, fn, text in load_docs():
        trail: list = []
        cur = {'file': fn, 'title': title, 'heading': title, 'trail': [title], 'anchor': '', 'lines': []}
        fence = False
        for line in text.splitlines():
            if line.lstrip().startswith('```'):
                fence = not fence
            m = None if fence else re.match(r'^(#{1,3})\s+(.+?)\s*#*\s*$', line)
            if m:
                if cur['lines'] or cur['anchor']:
                    out.append(cur)
                level, head = len(m.group(1)), m.group(2).strip()
                trail = trail[:level - 1] + [head]
                cur = {'file': fn, 'title': title, 'heading': head, 'trail': list(trail), 'anchor': anchor(head),
                       'lines': []}
                continue
            cur['lines'].append(line)
        out.append(cur)
    for s in out:
        s['text'] = '\n'.join(s.pop('lines')).strip()
    out = [s for s in out if s['text'] or s['anchor']]
    _CACHE.update(key=key, sections=out)
    return out


def search(query: str, limit: int = 5) -> list:
    """The best sections for query, best first: (score, section). Empty when no word of the query is in the manual."""
    secs = sections()
    q = [t for t in dict.fromkeys(tokens(query))]
    if not q or not secs:
        return []
    docs = [tokens(' '.join([s['heading']] * 3 + [s['text']])) for s in secs]
    n = len(docs)
    avg = sum(len(d) for d in docs) / n
    df = {t: sum(1 for d in docs if t in set(d)) for t in q}
    scored = []
    for s, d in zip(secs, docs):
        tf: dict = {}
        for t in d:
            if t in df:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in q:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * len(d) / avg))
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return scored[:limit]
