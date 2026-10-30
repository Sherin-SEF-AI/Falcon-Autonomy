"""The grounding check (falcon_autonomy.assistant.grounding): which numbers of an answer are checked, which sources
count, and the rounding, percentage and suffix tolerance."""
import json

from falcon_autonomy.assistant import grounding as G


def tool(result, call_args=None):
    msgs = []
    if call_args is not None:
        msgs.append({'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'c', 'type': 'function', 'function': {'name': 'x', 'arguments': json.dumps(call_args)}}]})
    msgs.append({'role': 'tool', 'tool_call_id': 'c', 'content': json.dumps(result)})
    return msgs


def flagged(answer, msgs):
    return [n['text'] for n in G.check(answer, msgs)]


def test_what_is_checked():
    got = [n['text'] for n in G.numbers(
        'AP 0.73 on `run_7` (73.4 %), 1.2k frames, gpt-oss-120b, CARLA 0.9.16, 3D boxes, ResNet50, on 2026-10-30 at '
        '14:05, job #12, ~/runs/r12, run_2026_a, 5-10 m, -0.5 m, 12,345 steps\n1. first\n2. second, 2 options\n'
        '```\nseed = 99\n```')]
    assert got == ['0.73', '73.4 %', '1.2k', '5', '10', '-0.5', '12,345']
    # thousands grouped with a narrow, thin or no-break space (gpt-oss writes 3 013) are one number, not 3 and 013
    assert flagged('3 013 objects, 1 693 signs, 1 272 tiny', tool({'n': [3013, 1693, 1272]})) == []


def test_rounding_percentages_suffixes_and_counts():
    msgs = tool({'ap': 0.73421, 'frames': 1234, 'recall': 0.5, 'runs': ['a', 'b', 'c', 'd', 'e'],
                 'note': 'mean TTC 2.71 s over 40 events', 'big': 3_400_000})
    assert flagged('AP 0.73, or 0.734, or 73.4 %, or 73 %; 1.2k frames; recall 50 %; 5 runs; TTC 2.7 s over 40 '
                   'events; 3.4M points', msgs) == []
    assert flagged('AP 0.74 and 0.7343 and 1.3k frames, 6 runs, 41 events', msgs) == \
        ['0.74', '0.7343', '1.3k', '6', '41']


def test_sources_are_tool_results_arguments_and_the_user():
    msgs = [{'role': 'user', 'content': 'record 30 s with seed 7'}] + tool({'job_id': 12}, {'seconds': 30, 'seed': 7})
    msgs.append({'role': 'assistant', 'content': 'Your mAP is 0.66.'})       # the model's own words: not a source
    assert flagged('Queued job 12: a 30 s drive with seed 7.', msgs) == []
    assert flagged('Last time the mAP was 0.66.', msgs) == ['0.66']
    assert G.check('No numbers here, just words.', []) == []
    assert flagged('Two options: retrain or relabel; 1 of them is cheaper.', []) == []     # 0, 1, 2 stand alone
    assert flagged('It took 3 tries.', []) == ['3']
