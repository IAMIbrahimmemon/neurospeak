"""Exploratory, no bars: where loops start and whether a run leaves one. Uses saved traces only, and no GPU.
A window of 200 words is looped when at least 90% of its 8-grams already occurred anywhere earlier in the run.
A run has left a loop when a later window falls below 50%."""
import json, sys
SIZE, STEP, HIGH, LOW = 200, 50, 0.9, 0.5


def windows(w):
    first = {}
    for i in range(len(w) - 7):
        first.setdefault(tuple(w[i:i + 8]), i)
    is_old = [first[tuple(w[i:i + 8])] < i for i in range(len(w) - 7)]
    return [sum(is_old[i:i + SIZE]) / max(len(is_old[i:i + SIZE]), 1) for i in range(0, max(len(is_old) - SIZE, 1), STEP)]


def study(f):
    runs = []
    for l in open(f):
        r = json.loads(l)
        w = r["completion"].split()
        s = windows(w)
        on = next((i for i, x in enumerate(s) if x >= HIGH), None)
        runs.append(dict(entered=on is not None, onset=None if on is None else on * STEP, words=len(w), left=on is not None and any(x < LOW for x in s[on:]),
                         ends_looped=s[-1] >= HIGH, ended=not r["cut"]))
    on = sorted(x["onset"] for x in runs if x["entered"])
    return dict(n=len(runs), entered=sum(x["entered"] for x in runs), left=sum(x["left"] for x in runs), ended_inside_loop=sum(x["ends_looped"] for x in runs),
                finished=sum(x["ended"] for x in runs), finished_after_entering=sum(x["ended"] and x["entered"] for x in runs),
                onset_quartiles=[on[len(on) // 4], on[len(on) // 2], on[3 * len(on) // 4]] if on else None, median_words=sorted(x["words"] for x in runs)[len(runs) // 2])


if __name__ == "__main__":
    out = {f.split("/")[-1].replace("_traces.jsonl", ""): study(f) for f in sys.argv[1:]}
    json.dump(out, open("runs/research/loop_onset.json", "w"), indent=1)
    for k, v in out.items():
        print(k, v)
