from bisect import insort

def past_decisions(scores, start):
    history = []
    for j,s in enumerate(scores):
        if j >= start:
            midpoint = j//2
            cut = history[midpoint] if j%2 else .5*history[midpoint-1] + .5*history[midpoint]
            yield int(s>=cut)
        insort(history,s)
