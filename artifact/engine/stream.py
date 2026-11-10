from bisect import insort

def streamed(scores):
    history=[]
    for j,s in enumerate(scores):
        if j>=10:
            middle=j//2
            threshold=history[middle] if j%2 else .5*history[middle-1]+.5*history[middle]
            yield int(s>=threshold)
        insort(history,s)
