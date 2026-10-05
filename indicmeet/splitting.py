"""Energy-silence boundaries. No model downloads or GPU dependencies."""
import math

def split_samples(samples, sample_rate, maximum_seconds, silence_seconds=0.12):
    limit = max(1, int(sample_rate * maximum_seconds))
    length = len(samples)
    if length <= limit:
        return [(0, length)]
    window = max(1, int(sample_rate * .02))
    energies = [(i, math.sqrt(sum(float(x)**2 for x in samples[i:i+window]) / max(1,len(samples[i:i+window])))) for i in range(0,length,window)]
    peak = max(e for _,e in energies)
    threshold = min(.01, peak * .04)
    boundaries, first = [], None
    for i, energy in energies + [(length, peak + 1)]:
        if energy <= threshold and first is None:
            first = i
        elif energy > threshold and first is not None:
            if i-first >= sample_rate*silence_seconds:
                boundaries.append((first+i)//2)
            first = None
    if not boundaries:
        count = math.ceil(length/limit)
        return [(round(i*length/count),round((i+1)*length/count)) for i in range(count)]
    spans, start = [], 0
    while length-start > limit:
        candidates = [b for b in boundaries if start+window < b <= start+limit]
        end = max(candidates) if candidates else start+limit
        spans.append((start,end)); start=end
    spans.append((start,length))
    return spans
