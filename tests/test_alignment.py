from indicmeet.splitting import split_samples
def test_silence_preferred_and_limit_held():
    samples = [1.]*700+[0.]*200+[1.]*1400
    spans = split_samples(samples,100,10)
    assert spans[0][1] == 800
    assert all(end-start<=1000 for start,end in spans)
    assert spans[0][0]==0 and spans[-1][1]==len(samples)
def test_no_silence_equal_fallback():
    assert split_samples([1.]*2300,100,10)==[(0,767),(767,1533),(1533,2300)]
