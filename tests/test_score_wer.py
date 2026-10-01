from eval.score_wer import normalize, score_words


def test_tiny_word_error_score_matches_hand_calculation():
    reference = normalize("A, b c.")
    hypothesis = normalize("a x c d")

    result = score_words(reference, hypothesis)

    # b -> x is one substitution and d is one insertion: 2 errors / 3 ref words.
    assert result.reference_words == 3
    assert result.hypothesis_words == 4
    assert result.substitutions == 1
    assert result.deletions == 0
    assert result.insertions == 1
    assert result.wer == 2 / 3
