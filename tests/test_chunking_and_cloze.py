from text_chunking import apply_cloze, chunk_by_sentence_count


def test_chunking_sentence_boundaries_native():
    text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."
    chunks = chunk_by_sentence_count(text, native_mode=True)
    assert len(chunks) == 1
    assert chunks[0].endswith("Ten.")


def test_cloze_new_vs_old_words():
    masked, new_words = apply_cloze("apple banana orange", known_words={"apple"}, max_new_words=2)
    assert "apple" in masked
    assert masked.count("____") == 2
    assert new_words == ["banana", "orange"]
