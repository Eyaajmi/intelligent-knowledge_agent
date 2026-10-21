from app.chunking import split_text


def _sentences(n):
    return " ".join(f"Phrase numéro {i} sur la discipline au travail." for i in range(n))


def test_chunks_never_exceed_size():
    chunks = split_text(_sentences(80), chunk_size=300, overlap=80, min_chars=10)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)


def test_chunks_are_made_of_whole_sentences():
    chunks = split_text(_sentences(40), chunk_size=300, overlap=80, min_chars=10)
    assert all(c.endswith(".") and c.startswith("Phrase") for c in chunks)


def test_overlap_repeats_last_sentence_of_previous_chunk():
    chunks = split_text(_sentences(40), chunk_size=300, overlap=80, min_chars=10)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.split(". ")[-1].rstrip(".")
        assert last_sentence in nxt


def test_no_overlap_when_disabled():
    chunks = split_text(_sentences(40), chunk_size=300, overlap=0, min_chars=10)
    joined = " ".join(chunks)
    assert joined.count("Phrase numéro 7 ") == 1


def test_hyphenated_words_are_rejoined():
    text = "Les sanctions dis-\nciplinaires sont définies par le règlement intérieur de la société."
    assert "disciplinaires" in split_text(text, min_chars=10)[0]


def test_tiny_fragments_are_dropped():
    assert split_text("Page 3", min_chars=40) == []
    assert split_text("", min_chars=40) == []


def test_sentence_longer_than_chunk_is_split_on_words():
    text = " ".join(["mot"] * 1000)
    chunks = split_text(text, chunk_size=300, overlap=0, min_chars=10)
    assert len(chunks) >= 10 and all(len(c) <= 300 for c in chunks)


def test_line_wraps_are_not_treated_as_paragraphs():
    text = "Ceci est une phrase\nrépartie sur plusieurs\nlignes dans le PDF et elle doit rester entière."
    chunks = split_text(text, min_chars=10)
    assert len(chunks) == 1 and "\n" not in chunks[0]
