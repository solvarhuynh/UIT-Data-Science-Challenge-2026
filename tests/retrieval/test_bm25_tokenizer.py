from udsc2026.retrieval.sparse.tokenizer import tokenize_vi


def test_tokenizer_preserves_vietnamese_accents():
    tokens = tokenize_vi("Người lao động có quyền nghỉ phép.")
    assert any("động" in token for token in tokens)
    assert any("quyền" in token for token in tokens)


def test_legal_phrases_are_not_split_into_number_tokens():
    tokens = tokenize_vi("Khoản 2 Điều 5 quy định tại Điểm a Điều 10.")
    assert "Khoản 2" in tokens
    assert "Điều 5" in tokens
    assert "Điểm a" in tokens
    assert "5" not in tokens
    assert "10" not in tokens


def test_legal_document_number_is_preserved_as_one_token():
    tokens = tokenize_vi("Theo van ban 45/2019/QH14.")
    assert "45/2019/QH14" in tokens
    assert "45" not in tokens
    assert "2019" not in tokens
    assert "qh14" not in tokens


def test_legal_abbreviation_remains_searchable():
    tokens = tokenize_vi("BLLD va Nghi dinh.")
    assert any("bll" in token.casefold() for token in tokens)
