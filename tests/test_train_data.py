import pytest

transformers = pytest.importorskip("transformers")

from csasr.train.data import ASRDataset, TokenSpans, token_roles  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.WhisperTokenizer.from_pretrained("openai/whisper-small")
    except OSError:
        pytest.skip("tokenizer not cached")


def test_token_roles_zh_main(tok):
    text = "我明天要交 assignment 啦"
    ids = tok(text, add_special_tokens=False).input_ids
    roles = token_roles(text, ids, TokenSpans(tok))
    pieces = [tok.decode([i]) for i in ids]
    # every token of " assignment" is embedded (2); tokens covering 交 / 啦 are Ms (1)
    emb = "".join(p for p, r in zip(pieces, roles) if r == 2)
    assert emb.strip() == "assignment"
    assert max(roles) == 2 and 1 in roles


def test_token_roles_en_main_leading_space(tok):
    text = " then we go 吃饭 lah"
    ids = tok(text, add_special_tokens=False).input_ids
    roles = token_roles(text, ids, TokenSpans(tok))
    emb_bytes = b"".join(bytes(TokenSpans(tok).byte_decoder[c] for c in t)
                         for t, r in zip(tok.convert_ids_to_tokens(ids), roles) if r == 2)
    assert emb_bytes.decode("utf-8").strip() == "吃饭"


def test_encode_prefix(tok):
    ds = ASRDataset([], tok, None)
    seq, roles = ds.encode({"ref_raw": "我明天要交 assignment 啦", "ref": "我明天要交 assignment 啦", "main": "zh"})
    assert seq[1] == tok.convert_tokens_to_ids("<|zh|>")
    assert len(roles) == len(seq) - 1 and roles[:3] == [3, -1, -1]
