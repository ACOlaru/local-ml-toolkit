import pytest

from local_ml_toolkit.model import LocalTextClassifier


def test_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        LocalTextClassifier.__init__(object.__new__(LocalTextClassifier), batch_size=0)


def test_predict_many_empty_input_does_not_load_model():
    # Exercise the empty-input contract without downloading a checkpoint.
    classifier = object.__new__(LocalTextClassifier)
    assert classifier.predict_many([]) == []


def test_predict_many_rejects_blank_text():
    classifier = object.__new__(LocalTextClassifier)
    with pytest.raises(ValueError, match="non-empty"):
        classifier.predict_many(["valid", "  "])
