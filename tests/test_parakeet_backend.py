import unittest
from types import SimpleNamespace

from whisper_live.backend.parakeet_backend import split_sentences


def result(*tokens):
    return SimpleNamespace(tokens=list(tokens), timestamps=[i * 0.5 for i in range(len(tokens))])


class TestSplitSentences(unittest.TestCase):
    def test_splits_on_sentence_end_before_new_word(self):
        segments = split_sentences(result(" Hallo", " du.", " Wie", " geht's?", " Gut"))
        self.assertEqual([s.text for s in segments], ["Hallo du.", "Wie geht's?", "Gut"])
        self.assertEqual([(s.start, s.end) for s in segments], [(0.0, 1.0), (1.0, 2.0), (2.0, 2.0)])

    def test_keeps_dots_inside_words(self):
        segments = split_sentences(result(" unter", " www", ".", "wiki", ".", "ch"))
        self.assertEqual([s.text for s in segments], ["unter www.wiki.ch"])

    def test_trailing_punctuation_does_not_close_last_sentence(self):
        segments = split_sentences(result(" Der", " Satz", " im.", " .", "."))
        self.assertEqual([s.text for s in segments], ["Der Satz im."])

    def test_ignores_leading_punctuation(self):
        segments = split_sentences(result(" .", " Hallo."))
        self.assertEqual([s.text for s in segments], ["Hallo."])


if __name__ == "__main__":
    unittest.main()
