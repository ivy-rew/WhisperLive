import unittest

from whisper_live.backend.parakeet_backend import drop_context, split_sentences


def result(*tokens):
    return list(tokens), [i * 0.5 for i in range(len(tokens))]


class TestSplitSentences(unittest.TestCase):
    def test_splits_on_sentence_end_before_new_word(self):
        segments = split_sentences(*result(" Hallo", " du.", " Wie", " geht's?", " Gut"))
        self.assertEqual([s.text for s in segments], [" Hallo du.", " Wie geht's?", " Gut"])
        self.assertEqual([(s.start, s.end) for s in segments], [(0.0, 1.0), (1.0, 2.0), (2.0, 2.0)])

    def test_keeps_dots_inside_words(self):
        segments = split_sentences(*result(" unter", " www", ".", "wiki", ".", "ch"))
        self.assertEqual([s.text for s in segments], [" unter www.wiki.ch"])

    def test_trailing_punctuation_does_not_close_last_sentence(self):
        segments = split_sentences(*result(" Der", " Satz", " im.", " .", "."))
        self.assertEqual([s.text for s in segments], [" Der Satz im."])

    def test_ignores_leading_punctuation(self):
        segments = split_sentences(*result(" .", " Hallo."))
        self.assertEqual([s.text for s in segments], [" Hallo."])


class TestDropContext(unittest.TestCase):
    def test_keeps_tokens_after_context_from_word_start(self):
        tokens, timestamps = drop_context([" Ja", ".", " Und", "ers", " geht"], [0.0, 0.4, 0.96, 1.2, 1.5], 1.0)
        self.assertEqual(tokens, [" Und", "ers", " geht"])
        self.assertEqual([round(t, 2) for t in timestamps], [0.0, 0.2, 0.5])

    def test_skips_word_that_started_in_context(self):
        tokens, _ = drop_context([" Schwei", "z", " liegt"], [0.8, 1.1, 1.3], 1.0)
        self.assertEqual(tokens, [" liegt"])

    def test_without_context_keeps_everything(self):
        tokens, _ = drop_context([" Hallo"], [0.0], 0.0)
        self.assertEqual(tokens, [" Hallo"])


if __name__ == "__main__":
    unittest.main()
