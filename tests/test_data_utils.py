import unittest
from src.data_utils import tokenize, build_vocab, encode, NextTokenDataset

class DataUtilsTests(unittest.TestCase):
    def test_tokenize_punctuation(self):
        self.assertEqual(tokenize("Hello, world!"), ["hello", ",", "world", "!"])
    def test_unknown_token(self):
        vocab = build_vocab(["hello", "hello", "world"], max_vocab=10, min_freq=2)
        self.assertEqual(encode(["hello", "missing"], vocab)[1], vocab["<unk>"])
    def test_sequence_count_and_shapes(self):
        ds = NextTokenDataset([1,2,3,4,5], seq_len=2)
        self.assertEqual(len(ds), 3)
        x, y = ds[0]
        self.assertEqual(list(x.shape), [2])
        self.assertEqual(int(y), 3)

if __name__ == "__main__":
    unittest.main()
