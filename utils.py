"""
utils.py
--------
Vocabulary definition and the two Keras StringLookup layers used everywhere:

  char_to_num  →  maps a character string to an integer index
  num_to_char  →  maps an integer index back to a character string

Both layers are module-level singletons so every file that does
`from utils import char_to_num, num_to_char` shares the same object.
"""

import tensorflow as tf
from config import VOCAB

# ── Lookup layers ─────────────────────────────────────────────────────────────

char_to_num = tf.keras.layers.StringLookup(
    vocabulary=VOCAB,
    oov_token=""       # out-of-vocabulary token (index 0)
)

num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    oov_token="",
    invert=True        # integer → character direction
)

VOCAB_SIZE = char_to_num.vocabulary_size()   # 40 characters + 1 OOV = 41

# ── Quick sanity-check (runs only when this file is executed directly) ────────

if __name__ == "__main__":
    print(f"Vocabulary ({VOCAB_SIZE} tokens):")
    print(char_to_num.get_vocabulary())
    print()
    # Round-trip test
    test_word = list("lipnet")
    encoded = char_to_num(test_word)
    decoded = num_to_char(encoded)
    print(f"Encoded 'lipnet': {encoded.numpy()}")
    print(f"Decoded back:     {[x.decode() for x in decoded.numpy()]}")
