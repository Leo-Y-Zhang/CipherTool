# Example ciphertexts

Three messages to try the toolkit on, in rough order of difficulty. Each is
a real encryption of ordinary English prose, printed in five-letter groups
the way a competition presents them -- so the spacing tells you nothing.

| File | Cipher | Try | `auto fast` takes |
|---|---|---|---|
| `01_caesar.txt` | Caesar shift | `auto fast` | ~2s |
| `02_vigenere.txt` | Vigenere, 6-letter key | `analyse` first, then `auto fast` | ~4s |
| `03_transposition.txt` | Columnar, 7-letter key | `analyse` -- note the chi-squared | ~2s |

All three are solved by `auto fast` with `strong` confidence. If one of them
ever stops solving, that is a real regression and worth investigating.

**Worth doing in this order**, because it shows you what the tool is for:

1. Run `analyse` on each one *before* solving it. Look at the Index of
   Coincidence and the chi-squared figure. Those two numbers alone tell you
   which family each message belongs to -- and you can read that off the
   report yourself, without the tool guessing for you.
2. Then run `auto fast` and see whether the heuristics agreed with you.

`03_transposition.txt` is the interesting one: its letter frequencies are
*exactly* English, because a transposition only moves letters and never
changes them. The chi-squared stays small while the text stays unreadable.
No other family does that.

The answers are not written down here on purpose. Read the plaintext the
tool gives you and judge it yourself -- that is the habit the whole toolkit
is built around.
