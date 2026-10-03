# Draft question for the organisers

This asks the only people who can answer whether `cipher_tool` may be used in
the 2026 challenge. **It has to be sent by a team member.** Nothing is sent on
the team's behalf, and the answer is not guessed at in the meantime -- until
there is one, the toolkit is treated as not cleared (see
[RULES_COMPLIANCE.md](../RULES_COMPLIANCE.md#the-ai-clause-unresolved)).

Send it from a school or personal address, not from a shared one, and keep the
disclosure in it. The disclosure is the reason this is a question and not a
breach.

---

**To:** the National Cipher Challenge organisers, via the contact address on
`cipherchallenge.org`

**Subject:** Rule 13 and a tool we wrote ourselves with some AI assistance

---

Dear organisers,

I would like to check a point of rule 13 before the challenge gets going,
rather than after.

Over the last few months I wrote a classical cryptanalysis toolkit of my own:
fifteen ciphers and their attacks, pure Python with no third-party
cryptanalysis code, no dependencies and no network access. It is mine in the
sense the rule means -- I did not take a solver from the web and I do not call
any outside service.

The part I want to ask about is how some of it came to be written. I used an
AI coding assistant during the project, and I have kept that disclosed in the
repository throughout. Looking at the history properly, nine of fifty-six
commits had AI involvement. Most are corrections to code I had already
written -- stale figures in documentation, an error message instead of a
crash, a malformed input rejected -- but two changed how the software behaves
when solving, and the files touched include the two modules every attack
depends on, the frequency statistics and the language-model scoring.

Rule 13 says entrants may use "any software that you write yourself", and also
that we should not use AI "to decipher messages or to write software to do
so". I can see a reading in which my toolkit is my own software and the AI
assistance was incidental to code I wrote, and a reading in which the second
half of the rule covers it squarely. I do not want to guess which you intend.

So: may I use this toolkit in the 2026 challenge, or should I enter without
it? I am entirely happy to enter without it -- I would rather ask and be told
no than use it and find out later. If it helps, I can share the repository and
the commit history so you can see exactly what was AI-assisted and what was
not.

Thank you for your time, and for running the challenge.

Yours sincerely,

Leo Y. Zhang

---

## If the answer is no

Nothing needs undoing. The toolkit stays a public piece of engineering, the
challenge is entered with spreadsheets, text editors and the official BOSS
tools as rule 13 provides, and `RULES_COMPLIANCE.md` records why. Close issue
#8 with the organisers' reply quoted in it.

## If the answer is yes

Quote the reply in `RULES_COMPLIANCE.md` under the status table, with the date
and who sent it, and re-date the status to cleared. A permission that is not
written down where the next reader will find it will be forgotten by next
season.
