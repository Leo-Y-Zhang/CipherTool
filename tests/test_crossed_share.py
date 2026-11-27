"""The crossed-coordinate stage gets enough of a deep budget to finish.

On 2025 challenge 10B two restarts take about 17 seconds. At weight 6 the
stage's share of a 270 second deep run was about 10 seconds, so a slow
restart was cut off and the pipeline answered with a substitution. It
refuses other texts in a hundredth of a second, so a larger share costs
nothing when the cipher is something else.
"""

from __future__ import annotations

import unittest

from cipher_tool.auto import build_stages


class TestCrossedShare(unittest.TestCase):
    def test_weight_covers_two_restarts(self) -> None:
        stage = next(s for s in build_stages("deep", 5, 1)
                     if s.name == "crossed-coordinate digraph")
        self.assertGreaterEqual(stage.weight, 12.0)


if __name__ == "__main__":
    unittest.main()
