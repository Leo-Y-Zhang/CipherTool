"""cipher_tool -- an offline classical cryptanalysis toolkit.

Written from scratch for a school team entering the National Cipher
Challenge. Every cipher and every attack in this package is implemented here
in plain Python; no third-party cryptography, cryptanalysis or numerical
package is used, and the toolkit never touches the network.

See RULES_COMPLIANCE.md before using this in a live competition round.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__", "DISCLAIMER"]

#: Printed by the CLI at the end of every solve. Deliberately unglamorous.
DISCLAIMER = (
    "This is a locally written cryptanalysis toolkit. Competition eligibility "
    "depends on the current National Cipher Challenge rules. Verify the "
    "current rules before using it in a live round."
)
