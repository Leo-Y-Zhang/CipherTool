"""The Hill cipher: block substitution by matrix multiplication modulo 26.

The cipher
----------
Lester Hill's 1929 cipher is the first classical cipher that is genuinely
*polygraphic* in a linear-algebra sense. Fix a block size ``n`` and an
``n x n`` matrix ``K`` of integers modulo 26. Cut the plaintext into blocks of
``n`` letters, write each block as a column vector ``p`` of numbers (A=0 ...
Z=25), and encipher it as

    c = K p   (mod 26)

Every ciphertext letter of a block therefore depends on *every* plaintext
letter of that block. That is what makes Hill strong against the frequency
attacks that break Caesar, affine and simple substitution: with n = 2 the
single-letter distribution of the ciphertext is close to flat, because each
ciphertext letter is a mixture of two plaintext letters.

Decryption needs the matrix inverse modulo 26:

    p = K^-1 c   (mod 26)

and the whole difficulty of implementing Hill correctly is that "inverse"
here means inverse in the ring of integers modulo 26, not inverse over the
real numbers. See :func:`matrix_inverse`.

Why 26 is an awkward modulus
----------------------------
26 = 2 x 13 is not prime, so the integers modulo 26 form a ring with zero
divisors, not a field. A matrix ``M`` is invertible modulo 26 if and only if
its determinant is a *unit* modulo 26, that is

    gcd(det M, 26) = 1

which, because the only prime factors of 26 are 2 and 13, means exactly:
the determinant must be odd and must not be a multiple of 13. Twelve of the
twenty-six residues qualify (1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25).
A matrix whose determinant is 0, 2, 13 or 24 has no inverse modulo 26 and is
useless as a Hill key: :func:`matrix_inverse` raises
:class:`NotInvertibleError` and names the offending determinant rather than
quietly returning nonsense.

The attacks implemented here
----------------------------
**Known plaintext.** This is the real break, and it is devastating. Hill is
linear, so ``n`` matched plaintext/ciphertext blocks give ``n`` simultaneous
vector equations that determine the key outright:

    K P = C   =>   K = C P^-1   (mod 26)

where ``P`` and ``C`` have the matched blocks as their columns. No searching
is involved; the only thing that can go wrong is that the particular blocks
chosen give a singular ``P``, and then we simply try different blocks. See
:func:`known_plaintext_attack`.

**Exhaustive search, 2 x 2 only.** With n = 2 there are 26^4 = 456,976
matrices, of which 157,248 are invertible modulo 26. That is small enough to
test every one of them, so :func:`solve` does exactly that and ranks the
results. It is a two-stage search: a cheap digraph model shortlists keys, and
the full order-3 model rescores the shortlist. See :func:`solve`.

**Nothing else.** For n = 3 the space is 26^9 = 5,429,503,678,976 matrices
and for n = 4 it is 26^16, which is roughly 4.4 x 10^22. Neither is
searchable, by this toolkit or by anything else, so :func:`solve` refuses to
pretend: with n >= 3 it returns no candidates and says in its notes that a
crib or a supplied key is the only way in. Statistical attacks on 3 x 3 Hill
(hill climbing on trigraph statistics) exist in the literature but need far
more ciphertext than a competition round supplies, and we have not written
one; claiming otherwise would be worse than useless.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Any, Iterable, Sequence

from .candidates import Candidate, CandidateSet
from .normalize import (
    ALPHABET,
    ALPHABET_SIZE,
    NormalizedText,
    clean_key,
    letters_only,
    normalize,
)
from .scoring import EnglishScorer, annotate, default_scorer
from .statistics import chi_squared_english, index_of_coincidence, prime_factors

#: Letter used to pad a short final block during encryption.
FILLER = "X"

#: Type alias for a matrix given by a caller. Rows of integers.
Matrix = list[list[int]]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NotInvertibleError(ValueError):
    """A matrix has no inverse modulo the working modulus.

    Raised instead of returning something plausible-looking, because a Hill
    key with a non-unit determinant is not a "nearly valid" key: the encryption
    map is not a bijection at all, several plaintexts collide onto the same
    ciphertext, and no decryption exists. The message names the determinant
    and the factor it shares with the modulus so the operator can see exactly
    which rule was broken.

    Subclasses :class:`ValueError` so callers that only care about "bad key"
    can catch the broad type.
    """

    def __init__(self, determinant_value: int, modulus: int, detail: str = "") -> None:
        self.determinant = determinant_value
        self.modulus = modulus
        self.common_factor = math.gcd(determinant_value, modulus)
        message = (
            f"matrix is not invertible modulo {modulus}: its determinant is "
            f"{determinant_value}, which shares the factor "
            f"{self.common_factor} with {modulus}"
        )
        if modulus == ALPHABET_SIZE:
            message += (
                " (26 = 2 x 13, so a usable Hill determinant must be odd and "
                "not a multiple of 13; the usable values are "
                "1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25)"
            )
        if detail:
            message += ". " + detail
        super().__init__(message)


# ---------------------------------------------------------------------------
# Modular arithmetic
# ---------------------------------------------------------------------------


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm: return ``(g, x, y)`` with ``ax + by = g``.

    ``g`` is ``gcd(a, b)``. The algorithm is ordinary Euclid -- repeatedly
    replace ``(a, b)`` by ``(b, a mod b)`` until the remainder is zero -- with
    two extra columns carried alongside that express each remainder as an
    integer combination of the two original inputs.

    Write the state as ``r = a*s + b*t``. We start with two such rows that are
    true by inspection::

        r0 = a,  s0 = 1,  t0 = 0      (a = a*1 + b*0)
        r1 = b,  s1 = 0,  t1 = 1      (b = a*0 + b*1)

    Each step subtracts ``q`` copies of the second row from the first, where
    ``q = r0 // r1``. Subtracting whole rows keeps the invariant
    ``r = a*s + b*t`` true, and the ``r`` column is exactly the Euclidean
    sequence, so it terminates. When the remainder hits zero the previous row
    holds the gcd together with the coefficients that produce it.

    >>> extended_gcd(240, 46)
    (2, -9, 47)
    """
    old_remainder, remainder = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while remainder != 0:
        quotient = old_remainder // remainder
        old_remainder, remainder = remainder, old_remainder - quotient * remainder
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return old_remainder, old_s, old_t


def modular_inverse(a: int, m: int) -> int:
    """Return the ``x`` in ``0..m-1`` with ``a * x == 1 (mod m)``.

    Exists if and only if ``gcd(a, m) = 1``. The extended Euclidean algorithm
    gives integers ``x`` and ``y`` with

        a*x + m*y = gcd(a, m)

    and when that gcd is 1, reading the identity modulo ``m`` kills the ``m*y``
    term and leaves ``a*x == 1 (mod m)``. So ``x`` reduced into range is the
    inverse. When the gcd is not 1 no inverse exists, because ``a`` times
    anything is always a multiple of that gcd and so can never be 1.

    Raises ``ValueError`` naming the gcd when there is no inverse.
    """
    if m < 2:
        raise ValueError(f"modulus must be at least 2, got {m}")
    reduced = a % m
    gcd_value, x, _y = extended_gcd(reduced, m)
    if gcd_value != 1:
        raise ValueError(
            f"{a} has no inverse modulo {m}: gcd({reduced}, {m}) = {gcd_value}, "
            "and only values coprime to the modulus are invertible"
        )
    return x % m


# ---------------------------------------------------------------------------
# Matrix basics, written out rather than imported
# ---------------------------------------------------------------------------


def validate_matrix(
    matrix: Sequence[Sequence[int]], *, modulus: int = ALPHABET_SIZE
) -> Matrix:
    """Check that *matrix* is a square grid of integers and reduce it.

    Returns a fresh list-of-lists with every entry reduced into ``0..modulus-1``
    so the rest of the module can assume clean input. Booleans are rejected
    even though Python treats them as integers, because ``True`` in a key is
    always a mistake rather than an intention.
    """
    if modulus < 2:
        raise ValueError(f"modulus must be at least 2, got {modulus}")
    if isinstance(matrix, (str, bytes)):
        raise ValueError(
            "matrix must be a sequence of rows, not a string; use "
            "key_from_string() to build a matrix from a keyword"
        )
    rows = list(matrix)
    if not rows:
        raise ValueError("matrix is empty; a Hill key needs at least one row")
    size = len(rows)
    cleaned: Matrix = []
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise ValueError(f"row {row_index} of the matrix is a string, not numbers")
        values = list(row)
        if len(values) != size:
            raise ValueError(
                f"matrix must be square: it has {size} rows but row {row_index} "
                f"has {len(values)} entries"
            )
        clean_row: list[int] = []
        for column_index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"matrix entry at row {row_index}, column {column_index} is "
                    f"{value!r}; every entry must be a plain integer"
                )
            clean_row.append(value % modulus)
        cleaned.append(clean_row)
    return cleaned


def identity_matrix(size: int) -> Matrix:
    """The ``size x size`` identity matrix."""
    if size < 1:
        raise ValueError(f"matrix size must be at least 1, got {size}")
    return [[1 if row == column else 0 for column in range(size)] for row in range(size)]


def transpose(matrix: Sequence[Sequence[int]]) -> Matrix:
    """Reflect a matrix in its leading diagonal: entry (i, j) becomes (j, i)."""
    rows = list(matrix)
    if not rows:
        return []
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("cannot transpose a ragged matrix; every row must be the same length")
    return [[rows[row][column] for row in range(len(rows))] for column in range(width)]


def matrix_multiply(
    a: Sequence[Sequence[int]],
    b: Sequence[Sequence[int]],
    modulus: int = ALPHABET_SIZE,
) -> Matrix:
    """Matrix product ``a b`` reduced modulo *modulus*.

    Entry ``(i, j)`` of the product is the dot product of row ``i`` of ``a``
    with column ``j`` of ``b``::

        (ab)[i][j] = sum over k of a[i][k] * b[k][j]

    which requires ``a`` to have exactly as many columns as ``b`` has rows.
    Reducing modulo 26 at the end is equivalent to reducing at every step,
    because addition and multiplication both commute with taking remainders;
    we reduce once at the end so the arithmetic is exact until then.
    """
    if modulus < 2:
        raise ValueError(f"modulus must be at least 2, got {modulus}")
    left = [list(row) for row in a]
    right = [list(row) for row in b]
    if not left or not right:
        raise ValueError("cannot multiply an empty matrix")
    inner = len(left[0])
    if any(len(row) != inner for row in left):
        raise ValueError("left matrix is ragged; every row must be the same length")
    if len(right) != inner:
        raise ValueError(
            f"shape mismatch: left matrix has {inner} columns but right matrix "
            f"has {len(right)} rows; they cannot be multiplied"
        )
    width = len(right[0])
    if any(len(row) != width for row in right):
        raise ValueError("right matrix is ragged; every row must be the same length")

    product: Matrix = []
    for row in left:
        new_row = []
        for column in range(width):
            total = 0
            for k in range(inner):
                total += row[k] * right[k][column]
            new_row.append(total % modulus)
        product.append(new_row)
    return product


def matrix_vector(
    matrix: Sequence[Sequence[int]],
    vector: Sequence[int],
    modulus: int = ALPHABET_SIZE,
) -> list[int]:
    """Multiply *matrix* by the column vector *vector*, modulo *modulus*.

    This is the whole Hill cipher in one line of mathematics::

        result[i] = sum over j of matrix[i][j] * vector[j]   (mod 26)

    Written separately from :func:`matrix_multiply` because it is the inner
    loop of encryption and decryption and deserves not to allocate a matrix
    per block.
    """
    if modulus < 2:
        raise ValueError(f"modulus must be at least 2, got {modulus}")
    rows = [list(row) for row in matrix]
    values = list(vector)
    if not rows:
        raise ValueError("cannot multiply by an empty matrix")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("matrix is ragged; every row must be the same length")
    if len(values) != width:
        raise ValueError(
            f"shape mismatch: matrix has {width} columns but the vector has "
            f"{len(values)} entries"
        )
    return [
        sum(row[j] * values[j] for j in range(width)) % modulus for row in rows
    ]


def minor(matrix: Sequence[Sequence[int]], row: int, column: int) -> Matrix:
    """The submatrix left after deleting one row and one column.

    Used by cofactor expansion. Deleting row ``i`` and column ``j`` of an
    ``n x n`` matrix leaves an ``(n-1) x (n-1)`` matrix whose determinant is
    called the ``(i, j)`` minor.
    """
    return [
        [value for column_index, value in enumerate(source_row) if column_index != column]
        for row_index, source_row in enumerate(matrix)
        if row_index != row
    ]


def determinant(
    matrix: Sequence[Sequence[int]], modulus: int | None = ALPHABET_SIZE
) -> int:
    """Determinant of a square matrix, optionally reduced modulo *modulus*.

    Pass ``modulus=None`` for the ordinary integer determinant.

    How it is computed
    ------------------
    * 1 x 1: the single entry.
    * 2 x 2: ``ad - bc`` directly. Geometrically this is the signed area of
      the parallelogram spanned by the two rows; algebraically it is the only
      term structure the Leibniz formula allows for n = 2.
    * n x n: **cofactor expansion along the first row**::

          det(M) = sum over j of (-1)^j * M[0][j] * det(minor(M, 0, j))

      Read that as: pick the entry in the first row and column ``j``, delete
      its row and column, and weight the determinant of what is left by that
      entry and by an alternating sign. The alternating sign is not decoration
      -- it is what makes the result change sign when two rows are swapped,
      which is the defining property of the determinant.

    Cofactor expansion costs O(n!) work, which would be hopeless for large
    matrices. Hill keys in this toolkit are at most 4 x 4 (24 terms), so the
    recursion is simpler and safer than writing a modular Gaussian
    elimination, which is awkward anyway because 26 is not prime and pivots
    cannot always be scaled to 1.
    """
    rows = [list(row) for row in matrix]
    if not rows:
        raise ValueError("determinant is undefined for an empty matrix")
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError(
            f"determinant needs a square matrix; got {size} rows with lengths "
            f"{[len(row) for row in rows]}"
        )

    if size == 1:
        value = rows[0][0]
    elif size == 2:
        # The 2 x 2 case in full: | a b ; c d | -> ad - bc.
        value = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
    else:
        value = 0
        sign = 1
        for column in range(size):
            entry = rows[0][column]
            if entry:  # a zero entry contributes nothing, so skip the recursion
                value += sign * entry * determinant(minor(rows, 0, column), None)
            sign = -sign

    if modulus is None:
        return value
    if modulus < 2:
        raise ValueError(f"modulus must be at least 2, got {modulus}")
    return value % modulus


def cofactor_matrix(
    matrix: Sequence[Sequence[int]], modulus: int | None = ALPHABET_SIZE
) -> Matrix:
    """Matrix of signed minors: ``C[i][j] = (-1)^(i+j) * det(minor(M, i, j))``.

    For a 1 x 1 matrix the minor is the empty matrix, whose determinant is 1
    by the usual empty-product convention, so the cofactor matrix is ``[[1]]``.
    That convention is what makes ``adj(M) M = det(M) I`` hold for n = 1 too.
    """
    rows = [list(row) for row in matrix]
    size = len(rows)
    if size == 0:
        raise ValueError("cofactor matrix is undefined for an empty matrix")
    if any(len(row) != size for row in rows):
        raise ValueError("cofactor matrix needs a square matrix")
    if size == 1:
        result = [[1]]
    else:
        result = [
            [
                ((-1) ** (i + j)) * determinant(minor(rows, i, j), None)
                for j in range(size)
            ]
            for i in range(size)
        ]
    if modulus is None:
        return result
    return [[value % modulus for value in row] for row in result]


def adjugate(
    matrix: Sequence[Sequence[int]], modulus: int | None = ALPHABET_SIZE
) -> Matrix:
    """The adjugate (classical adjoint): the transpose of the cofactor matrix.

    Its defining property, and the only reason we care about it, is the
    identity that holds over any commutative ring::

        adj(M) M = M adj(M) = det(M) I

    That identity is what turns "divide by the determinant" into a legal
    operation modulo 26 whenever the determinant happens to be invertible.
    """
    return transpose(cofactor_matrix(matrix, modulus))


def matrix_inverse(
    matrix: Sequence[Sequence[int]], modulus: int = ALPHABET_SIZE
) -> Matrix:
    """Inverse of *matrix* modulo *modulus*, via the adjugate.

    The mathematics
    ---------------
    Over the integers modulo ``m`` we cannot "divide" in general, so the usual
    ``inv(M) = adj(M) / det(M)`` has to be rewritten as a multiplication::

        inv(M) = det(M)^-1 * adj(M)   (mod m)

    where ``det(M)^-1`` is the *modular* inverse of the determinant. Checking
    that this works takes one line, using the adjugate identity::

        (det(M)^-1 adj(M)) M = det(M)^-1 (adj(M) M)
                             = det(M)^-1 det(M) I
                             = I

    Existence
    ---------
    The construction needs ``det(M)^-1`` to exist, so the inverse exists **if
    and only if** ``gcd(det M, m) = 1``. For the alphabet modulus this is the
    familiar Hill rule:

        26 = 2 x 13

    so the determinant must be coprime to both 2 and 13, that is **odd and not
    a multiple of 13**. A determinant of 0, 2, 13 or 24 fails. The "only if"
    half is worth stating too: if ``gcd(det M, m) = g > 1`` then for any
    candidate inverse ``N`` we would need ``det(N) det(M) = det(I) = 1``, and
    the left side is a multiple of ``g``, so no such ``N`` exists.

    Raises
    ------
    NotInvertibleError
        When the determinant is not a unit modulo *modulus*. The error names
        the determinant and the shared factor.
    """
    reduced = validate_matrix(matrix, modulus=modulus)
    det_value = determinant(reduced, modulus)
    if math.gcd(det_value, modulus) != 1:
        raise NotInvertibleError(
            det_value,
            modulus,
            "Choose a different key matrix; this one maps several different "
            "plaintext blocks onto the same ciphertext block, so even the "
            "sender could not decipher it.",
        )
    det_inverse = modular_inverse(det_value, modulus)
    adjugate_matrix = adjugate(reduced, modulus)
    return [[(det_inverse * value) % modulus for value in row] for row in adjugate_matrix]


def is_invertible(
    matrix: Sequence[Sequence[int]], modulus: int = ALPHABET_SIZE
) -> bool:
    """True when *matrix* can be used as a Hill key modulo *modulus*."""
    reduced = validate_matrix(matrix, modulus=modulus)
    return math.gcd(determinant(reduced, modulus), modulus) == 1


# ---------------------------------------------------------------------------
# Keys as text
# ---------------------------------------------------------------------------


def key_from_string(key: str, *, modulus: int = ALPHABET_SIZE) -> Matrix:
    """Build a square Hill matrix from a keyword, filling row by row.

    ``key_from_string("HILL")`` gives ``[[7, 8], [11, 11]]``: H=7, I=8, L=11,
    L=11 laid out across the first row and then the second. A keyword of
    ``n*n`` letters therefore gives an ``n x n`` key, which is why real Hill
    keys are almost always perfect squares in length -- four letters, nine
    letters, sixteen letters.

    Non-letters are stripped first, so ``"hill"``, ``"H I L L"`` and
    ``"H.I.L.L!"`` all give the same matrix.

    The matrix is **not** checked for invertibility here: a caller may well
    want to look at a bad key and be told exactly why it is bad. Encryption
    and decryption do the checking.

    Raises
    ------
    ValueError
        If the cleaned key is empty, is not a perfect square in length, or
        would give a 1 x 1 matrix (which is not a Hill cipher at all, just a
        multiplicative cipher -- use the affine module for that).
    """
    cleaned = clean_key(key)
    if not cleaned:
        raise ValueError(
            f"key {key!r} contains no letters; a Hill key must be a run of "
            "A-Z letters whose length is a perfect square (4, 9, 16, ...)"
        )
    size = math.isqrt(len(cleaned))
    if size * size != len(cleaned):
        nearest_low = size * size
        nearest_high = (size + 1) * (size + 1)
        raise ValueError(
            f"key {cleaned!r} has {len(cleaned)} letters, which is not a "
            f"perfect square, so it cannot fill a square matrix; the nearest "
            f"usable lengths are {nearest_low} and {nearest_high}"
        )
    if size < 2:
        raise ValueError(
            f"key {cleaned!r} would give a 1 x 1 matrix, which is a "
            "multiplicative (affine) cipher rather than a Hill cipher; use a "
            "key of 4, 9 or 16 letters"
        )
    values = [ord(character) - 65 for character in cleaned]
    return [
        [values[row * size + column] % modulus for column in range(size)]
        for row in range(size)
    ]


def matrix_to_string(matrix: Sequence[Sequence[int]]) -> str:
    """Render a matrix as the keyword that would produce it, row by row.

    The inverse of :func:`key_from_string`. Entries outside 0..25 are reduced
    first, so this always produces letters.
    """
    reduced = validate_matrix(matrix)
    return "".join(ALPHABET[value] for row in reduced for value in row)


def describe_matrix(matrix: Sequence[Sequence[int]]) -> str:
    """Copy-pasteable one-line description used as ``Candidate.key``.

    ``describe_matrix([[7, 8], [11, 11]])`` gives
    ``"key=HILL matrix=[[7,8],[11,11]]"``.
    """
    reduced = validate_matrix(matrix)
    rows = ",".join(
        "[" + ",".join(str(value) for value in row) + "]" for row in reduced
    )
    return f"key={matrix_to_string(reduced)} matrix=[{rows}]"


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


def encrypt(
    text: str, matrix: Sequence[Sequence[int]], *, filler: str = FILLER
) -> str:
    """Encrypt. Operates on letters only; returns letters only, uppercase.

    The text is cut into blocks of ``n = len(matrix)`` letters and each block
    is multiplied by the key matrix modulo 26.

    Padding
    -------
    The final block is padded to the full block length with *filler*
    (``"X"`` by default), because the matrix cannot act on a short vector.
    **The padding is not recorded anywhere.** Decryption has no way to know
    whether a trailing ``X`` was in the message or was added here, so
    :func:`decrypt` returns it as-is and the reader must judge. This is a
    genuine property of the cipher, not a shortcoming of this implementation.

    Raises
    ------
    ValueError
        If the matrix is not square or is not invertible modulo 26 (an
        un-invertible key produces a ciphertext that nobody, including the
        sender, can decipher), or if *filler* is not a single A-Z letter.
    """
    reduced = validate_matrix(matrix)
    size = len(reduced)
    # Reject an unusable key at encryption time rather than letting the
    # operator discover at decryption time that the message is lost.
    if not is_invertible(reduced):
        raise NotInvertibleError(
            determinant(reduced),
            ALPHABET_SIZE,
            "Refusing to encrypt with a key that cannot be inverted: the "
            "resulting ciphertext could never be deciphered.",
        )
    pad = _validate_filler(filler)

    letters = letters_only(text)
    if not letters:
        return ""
    remainder = len(letters) % size
    if remainder:
        letters += pad * (size - remainder)

    values = [ord(character) - 65 for character in letters]
    out: list[str] = []
    for start in range(0, len(values), size):
        block = values[start : start + size]
        for value in matrix_vector(reduced, block):
            out.append(ALPHABET[value])
    return "".join(out)


def decrypt(text: str, matrix: Sequence[Sequence[int]]) -> str:
    """Exact inverse of :func:`encrypt` for the same key.

    Computes ``K^-1`` once and applies it to every block. Any filler letters
    that :func:`encrypt` added are returned unchanged, because nothing in the
    ciphertext distinguishes them from real plaintext.

    Raises
    ------
    NotInvertibleError
        If the key matrix has no inverse modulo 26.
    ValueError
        If the ciphertext does not divide exactly into blocks. A genuine Hill
        ciphertext always does, so a leftover partial block means the text has
        been truncated or mistranscribed, and guessing at the missing letters
        would produce plausible-looking nonsense.
    """
    reduced = validate_matrix(matrix)
    size = len(reduced)
    inverse = matrix_inverse(reduced)

    letters = letters_only(text)
    if not letters:
        return ""
    if len(letters) % size:
        raise ValueError(
            f"ciphertext has {len(letters)} letters, which is not a multiple of "
            f"the block size {size} (it leaves {len(letters) % size} letters "
            "over). A Hill ciphertext is always a whole number of blocks, so "
            "either the block size is wrong or the text is incomplete."
        )

    values = [ord(character) - 65 for character in letters]
    out: list[str] = []
    for start in range(0, len(values), size):
        block = values[start : start + size]
        for value in matrix_vector(inverse, block):
            out.append(ALPHABET[value])
    return "".join(out)


def _validate_filler(filler: str) -> str:
    """A filler must be exactly one A-Z letter."""
    cleaned = letters_only(filler)
    if len(cleaned) != 1 or len(filler) != 1:
        raise ValueError(
            f"filler must be a single A-Z letter, got {filler!r}"
        )
    return cleaned


# ---------------------------------------------------------------------------
# How big is the key space, really
# ---------------------------------------------------------------------------


def search_space_size(size: int, modulus: int = ALPHABET_SIZE) -> int:
    """Total number of ``size x size`` matrices modulo *modulus*.

    Simply ``modulus ** (size * size)``: 456,976 for 2 x 2 and
    5,429,503,678,976 for 3 x 3.
    """
    if size < 1:
        raise ValueError(f"matrix size must be at least 1, got {size}")
    return modulus ** (size * size)


@lru_cache(maxsize=16)
def invertible_matrix_count(size: int, modulus: int = ALPHABET_SIZE) -> int:
    """How many ``size x size`` matrices are invertible modulo *modulus*.

    Counted exactly, not sampled. Two facts do the work.

    **1. Chinese Remainder Theorem.** If ``m = p1^k1 * p2^k2 * ...`` then the
    ring of integers modulo ``m`` splits as a product of the rings modulo each
    prime power, and a matrix is invertible modulo ``m`` exactly when it is
    invertible modulo every prime power. So the count multiplies::

        |GL(n, Z/m)| = product over prime powers of |GL(n, Z/p^k)|

    **2. Counting over a prime.** Over the field with ``p`` elements, an
    invertible matrix is one whose rows are linearly independent. Choose the
    first row: any nonzero vector, ``p^n - 1`` ways. Choose the second: any
    vector outside the line spanned by the first, ``p^n - p`` ways. In general
    the ``i``-th row must avoid the ``p^i`` vectors already spanned::

        |GL(n, Z/p)| = product for i = 0..n-1 of (p^n - p^i)

    Lifting from ``Z/p`` to ``Z/p^k`` multiplies by ``p^((k-1) n^2)``, because
    reduction modulo ``p`` is a surjection whose fibres all have that size and
    invertibility only depends on the reduction.

    For the alphabet case: 26 = 2 x 13, so
    ``|GL(2, Z/26)| = (4-1)(4-2) x (169-1)(169-13) = 6 x 26208 = 157,248``,
    which is 34.4 per cent of the 456,976 matrices that exist.
    """
    if size < 1:
        raise ValueError(f"matrix size must be at least 1, got {size}")
    if modulus < 2:
        raise ValueError(f"modulus must be at least 2, got {modulus}")
    factors: dict[int, int] = {}
    for prime in prime_factors(modulus):
        factors[prime] = factors.get(prime, 0) + 1

    total = 1
    for prime, power in factors.items():
        order = 1
        for i in range(size):
            order *= prime**size - prime**i
        total *= order * prime ** ((power - 1) * size * size)
    return total


def brute_force_feasible(size: int) -> bool:
    """Is exhaustive search over ``size x size`` Hill keys realistic?

    True only for ``size == 2``. The 157,248 invertible 2 x 2 matrices can all
    be scored in a couple of seconds of pure Python. The 3 x 3 case has about
    1.6 x 10^12 invertible matrices; at the speed this toolkit achieves that
    is roughly four hundred thousand years, so the honest answer is no.
    """
    return size == 2


# ---------------------------------------------------------------------------
# The real attack: known plaintext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownPlaintextAttempt:
    """One try at solving for the key from a particular choice of blocks."""

    blocks: tuple[int, ...]
    letter_offsets: tuple[int, ...]
    determinant: int
    outcome: str

    def describe(self) -> str:
        """One line naming the blocks tried, det(P) and what went wrong."""
        return (
            f"blocks {list(self.blocks)} (letters "
            f"{list(self.letter_offsets)}), det(P)={self.determinant}: "
            f"{self.outcome}"
        )


@dataclass(frozen=True)
class KnownPlaintextResult:
    """What :func:`known_plaintext_attack` found, and what it tried.

    ``matrix`` is ``None`` when the attack failed. ``attempts`` always lists
    every block selection that was tried and why it did not work, so a failure
    is a report rather than a shrug.
    """

    size: int
    matrix: Matrix | None
    blocks_used: tuple[int, ...] | None
    verified: bool
    blocks_available: int
    attempts: tuple[KnownPlaintextAttempt, ...] = ()
    reason: str = ""

    @property
    def succeeded(self) -> bool:
        """True only when a key was recovered *and* it reproduced the crib.

        Both halves matter: the algebra can hand back a well-formed matrix
        from a misaligned crib, and only the verification step tells the two
        cases apart.
        """
        return self.matrix is not None and self.verified

    def describe(self) -> str:
        """Human-readable summary for a report."""
        if self.succeeded:
            assert self.matrix is not None
            return (
                f"Recovered a {self.size}x{self.size} key from blocks "
                f"{list(self.blocks_used or ())} and verified it against all "
                f"{self.blocks_available} matched blocks: "
                f"{describe_matrix(self.matrix)}"
            )
        lines = [f"No key recovered from {self.blocks_available} matched blocks."]
        if self.reason:
            lines.append(self.reason)
        for attempt in self.attempts[:12]:
            lines.append("  " + attempt.describe())
        if len(self.attempts) > 12:
            lines.append(f"  ... and {len(self.attempts) - 12} further attempts")
        return "\n".join(lines)


def known_plaintext_attack(
    cipher: str,
    plain: str,
    size: int,
    *,
    maximum_attempts: int = 400,
) -> KnownPlaintextResult:
    """Solve for the Hill key from matched plaintext and ciphertext.

    *cipher* and *plain* must be the **same stretch of the message**, and must
    both start on a block boundary of the real encryption -- Hill blocks are
    counted from the first letter of the message, so a crib that begins at
    letter 7 of a 2 x 2 message is useless until you shift it to letter 6 or 8.
    Only whole blocks are used; any trailing letters are ignored.

    The linear algebra
    ------------------
    Encryption acts one block at a time: ``c_j = K p_j (mod 26)``, where
    ``p_j`` and ``c_j`` are column vectors of ``n`` letters. Take any ``n``
    matched blocks and stack them side by side as the **columns** of two
    ``n x n`` matrices::

        P = [ p_{j1} | p_{j2} | ... | p_{jn} ]
        C = [ c_{j1} | c_{j2} | ... | c_{jn} ]

    Matrix multiplication acts on columns independently, so applying ``K`` to
    every column at once is exactly ``K P = C``. If ``P`` happens to be
    invertible modulo 26 we may multiply on the right by ``P^-1``::

        K = C P^-1   (mod 26)

    and the key falls out with no searching at all. That is why Hill is
    considered broken: ``n^2`` known letters are enough, and a competition crib
    of a dozen letters usually supplies them.

    When it fails, and what we do about it
    --------------------------------------
    The construction needs ``gcd(det P, 26) = 1``. Roughly a third of block
    pairs fail that test -- for instance any two plaintext blocks that are
    equal, or a plaintext like "ABAB" whose blocks repeat -- so we try
    different selections of blocks:

    1. every window of ``n`` **consecutive** blocks, in order;
    2. then every combination of ``n`` blocks drawn from the first few, which
       catches texts where consecutive blocks are stubbornly dependent.

    Each recovered key is then **verified** by re-encrypting the whole matched
    plaintext and comparing with the matched ciphertext. That check is what
    catches the other failure mode: a crib placed at the wrong offset, or the
    wrong block size, produces a perfectly well-formed matrix that decrypts
    nothing. Such a key is reported as a failed attempt, never as a solution.

    Raises
    ------
    ValueError
        If *size* is less than 2, if the two texts contain different numbers of
        letters, or if there are fewer than ``size`` whole matched blocks (with
        fewer than ``n`` blocks the system is underdetermined and any answer
        would be a guess).
    """
    if size < 2:
        raise ValueError(f"block size must be at least 2, got {size}")
    if maximum_attempts < 1:
        raise ValueError(f"maximum_attempts must be at least 1, got {maximum_attempts}")

    cipher_letters = letters_only(cipher)
    plain_letters = letters_only(plain)
    if len(cipher_letters) != len(plain_letters):
        raise ValueError(
            f"the matched texts must be the same length: got "
            f"{len(plain_letters)} plaintext letters and "
            f"{len(cipher_letters)} ciphertext letters. They must describe the "
            "same stretch of the message, letter for letter."
        )

    block_count = len(plain_letters) // size
    if block_count < size:
        raise ValueError(
            f"a {size}x{size} Hill key needs at least {size} matched blocks, "
            f"that is {size * size} letters; got {len(plain_letters)} letters "
            f"({block_count} whole blocks). With fewer blocks the equations do "
            "not determine the key."
        )

    usable = block_count * size
    plain_values = [ord(character) - 65 for character in plain_letters[:usable]]
    cipher_values = [ord(character) - 65 for character in cipher_letters[:usable]]
    plain_blocks = [
        plain_values[i * size : (i + 1) * size] for i in range(block_count)
    ]
    cipher_blocks = [
        cipher_values[i * size : (i + 1) * size] for i in range(block_count)
    ]

    attempts: list[KnownPlaintextAttempt] = []
    singular = 0

    for chosen in _block_selections(block_count, size, maximum_attempts):
        # Columns are blocks, so P[row][column] = block[column][row].
        p_matrix = [[plain_blocks[j][r] for j in chosen] for r in range(size)]
        det_value = determinant(p_matrix)
        offsets = tuple(j * size for j in chosen)

        if math.gcd(det_value, ALPHABET_SIZE) != 1:
            singular += 1
            attempts.append(
                KnownPlaintextAttempt(
                    chosen,
                    offsets,
                    det_value,
                    "P is singular modulo 26 (its determinant shares a factor "
                    f"{math.gcd(det_value, ALPHABET_SIZE)} with 26), so these "
                    "blocks do not determine the key",
                )
            )
            continue

        c_matrix = [[cipher_blocks[j][r] for j in chosen] for r in range(size)]
        key = matrix_multiply(c_matrix, matrix_inverse(p_matrix))

        # Verification: does this key reproduce EVERY matched block, not just
        # the ones it was built from? If not, the crib is misaligned or the
        # block size is wrong, and the key is an artefact.
        if not is_invertible(key):
            attempts.append(
                KnownPlaintextAttempt(
                    chosen,
                    offsets,
                    det_value,
                    f"solved to a key with determinant {determinant(key)}, "
                    "which is not invertible modulo 26 -- so it cannot be the "
                    "real key and the matched texts do not line up",
                )
            )
            continue

        reencrypted = "".join(
            ALPHABET[value]
            for block in plain_blocks
            for value in matrix_vector(key, block)
        )
        if reencrypted == cipher_letters[:usable]:
            attempts.append(
                KnownPlaintextAttempt(chosen, offsets, det_value, "solved and verified")
            )
            return KnownPlaintextResult(
                size=size,
                matrix=key,
                blocks_used=chosen,
                verified=True,
                blocks_available=block_count,
                attempts=tuple(attempts),
            )

        matching = sum(
            1 for i in range(usable) if reencrypted[i] == cipher_letters[i]
        )
        attempts.append(
            KnownPlaintextAttempt(
                chosen,
                offsets,
                det_value,
                f"solved to {describe_matrix(key)} but re-encrypting the known "
                f"plaintext reproduces only {matching} of {usable} ciphertext "
                "letters, so the matched texts are not aligned",
            )
        )

    reason = (
        f"Tried {len(attempts)} selections of {size} blocks out of "
        f"{block_count} available; {singular} were singular modulo 26 and "
        f"{len(attempts) - singular} produced a key that failed verification. "
        "The usual causes are a crib placed at the wrong offset (Hill blocks "
        f"are counted from letter 0 in steps of {size}), the wrong block size, "
        "or a transcription error in the crib."
    )
    return KnownPlaintextResult(
        size=size,
        matrix=None,
        blocks_used=None,
        verified=False,
        blocks_available=block_count,
        attempts=tuple(attempts),
        reason=reason,
    )


def _block_selections(
    block_count: int, size: int, limit: int
) -> Iterable[tuple[int, ...]]:
    """Which sets of *size* blocks to try, best bet first.

    Consecutive windows come first: a crib is usually a short contiguous run,
    and the earliest blocks are the ones the operator is most confident about.
    Non-consecutive combinations follow, because a plaintext with repeating
    structure ("ABABAB", "THETHE") can make every consecutive window singular
    while a wider spread is fine.
    """
    seen: set[tuple[int, ...]] = set()
    produced = 0

    for start in range(block_count - size + 1):
        chosen = tuple(range(start, start + size))
        seen.add(chosen)
        yield chosen
        produced += 1
        if produced >= limit:
            return

    # Widen the net. Restricted to the first few blocks so the number of
    # combinations stays small: C(10, 3) is 120, C(10, 4) is 210.
    pool = min(block_count, 10)
    for chosen in combinations(range(pool), size):
        if chosen in seen:
            continue
        yield chosen
        produced += 1
        if produced >= limit:
            return


# ---------------------------------------------------------------------------
# Search machinery for the 2 x 2 exhaustive attack
# ---------------------------------------------------------------------------


class HillSearchResult(CandidateSet):
    """A :class:`CandidateSet` that also carries notes about the search.

    Hill is the one cipher in this toolkit where "we did not even try" is a
    legitimate and important outcome -- a 3 x 3 key cannot be searched for --
    and an empty candidate list on its own would not say why. ``notes`` holds
    those plain-English explanations. Everything else behaves exactly like a
    ``CandidateSet``.
    """

    def __init__(
        self,
        candidates: Iterable[Candidate] = (),
        notes: Iterable[str] = (),
    ) -> None:
        super().__init__(candidates)
        self.notes: tuple[str, ...] = tuple(notes)


@lru_cache(maxsize=4)
def _digraph_log_probabilities(scorer: EnglishScorer) -> tuple[float, ...]:
    """log10 P(digraph) for all 676 digraphs, from the shared English model.

    Indexed ``first * 26 + second``. Obtained by asking the scorer to score
    each two-letter string, which under its interpolated model is exactly
    ``log P(x1) + log P(x2 | x1)``.

    Why a digraph model at all, when the toolkit has an order-3 one? Because
    the exhaustive 2 x 2 search scores 157,248 keys, and a full order-3 pass
    over the whole ciphertext for each of them would be a hundred times more
    arithmetic than we can afford. A 2 x 2 Hill cipher enciphers *aligned
    plaintext digraphs* independently, so a digraph model is precisely matched
    to the structure of the cipher: it is the strongest signal that can be
    read off a single block. The order-3 model then rescores the shortlist,
    where its cross-block context actually costs nothing.

    Cached per scorer instance; the shared scorer means this is built once.
    """
    return tuple(
        scorer.score(ALPHABET[first] + ALPHABET[second])
        for first in range(ALPHABET_SIZE)
        for second in range(ALPHABET_SIZE)
    )


@lru_cache(maxsize=1)
def _unit_table(modulus: int = ALPHABET_SIZE) -> tuple[int, ...]:
    """Modular inverses 0..modulus-1, with ``-1`` where no inverse exists.

    Lets the hot loop replace an extended-Euclid call with one list lookup.
    """
    table = []
    for value in range(modulus):
        table.append(modular_inverse(value, modulus) if math.gcd(value, modulus) == 1 else -1)
    return tuple(table)


def _row_order(digraphs: Sequence[float]) -> list[int]:
    """Visit matrix rows in descending order of English digraph probability.

    Justification for this ordering, since an arbitrary one would have to be
    declared as such: Hill keys in the wild are almost always *written down as
    a keyword* -- that is what :func:`key_from_string` exists for -- so the
    entries of a real key spell English text, and each row of the matrix is an
    English digraph. Visiting rows in digraph order therefore tests
    keyword-shaped keys such as ``HILL`` or ``CATS`` in the first few per cent
    of the search.

    This changes nothing when the search runs to completion; it matters only
    when a time budget cuts it short, and in that case the bias is recorded in
    the diagnostics so nobody reads a partial search as a clean sweep.
    """
    return sorted(range(len(digraphs)), key=lambda index: -digraphs[index])


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    size: int | None = None,
    matrix: Sequence[Sequence[int]] | None = None,
    key: str | None = None,
    crib: str | None = None,
    crib_at: int = 0,
    time_budget: float | None = None,
    sample_blocks: int = 200,
    shortlist: int = 40,
) -> HillSearchResult:
    """Attack a Hill ciphertext and return RANKED CANDIDATES, never one answer.

    Three routes in, tried in this order:

    ``matrix=`` or ``key=``
        A key was supplied. Decrypt with it, score it, return one candidate.
        Nothing is searched and nothing is claimed beyond the score.

    ``crib=``
        A stretch of known plaintext was supplied, starting at ciphertext
        letter *crib_at* (which must be a multiple of the block size, because
        Hill blocks are counted from the start of the message).
        :func:`known_plaintext_attack` solves for the key algebraically and
        verifies it. This is the only attack that works on a 3 x 3 or 4 x 4
        key. If it fails and the block size is 2, the exhaustive search runs
        as a fallback.

    neither, with ``size=2``
        Exhaustive search over all 157,248 invertible 2 x 2 matrices, in two
        stages: a digraph model scores every key over the first *sample_blocks*
        blocks, and the toolkit's order-3 model rescores the best *shortlist*
        of them over the whole text. On this machine the full sweep takes a
        couple of seconds.

    neither, with ``size >= 3``
        **No search is attempted and none is possible.** 26^9 is over five
        trillion matrices for a 3 x 3 key. The returned set is empty and
        ``.notes`` explains that a crib or a supplied key is the only way in.
        This is a deliberate refusal, not an oversight.

    Diagnostics on every candidate record how much of the key space was really
    covered, whether the time budget stopped the search, the determinant of
    the proposed key, and how many trailing letters did not fit into whole
    blocks.

    The search is exhaustive and deterministic; no randomness is involved, so
    there is no seed to set and repeated runs give identical output.

    Parameters
    ----------
    time_budget:
        Seconds. The exhaustive search checks the clock between rows of the
        outer loop and stops cleanly, recording
        ``diagnostics["time_budget_hit"] = True`` and the fraction of the key
        space it managed to cover.
    """
    engine = scorer or default_scorer()
    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters
    notes: list[str] = []

    if matrix is not None and key is not None:
        raise ValueError(
            "give either matrix= or key=, not both; they are two ways of "
            "writing the same thing (see key_from_string)"
        )

    supplied: Matrix | None = None
    if key is not None:
        supplied = key_from_string(key)
    elif matrix is not None:
        supplied = validate_matrix(matrix)

    if supplied is not None:
        inferred = len(supplied)
        if size is not None and size != inferred:
            raise ValueError(
                f"size={size} contradicts the supplied key, which is "
                f"{inferred}x{inferred}"
            )
        block_size = inferred
    else:
        block_size = 2 if size is None else size
    if block_size < 2:
        raise ValueError(
            f"block size must be at least 2, got {block_size}; a 1 x 1 Hill "
            "key is a multiplicative cipher, not a Hill cipher"
        )

    if not letters:
        return HillSearchResult(
            notes=["The input contains no letters, so there is nothing to solve."]
        )

    usable = len(letters) - (len(letters) % block_size)
    dropped = len(letters) - usable
    if dropped:
        notes.append(
            f"The text has {len(letters)} letters, which is not a multiple of "
            f"the block size {block_size}; the last {dropped} letter(s) cannot "
            "form a whole block and were ignored. A genuine Hill ciphertext "
            "divides exactly, so check the block size and the transcription."
        )
    if usable == 0:
        notes.append(
            f"Fewer than {block_size} letters: no whole block exists."
        )
        return HillSearchResult(notes=notes)

    body = letters[:usable]
    shared = {
        "block_size": block_size,
        "ciphertext_letters": len(letters),
        "letters_used": usable,
        "ciphertext_ic": index_of_coincidence(body),
        "ciphertext_chi2_per_letter": chi_squared_english(body),
    }
    if dropped:
        shared["trailing_letters_ignored"] = dropped

    # -- route 1: a key was handed to us ----------------------------------
    if supplied is not None:
        candidate = _candidate_for(
            supplied,
            body,
            normalized,
            usable == len(letters),
            engine,
            method=f"Hill {block_size}x{block_size} (supplied key)",
            extra={
                **shared,
                "search": "none -- the key was supplied, not found",
                "determinant": determinant(supplied),
            },
        )
        notes.append(
            "The key was supplied, so the score below is a measurement of that "
            "one key and not evidence that it is correct. Read the plaintext."
        )
        return HillSearchResult([candidate], notes)

    # -- route 2: a crib ---------------------------------------------------
    if crib is not None:
        crib_notes, crib_candidate = _solve_with_crib(
            body, crib, crib_at, block_size, normalized,
            usable == len(letters), engine, shared,
        )
        notes.extend(crib_notes)
        if crib_candidate is not None:
            return HillSearchResult([crib_candidate], notes)
        if not brute_force_feasible(block_size):
            notes.append(
                f"A {block_size}x{block_size} key cannot be searched for "
                f"({search_space_size(block_size):,} matrices), so the crib was "
                "the only available attack and no candidates can be offered."
            )
            return HillSearchResult(notes=notes)
        notes.append(
            "Falling back to the exhaustive 2x2 search, which does not need "
            "the crib."
        )

    # -- route 3: exhaustive search, 2 x 2 only ---------------------------
    if not brute_force_feasible(block_size):
        notes.append(
            f"No attack was run. A {block_size}x{block_size} Hill key is one of "
            f"{search_space_size(block_size):,} matrices "
            f"({invertible_matrix_count(block_size):,} of them invertible), so "
            "exhaustive search is out of the question -- not slow, impossible. "
            "This toolkit has no statistical attack on Hill keys larger than "
            "2x2. Supply a key with key=/matrix=, or supply a crib of at least "
            f"{block_size * block_size} known plaintext letters with crib= and "
            "crib_at=, and the key can be solved for algebraically."
        )
        return HillSearchResult(notes=notes)

    return _exhaustive_2x2(
        body, normalized, usable == len(letters), engine,
        top=top, time_budget=time_budget, sample_blocks=sample_blocks,
        shortlist=shortlist, shared=shared, notes=notes,
    )


def _solve_with_crib(
    body: str,
    crib: str,
    crib_at: int,
    block_size: int,
    normalized: NormalizedText,
    full_length: bool,
    engine: EnglishScorer,
    shared: dict[str, Any],
) -> tuple[list[str], Candidate | None]:
    """Run the known-plaintext attack from a crib and build a candidate."""
    notes: list[str] = []
    crib_letters = letters_only(crib)
    if not crib_letters:
        raise ValueError("crib contains no letters")
    if crib_at < 0:
        raise ValueError(f"crib_at must not be negative, got {crib_at}")
    if crib_at % block_size:
        raise ValueError(
            f"crib_at={crib_at} is not a multiple of the block size "
            f"{block_size}. Hill blocks are counted from letter 0 of the "
            f"message in steps of {block_size}, so a crib must start on a "
            f"block boundary; the nearest are "
            f"{crib_at - crib_at % block_size} and "
            f"{crib_at - crib_at % block_size + block_size}."
        )
    if crib_at + len(crib_letters) > len(body):
        raise ValueError(
            f"the crib runs from letter {crib_at} to "
            f"{crib_at + len(crib_letters)}, past the end of the usable "
            f"ciphertext ({len(body)} letters)"
        )

    matched_cipher = body[crib_at : crib_at + len(crib_letters)]
    result = known_plaintext_attack(matched_cipher, crib_letters, block_size)
    if not result.succeeded:
        notes.append("The known-plaintext attack failed. " + result.describe())
        return notes, None

    assert result.matrix is not None
    candidate = _candidate_for(
        result.matrix,
        body,
        normalized,
        full_length,
        engine,
        method=f"Hill {block_size}x{block_size} (known-plaintext attack)",
        extra={
            **shared,
            "search": "algebraic -- solved from the crib, nothing was guessed",
            "determinant": determinant(result.matrix),
            "crib_at": crib_at,
            "crib_letters": len(crib_letters),
            "crib_blocks_used": list(result.blocks_used or ()),
            "crib_attempts": len(result.attempts),
            "crib_verified_against_blocks": result.blocks_available,
        },
    )
    notes.append(result.describe())
    notes.append(
        "This key was solved for, not searched for, and it reproduces every "
        "matched block of the crib. If the crib itself was right, the key is "
        "right; if the plaintext below still reads as nonsense, the crib was "
        "wrong."
    )
    return notes, candidate


def _exhaustive_2x2(
    body: str,
    normalized: NormalizedText,
    full_length: bool,
    engine: EnglishScorer,
    *,
    top: int,
    time_budget: float | None,
    sample_blocks: int,
    shortlist: int,
    shared: dict[str, Any],
    notes: list[str],
) -> HillSearchResult:
    """Score every invertible 2 x 2 key, then rescore the best with the full model.

    Stage 1 is where the time goes, so it is written to do as little work per
    key as possible:

    * The ciphertext is turned into a list of blocks once. For each of the 676
      possible matrix *rows* ``(p, q)`` we precompute ``(p*c0 + q*c1) mod 26``
      for every block. Applying a decryption matrix to the whole text is then
      two table lookups per block rather than four multiplies.
    * We enumerate *encryption* keys ``K`` and derive the decryption matrix
      from the closed-form 2 x 2 adjugate, ``K^-1 = det^-1 [[d, -b], [-c, a]]``,
      with the determinant inverse read from a 26-entry table. This costs
      almost nothing and lets us report the key the way the sender wrote it.
    * Keys with a non-invertible determinant are skipped without scoring.

    Stage 2 takes the ``shortlist`` best keys, decrypts the whole ciphertext
    with each and scores it with the order-3 model, which sees across block
    boundaries and so separates a genuinely English plaintext from one that
    merely has plausible digraphs.
    """
    if sample_blocks < 1:
        raise ValueError(f"sample_blocks must be at least 1, got {sample_blocks}")
    if shortlist < 1:
        raise ValueError(f"shortlist must be at least 1, got {shortlist}")

    digraphs = _digraph_log_probabilities(engine)
    units = _unit_table(ALPHABET_SIZE)
    order = _row_order(digraphs)

    values = [ord(character) - 65 for character in body]
    total_blocks = len(values) // 2
    used_blocks = min(total_blocks, sample_blocks)
    first = [values[2 * i] for i in range(used_blocks)]
    second = [values[2 * i + 1] for i in range(used_blocks)]

    # row_plain[r] holds the plaintext letter this row produces for each block;
    # row_plain26[r] is the same scaled by 26 so the two can be added into a
    # flat digraph index without a multiply in the inner loop.
    row_plain: list[list[int]] = []
    row_plain26: list[list[int]] = []
    for p in range(ALPHABET_SIZE):
        for q in range(ALPHABET_SIZE):
            row = [(p * c0 + q * c1) % ALPHABET_SIZE for c0, c1 in zip(first, second)]
            row_plain.append(row)
            row_plain26.append([26 * value for value in row])

    keep = max(shortlist, top, 1)
    heap: list[tuple[float, int, int, int, int]] = []
    threshold = float("-inf")

    lookup = digraphs.__getitem__
    add = int.__add__  # C-level elementwise add for map(); avoids a lambda

    started = time.monotonic()
    deadline = None if time_budget is None else started + time_budget
    scored = 0
    budget_hit = False
    rows_visited = 0

    for r0 in order:
        if deadline is not None and time.monotonic() >= deadline:
            budget_hit = True
            break
        rows_visited += 1
        a, b = divmod(r0, ALPHABET_SIZE)
        for r1 in order:
            c, d = divmod(r1, ALPHABET_SIZE)
            unit = units[(a * d - b * c) % ALPHABET_SIZE]
            if unit < 0:
                continue  # determinant not coprime to 26: not a usable key
            # K^-1 = det^-1 * [[d, -b], [-c, a]] for a 2 x 2 matrix.
            top_row = row_plain26[
                (unit * d % ALPHABET_SIZE) * ALPHABET_SIZE
                + (-unit * b % ALPHABET_SIZE)
            ]
            bottom_row = row_plain[
                (-unit * c % ALPHABET_SIZE) * ALPHABET_SIZE
                + (unit * a % ALPHABET_SIZE)
            ]
            total = sum(map(lookup, map(add, top_row, bottom_row)))
            scored += 1
            if len(heap) < keep:
                heapq.heappush(heap, (total, a, b, c, d))
                if len(heap) == keep:
                    threshold = heap[0][0]
            elif total > threshold:
                heapq.heapreplace(heap, (total, a, b, c, d))
                threshold = heap[0][0]

    elapsed = time.monotonic() - started
    invertible_total = invertible_matrix_count(2)
    coverage = scored / invertible_total if invertible_total else 0.0

    if budget_hit:
        notes.append(
            f"The time budget of {time_budget:g}s stopped the search after "
            f"{scored:,} of {invertible_total:,} invertible keys "
            f"({coverage:.1%}). Keys are visited in order of how English-like "
            "their rows read as digraphs, because real Hill keys are usually "
            "written as a keyword -- so the untested remainder is biased "
            "towards matrices that do NOT spell common letter pairs. If the "
            "key was not keyword-shaped, it may be in the part not reached."
        )

    diagnostics_common: dict[str, Any] = {
        **shared,
        "search": "exhaustive over invertible 2x2 matrices",
        "keys_scored": scored,
        "invertible_keys_total": invertible_total,
        "matrices_total": search_space_size(2),
        "key_space_covered": round(coverage, 4),
        "search_complete": not budget_hit,
        "stage1_blocks_scored": used_blocks,
        "stage1_blocks_available": total_blocks,
        "stage2_rescored": min(len(heap), keep),
        "search_seconds": round(elapsed, 3),
        "search_order": (
            "rows in descending English digraph probability "
            "(keyword-shaped keys first)"
        ),
    }
    if budget_hit:
        diagnostics_common["time_budget_hit"] = True

    if used_blocks < total_blocks:
        notes.append(
            f"Stage 1 ranked keys on the first {used_blocks} of "
            f"{total_blocks} blocks ({used_blocks * 2} letters). That is ample "
            "to separate the right key, but the shortlist -- not the whole key "
            "space -- was rescored on the full text."
        )

    result = HillSearchResult(notes=notes)
    for stage1_score, a, b, c, d in heapq.nlargest(keep, heap):
        candidate = _candidate_for(
            [[a, b], [c, d]],
            body,
            normalized,
            full_length,
            engine,
            method="Hill 2x2 (exhaustive search)",
            extra={
                **diagnostics_common,
                "determinant": (a * d - b * c) % ALPHABET_SIZE,
                "stage1_digraph_score": round(stage1_score, 2),
            },
        )
        result.add(candidate)

    trimmed = HillSearchResult(result.top(max(top, 1)), notes)
    return trimmed


def _candidate_for(
    key_matrix: Sequence[Sequence[int]],
    body: str,
    normalized: NormalizedText,
    full_length: bool,
    engine: EnglishScorer,
    *,
    method: str,
    extra: dict[str, Any],
) -> Candidate:
    """Decrypt *body* with *key_matrix*, score it and package the evidence."""
    plaintext = decrypt(body, key_matrix)
    diagnostics = dict(extra)
    annotate(diagnostics, plaintext, engine)
    # relayout needs one plaintext character per letter of the original input,
    # which only holds when no trailing letters were dropped. Hill preserves
    # length otherwise, unlike a transposition or fractionating cipher.
    display = normalized.relayout(plaintext) if full_length else None
    return Candidate(
        method=method,
        key=describe_matrix(key_matrix),
        score=engine.score(plaintext),
        plaintext=plaintext,
        diagnostics=diagnostics,
        display=display,
    )
