"""Automated competition-compliance audit.

RULES_COMPLIANCE.md makes a set of claims about this toolkit. This file turns
the checkable ones into tests, so that the claims cannot quietly rot as the
code changes. Every assertion here corresponds to a numbered requirement in
that document.

What is checked here
--------------------
1. No third-party imports anywhere: every import is either the standard
   library or another module of this package.
2. No networking capability at all -- statically (no networking module is
   imported) and dynamically (opening a socket during a full solve raises).
3. No process execution, browser control or dynamic code loading.
4. No hard-coded URLs in the source.
5. No declared dependencies in ``pyproject.toml``.
6. No automatic submission of answers anywhere.
7. Source is pure ASCII and every module is documented.

What CANNOT be checked here
---------------------------
Whether the current year's competition rules permit this toolkit at all.
That is a human judgement and must be made against the published rules
before the toolkit is used in a live round.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "cipher_tool"

#: Modules whose presence would mean the toolkit can reach the outside world,
#: run other programs, or load code at runtime. None of these may appear.
FORBIDDEN_MODULES = frozenset({
    "aiohttp", "asyncio", "ctypes", "ftplib", "http", "httpx", "imaplib",
    "importlib.util", "nntplib", "poplib", "requests", "smtplib", "socket",
    "socketserver", "ssl", "subprocess", "telnetlib", "urllib", "urllib3",
    "webbrowser", "xmlrpc", "wsgiref", "selenium", "playwright", "openai",
    "anthropic", "google", "boto3", "paramiko", "pycurl",
})

#: Function names that would indicate dynamic code loading or execution.
FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "__import__"})

#: Words that would suggest automatic submission to the competition site.
SUBMISSION_WORDS = ("submit_answer", "post_answer", "upload_solution",
                    "auto_submit", "submit_to_competition")


def source_files() -> list[Path]:
    """Every Python file that ships as part of the toolkit."""
    return sorted(PACKAGE.rglob("*.py"))


def all_project_files() -> list[Path]:
    """Package source, tests and the helper scripts at the repository root."""
    return (
        source_files()
        + sorted((ROOT / "tests").rglob("*.py"))
        + [path for path in (ROOT / "conftest.py", ROOT / "run_tests.py")
           if path.exists()]
    )


def imported_modules(path: Path) -> set[str]:
    """Top-level module names imported by *path*, absolute imports only.

    Relative imports (``from .caesar import ...``) are internal and are
    deliberately not reported.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


class TestNoThirdPartyDependencies(unittest.TestCase):
    def test_every_import_is_standard_library_or_our_own(self) -> None:
        allowed = set(sys.stdlib_module_names) | {"cipher_tool", "tests"}
        offenders: dict[str, set[str]] = {}
        for path in all_project_files():
            outside = imported_modules(path) - allowed
            if outside:
                offenders[path.name] = outside
        self.assertEqual(
            offenders, {},
            "third-party imports found -- the toolkit must be standard "
            f"library only: {offenders}",
        )

    def test_pyproject_declares_no_runtime_dependencies(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("dependencies = []", text.replace(" ", " "))

    def test_no_requirements_file_sneaked_in(self) -> None:
        for name in ("requirements.txt", "Pipfile", "poetry.lock",
                     "environment.yml"):
            self.assertFalse(
                (ROOT / name).exists(),
                f"{name} exists; dependencies must stay declared in "
                "pyproject.toml and must stay empty",
            )


class TestOffline(unittest.TestCase):
    def test_no_networking_module_is_imported(self) -> None:
        offenders: dict[str, set[str]] = {}
        for path in all_project_files():
            if path.name == "test_compliance.py":
                continue  # this file names them in order to forbid them
            banned = imported_modules(path) & FORBIDDEN_MODULES
            if banned:
                offenders[path.name] = banned
        self.assertEqual(offenders, {},
                         f"forbidden modules imported: {offenders}")

    def test_no_urls_in_the_source(self) -> None:
        offenders: list[str] = []
        for path in all_project_files():
            if path.name == "test_compliance.py":
                continue
            text = path.read_text(encoding="utf-8")
            for marker in ("http://", "https://", "ftp://", "www."):
                if marker in text:
                    offenders.append(f"{path.name}: {marker}")
        self.assertEqual(offenders, [],
                         f"hard-coded addresses in source: {offenders}")

    def test_a_full_solve_never_opens_a_socket(self) -> None:
        """Dynamic proof, not just a grep over imports.

        Break every way of opening a connection, then run the whole
        cross-solver pipeline. If any code path tried to reach the network
        this test would fail with the RuntimeError below.
        """
        import socket

        def refuse(*args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "the toolkit attempted to use the network, which it must "
                "never do"
            )

        patched = {
            "socket": socket.socket,
            "create_connection": socket.create_connection,
            "getaddrinfo": socket.getaddrinfo,
            "gethostbyname": socket.gethostbyname,
        }
        for name in patched:
            setattr(socket, name, refuse)
        try:
            from cipher_tool.auto import auto_solve

            ciphertext = (
                "WKHUH LVQRW KLQJV RIDWD OWRFK DUDFW HUDVK DOIIL QLVKH "
                "GWDVN VDQGW KHPHV VDJHZ DVVHQ WIURP WKHKD UERXU"
            )
            result = auto_solve(ciphertext, effort="fast", top=3)
            self.assertTrue(result.candidates.ranked())
        finally:
            for name, original in patched.items():
                setattr(socket, name, original)


class TestNoRuleCircumvention(unittest.TestCase):
    def test_no_dynamic_code_execution(self) -> None:
        offenders: list[str] = []
        for path in all_project_files():
            if path.name == "test_compliance.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_CALLS:
                        offenders.append(f"{path.name}:{node.lineno} "
                                         f"{node.func.id}()")
        self.assertEqual(offenders, [],
                         f"dynamic code execution found: {offenders}")

    def test_nothing_submits_answers_anywhere(self) -> None:
        offenders: list[str] = []
        for path in all_project_files():
            if path.name == "test_compliance.py":
                continue
            text = path.read_text(encoding="utf-8").lower()
            for word in SUBMISSION_WORDS:
                if word in text:
                    offenders.append(f"{path.name}: {word}")
        self.assertEqual(offenders, [],
                         f"possible submission automation: {offenders}")

    def test_the_disclaimer_exists_and_does_not_overclaim(self) -> None:
        from cipher_tool import DISCLAIMER

        self.assertIn("locally written", DISCLAIMER)
        self.assertIn("Verify the current rules", DISCLAIMER)
        for overclaim in ("approved", "compliant", "permitted", "endorsed",
                          "official"):
            self.assertNotIn(overclaim, DISCLAIMER.lower(),
                             f"the disclaimer must not imply approval "
                             f"('{overclaim}')")


class TestSourceQuality(unittest.TestCase):
    def test_all_source_is_ascii(self) -> None:
        offenders: list[str] = []
        for path in all_project_files():
            text = path.read_text(encoding="utf-8")
            for number, line in enumerate(text.splitlines(), start=1):
                if any(ord(char) > 127 for char in line):
                    offenders.append(f"{path.name}:{number}")
        self.assertEqual(offenders, [], f"non-ASCII source lines: {offenders}")

    def test_every_module_has_a_docstring(self) -> None:
        missing = [
            path.name
            for path in source_files()
            if not ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
        ]
        self.assertEqual(missing, [], f"modules without a docstring: {missing}")

    def test_every_public_function_has_a_docstring(self) -> None:
        """Public API only: module-level functions and methods.

        Functions nested inside another function are local helpers, not part
        of anything a teammate can call, so they are not required to carry a
        docstring.
        """
        missing: list[str] = []
        for path in source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            bodies = [tree.body]
            bodies += [
                node.body for node in tree.body if isinstance(node, ast.ClassDef)
            ]
            for body in bodies:
                for node in body:
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if node.name.startswith("_"):
                        continue
                    if not ast.get_docstring(node):
                        missing.append(f"{path.name}:{node.name}")
        self.assertEqual(missing, [],
                         f"public functions without a docstring: {missing}")

    def test_documentation_files_exist(self) -> None:
        for name in ("README.md", "RULES_COMPLIANCE.md", "ALGORITHMS.md",
                     "CHANGELOG.md"):
            self.assertTrue((ROOT / name).exists(), f"{name} is missing")


if __name__ == "__main__":
    unittest.main()
