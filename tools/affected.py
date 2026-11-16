"""Run the tests that the files changed since the last commit can break, and no others.

Usage: `make test`, or `uv run python tools/affected.py [--list] [--jobs N] [-- pytest arguments]`.
`--list` prints the test files and runs nothing.

The answer comes from the imports, read and not run. A test file depends on each program module it
imports, on each module that one imports in turn, and on the same for each helper it takes from another
test file. A changed module selects the test files that depend on it.

**In a test file every import counts**, an import inside a function too. So does a text equal to a
module's name, since the screens are imported by name from a list. A helper counts for only the modules
that the helper itself uses: a test that takes `StubPage` from the harness does not depend on every
screen the harness can build.

**Between program modules only an import at the top of the file counts.** The imports inside functions
are the jumps from one screen to another (a screen opens the device screen, which can open every
screen), and with them each module reaches all the others and each change selects every test. So a
fault that crosses such a jump is found by the tests of the module that changed, or by `make test ALL=1`
before the commit. A change with no import between it and a test is not seen either: a data file, a
setting read from the environment.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
EVERYTHING = {"pyproject.toml", "uv.lock", "tests/conftest.py"}
"""A change to one of these can change any test."""
PARALLEL_FROM = 5
"""From this many test files the run uses every performance core. Below it, starting the workers costs
more than it saves."""


def program_modules() -> dict[str, Path]:
    """Every program module by its dotted name, and the tools by their file name."""
    found = {}
    for src in sorted(ROOT.glob("packages/*/src")):
        for path in sorted(src.rglob("*.py")):
            parts = list(path.relative_to(src).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            found[".".join(parts)] = path
    for path in sorted((ROOT / "tools").glob("*.py")):
        found[path.stem] = path
    return found


MODULES = program_modules()
PACKAGES = {name for name, path in MODULES.items() if path.name == "__init__.py"}
BY_LAST_NAME: dict[str, set[str]] = {}
for _name in MODULES:
    BY_LAST_NAME.setdefault(_name.rpartition(".")[2], set()).add(_name)
TEST_FILES = {path.stem: path for path in sorted(TESTS.glob("*.py"))}


@lru_cache(maxsize=None)
def tree_of(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def longest_module(dotted: str) -> str | None:
    """`nfb.dsp.envelope.design_cfir` is the module `nfb.dsp.envelope`."""
    parts = dotted.split(".")
    while parts:
        if ".".join(parts) in MODULES:
            return ".".join(parts)
        parts.pop()
    return None


def imported(node: ast.AST, here: str = "",
             literals: bool = True) -> tuple[set[str], set[tuple[str, str | None]]]:
    """What the statements under `node` import: program modules, and `(test file, name)` pairs. A name of
    None is the whole test file. `here` is the importing module's dotted name, for a relative import.
    With `literals`, a text equal to a module's name is an import of it."""
    programs: set[str] = set()
    tests: set[tuple[str, str | None]] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import):
            for alias in sub.names:
                if alias.name in TEST_FILES:
                    tests.add((alias.name, None))
                elif (module := longest_module(alias.name)):
                    programs.add(module)
        elif isinstance(sub, ast.ImportFrom):
            base = sub.module or ""
            if sub.level:
                package = here.split(".") if here in PACKAGES else here.split(".")[:-1]
                package = package[:len(package) - (sub.level - 1)]
                base = ".".join(package + ([base] if base else []))
            if base in TEST_FILES and not sub.level:
                tests.update((base, alias.name) for alias in sub.names)
                continue
            for alias in sub.names:
                module = longest_module(f"{base}.{alias.name}") if base else longest_module(alias.name)
                if module:
                    programs.add(module)
        elif literals and isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            programs.update(BY_LAST_NAME.get(sub.value, ()))
    return programs, tests


def imported_at_the_top(module: str) -> set[str]:
    """The program modules that `module` imports when it loads: not those a function imports when called."""
    found: set[str] = set()
    for node in tree_of(MODULES[module]).body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.If, ast.Try)):
            found |= imported(node, module, literals=False)[0]
    return found


@lru_cache(maxsize=None)
def reaches(module: str) -> frozenset[str]:
    """The module and every program module that loads with it, however far down."""
    seen, queue = {module}, [module]
    while queue:
        for found in imported_at_the_top(queue.pop()):
            if found not in seen:
                seen.add(found)
                queue.append(found)
    return frozenset(seen)


def top_level(tree: ast.Module) -> dict[str, list[ast.stmt]]:
    """Each name a test file binds at its top level, with the statements that bind it."""
    names: dict[str, list[ast.stmt]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound = [node.name]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound = [(alias.asname or alias.name).split(".")[0] for alias in node.names]
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound = [n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)]
        else:
            continue
        for name in bound:
            names.setdefault(name, []).append(node)
    return names


def name_needs(stem: str, name: str | None, seen: set | None = None) -> tuple[set[str], set[str]]:
    """What one name of a test file needs: program modules, and test files. With no name, the whole file.

    A name needs what its own statements import, and what every other top-level name they use needs.
    """
    seen = set() if seen is None else seen
    if (stem, name) in seen:
        return set(), set()
    seen.add((stem, name))
    tree = tree_of(TEST_FILES[stem])
    names = top_level(tree)
    if name is None:
        nodes: list[ast.AST] = [tree]
    else:
        nodes = list(names.get(name, []))
        queue = list(nodes)
        while queue:
            for sub in ast.walk(queue.pop()):
                if isinstance(sub, ast.Name):
                    for other in names.get(sub.id, []):
                        if other not in nodes:
                            nodes.append(other)
                            queue.append(other)
    programs, files = set(), {stem}
    for node in nodes:
        found, from_tests = imported(node)
        if isinstance(node, ast.ImportFrom) and name is not None and node.module in TEST_FILES:
            # `from harness import a, b` reached through one name: only that name.
            from_tests = {(node.module, alias.name) for alias in node.names
                          if (alias.asname or alias.name) in _used_names(nodes, names) | {name}}
        programs |= found
        for other, wanted in from_tests:
            more_programs, more_files = name_needs(other, wanted, seen)
            programs |= more_programs
            files |= more_files
    return programs, files


def _used_names(nodes, names) -> set[str]:
    return {sub.id for node in nodes if not isinstance(node, (ast.Import, ast.ImportFrom))
            for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def needs(stem: str) -> tuple[frozenset[str], frozenset[str]]:
    """Every program module and every test file that can change what this test file does."""
    programs, files = name_needs(stem, None)
    return frozenset(m for program in programs for m in reaches(program)), frozenset(files)


def changed_files() -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain", "-uall"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    paths = []
    for line in out.splitlines():
        paths += [part.strip().strip('"') for part in line[3:].split(" -> ")]
    return paths


def affected(changed: list[str]) -> tuple[list[Path], str]:
    """The test files to run for these changed paths, and one line that says why."""
    tests = sorted(path for stem, path in TEST_FILES.items() if stem.startswith("test_"))
    path_to_module = {str(path.relative_to(ROOT)): name for name, path in MODULES.items()}
    modules, files, unknown = set(), set(), []
    for path in changed:
        if path in EVERYTHING:
            return tests, f"{path} changed, and it can change any test"
        if path in path_to_module:
            modules.add(path_to_module[path])
        elif path.startswith("tests/") and path.endswith(".py"):
            files.add(Path(path).stem)
        elif path.startswith("packages/") and path.endswith(".py"):
            unknown.append(path)
    if unknown:
        return tests, f"{unknown[0]} is not there now, so its dependents cannot be found"
    if not modules and not files:
        return [], "no program file and no test file differs from the last commit"
    chosen = []
    for path in tests:
        programs, test_files = needs(path.stem)
        if programs & modules or test_files & files:
            chosen.append(path)
    what = sorted(modules) + sorted(f"tests/{stem}.py" for stem in files)
    return chosen, f"changed: {', '.join(what)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the test files and run nothing")
    parser.add_argument("--jobs", default="auto", help="worker processes for a run of many files")
    parser.add_argument("pytest_args", nargs="*", help="passed to pytest, after `--`")
    args = parser.parse_args(argv)

    chosen, why = affected(changed_files())
    total = sum(1 for stem in TEST_FILES if stem.startswith("test_"))
    print(f"{len(chosen)} of {total} test files ({why})", file=sys.stderr)
    if args.list or not chosen:
        for path in chosen:
            print(path.relative_to(ROOT))
        if not chosen:
            print("Nothing to run. `make test ALL=1` runs every test.", file=sys.stderr)
        return 0
    command = [sys.executable, "-m", "pytest", *args.pytest_args]
    if len(chosen) >= PARALLEL_FROM:
        command += ["-n", args.jobs, "--dist", "loadgroup"]
    command += [str(path.relative_to(ROOT)) for path in chosen]
    os.chdir(ROOT)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    sys.exit(main())
