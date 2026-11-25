"""Runner mínimo compartido: cada archivo de tests se ejecuta con python tests/<archivo>.py."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(namespace: dict) -> int:
    tests = [value for name, value in sorted(namespace.items()) if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - un test que explota es un fallo, no aborta el resto
            failures += 1
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            print(f"FAIL {test.__name__}: {detail}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests pasaron")
    return 1 if failures else 0
