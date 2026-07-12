from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

src_str = str(SRC)
if src_str in sys.path:
    sys.path.remove(src_str)
sys.path.insert(0, src_str)

for module_name in list(sys.modules):
    if module_name == "tokencode_sdk" or module_name.startswith("tokencode_sdk."):
        sys.modules.pop(module_name)
