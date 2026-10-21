"""Write the API's OpenAPI spec to openapi.json (the committed contract)."""

import json
from pathlib import Path

from chordanalyzer.api.app import create_app
from chordanalyzer.settings import Settings

out = Path(__file__).resolve().parents[1] / "openapi.json"
spec = create_app(Settings()).openapi()
out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out}")
