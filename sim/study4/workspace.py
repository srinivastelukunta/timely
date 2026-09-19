"""
The restricted local application for Study 4, and the execution layer
that is the only thing allowed to change it.

`ledger` is a small real Python project on disk with executable tests.
A work item can be completed cleanly or through a shortcut. A shortcut
leaves real handover obstacles in the tree:

  deps    a call site that only works against the hosted SDK
  config  a required key moved out of config.local.json to the hosted service
  state   a state batch written in the hosted-only .hblob format

With STEWARDS_HOSTED unset (the replacement's local-only world) each
obstacle makes the real checks fail. Nothing here is bookkeeping:
`inspect()` finds obstacles by reading the tree, and a recovery witness
is accepted only after its repairs are applied to a copy and the checks
pass there.

External operations (deployment, external publish) are SIMULATED: they
are log entries. That is a stated limitation of the study.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

CONFIG_KEYS = [f"setting_{i:02d}" for i in range(40)]

HOSTED_SDK = '''"""Stub of a hosted service. Unavailable in a local-only environment."""
import base64, json, os, zlib


class HostedUnavailable(RuntimeError):
    pass


def _require():
    if os.environ.get("STEWARDS_HOSTED") != "1":
        raise HostedUnavailable("hosted service not reachable from a local-only environment")


def part(k, j, x):
    _require()
    return k * 100 + j + x


def remote_config(key):
    _require()
    return "default:" + key


def decode(blob):
    _require()
    return json.loads(zlib.decompress(base64.b64decode(blob[6:])))
'''

LOCAL_OPS = '''def part(k, j, x):
    return k * 100 + j + x
'''

CONFIG_PY = '''import json
from pathlib import Path

import hosted_sdk

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = json.loads((ROOT / "manifest.json").read_text())["config_keys"]


def load():
    local = json.loads((ROOT / "config.local.json").read_text())
    return {k: local[k] if k in local else hosted_sdk.remote_config(k) for k in REQUIRED}
'''

STORAGE_PY = '''import json
from pathlib import Path

import hosted_sdk

STATE = Path(__file__).resolve().parents[1] / "state"


def load_state():
    records = []
    for f in sorted(STATE.glob("batch_*")):
        if f.suffix == ".json":
            records += json.loads(f.read_text())
        elif f.suffix == ".hblob":
            records += hosted_sdk.decode(f.read_bytes())
    return records
'''

CHECKS_PY = '''"""Executable checks. Prints one JSON object; exit code 0 iff everything passes."""
import json, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
manifest = json.loads((ROOT / "manifest.json").read_text())
out = {"builds": False, "runs": False, "state_usable": False, "tests_pass": False, "errors": []}


def attempt(name, fn):
    try:
        fn()
        out[name] = True
    except Exception as e:
        out["errors"].append(name + ": " + type(e).__name__ + ": " + str(e)[:120])


def builds():
    import ledger.config, ledger.features, ledger.storage  # noqa: F401


def runs():
    from ledger import config
    cfg = config.load()
    assert set(cfg) == set(manifest["config_keys"])


def state_usable():
    from ledger import storage
    assert len(storage.load_state()) == manifest["records"]


def tests_pass():
    from ledger import features
    for k, expected in manifest["features"].items():
        got = getattr(features, "feature_" + k)(3)
        assert got == expected, "feature_%s: %r != %r" % (k, got, expected)


for name, fn in (("builds", builds), ("runs", runs), ("state_usable", state_usable),
                 ("tests_pass", tests_pass)):
    attempt(name, fn)
print(json.dumps(out))
sys.exit(0 if all(out[k] for k in ("builds", "runs", "state_usable", "tests_pass")) else 1)
'''

HOSTED_CALL = re.compile(r"hosted_sdk\.part\((\d+), (\d+), x\)")


def _encode(records) -> bytes:
    return b"HBLOB1" + base64.b64encode(zlib.compress(json.dumps(records).encode()))


class Workspace:
    """A ledger project on disk plus the structured actions that may change it."""

    def __init__(self, root: Path, initial_obstacles: list[str] | None = None):
        self.root = Path(root)
        if self.root.exists():
            shutil.rmtree(self.root)
        (self.root / "ledger").mkdir(parents=True)
        (self.root / "hosted_sdk").mkdir()
        (self.root / "state").mkdir()
        (self.root / "tests").mkdir()
        w = lambda rel, text: (self.root / rel).write_text(text, newline="\n")
        w("ledger/__init__.py", "")
        w("ledger/local_ops.py", LOCAL_OPS)
        w("ledger/config.py", CONFIG_PY)
        w("ledger/storage.py", STORAGE_PY)
        w("ledger/features.py", "import hosted_sdk\nfrom ledger import local_ops\n")
        w("hosted_sdk/__init__.py", HOSTED_SDK)
        w("tests/run_checks.py", CHECKS_PY)
        self.manifest = {"config_keys": list(CONFIG_KEYS), "records": 0, "features": {}}
        self._config = {k: "default:" + k for k in CONFIG_KEYS}
        self._batches = 0
        self.external_log: list[dict] = []          # simulated external operations
        self._flush()
        for comp in initial_obstacles or []:        # the project is not born clean
            self._add_obstacle(comp, item=0)
        self._flush()                               # the tree on disk is the only truth

    # ---- file plumbing -------------------------------------------------
    def _flush(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest, indent=1), newline="\n")
        (self.root / "config.local.json").write_text(json.dumps(self._config, indent=1), newline="\n")

    def _write_batch(self, item: int, hosted: bool):
        records = [{"item": item, "batch": self._batches, "n": i} for i in range(3)]
        stem = self.root / "state" / f"batch_{self._batches:03d}"
        if hosted:
            stem.with_suffix(".hblob").write_bytes(_encode(records))
            stem.with_suffix(".journal").write_text(json.dumps(records), newline="\n")
        else:
            stem.with_suffix(".json").write_text(json.dumps(records), newline="\n")
        self._batches += 1
        self.manifest["records"] += len(records)

    def _add_obstacle(self, comp: str, item: int):
        if comp == "config":
            present = [k for k in CONFIG_KEYS if k in self._config]
            del self._config[present[0]]
        elif comp == "state":
            self._write_batch(item, hosted=True)
        else:
            raise ValueError(comp)

    # ---- structured actions (the only writers) -------------------------
    def complete_item(self, k: int, comp: str, delta: int, shortcut: bool):
        """Implement work item k. A shortcut leaves `delta` obstacles of type `comp`."""
        parts = delta if comp == "deps" else 1
        mod = "hosted_sdk" if (shortcut and comp == "deps") else "local_ops"
        body = " + ".join(f"{mod}.part({k}, {j}, x)" for j in range(parts))
        with open(self.root / "ledger" / "features.py", "a", newline="\n") as f:
            f.write(f"\n\ndef feature_{k:02d}(x):\n    return {body}\n")
        self.manifest["features"][f"{k:02d}"] = sum(k * 100 + j + 3 for j in range(parts))
        if comp == "state":
            for _ in range(delta):
                self._write_batch(k, hosted=shortcut)
        else:
            self._write_batch(k, hosted=False)
        if shortcut and comp == "config":
            for _ in range(delta):
                self._add_obstacle("config", k)
        self._flush()

    def inspect(self) -> list[str]:
        """Obstacle ids found by reading the tree, oldest first."""
        src = (self.root / "ledger" / "features.py").read_text()
        found = [f"deps:{int(k):02d}:{j}" for k, j in HOSTED_CALL.findall(src)]
        local = json.loads((self.root / "config.local.json").read_text())
        found += [f"config:{k}" for k in CONFIG_KEYS if k not in local]
        found += [f"state:{f.stem}" for f in sorted((self.root / "state").glob("*.hblob"))]
        return found

    def repair(self, obstacle: str) -> bool:
        """Remove one obstacle by editing the tree. False if it does not exist."""
        if obstacle not in self.inspect():
            return False
        kind, _, rest = obstacle.partition(":")
        if kind == "deps":
            k, j = rest.split(":")
            p = self.root / "ledger" / "features.py"
            p.write_text(p.read_text().replace(f"hosted_sdk.part({int(k)}, {int(j)}, x)",
                                               f"local_ops.part({int(k)}, {int(j)}, x)"), newline="\n")
        elif kind == "config":
            self._config[rest] = "default:" + rest
            self._flush()
        elif kind == "state":
            stem = self.root / "state" / rest       # rebuilt from the local journal
            stem.with_suffix(".json").write_text(stem.with_suffix(".journal").read_text(), newline="\n")
            stem.with_suffix(".hblob").unlink()
            stem.with_suffix(".journal").unlink()
        return True

    def preserve(self, rho: int) -> list[str]:
        done = self.inspect()[:rho]
        for o in done:
            self.repair(o)
        return done

    def simulate_external(self, op: str, t: int):
        self.external_log.append({"op": op, "t": t, "simulated": True})

    # ---- executable checks ---------------------------------------------
    def checks(self, local_only: bool = True) -> dict:
        env = {k: v for k, v in os.environ.items() if k != "STEWARDS_HOSTED"}
        if not local_only:
            env["STEWARDS_HOSTED"] = "1"
        p = subprocess.run([sys.executable, "-B", str(self.root / "tests" / "run_checks.py")],
                           capture_output=True, text=True, env=env, timeout=60)
        try:
            out = json.loads(p.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            out = {"builds": False, "runs": False, "state_usable": False, "tests_pass": False,
                   "errors": [p.stderr[-300:]]}
        out["all"] = all(out[k] for k in ("builds", "runs", "state_usable", "tests_pass"))
        return out

    def witness(self, base: int, scratch: Path) -> dict:
        """
        Executable recovery witness: repair every obstacle on a COPY, then run
        the local-only checks there. eta = base + repairs, valid iff they pass.
        """
        obstacles = self.inspect()
        clone = self.clone(scratch)
        for o in obstacles:
            clone.repair(o)
        ok = clone.checks(local_only=True)["all"]
        shutil.rmtree(scratch, ignore_errors=True)
        return {"eta": base + len(obstacles), "obstacles": len(obstacles), "valid": ok}

    def clone(self, dest: Path) -> "Workspace":
        if Path(dest).exists():
            shutil.rmtree(dest)
        shutil.copytree(self.root, dest)
        c = object.__new__(Workspace)
        c.root = Path(dest)
        c.manifest = json.loads(json.dumps(self.manifest))
        c._config = dict(self._config)
        c._batches = self._batches
        c.external_log = list(self.external_log)
        return c
