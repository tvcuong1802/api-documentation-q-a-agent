"""Knowledge-base loading — bundled data only, for now.

The template ships its own knowledge data and always uses it. There is deliberately
no operator-supplied channel yet: how an adopting organisation plugs in its own data
is a platform-level convention, and defining it is CoE's call, not one template's.
Inventing a private convention here would mean sixty templates each inventing a
different one, and all sixty needing rework once the real one lands.

What this module does provide is the seam. `load_kb` already returns *where* the data
came from, so the day a channel exists it is one branch in one function rather than a
change spread across every node that reads knowledge data.

The origin travels with the data on purpose. A template quietly running on its own
sample data looks exactly like one running on real data — that indistinguishability
is the whole failure this work is about, so callers surface `describe()` in the
output rather than letting the reader assume.

── When the channel arrives ────────────────────────────────────────────────────
Constraints already measured, so whoever wires it does not rediscover them:

  * The Admin console offers environment variables only — no file upload, no volume,
    no configmap. The Pod's filesystem holds what the image holds, nothing else.
  * Values must be read through `ctx.secrets` / the SecretProvider, never
    `os.environ` directly (README §Initialization; the framework contract).
  * ORDER: the runner calls `compile()` BEFORE `provision_secrets()`
    (marketplace_app.py:96 then :107). `register_nodes()` therefore hands each node
    its own copy of the data before any provider exists — verified by running it, a
    reload afterwards leaves the nodes on the old copy. So a provider-sourced KB is
    not just a change to this function: the consuming nodes must read lazily too.
    Worth raising with CoE when the channel is designed, because it constrains any
    mechanism that arrives after compile time.
  * An env var carries ~32KB comfortably (measured against this template's own
    pattern set). A RAG-scale corpus does not fit, and no channel for one exists.

The shape it would take:

    raw = ctx.secrets.get(env_var(name))      # what the operator configured
    if raw:
        return KB(_parse(raw), "configured", name)
    return KB(_parse(bundled_path.read_text()), "bundled", name)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple


class KB(NamedTuple):
    """Loaded knowledge data plus where it came from."""

    data: dict[str, Any]
    origin: str  # "bundled" today; "configured" once a channel exists
    name: str

    @property
    def is_bundled(self) -> bool:
        return self.origin == "bundled"

    def describe(self) -> str:
        """One line for the output, so the reader knows what the answer rests on.

        States where the data came from, not how good it is. This template's own
        pattern set cites CAA and PMDA guidance with a revision date — calling that
        "sample data" would be false. Whether a given set is current and right for
        the reader's jurisdiction is their judgement, and the line invites it rather
        than making it for them.
        """
        if self.origin == "bundled":
            return (
                f"Reference set: `{self.name}`, bundled with this template. "
                "Confirm it is current and applicable before relying on the result."
            )
        return f"Reference set: `{self.name}`, supplied by configuration."


def env_var(name: str) -> str:
    """`keihin_prohibited_patterns` -> `AGENT_KB_KEIHIN_PROHIBITED_PATTERNS`.

    Unused while there is no channel, and kept so the naming is settled in one place
    instead of being argued per template later.
    """
    return "AGENT_KB_" + name.upper().replace("-", "_")


def _parse(text: str) -> dict[str, Any]:
    """Accept JSON or YAML — JSON is a subset, so one parser covers both.

    Parse failures are re-raised as ValueError: yaml.YAMLError is not one, so a
    caller catching ValueError alone would let a raw parser traceback escape.
    """
    # importlib, not `import yaml`: PyYAML ships no type stubs, so `mypy src/` in CI
    # fails with "Library stubs not installed for yaml" on every repo whose dev extras
    # omit types-PyYAML. Adding the stub to each pyproject fixes it once per repo; not
    # needing it fixes it once here. Measured across the fleet 2026-09-03.
    import importlib  # noqa: PLC0415

    yaml = importlib.import_module("yaml")

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(str(exc).splitlines()[0]) from exc
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping at the top level, got {type(data).__name__}")
    return data


def load_kb(name: str, bundled_path: Path, provider: Any = None) -> KB:
    """Load `name` from the data the template ships.

    `provider` is accepted and not yet used. It is the `SecretProvider` the runner
    hands to `provision_secrets()` — a `ChainedSecretProvider(EnvProvider, Dotenv‑
    Provider)`, i.e. the only channel through which the platform delivers per-agent
    configuration today. Taking it now costs one parameter; leaving it out costs a
    call-site edit in every template on the day a channel is defined, and there are
    sixty of them.

    When that day comes the change is confined to this function:

        raw = provider.get(env_var(name)) if provider else None
        if raw:
            return KB(_parse(raw), "configured", name)

    Raises rather than returning empty when the file is missing. An agent whose
    knowledge data is absent cannot answer its question, and returning `{}` is what
    turns "nothing was checked" into "nothing was found".
    """
    if not bundled_path.exists():
        raise FileNotFoundError(
            f"No knowledge data for `{name}`: {bundled_path} is missing. "
            "Without it this agent has nothing to check against."
        )
    return KB(_parse(bundled_path.read_text(encoding="utf-8")), "bundled", name)
