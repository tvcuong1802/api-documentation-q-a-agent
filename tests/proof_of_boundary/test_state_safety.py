# PB-2 + PB-5: State Safety Verification
# Verifies that State contains only msgpack-safe types (no Pydantic, dataclass, JWT)

import ast
import os
import re
import pytest


CREDENTIAL_FIELD_PATTERNS = re.compile(
    r"(jwt|token|api_key|secret|password|credential|connection_string)", re.IGNORECASE
)

PROHIBITED_TYPE_ANNOTATIONS = [
    "BaseModel",
    "InvocationContext",
]


def _scan_state_file(filepath: str) -> list[str]:
    """Scan a state definition file for safety violations."""
    with open(filepath, "r") as f:
        source = f.read()
        tree = ast.parse(source, filename=filepath)

    violations = []

    for node in ast.walk(tree):
        # Check class definitions that look like State
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    field_name = item.target.id

                    # Check for credential-like field names. Boolean detection/result
                    # flags (e.g. credential_leak_detected) store no secret value — they
                    # are the OUTPUT of a security scan — so they are not violations.
                    _FLAG_SUFFIXES = (
                        "_detected",
                        "_detection",
                        "_enabled",
                        "_disabled",
                        "_check",
                        "_required",
                        "_violation",
                        "_blocked",
                    )
                    is_flag = field_name.endswith(_FLAG_SUFFIXES) or (
                        isinstance(item.annotation, ast.Name) and item.annotation.id == "bool"
                    )
                    if CREDENTIAL_FIELD_PATTERNS.search(field_name) and not is_flag:
                        violations.append(f"{filepath}:{item.lineno} — Credential-like field name: {field_name}")

                    # Check for prohibited type annotations
                    if item.annotation:
                        annotation_str = ast.dump(item.annotation)
                        for prohibited in PROHIBITED_TYPE_ANNOTATIONS:
                            if prohibited in annotation_str:
                                violations.append(f"{filepath}:{item.lineno} — Prohibited type in State: {prohibited}")

                        # Flat-TypedDict rule: no nested list[dict[...]] in State.
                        # Such fields are not reliably msgpack-safe (Any-typed values
                        # can corrupt the checkpoint); they MUST be JSON-serialized to
                        # str via src.utils.serde. See docs/04_security_review.md.
                        if _is_nested_list_of_dict(item.annotation):
                            violations.append(
                                f"{filepath}:{item.lineno} — Nested list[dict] in State "
                                f"(use str + src.utils.serde): {field_name}"
                            )

    return violations


def _is_nested_list_of_dict(annotation: ast.expr) -> bool:
    """Return True for annotations shaped like list[dict] / list[dict[...]]."""
    if not (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "list"
    ):
        return False
    inner = annotation.slice
    if isinstance(inner, ast.Name):
        return inner.id == "dict"
    if isinstance(inner, ast.Subscript) and isinstance(inner.value, ast.Name):
        return inner.value.id == "dict"
    return False


class TestStateSafety:
    """PB-2/PB-5: State must be msgpack-safe with no credentials."""

    def test_state_file_safety(self):
        """State definition must not contain credential fields or prohibited types."""
        state_file = os.path.join(os.path.dirname(__file__), "..", "..", "src", "schemas", "state.py")
        if not os.path.exists(state_file):
            pytest.skip("src/schemas/state.py not found")

        violations = _scan_state_file(state_file)

        assert violations == [], "State safety violations found:\n" + "\n".join(violations)
