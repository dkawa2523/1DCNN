"""
Lightweight configuration loader with optional PyYAML dependency.

If PyYAML is available it is used for full YAML support.  Otherwise a
minimal indentation-based parser is employed that understands the
subset of YAML used in this project (nested mappings, lists, numbers,
booleans and strings).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token.startswith(("'", '"')) and token.endswith(("'", '"')) and len(token) >= 2:
        return token[1:-1]
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in token or "e" in token.lower():
            return float(token)
        return int(token, 10)
    except ValueError:
        return token


def _prepare_lines(text: str) -> List[Tuple[int, str]]:
    prepared: List[Tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        prepared.append((indent, content))
    return prepared


def _parse_block(lines: List[Tuple[int, str]], start: int, base_indent: int) -> Tuple[int, Any]:
    mapping: dict[str, Any] = {}
    sequence: List[Any] | None = None
    index = start

    while index < len(lines):
        indent, content = lines[index]
        if indent < base_indent:
            break

        if content.startswith("- "):
            if mapping:
                raise ValueError("Mixed mapping and sequence not supported in this minimal parser.")
            if sequence is None:
                sequence = []
            value_str = content[2:].strip()
            if value_str:
                sequence.append(_parse_scalar(value_str))
                index += 1
            else:
                index, value = _parse_block(lines, index + 1, indent + 2)
                sequence.append(value)
        else:
            if sequence is not None:
                raise ValueError("Mixed sequence and mapping structure encountered.")
            if ":" not in content:
                raise ValueError(f"Invalid line: '{content}'")
            key, value_str = content.split(":", 1)
            key = key.strip()
            value_str = value_str.strip()
            if value_str:
                mapping[key] = _parse_scalar(value_str)
                index += 1
            else:
                index, value = _parse_block(lines, index + 1, indent + 2)
                mapping[key] = value

    if sequence is not None:
        return index, sequence
    return index, mapping


def simple_yaml_load(text: str) -> Any:
    lines = _prepare_lines(text)
    _, data = _parse_block(lines, 0, 0)
    return data


def load_config(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return simple_yaml_load(text)
    else:
        return yaml.safe_load(text)


def dump_config(data: Any, path: str | Path) -> None:
    path = Path(path)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        import json
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
