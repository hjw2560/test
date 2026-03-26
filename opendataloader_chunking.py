"""
OpenDataLoader chunking: structure-aware chunk creation from OpenDataLoader JSON + PDF.

Used when config.PDF_PARSER=opendataloader.
Returns the same base chunk format as the Docling path:
    {
        "text": str,
        "pages": list[int],
        "page_bboxes": dict[int, {"l": float, "b": float, "r": float, "t": float}],
        "file_name": str,
    }

Production-oriented principles:
  - Keep chunks small and retrieval-friendly, but always carry section context.
  - Split by document structure first: heading, paragraph, list, table, image/caption.
  - Convert tables into logical row/group chunks instead of one huge blob.
  - Preserve exact lexical values for BM25 while keeping semantic context for vector search.
  - Avoid mandatory LLM calls during ingest; image captioning stays optional.
"""

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import pypdfium2 as pdfium
from docling.utils.locks import pypdfium2_lock
from qwen_vl_utils import process_vision_info

import logging_config
from models.doc_file_model import DocFile
from services.chunking_service import union_bbox

logger = logging_config.get_logger(__name__)

Y_TOLERANCE = 3.0
INDENT_THRESHOLD = 10.0
HEADER_SCAN_MAX_ROWS = 3
MIN_SPLIT_BODY_SIZE = 220


# ── 공통 텍스트/메타 헬퍼 ─────────────────────────────────────────────────────

def _normalize_inline_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _normalize_multiline_text(text: Optional[str]) -> str:
    if not text:
        return ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line)


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _bbox_to_dict(bbox: Any) -> Optional[Dict[str, float]]:
    if isinstance(bbox, list) and len(bbox) >= 4:
        return {"l": bbox[0], "b": bbox[1], "r": bbox[2], "t": bbox[3]}
    if isinstance(bbox, dict):
        return bbox
    return None


def _page_no_to_int(page_no: Any) -> Any:
    if isinstance(page_no, float) and page_no.is_integer():
        return int(page_no)
    if isinstance(page_no, int):
        return page_no
    return page_no


def _prov_from_element(page_no: Any, bbox: Any) -> List[Dict[str, Any]]:
    bbox_dict = _bbox_to_dict(bbox)
    if page_no is None or not bbox_dict:
        return []
    return [{"page_no": _page_no_to_int(page_no), "bbox": bbox_dict}]


def _merge_pages_and_bboxes(provs: List[List[Dict[str, Any]]]) -> tuple:
    pages_bboxes: Dict[Any, List[Dict[str, float]]] = {}
    for prov_list in provs:
        if not prov_list:
            continue
        for prov in prov_list:
            if not isinstance(prov, dict):
                continue
            page_no = prov.get("page_no") or prov.get("page number")
            bbox = prov.get("bbox")
            if page_no is None or not bbox:
                continue
            pages_bboxes.setdefault(page_no, []).append(bbox)

    page_union = {}
    for page_no, boxes in pages_bboxes.items():
        if boxes:
            page_union[page_no] = union_bbox(boxes)

    def _sort_key(v: Any):
        if isinstance(v, int):
            return (0, v)
        return (1, str(v))

    pages = sorted(pages_bboxes.keys(), key=_sort_key)
    return pages, page_union


def _split_text_with_overlap(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """
    길이가 긴 단일 본문을 문단/문장 경계 우선으로 분할합니다.
    경계 탐색이 실패하면 안전하게 문자 기준 분할로 폴백합니다.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    windows = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(text_len, start + max_chars)

        if end < text_len:
            split_candidates = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("; ", start, end),
                text.rfind(": ", start, end),
                text.rfind(", ", start, end),
                text.rfind(" ", start, end),
            ]
            best = max(split_candidates)
            if best > start + int(max_chars * 0.6):
                if text[best:best + 2] in ("\n\n", ". ", "; ", ": ", ", "):
                    end = best + 1
                else:
                    end = best

        chunk = text[start:end].strip()
        if chunk:
            windows.append(chunk)

        if end >= text_len:
            break

        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        while next_start < text_len and text[next_start].isspace():
            next_start += 1
        start = next_start

    return windows


def _format_chunk_text(prefix_lines: List[str], body: str) -> str:
    prefix = "\n".join(line.strip() for line in prefix_lines if line and line.strip()).strip()
    body = body.strip()
    if prefix and body:
        return f"{prefix}\n\n{body}"
    return prefix or body


def _truncate_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _bool_setting(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_setting(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_runtime_setting(name: str, default: Any = None) -> Any:
    try:
        from django.conf import settings as django_settings  # type: ignore
        return getattr(django_settings, name, default)
    except Exception:
        return os.getenv(name, default)


# ── 테이블 파싱 헬퍼 ──────────────────────────────────────────────────────────

def _get_y(kid: dict) -> float:
    bb = kid.get("bounding box", [])
    return bb[1] if len(bb) >= 2 else 0.0


def _cell_kids(cell: dict) -> list:
    kids = [k for k in cell.get("kids", []) if _normalize_inline_text(k.get("content", ""))]
    return sorted(kids, key=lambda k: -_get_y(k))


def _cell_text(cell: dict) -> str:
    parts = [_normalize_inline_text(k.get("content", "")) for k in _cell_kids(cell)]
    return "\n".join(part for part in parts if part)


def _is_count(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?\*?", _normalize_inline_text(text)))


def _is_zero_bbox(cell: dict) -> bool:
    bb = cell.get("bounding box", [])
    if len(bb) < 4:
        return True
    return bb[0] == bb[2] and bb[1] == bb[3]


def _row_texts(cells: List[dict], forward_fill_merged: bool = False) -> List[str]:
    values = []
    last_non_empty = ""
    for cell in cells:
        value = _cell_text(cell)
        if not value and forward_fill_merged and _is_zero_bbox(cell) and last_non_empty:
            value = last_non_empty
        if value:
            last_non_empty = value
        values.append(value)
    return values


def _looks_like_header_label(text: str) -> bool:
    text = _normalize_inline_text(text)
    if not text:
        return False
    if len(text) > 80:
        return False
    if re.search(r"[.!?]$", text):
        return False
    return len(text.split()) <= 8


def _infer_header_row_count(rows: List[dict]) -> int:
    """
    멀티헤더 테이블을 위한 헤더 row 개수 추정.

    대표 패턴:
      row1: Location | No. | Combination | "" | ""
      row2: "" | "" | Mooring drum | Warping head | Remark
    """
    if not rows:
        return 0
    if len(rows) == 1:
        return 1

    header_count = 1
    max_scan = min(len(rows), HEADER_SCAN_MAX_ROWS)
    prev_cells = rows[0].get("cells", [])

    for idx in range(1, max_scan):
        cells = rows[idx].get("cells", [])
        texts = _row_texts(cells, forward_fill_merged=False)
        non_empty = [t for t in texts if t]
        if not non_empty:
            continue

        numeric_ratio = sum(1 for t in non_empty if _is_count(t)) / max(1, len(non_empty))
        merged_like = any(_is_zero_bbox(c) and not _cell_text(c) for c in cells)
        previous_row_has_merge_gap = any(_is_zero_bbox(c) and not _cell_text(c) for c in prev_cells)
        short_label_ratio = sum(1 for t in non_empty if _looks_like_header_label(t)) / max(1, len(non_empty))
        leading_empty = 0
        for value in texts:
            if value:
                break
            leading_empty += 1

        is_header_continuation = (
            numeric_ratio <= 0.25
            and (
                merged_like
                or previous_row_has_merge_gap
                or leading_empty >= 1
                or short_label_ratio >= 0.8
            )
        )

        if not is_header_continuation:
            break

        header_count += 1
        prev_cells = cells

    return max(1, min(header_count, len(rows)))


def _compose_headers(header_rows: List[dict]) -> List[str]:
    layered_rows = []
    max_cols = 0

    for row in header_rows:
        cells = row.get("cells", [])
        row_values = _row_texts(cells, forward_fill_merged=True)
        layered_rows.append(row_values)
        max_cols = max(max_cols, len(row_values))

    headers = []
    for col_idx in range(max_cols):
        parts = []
        for row_values in layered_rows:
            value = _normalize_inline_text(row_values[col_idx] if col_idx < len(row_values) else "")
            if value and (not parts or parts[-1] != value):
                parts.append(value)
        headers.append(" > ".join(parts) if parts else f"col{col_idx + 1}")
    return headers


def _flatten_cell_for_line(cell: dict) -> str:
    return _normalize_inline_text(_cell_text(cell).replace("\n", " / "))


def _row_non_empty_texts(row: dict) -> List[str]:
    return [value for value in _row_texts(row.get("cells", []), forward_fill_merged=False) if value]


def _is_property_sheet_table(rows: List[dict]) -> bool:
    """
    세로형 property/value 사양 표 감지.

    전형 패턴:
      row1: Quantity | One (1) | ""
      row2: Type | Self-contained... | ""
      row6: Speed | Hoisting | Appx. 10 m/min...
      row7: ""    | Slewing  | Manufacturer's standard

    즉 첫 컬럼이 속성명 역할을 많이 하고, 값이 뒤 컬럼에 세로로 이어집니다.
    """
    if len(rows) < 4:
        return False

    first_col_label_rows = 0
    continued_label_rows = 0
    three_value_rows = 0
    total_considered = 0

    for row in rows:
        cells = row.get("cells", [])
        if not cells:
            continue
        total_considered += 1
        col1 = _cell_text(cells[0]) if len(cells) > 0 else ""
        col2 = _cell_text(cells[1]) if len(cells) > 1 else ""
        col3 = _cell_text(cells[2]) if len(cells) > 2 else ""

        if col1 and col2:
            first_col_label_rows += 1
        if (not col1) and col2 and col3:
            continued_label_rows += 1
        if col1 and col2 and col3:
            three_value_rows += 1

    if total_considered == 0:
        return False

    return (
        first_col_label_rows >= max(3, total_considered // 3)
        and (continued_label_rows >= 1 or three_value_rows >= 1)
    )


def _property_sheet_header_count(rows: List[dict]) -> int:
    """
    property sheet는 보통 별도 헤더가 없으므로 기본 0.
    다만 첫 row가 명백히 title/header처럼 보이면 1만 허용.
    """
    if not rows:
        return 0
    first_texts = _row_non_empty_texts(rows[0])
    if len(first_texts) >= 2:
        return 0
    if first_texts and _looks_like_header_label(first_texts[0]):
        return 1
    return 0


def _table_stats(rows: List[dict]) -> Dict[str, int]:
    total_cells = 0
    merged_cells = 0
    multiline_cells = 0
    non_empty_cells = 0
    max_cols = 0

    for row in rows:
        cells = row.get("cells", [])
        max_cols = max(max_cols, len(cells))
        for cell in cells:
            total_cells += 1
            if _is_zero_bbox(cell):
                merged_cells += 1
            value = _cell_text(cell)
            if value:
                non_empty_cells += 1
            if len(_cell_kids(cell)) > 1 or "\n" in value:
                multiline_cells += 1

    return {
        "row_count": len(rows),
        "column_count": max_cols,
        "total_cells": total_cells,
        "merged_cells": merged_cells,
        "multiline_cells": multiline_cells,
        "non_empty_cells": non_empty_cells,
        "empty_cells": max(0, total_cells - non_empty_cells),
    }


def _is_generic_header(header: str) -> bool:
    return bool(re.fullmatch(r"col\d+", _normalize_inline_text(header), re.IGNORECASE))


def _headers_are_weak(headers: List[str]) -> bool:
    if not headers:
        return True
    meaningful = [header for header in headers if header and not _is_generic_header(header)]
    return len(meaningful) < max(1, len(headers) // 2)


def _linearize_table_rows(rows: List[dict]) -> List[Dict[str, Any]]:
    linear_rows = []
    carry_values: List[str] = []

    for row in rows:
        cells = row.get("cells", [])
        if len(carry_values) < len(cells):
            carry_values.extend([""] * (len(cells) - len(carry_values)))

        values = []
        has_value = False
        for idx, cell in enumerate(cells):
            value = _flatten_cell_for_line(cell)
            if not value and _is_zero_bbox(cell) and idx < len(carry_values):
                value = carry_values[idx]
            if value:
                carry_values[idx] = value
                has_value = True
            values.append(value)

        if has_value:
            linear_rows.append({
                "row_number": row.get("row number"),
                "values": values,
            })

    return linear_rows


def _table_layout_family(table_type: str, headers: List[str], stats: Dict[str, int]) -> str:
    if table_type == "property_sheet":
        return "property_sheet"
    if table_type == "hierarchical":
        return "hierarchical"
    if table_type == "subrow":
        return "grouped_list"
    if table_type == "count":
        return "matrix"
    if _headers_are_weak(headers):
        return "freeform_layout"
    if stats.get("merged_cells", 0) > max(1, stats.get("total_cells", 0) // 4):
        return "grouped_list"
    return "matrix"


def _table_info_lines(simple: Dict[str, Any], content_type: str) -> List[str]:
    stats = simple.get("stats", {})
    return [
        f"content_type={content_type}",
        f"table_layout={simple.get('layout_family', simple.get('type', 'unknown'))}",
        f"table_shape={stats.get('row_count', 0)}x{stats.get('column_count', 0)}",
    ]


def _to_field_key(text: str) -> str:
    text = _normalize_inline_text(text).lower()
    text = text.replace(">", " ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "field"


def _header_key_map(headers: List[str]) -> Dict[str, str]:
    used = set()
    mapping = {}
    for idx, header in enumerate(headers):
        base = _to_field_key(header if header and not _is_generic_header(header) else f"col_{idx + 1}")
        key = base
        suffix = 2
        while key in used:
            key = f"{base}_{suffix}"
            suffix += 1
        used.add(key)
        mapping[header] = key
    return mapping


def _kv_value(value: Any) -> str:
    value = _normalize_multiline_text("" if value is None else str(value))
    return value.replace("\n", " ; ")


def _join_non_empty(parts: List[str], sep: str = " ; ") -> str:
    return sep.join([part for part in parts if _normalize_inline_text(part)])


def _humanize_key(key: str) -> str:
    key = _normalize_inline_text(key).replace("_", " ").strip()
    return key[:1].upper() + key[1:] if key else "Value"


def _property_sheet_narrative_chunks(simple: Dict[str, Any]) -> List[str]:
    chunks = []
    for record in simple.get("records", []):
        values = record.get("values", {})
        if not values:
            continue
        for key, value in values.items():
            value = _kv_value(value)
            if not value:
                continue
            chunks.append(f"{_humanize_key(key)}: {value}")
    return chunks


def _generic_narrative_chunks(simple: Dict[str, Any]) -> List[str]:
    headers = simple.get("headers", [])
    key_map = _header_key_map(headers)
    chunks = []

    if simple.get("layout_family") == "freeform_layout":
        for row in simple.get("linear_rows", []):
            joined = " | ".join(value for value in row.get("values", []) if value != "")
            if joined:
                chunks.append(joined)
        return chunks

    for record in simple.get("records", []):
        values = record.get("values", {})
        parts = []
        for header in headers or list(values.keys()):
            value = _normalize_multiline_text(values.get(header, ""))
            if not value:
                continue
            field_key = key_map.get(header) or _to_field_key(header)
            parts.append(f"{_humanize_key(field_key)}: {_kv_value(value)}")
        if parts:
            chunks.append("\n".join(parts))
    return chunks


def _table_summary_narrative(simple: Dict[str, Any], raw_chunk: str) -> str:
    llm_summary = _summarize_table_via_chat_api(simple, raw_chunk)
    if llm_summary:
        return llm_summary
    heuristic = _heuristic_table_summary(simple)
    if not heuristic:
        return ""
    lines = []
    for line in heuristic.splitlines():
        line = _normalize_inline_text(line)
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "summary":
                lines.append(value)
            elif key.startswith("sample"):
                lines.append(f"Examples: {value}")
            elif key == "structure_note":
                lines.append(f"Structure note: {value}")
            elif key == "ambiguity_note":
                lines.append(f"Ambiguity note: {value}")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def _build_table_raw_chunk(simple: Dict[str, Any]) -> str:
    headers = simple.get("headers", [])
    data_lines = []
    for row in simple.get("linear_rows", []):
        joined = " | ".join(value for value in row.get("values", []) if value != "")
        if joined:
            row_no = row.get("row_number")
            key = f"row_{row_no}" if row_no is not None else "row"
            data_lines.append(f"{key}={joined}".strip())

    lines = _table_info_lines(simple, "table_raw")
    if headers:
        lines.append("headers=" + " | ".join(headers))
    if data_lines:
        lines.extend(data_lines)
    return "\n".join(line for line in lines if line).strip()


def _build_table_record_chunks(simple: Dict[str, Any]) -> List[str]:
    headers = simple.get("headers", [])
    key_map = _header_key_map(headers)
    chunks = []

    if simple.get("layout_family") == "property_sheet":
        for record in simple.get("records", []):
            values = record.get("values", {})
            lines = _table_info_lines(simple, "table_record")
            for key, value in values.items():
                if value:
                    lines.append(f"{key}={_kv_value(value)}")
            text = "\n".join(line for line in lines if line).strip()
            if text:
                chunks.append(text)
        return chunks

    if simple.get("layout_family") == "freeform_layout":
        for row in simple.get("linear_rows", []):
            joined = " | ".join(value for value in row.get("values", []) if value != "")
            if not joined:
                continue
            lines = _table_info_lines(simple, "table_layout")
            lines.append(f"layout_row={joined}")
            chunks.append("\n".join(line for line in lines if line).strip())
        return chunks

    for record in simple.get("records", []):
        values = record.get("values", {})
        lines = _table_info_lines(simple, "table_record")
        for header in headers or list(values.keys()):
            value = _normalize_multiline_text(values.get(header, ""))
            if value:
                field_key = key_map.get(header) or _to_field_key(header)
                lines.append(f"{field_key}={_kv_value(value)}")
        text = "\n".join(line for line in lines if line).strip()
        if text:
            chunks.append(text)

    return chunks


def _should_generate_table_summary(simple: Dict[str, Any]) -> bool:
    stats = simple.get("stats", {})
    if simple.get("layout_family") == "property_sheet":
        return True
    if simple.get("layout_family") == "freeform_layout":
        return True
    if simple.get("header_row_count", 0) >= 2:
        return True
    if stats.get("merged_cells", 0) >= 2:
        return True
    if stats.get("multiline_cells", 0) >= 2:
        return True
    if stats.get("row_count", 0) >= 6:
        return True
    if simple.get("type") in {"hierarchical", "subrow"}:
        return True
    return False


def _build_table_summary_prompt(simple: Dict[str, Any], raw_chunk: str) -> str:
    headers = simple.get("headers", [])
    prompt_lines = [
        "다음 표를 검색용으로 요약하라.",
        "표에 없는 내용은 추론하지 말고, 불확실하면 불확실하다고 명시하라.",
        "반드시 포함할 항목:",
        "1. 표의 목적",
        "2. 행 기준 분류",
        "3. 열 기준 분류",
        "4. 주요 수치 또는 속성",
        "5. 예외/remark",
        "6. 구조가 애매한 부분",
        "",
        "표 메타:",
        *(_table_info_lines(simple, "table_summary")),
    ]
    if headers:
        prompt_lines.append("헤더: " + " | ".join(headers))
    prompt_lines.extend([
        "",
        "표 원문:",
        _truncate_text(raw_chunk, 5000),
    ])
    return "\n".join(prompt_lines).strip()


def _summarize_table_via_chat_api(simple: Dict[str, Any], raw_chunk: str) -> str:
    use_llm_summary = _bool_setting(_get_runtime_setting("OPENDATALOADER_USE_CHAT_TABLE_SUMMARY", False), False)
    if not use_llm_summary:
        return ""

    api_root = (_get_runtime_setting("CHAT_QA_CHAT_JSON_URL", "") or "").strip()
    db_name = (_get_runtime_setting("MILVUS_DB_NAME", "") or "").strip()
    if not api_root or not db_name:
        return ""

    payload = {
        "query": _build_table_summary_prompt(simple, raw_chunk),
        "db_name": db_name,
        "top_k": 1,
        "own": "PRJT_000000",
        "grop_id": None,
        "chat_id": 0,
        "llm_params": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": _int_setting(_get_runtime_setting("OPENDATALOADER_TABLE_SUMMARY_MAX_TOKENS", 500), 500),
            "prompt": None,
        },
        "use_web_search": False,
        "fast_mode": True,
    }
    req = urllib.request.Request(
        api_root.rstrip("/") + "/api/v1/chat_json",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
            return _normalize_multiline_text((body.get("result") or "").strip())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        logger.info("OpenDataLoader table summary failed for table %s: %s", simple.get("table_id"), exc)
        return ""


def _heuristic_table_summary(simple: Dict[str, Any]) -> str:
    if simple.get("layout_family") == "property_sheet":
        summary_lines = _table_info_lines(simple, "table_summary")
        records = simple.get("records", [])
        summary_lines.append("summary=property/value 형식의 사양 표이며 첫 번째 컬럼은 속성명, 뒤 컬럼은 값 또는 하위 속성-값을 나타냅니다.")
        if records:
            sample_parts = []
            for record in records[:3]:
                values = record.get("values", {})
                display = _join_non_empty([f"{k}={v}" for k, v in values.items()])
                if display:
                    sample_parts.append(display)
            if sample_parts:
                summary_lines.append("sample=" + " || ".join(sample_parts))
        return "\n".join(summary_lines).strip()

    headers = [header for header in simple.get("headers", []) if header and not _is_generic_header(header)]
    stats = simple.get("stats", {})
    records = simple.get("records", [])
    layout_family = simple.get("layout_family", simple.get("type", "unknown"))

    summary_lines = _table_info_lines(simple, "table_summary")

    if headers:
        lead = headers[0]
        rest = ", ".join(headers[1:4]) if len(headers) > 1 else "세부 값"
        summary_lines.append(f"summary={lead} 기준으로 정리된 표이며 주요 항목은 {rest} 입니다.")
    else:
        summary_lines.append("summary=비정형 레이아웃 표이므로 원문 행 기준으로 해석하는 것이 안전합니다.")

    summary_lines.append(f"structure_note=layout={layout_family}")

    if simple.get("header_row_count", 0) >= 2 or stats.get("merged_cells", 0) > 0:
        summary_lines.append("ambiguity_note=multi_header_or_merged_cells_detected")

    if records:
        key_map = _header_key_map(simple.get("headers", []))
        sample_idx = 1
        for record in records[:2]:
            values = record.get("values", {})
            parts = []
            for header in simple.get("headers", [])[:4]:
                value = _normalize_inline_text(values.get(header, ""))
                if value:
                    field_key = key_map.get(header) or _to_field_key(header)
                    parts.append(f"{field_key}={value}")
            if parts:
                summary_lines.append(f"sample_{sample_idx}=" + " ; ".join(parts))
                sample_idx += 1

    return "\n".join(summary_lines).strip()


def _detect_count_col_idx(data_rows: list, num_cols: int) -> int:
    best_idx = -1
    best_ratio = 0.8
    for col_idx in range(1, num_cols):
        all_contents = []
        for row in data_rows:
            cells = row.get("cells", [])
            if col_idx >= len(cells):
                continue
            for kid in cells[col_idx].get("kids", []):
                content = _normalize_inline_text(kid.get("content", ""))
                if content:
                    all_contents.append(content)
        if not all_contents:
            continue
        ratio = sum(1 for content in all_contents if _is_count(content)) / len(all_contents)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = col_idx
    return best_idx


def _build_items_count(count_kids: list, item_kids: list, label: str) -> list:
    count_map = {}
    for kid in count_kids:
        value = _normalize_inline_text(kid.get("content", ""))
        if _is_count(value):
            count_map[round(_get_y(kid))] = value

    items = []
    for kid in item_kids:
        item_name = _normalize_inline_text(kid.get("content", ""))
        if not item_name:
            continue
        y = round(_get_y(kid))
        matched_count = None
        for count_y, count_value in count_map.items():
            if abs(count_y - y) <= Y_TOLERANCE:
                matched_count = count_value
                break
        if matched_count is not None:
            items.append({label: item_name, "count": matched_count})
        elif items:
            items[-1][label] = (items[-1][label].rstrip() + " " + item_name).strip()
        else:
            items.append({label: item_name})
    return items


def _find_subrow_anchor_col(cells: list) -> int:
    for idx in range(1, len(cells) - 1):
        kids_left = [k for k in cells[idx].get("kids", []) if _normalize_inline_text(k.get("content", ""))]
        kids_right = [k for k in cells[idx + 1].get("kids", []) if _normalize_inline_text(k.get("content", ""))]
        if len(kids_left) > 1 and len(kids_right) > 1:
            numeric_ratio = sum(1 for kid in kids_left if _is_count(kid.get("content", ""))) / len(kids_left)
            if numeric_ratio > 0.8:
                continue
            return idx
    return -1


def _detect_table_type(data_rows: list, num_cols: int) -> tuple:
    for row in data_rows:
        anchor = _find_subrow_anchor_col(row.get("cells", []))
        if anchor >= 0:
            return ("subrow", anchor)

    count_col = _detect_count_col_idx(data_rows, num_cols)
    if count_col >= 0:
        item_col = count_col + 1 if count_col + 1 < num_cols else count_col - 1
        return ("count", count_col, item_col)

    return ("text",)


def _expand_row_to_subrows(cells: list, headers: list, anchor_col: int, carry_values: List[str]) -> list:
    prefix_cols = {}
    for col_idx in range(0, anchor_col):
        if col_idx >= len(cells):
            value = carry_values[col_idx] if col_idx < len(carry_values) else ""
        else:
            value = _cell_text(cells[col_idx])
            if not value and _is_zero_bbox(cells[col_idx]) and col_idx < len(carry_values):
                value = carry_values[col_idx]
        prefix_cols[headers[col_idx]] = value

    anchor_kids = _cell_kids(cells[anchor_col]) if anchor_col < len(cells) else []
    if not anchor_kids:
        return []

    match_col = anchor_col + 1
    match_kids = _cell_kids(cells[match_col]) if match_col < len(cells) else []
    match_y_map = {round(_get_y(kid)): _normalize_inline_text(kid.get("content", "")) for kid in match_kids}

    shared_cols = {}
    for col_idx in range(match_col + 1, len(headers)):
        if col_idx < len(cells):
            shared_cols[headers[col_idx]] = " / ".join(
                _normalize_inline_text(kid.get("content", ""))
                for kid in _cell_kids(cells[col_idx])
                if _normalize_inline_text(kid.get("content", ""))
            )
        else:
            shared_cols[headers[col_idx]] = ""

    records = []
    for kid in anchor_kids:
        anchor_val = _normalize_inline_text(kid.get("content", ""))
        anchor_y = round(_get_y(kid))

        matched_value = ""
        best_dist = float("inf")
        for candidate_y, candidate_value in match_y_map.items():
            dist = abs(candidate_y - anchor_y)
            if dist < best_dist:
                best_dist = dist
                matched_value = candidate_value
        if best_dist > Y_TOLERANCE * 5:
            matched_value = ""

        values = dict(prefix_cols)
        values[headers[anchor_col]] = anchor_val
        if match_col < len(headers):
            values[headers[match_col]] = matched_value
        values.update(shared_cols)
        records.append(values)

    return records


def _is_hierarchical_table(data_rows: list, num_cols: int) -> bool:
    for row in data_rows:
        cells = row.get("cells", [])
        if len(cells) < 2:
            continue
        if _is_zero_bbox(cells[0]) and bool(_cell_text(cells[1])):
            return True
    return False


def _hierarchical_to_records(data_rows: list) -> list:
    records = []
    last_category = ""

    for row in data_rows:
        cells = row.get("cells", [])
        if not cells:
            continue

        col1_dummy = _is_zero_bbox(cells[0])
        category = _cell_text(cells[0]) if not col1_dummy else ""
        subcategory = _cell_text(cells[1]) if len(cells) > 1 else ""
        value = _cell_text(cells[2]) if len(cells) > 2 else ""

        if category:
            last_category = category
        if not category:
            category = last_category

        if not subcategory and not value:
            continue

        records.append({
            "row_number": row.get("row number"),
            "values": {
                "category": category,
                "subcategory": subcategory,
                "value": value,
            },
        })

    return records


def _generic_records_from_rows(data_rows: List[dict], headers: List[str]) -> List[dict]:
    records = []
    carry_values = [""] * len(headers)

    for row in data_rows:
        cells = row.get("cells", [])
        values = {}
        has_value = False

        for col_idx, header in enumerate(headers):
            cell = cells[col_idx] if col_idx < len(cells) else {}
            value = _cell_text(cell) if isinstance(cell, dict) else ""
            if not value and isinstance(cell, dict) and _is_zero_bbox(cell) and col_idx < len(carry_values):
                value = carry_values[col_idx]
            if value:
                carry_values[col_idx] = value
                has_value = True
            values[header] = value

        if has_value:
            records.append({"row_number": row.get("row number"), "values": values})

    return records


def _subrow_records_from_rows(data_rows: List[dict], headers: List[str], default_anchor_col: int) -> List[dict]:
    records = []
    carry_values = [""] * len(headers)

    for row in data_rows:
        cells = row.get("cells", [])
        if not cells:
            continue

        anchor_col = _find_subrow_anchor_col(cells)
        if anchor_col < 0:
            anchor_col = default_anchor_col

        if anchor_col >= 0 and anchor_col < len(cells):
            expanded_values = _expand_row_to_subrows(cells, headers, anchor_col, carry_values)
            if expanded_values:
                for values in expanded_values:
                    for idx in range(min(anchor_col, len(headers))):
                        header = headers[idx]
                        if values.get(header):
                            carry_values[idx] = values[header]
                        elif carry_values[idx]:
                            values[header] = carry_values[idx]
                    records.append({"row_number": row.get("row number"), "values": values})
                continue

        generic = _generic_records_from_rows([row], headers)
        if generic:
            record = generic[0]
            for idx, header in enumerate(headers):
                if record["values"].get(header):
                    carry_values[idx] = record["values"][header]
            records.append(record)

    return records


def _count_records_from_rows(data_rows: List[dict], headers: List[str], count_col: int, item_col: int) -> List[dict]:
    records = []
    category_header = headers[0] if headers else "category"
    item_header = headers[item_col] if item_col < len(headers) else f"col{item_col + 1}"
    count_header = headers[count_col] if count_col < len(headers) else "count"
    last_category = ""

    for row in data_rows:
        cells = row.get("cells", [])
        if len(cells) <= max(count_col, item_col):
            continue

        category = _cell_text(cells[0]) if cells else ""
        if not category and cells and _is_zero_bbox(cells[0]):
            category = last_category
        elif category:
            last_category = category

        items = _build_items_count(_cell_kids(cells[count_col]), _cell_kids(cells[item_col]), item_header)
        if not items:
            generic = _generic_records_from_rows([row], headers)
            records.extend(generic)
            continue

        for item in items:
            values = {
                category_header: category,
                item_header: item.get(item_header, ""),
                count_header: item.get("count", ""),
            }
            records.append({"row_number": row.get("row number"), "values": values})

    return records


def _property_sheet_records_from_rows(data_rows: List[dict]) -> List[dict]:
    records = []
    current_group_key = ""
    current_group_label = ""

    for row in data_rows:
        cells = row.get("cells", [])
        if not cells:
            continue

        col1 = _normalize_multiline_text(_cell_text(cells[0]) if len(cells) > 0 else "")
        col2 = _normalize_multiline_text(_cell_text(cells[1]) if len(cells) > 1 else "")
        col3 = _normalize_multiline_text(_cell_text(cells[2]) if len(cells) > 2 else "")

        if col1:
            current_group_label = col1
            current_group_key = _to_field_key(col1)

        if col1 and col2 and not col3:
            records.append({
                "row_number": row.get("row number"),
                "group_key": current_group_key,
                "group_label": current_group_label,
                "values": {
                    current_group_key or "property": col2,
                },
            })
            continue

        if col1 and col2 and col3:
            sub_key = _to_field_key(col2)
            property_key = _join_non_empty([current_group_key, sub_key], sep="_") or "property"
            records.append({
                "row_number": row.get("row number"),
                "group_key": current_group_key,
                "group_label": current_group_label,
                "sub_key": sub_key,
                "sub_label": col2,
                "values": {
                    property_key: col3,
                },
            })
            continue

        if not col1 and col2 and col3 and current_group_key:
            sub_key = _to_field_key(col2)
            property_key = _join_non_empty([current_group_key, sub_key], sep="_") or "property"
            records.append({
                "row_number": row.get("row number"),
                "group_key": current_group_key,
                "group_label": current_group_label,
                "sub_key": sub_key,
                "sub_label": col2,
                "values": {
                    property_key: col3,
                },
            })
            continue

        fallback_parts = []
        if col1:
            fallback_parts.append(col1)
        if col2:
            fallback_parts.append(col2)
        if col3:
            fallback_parts.append(col3)
        if fallback_parts:
            records.append({
                "row_number": row.get("row number"),
                "group_key": current_group_key,
                "group_label": current_group_label,
                "values": {
                    "property_text": " ; ".join(fallback_parts),
                },
            })

    return records


def table_to_simple(table: dict) -> dict:
    rows = table.get("rows", [])
    if not rows:
        return {}

    is_property_sheet = _is_property_sheet_table(rows)
    header_row_count = _property_sheet_header_count(rows) if is_property_sheet else _infer_header_row_count(rows)
    headers = _compose_headers(rows[:header_row_count])
    data_rows = rows[header_row_count:]

    result = {
        "table_id": table.get("id"),
        "headers": headers,
        "header_row_count": header_row_count,
        "records": [],
    }

    if not data_rows:
        return result

    if is_property_sheet:
        result["type"] = "property_sheet"
        result["records"] = _property_sheet_records_from_rows(data_rows)
        result["layout_family"] = "property_sheet"
        return result

    if _is_hierarchical_table(data_rows, len(headers)):
        result["type"] = "hierarchical"
        result["records"] = _hierarchical_to_records(data_rows)
        return result

    table_type = _detect_table_type(data_rows, len(headers))

    if table_type[0] == "subrow":
        _, anchor_col = table_type
        result["type"] = "subrow"
        result["records"] = _subrow_records_from_rows(data_rows, headers, anchor_col)
    elif table_type[0] == "count":
        _, count_col, item_col = table_type
        result["type"] = "count"
        result["records"] = _count_records_from_rows(data_rows, headers, count_col, item_col)
    else:
        result["type"] = "text"
        result["records"] = _generic_records_from_rows(data_rows, headers)

    return result


def to_milvus_chunks(table: dict) -> list:
    """
    테이블을 retrieval-friendly 청크들로 변환합니다.

    설계 포인트:
      - 전체 표를 한 덩어리로 넣지 않고, 논리 row/record 단위로 나눕니다.
      - 헤더 계층을 보존해서 BM25/semantic 검색 둘 다 잘 먹게 합니다.
      - exact token (ID, count, spec value) 이 손실되지 않도록 "Header: Value" 형태로 출력합니다.
    """
    simple = table_to_simple(table)
    if not simple:
        return []

    chunks = []

    raw_chunk = _build_table_raw_chunk(simple)
    if raw_chunk:
        chunks.append(raw_chunk)

    if simple.get("layout_family") == "property_sheet":
        chunks.extend(_property_sheet_narrative_chunks(simple))
    else:
        chunks.extend(_generic_narrative_chunks(simple))

    if _should_generate_table_summary(simple):
        summary_chunk = _table_summary_narrative(simple, raw_chunk)
        if summary_chunk:
            chunks.append(summary_chunk)

    return _dedupe_preserve_order([chunk.strip() for chunk in chunks if chunk and chunk.strip()])


def to_milvus_content(table: dict) -> str:
    return "\n\n".join(to_milvus_chunks(table))


# ── 내부 유틸 ─────────────────────────────────────────────────────────────────

def _extract_heading_level(text: str) -> Optional[int]:
    text = _normalize_inline_text(text)

    chapter_like = re.match(r"^(chapter|section|appendix)\s+([A-Z]+|\d+)\b", text, re.IGNORECASE)
    if chapter_like:
        return 1

    dotted = re.match(r"^(\d+(?:\.\d+)+)\s+\S", text)
    if dotted:
        return dotted.group(1).count(".") + 1

    coded = re.match(r"^(\d{2,})\s+\S", text)
    if coded:
        digits = coded.group(1)
        # 선박 사양서에서 자주 보이는 2/3/4자리 코드형 heading:
        # 31 -> level 2, 311 -> level 3, 3111 -> level 4
        return min(len(digits), 4)

    enumerated = re.match(r"^(\d+)[\)\-]\s+\S", text)
    if enumerated:
        return 5

    roman = re.match(r"^(?:[IVXLCM]+)\.\s+\S", text)
    if roman:
        return 1
    alpha = re.match(r"^[A-Z]\.\s+\S", text)
    if alpha:
        return 2
    return None


def _is_section_heading(text: str, font: str = "") -> bool:
    stripped = _normalize_inline_text(text)
    if not stripped:
        return False

    if _extract_heading_level(stripped) is not None:
        return True

    word_count = len(stripped.split())
    is_bold = "bold" in (font or "").lower()
    title_like = stripped == stripped.upper() or stripped == stripped.title()
    ends_like_sentence = bool(re.search(r"[.!?]$", stripped))

    return (
        is_bold
        and word_count <= 12
        and len(stripped) <= 120
        and title_like
        and not ends_like_sentence
        and len(re.findall(r"[A-Za-z]", stripped)) >= 3
    )


def _update_section_stack(section_stack: List[Dict[str, Any]], heading_text: str, level: Optional[int]) -> None:
    if level is None:
        level = section_stack[-1]["level"] if section_stack else 1
    while section_stack and section_stack[-1]["level"] >= level:
        section_stack.pop()
    section_stack.append({"level": level, "title": _normalize_inline_text(heading_text)})


def _section_path(section_stack: List[Dict[str, Any]]) -> str:
    if not section_stack:
        return ""
    titles = [item["title"] for item in section_stack if item.get("title")]
    if not titles:
        return ""
    # 너무 긴 prefix는 검색 효율과 가독성을 같이 떨어뜨리므로 최근 섹션만 유지합니다.
    return " > ".join(titles[-3:])


def _collect_elements_with_content(node: Any, out: list):
    """
    JSON 순서를 가능한 한 유지하면서 의미 있는 요소를 수집합니다.
    table 은 rows/cells 내부로 더 내려가지 않고 테이블 자체를 하나의 요소로 취급합니다.
    """
    if isinstance(node, list):
        for item in node:
            _collect_elements_with_content(item, out)
        return

    if not isinstance(node, dict):
        return

    node_type = (node.get("type") or "").lower()
    if node_type == "table":
        out.append(node)
        return

    if (
        node_type in {"picture", "image", "caption", "formula", "list", "heading", "paragraph", "text"}
        or node.get("content") is not None
        or node.get("text") is not None
    ):
        out.append(node)

    for key in ("kids", "list items", "elements", "content"):
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                _collect_elements_with_content(child, out)


def _crop_bbox_pdf_points(image, left, bottom, right, top, page_height: float, scale: float = 1.5):
    s = scale * 1.5
    upper_pt = page_height - top
    lower_pt = page_height - bottom
    return image.crop((
        max(0, left * s),
        max(0, upper_pt * s),
        min(image.width, right * s),
        min(image.height, lower_pt * s),
    ))


# ── 공개 함수: OpenDataLoader JSON → 청크 리스트 ──────────────────────────────

def parse_opendataloader_json_and_chunk(
    document: DocFile,
    json_doc_path: str,
    pdf_path: str,
    picture_crop_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    enable_image_caption: bool = True,
    qwen_model=None,
    qwen_processor=None,
):
    """
    OpenDataLoader JSON과 PDF에서 구조 기반 의미론적 청크를 생성합니다.

    핵심 변경점:
      1) 단일 current_heading 대신 section stack 유지
      2) 본문은 section 단위로 버퍼링 후 크기 기반 병합
      3) table 은 logical row/record 단위로 분리
      4) caption 은 가능한 경우 직후 table/image에 부착
      5) 큰 단일 블록만 overlap 분할
    """
    with open(json_doc_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    elements = []
    if isinstance(data, dict):
        _collect_elements_with_content(data, elements)
        if not elements:
            raw = data.get("kids", data.get("elements", data.get("content", [])))
            if isinstance(raw, list):
                _collect_elements_with_content(raw, elements)
    elif isinstance(data, list):
        _collect_elements_with_content(data, elements)

    logger.info("[STAGE 2-3] OpenDataLoader JSON 로드 - 요소 %s개 추출", len(elements))

    pdf_path = Path(pdf_path)
    picture_crop_dir = Path(picture_crop_dir)
    picture_output_dir = picture_crop_dir / pdf_path.stem
    picture_output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    text_blocks: List[Dict[str, Any]] = []
    section_stack: List[Dict[str, Any]] = []
    pending_caption: Optional[Dict[str, Any]] = None
    prev_body_bbox_x = None
    image_count = 0
    scale = 1.5

    try:
        pdf_doc = pdfium.PdfDocument(pdf_path)
    except Exception as exc:
        logger.warning("OpenDataLoader chunking: could not open PDF for images: %s", exc)
        pdf_doc = None

    def _append_chunk(body_text: str, provs: List[List[Dict[str, Any]]], prefix_lines: Optional[List[str]] = None, file_name: str = "n/a"):
        prefix_lines = prefix_lines or []
        available_body_size = max(MIN_SPLIT_BODY_SIZE, chunk_size - max(0, len("\n".join(prefix_lines)) + 2))
        body_pieces = _split_text_with_overlap(body_text, available_body_size, chunk_overlap)

        for piece in body_pieces:
            chunk_text = _format_chunk_text(prefix_lines, piece)
            pages, page_bboxes = _merge_pages_and_bboxes(provs)
            all_chunks.append({
                "text": chunk_text,
                "pages": pages,
                "page_bboxes": page_bboxes,
                "file_name": file_name,
            })

    def _flush_text_blocks():
        nonlocal text_blocks
        if not text_blocks:
            return

        section_line = _section_path(section_stack)
        current_texts = []
        current_provs: List[List[Dict[str, Any]]] = []
        current_len = 0

        for block in text_blocks:
            block_text = block["text"].strip()
            if not block_text:
                continue

            prefix_lines = [section_line] if section_line else []
            candidate_body = "\n\n".join(current_texts + [block_text]) if current_texts else block_text
            candidate_len = len(_format_chunk_text(prefix_lines, candidate_body))

            if current_texts and candidate_len > chunk_size:
                _append_chunk("\n\n".join(current_texts), current_provs, prefix_lines=prefix_lines)
                current_texts = []
                current_provs = []
                current_len = 0

            current_texts.append(block_text)
            current_provs.append(block["prov"])
            current_len += len(block_text)

            if current_len >= chunk_size:
                _append_chunk("\n\n".join(current_texts), current_provs, prefix_lines=prefix_lines)
                current_texts = []
                current_provs = []
                current_len = 0

        if current_texts:
            _append_chunk("\n\n".join(current_texts), current_provs, prefix_lines=[section_line] if section_line else [])

        text_blocks = []

    def _flush_pending_caption_as_text():
        nonlocal pending_caption
        if not pending_caption:
            return
        text_blocks.append({
            "text": pending_caption["text"],
            "prov": pending_caption["prov"],
        })
        pending_caption = None

    def _consume_pending_caption(page_no: Any) -> Optional[Dict[str, Any]]:
        nonlocal pending_caption
        if not pending_caption:
            return None
        pending_page = pending_caption.get("page_no")
        if pending_page == _page_no_to_int(page_no):
            caption = pending_caption
            pending_caption = None
            return caption
        _flush_pending_caption_as_text()
        return None

    for element in elements:
        if not isinstance(element, dict):
            continue

        element_type = (element.get("type") or "").lower()
        page_no = element.get("page number") or element.get("page_no") or 1
        page_no = _page_no_to_int(page_no)
        bbox = element.get("bounding box") or element.get("bbox")
        prov = _prov_from_element(page_no, bbox)

        if element_type not in ("paragraph", "heading", "text", "list", "table", "formula", "caption", "picture", "image", ""):
            continue

        # caption 은 직후 자산(table/image)에 부착하기 위해 잠시 보관
        if element_type == "caption":
            caption_text = _normalize_multiline_text(element.get("content") or element.get("text") or "")
            if caption_text:
                if pending_caption and pending_caption.get("page_no") == page_no:
                    pending_caption["text"] = f"{pending_caption['text']}\n{caption_text}".strip()
                    pending_caption["prov"].extend(prov)
                else:
                    pending_caption = {"text": caption_text, "prov": prov, "page_no": page_no}
            continue

        # 이미지/그림은 별도 파이프라인에서 처리하므로 여기서는 완전히 제외
        if element_type in ("picture", "image"):
            pending_caption = None
            prev_body_bbox_x = None
            continue

        # table 처리
        if element_type == "table":
            caption = _consume_pending_caption(page_no)
            _flush_text_blocks()

            prefix_lines = []
            section_line = _section_path(section_stack)
            if section_line:
                prefix_lines.append(section_line)
            if caption and caption.get("text"):
                prefix_lines.append(f"Caption: {caption['text']}")

            for table_chunk in to_milvus_chunks(element):
                if not table_chunk.strip():
                    continue
                provs = [prov]
                if caption:
                    provs.append(caption["prov"])
                _append_chunk(table_chunk, provs, prefix_lines=prefix_lines)

            prev_body_bbox_x = None
            continue

        # 일반 텍스트 전 진입 시, 미소비 caption 이 남아 있으면 본문으로 흡수
        _flush_pending_caption_as_text()

        text = _normalize_multiline_text(element.get("content") or element.get("text") or "")
        if not text:
            continue

        font = element.get("font", "")
        is_heading = element_type == "heading" or _is_section_heading(text, font)
        if is_heading:
            _flush_text_blocks()
            _update_section_stack(section_stack, text, _extract_heading_level(text))
            prev_body_bbox_x = None
            continue

        current_bbox_x = bbox[0] if isinstance(bbox, list) and len(bbox) >= 1 else None
        is_continuation = (
            current_bbox_x is not None
            and prev_body_bbox_x is not None
            and current_bbox_x > prev_body_bbox_x + INDENT_THRESHOLD
        )

        if is_continuation and text_blocks:
            text_blocks[-1]["text"] = f"{text_blocks[-1]['text']}\n{text}".strip()
            text_blocks[-1]["prov"].extend(prov)
        else:
            text_blocks.append({"text": text, "prov": prov})

        prev_body_bbox_x = current_bbox_x

    _flush_pending_caption_as_text()
    _flush_text_blocks()

    if pdf_doc is not None:
        try:
            pdf_doc.close()
        except Exception:
            pass

    logger.info(
        "[STAGE 2-3] OpenDataLoader JSON 청킹 완료 - 총 %s개 청크, 이미지 %s개",
        len(all_chunks),
        image_count,
    )
    return all_chunks
