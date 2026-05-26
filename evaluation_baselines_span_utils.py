import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ACTION_PATTERNS = {
    "email": [r"\bemail\b", r"\bmail\b", r"\bsend (an )?email\b"],
    "calendar": [r"\bcalendar\b", r"\bschedule\b", r"\bbook\b", r"\bset up (a )?meeting\b"],
    "phone": [r"\bcall\b", r"\bphone\b", r"\bdial\b"],
    "message": [r"\bslack\b", r"\bmessage\b", r"\bping\b", r"\btext\b"],
}


def split_context_and_tools(context: str) -> Tuple[str, str]:
    marker = "Available tools: "
    if marker in context:
        tool_output, tools_json = context.split(marker, 1)
        return tool_output.strip(), tools_json.strip()
    return context.strip(), ""


def parse_available_tools(context: str) -> List[Dict]:
    _, tools_json = split_context_and_tools(context)
    if not tools_json:
        return []
    try:
        parsed = json.loads(tools_json)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def parse_tool_blocks(context: str) -> List[Tuple[str, object]]:
    left, _ = split_context_and_tools(context)
    s = left.strip()
    blocks: List[Tuple[str, object]] = []
    decoder = json.JSONDecoder()
    i = 0
    n = len(s)

    while i < n:
        while i < n and s[i] in " \n\t.":
            i += 1
        if i >= n:
            break

        colon = s.find(":", i)
        if colon == -1:
            break

        name = s[i:colon].strip()
        j = colon + 1
        while j < n and s[j].isspace():
            j += 1
        if not name or j >= n:
            break

        try:
            payload, end = decoder.raw_decode(s, j)
            blocks.append((name, payload))
            i = end
            continue
        except Exception:
            match = re.search(r'(?:\n|\.\s+)(?=[^\n:]{1,80}:\s*[\[{\"])', s[j:])
            end = j + (match.start() if match else len(s[j:]))
            raw = s[j:end].strip()
            blocks.append((name, raw))
            i = end

    return blocks


def humanize_key(key: str) -> str:
    key = str(key).replace("_", " ").replace("-", " ")
    key = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key.lower()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def content_tokens(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_./%:-]+", str(text).lower())
    return [t for t in toks if len(t) >= 3]


def is_large_blob(text: str) -> bool:
    text = str(text)
    if text.startswith("data:image/"):
        return True
    return len(text) > 160 and re.fullmatch(r"[A-Za-z0-9+/=._:-]+", text) is not None


def format_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)

    text = str(value).strip().replace("\n", " ")
    if is_large_blob(text):
        return "[large binary/string omitted]"
    if len(text) > 220:
        return text[:217] + "..."
    return text


def render_json(value: object, indent: int = 0, max_items: int = 4) -> List[str]:
    sp = "  " * indent
    lines: List[str] = []

    if isinstance(value, dict):
        for k, v in value.items():
            key = humanize_key(k)
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{key}:")
                lines.extend(render_json(v, indent + 1, max_items=max_items))
            else:
                lines.append(f"{sp}{key}: {format_scalar(v)}")
        return lines

    if isinstance(value, list):
        for idx, item in enumerate(value[:max_items], 1):
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}- item {idx}:")
                lines.extend(render_json(item, indent + 1, max_items=max_items))
            else:
                lines.append(f"{sp}- item {idx}: {format_scalar(item)}")
        if len(value) > max_items:
            lines.append(f"{sp}- ... {len(value) - max_items} more items")
        return lines

    return [f"{sp}{format_scalar(value)}"]


def render_available_tools(context: str, max_tools: int = 8) -> List[str]:
    tools = parse_available_tools(context)
    if not tools:
        return []

    lines = ["Available tools:"]
    for tool in tools[:max_tools]:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        if len(desc) > 180:
            desc = desc[:177] + "..."
        lines.append(f"- {name}: {desc}")
    if len(tools) > max_tools:
        lines.append(f"- ... {len(tools) - max_tools} more tools")
    return lines


def normalize_tool_context(context: str, max_items: int = 4, max_tools: int = 8) -> str:
    blocks = parse_tool_blocks(context)
    lines: List[str] = []
    for name, payload in blocks:
        lines.append(f"Tool: {name}")
        lines.extend(render_json(payload, indent=1, max_items=max_items))
    lines.extend(render_available_tools(context, max_tools=max_tools))
    return "\n".join(lines)


def add_normalized_context_columns(df):
    df = df.copy()
    df["normalized_context"] = df["context"].apply(normalize_tool_context)
    df["normalized_tool_output"] = df["context"].apply(lambda x: split_context_and_tools(normalize_tool_context(x))[0])
    return df


def extract_percentages(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[+-]?\d+(?:\.\d+)?%", text)]


def extract_dates(text: str) -> List[Tuple[int, int, str]]:
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{4}/\d{2}/\d{2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
    ]
    matches = []
    for pat in patterns:
        matches.extend((m.start(), m.end(), m.group(0)) for m in re.finditer(pat, text))
    return matches


def extract_numbers(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\b\d+(?:\.\d+)?\b", text)]


def extract_quoted_strings(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r'"[^"\n]{3,}"', text)]


def extract_capitalized_phrases(text: str) -> List[Tuple[int, int, str]]:
    return [
        (m.start(), m.end(), m.group(0))
        for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][A-Za-z0-9.&'-]+){0,4})\b", text)
    ]


def merge_spans(spans: Sequence[Dict]) -> List[Dict]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
    merged = [dict(spans[0])]
    for span in spans[1:]:
        last = merged[-1]
        if span["start"] <= last["end"]:
            last["end"] = max(last["end"], span["end"])
            if len(span.get("text", "")) > len(last.get("text", "")):
                last["text"] = span.get("text", last.get("text", ""))
            last["score"] = max(last.get("score", 0.0), span.get("score", 0.0))
            last["type"] = last.get("type", span.get("type", "hallucination"))
        else:
            merged.append(dict(span))
    return merged


def normalize_gold_labels(labels: Sequence, text: Optional[str] = None) -> List[Dict]:
    normalized = []
    for label in labels or []:
        if isinstance(label, dict):
            start = int(label.get("start", -1))
            end = int(label.get("end", -1))
            label_type = label.get("type", "hallucination")
            snippet = label.get("text")
        elif isinstance(label, (list, tuple)) and len(label) >= 3:
            start = int(label[0])
            end = int(label[1])
            label_type = label[2]
            snippet = text[start:end] if text and start >= 0 and end > start else None
        else:
            continue
        if start < 0 or end <= start:
            continue
        normalized.append({
            "start": start,
            "end": end,
            "type": label_type,
            "text": snippet if snippet is not None else (text[start:end] if text else ""),
            "score": 1.0,
        })
    return merge_spans(normalized)


def spans_to_char_set(spans: Sequence[Dict]) -> set:
    chars = set()
    for span in spans:
        chars.update(range(int(span["start"]), int(span["end"])))
    return chars


def evaluate_span_predictions(gold_spans_list: Sequence[Sequence[Dict]], pred_spans_list: Sequence[Sequence[Dict]]) -> Dict[str, float]:
    tp = fp = fn = 0
    exact_matches = 0
    total = len(gold_spans_list)

    for gold_spans, pred_spans in zip(gold_spans_list, pred_spans_list):
        gold_chars = spans_to_char_set(gold_spans)
        pred_chars = spans_to_char_set(pred_spans)
        tp += len(gold_chars & pred_chars)
        fp += len(pred_chars - gold_chars)
        fn += len(gold_chars - pred_chars)

        gold_exact = {(s["start"], s["end"], s.get("type", "hallucination")) for s in gold_spans}
        pred_exact = {(s["start"], s["end"], s.get("type", "hallucination")) for s in pred_spans}
        exact_matches += int(gold_exact == pred_exact)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
        "exact_match_rate": exact_matches / total if total else 0.0,
    }


def build_span_eval_frame(df, pred_col: str) -> Dict[str, float]:
    gold = [normalize_gold_labels(labels, text=output) for labels, output in zip(df["hallucination_labels"], df["output"])]
    pred = df[pred_col].tolist()
    return evaluate_span_predictions(gold, pred)


def lexical_hallucination_spans(row) -> List[Dict]:
    output = row["output"]
    tool_output = row.get("normalized_tool_output", row["context"])
    available_tools = parse_available_tools(row["context"])
    tool_norm = normalize_text(tool_output)
    spans = []

    candidate_extractors = [
        ("percentage", extract_percentages),
        ("date", extract_dates),
        ("number", extract_numbers),
        ("quote", extract_quoted_strings),
        ("entity", extract_capitalized_phrases),
    ]

    for kind, extractor in candidate_extractors:
        for start, end, value in extractor(output):
            value_norm = normalize_text(value.strip('"'))
            if len(value_norm) < 2:
                continue
            if value_norm not in tool_norm:
                spans.append({
                    "start": start,
                    "end": end,
                    "text": output[start:end],
                    "type": "hallucination",
                    "score": 0.6 if kind in {"number", "entity"} else 0.8,
                })

    tool_blob = normalize_text(" ".join(
        f"{tool.get('name', '')} {tool.get('description', '')}" for tool in available_tools
    ))
    for affordance, patterns in ACTION_PATTERNS.items():
        supported = affordance in tool_blob
        for pattern in patterns:
            for match in re.finditer(pattern, output, flags=re.IGNORECASE):
                if not supported:
                    spans.append({
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(0),
                        "type": "missing_tool",
                        "score": 0.95,
                    })

    return merge_spans(spans)


def longest_streak(flags: Sequence[bool]) -> int:
    best = 0
    cur = 0
    for flag in flags:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best


def aggregate_span_features(spans: Sequence[Dict], text_length: int) -> Dict[str, float]:
    lengths = [max(0, span["end"] - span["start"]) for span in spans]
    scores = [float(span.get("score", 0.0)) for span in spans]
    coverage = sum(lengths) / max(1, text_length)
    return {
        "num_spans": len(spans),
        "max_span_score": max(scores) if scores else 0.0,
        "mean_span_score": float(np.mean(scores)) if scores else 0.0,
        "span_char_fraction": coverage,
        "max_span_len": max(lengths) if lengths else 0.0,
    }


def aggregate_lookback_features(ratios: Sequence[Dict]) -> Dict[str, float]:
    if not ratios:
        return {
            "mean_ratio": 0.5,
            "min_ratio": 0.5,
            "frac_low_03": 0.0,
            "frac_low_02": 0.0,
            "std_ratio": 0.0,
            "bottom3_mean": 0.5,
            "longest_low_streak": 0.0,
            "mean_ratio_numeric": 0.5,
            "frac_low_numeric": 0.0,
        }

    vals = np.array([r["lookback_ratio"] for r in ratios], dtype=float)
    bottom3 = np.sort(vals)[: min(3, len(vals))]
    numeric_mask = np.array([bool(re.search(r"\d", r["token"])) for r in ratios])

    if numeric_mask.any():
        numeric_vals = vals[numeric_mask]
        mean_ratio_numeric = float(numeric_vals.mean())
        frac_low_numeric = float((numeric_vals < 0.3).mean())
    else:
        mean_ratio_numeric = 0.5
        frac_low_numeric = 0.0

    low_flags = vals < 0.3
    return {
        "mean_ratio": float(vals.mean()),
        "min_ratio": float(vals.min()),
        "frac_low_03": float((vals < 0.3).mean()),
        "frac_low_02": float((vals < 0.2).mean()),
        "std_ratio": float(vals.std()),
        "bottom3_mean": float(bottom3.mean()),
        "longest_low_streak": float(longest_streak(low_flags.tolist())),
        "mean_ratio_numeric": mean_ratio_numeric,
        "frac_low_numeric": frac_low_numeric,
    }


def spans_from_lookback_ratios(answer: str, ratios: Sequence[Dict], low_threshold: float = 0.22, min_chars: int = 3) -> List[Dict]:
    spans = []
    cur = None

    for ratio in ratios:
        start = ratio.get("start")
        end = ratio.get("end")
        if start is None or end is None or end <= start:
            continue

        text = answer[start:end]
        is_content = bool(re.search(r"[A-Za-z0-9]", text))
        low = ratio["lookback_ratio"] < low_threshold

        if low and is_content:
            if cur is None:
                cur = {
                    "start": start,
                    "end": end,
                    "score_values": [1.0 - float(ratio["lookback_ratio"])],
                }
            else:
                if start <= cur["end"] + 1:
                    cur["end"] = end
                    cur["score_values"].append(1.0 - float(ratio["lookback_ratio"]))
                else:
                    spans.append(cur)
                    cur = {
                        "start": start,
                        "end": end,
                        "score_values": [1.0 - float(ratio["lookback_ratio"])],
                    }
        elif cur is not None:
            spans.append(cur)
            cur = None

    if cur is not None:
        spans.append(cur)

    normalized = []
    for span in spans:
        if span["end"] - span["start"] < min_chars:
            continue
        normalized.append({
            "start": span["start"],
            "end": span["end"],
            "text": answer[span["start"]:span["end"]],
            "type": "hallucination",
            "score": float(np.mean(span["score_values"])) if span["score_values"] else 0.0,
        })
    return merge_spans(normalized)
