#!/usr/bin/env python3
"""
Placeholder pipelines for ToolACE-style hallucination injection.

Goal:
- Provide clear scaffolding for 3 hallucination types from the assignment:
  1) hallucination (contradiction to tool context)
  2) overgeneration (unsupported extra details)
  3) missing_tool (suggesting an action that requires unavailable tool)

Notes:
- This file is intentionally minimal and modular.
- Replace TODO blocks with your local model calls, validators, and span alignment logic.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


Span = Tuple[int, int, str]


@dataclass
class Example:
    query: str
    context: str
    output: str
    hallucination_labels: List[Span]
    meta: Dict


def load_jsonl(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def dump_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_row(row: Dict) -> Example:
    # Supports your current notebook format: query/context/output.
    query = row.get("query", "")
    context = row.get("context", "")
    output = row.get("output", "")
    labels = row.get("hallucination_labels", [])
    meta = row.get("meta", {})
    return Example(query=query, context=context, output=output, hallucination_labels=labels, meta=meta)


def to_row(ex: Example) -> Dict:
    return {
        "query": ex.query,
        "context": ex.context,
        "output": ex.output,
        "hallucination_labels": ex.hallucination_labels,
        "meta": ex.meta,
    }


def find_span(text: str, snippet: str) -> Optional[Tuple[int, int]]:
    start = text.find(snippet)
    if start < 0:
        return None
    return start, start + len(snippet)


class LocalGenerator:
    """
    Placeholder for a local model wrapper.

    Replace generate(...) with your local inference call, e.g. transformers/vLLM.
    """

    def __init__(self, model_name_or_path: str = "TODO_LOCAL_MODEL"):
        self.model_name_or_path = model_name_or_path

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        # TODO: implement local generation.
        # Keep deterministic settings for reproducibility (temperature=0 / fixed seed).
        _ = (prompt, max_new_tokens)
        return "TODO_GENERATED_TEXT"


class LocalVerifier:
    """
    Placeholder for a local verifier (NLI or classifier).

    Suggested checks:
    - contradiction against context for hallucination type
    - not-supported by context for overgeneration
    - requested action requires unavailable tool for missing_tool
    """

    def is_valid(self, query: str, context: str, output: str, corruption_type: str) -> bool:
        # TODO: replace with real local checks.
        _ = (query, context, output, corruption_type)
        return True


class HallucinationInjector:
    def __init__(self, generator: LocalGenerator, verifier: LocalVerifier, seed: int = 42):
        self.generator = generator
        self.verifier = verifier
        random.seed(seed)

    def inject_hallucination(self, ex: Example) -> Example:
        """
        Type 1: contradiction between output and tool context.
        """
        prompt = (
            "You are editing an assistant answer.\n"
            "Task: introduce exactly one factual contradiction against the context.\n"
            "Keep the rest of the answer fluent and mostly unchanged.\n\n"
            f"Query:\n{ex.query}\n\n"
            f"Context:\n{ex.context}\n\n"
            f"Original answer:\n{ex.output}\n\n"
            "Return JSON with keys:\n"
            "mutated_output: string\n"
            "hallucinated_span: exact substring from mutated_output that contradicts context\n"
        )

        # TODO: parse real model JSON output here.
        mutated_output = ex.output + " [HALLUCINATION_PLACEHOLDER]"
        hallucinated_span = "[HALLUCINATION_PLACEHOLDER]"

        if not self.verifier.is_valid(ex.query, ex.context, mutated_output, "hallucination"):
            return ex

        span = find_span(mutated_output, hallucinated_span)
        labels: List[Span] = []
        if span is not None:
            labels.append((span[0], span[1], "hallucination"))

        return Example(
            query=ex.query,
            context=ex.context,
            output=mutated_output,
            hallucination_labels=labels,
            meta={**ex.meta, "corruption_type": "hallucination"},
        )

    def inject_overgeneration(self, ex: Example) -> Example:
        """
        Type 2: add unsupported details not present in context.
        """
        prompt = (
            "You are editing an assistant answer.\n"
            "Task: add one plausible detail that is NOT supported by the context.\n"
            "Do not add contradictions if possible; add unsupported extension.\n\n"
            f"Query:\n{ex.query}\n\n"
            f"Context:\n{ex.context}\n\n"
            f"Original answer:\n{ex.output}\n\n"
            "Return JSON with keys:\n"
            "mutated_output: string\n"
            "hallucinated_span: exact substring from mutated_output that is unsupported\n"
        )

        # TODO: parse real model JSON output here.
        mutated_output = ex.output + " [OVERGENERATION_PLACEHOLDER]"
        hallucinated_span = "[OVERGENERATION_PLACEHOLDER]"

        if not self.verifier.is_valid(ex.query, ex.context, mutated_output, "overgeneration"):
            return ex

        span = find_span(mutated_output, hallucinated_span)
        labels: List[Span] = []
        if span is not None:
            labels.append((span[0], span[1], "overgeneration"))

        return Example(
            query=ex.query,
            context=ex.context,
            output=mutated_output,
            hallucination_labels=labels,
            meta={**ex.meta, "corruption_type": "overgeneration"},
        )

    def inject_missing_tool(self, ex: Example) -> Example:
        """
        Type 3: suggest an action that requires a tool not in available tools.
        """
        prompt = (
            "You are editing an assistant answer.\n"
            "Task: add one action suggestion that requires a tool unavailable in the tool list.\n"
            "Use context as-is and keep answer natural.\n\n"
            f"Query:\n{ex.query}\n\n"
            f"Context (contains available tools):\n{ex.context}\n\n"
            f"Original answer:\n{ex.output}\n\n"
            "Return JSON with keys:\n"
            "mutated_output: string\n"
            "hallucinated_span: exact substring suggesting unavailable tool action\n"
        )

        # TODO: parse real model JSON output here.
        mutated_output = ex.output + " [MISSING_TOOL_PLACEHOLDER]"
        hallucinated_span = "[MISSING_TOOL_PLACEHOLDER]"

        if not self.verifier.is_valid(ex.query, ex.context, mutated_output, "missing_tool"):
            return ex

        span = find_span(mutated_output, hallucinated_span)
        labels: List[Span] = []
        if span is not None:
            labels.append((span[0], span[1], "missing_tool"))

        return Example(
            query=ex.query,
            context=ex.context,
            output=mutated_output,
            hallucination_labels=labels,
            meta={**ex.meta, "corruption_type": "missing_tool"},
        )


def process_dataset(
    rows: Sequence[Dict],
    injector: HallucinationInjector,
    corruption_type: str,
    apply_probability: float = 1.0,
) -> List[Dict]:
    out: List[Dict] = []
    for row in rows:
        ex = normalize_row(row)
        if random.random() > apply_probability:
            out.append(to_row(ex))
            continue

        if corruption_type == "hallucination":
            mutated = injector.inject_hallucination(ex)
        elif corruption_type == "overgeneration":
            mutated = injector.inject_overgeneration(ex)
        elif corruption_type == "missing_tool":
            mutated = injector.inject_missing_tool(ex)
        else:
            raise ValueError(f"Unknown corruption_type: {corruption_type}")

        out.append(to_row(mutated))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Placeholder hallucination injection pipelines")
    parser.add_argument("--input_jsonl", type=Path, required=True, help="Input dataset in JSONL")
    parser.add_argument(
        "--corruption_type",
        type=str,
        required=True,
        choices=["hallucination", "overgeneration", "missing_tool"],
        help="Which hallucination type to inject",
    )
    parser.add_argument("--output_jsonl", type=Path, required=True, help="Output dataset in JSONL")
    parser.add_argument("--apply_probability", type=float, default=1.0, help="Probability of mutation per sample")
    parser.add_argument("--model_name_or_path", type=str, default="TODO_LOCAL_MODEL")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_jsonl(args.input_jsonl)
    generator = LocalGenerator(model_name_or_path=args.model_name_or_path)
    verifier = LocalVerifier()
    injector = HallucinationInjector(generator=generator, verifier=verifier, seed=args.seed)

    mutated_rows = process_dataset(
        rows=rows,
        injector=injector,
        corruption_type=args.corruption_type,
        apply_probability=args.apply_probability,
    )
    dump_jsonl(args.output_jsonl, mutated_rows)

    print(f"Saved {len(mutated_rows)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
