"""Run the eval set against the RAG pipeline and report metrics."""
from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.rag import ask

EVAL_PATH = Path(__file__).parent / "eval_set.json"
console = Console()


def check_answer(answer: str, expected_any_of: list[str]) -> bool:
    a = answer.lower()
    return any(exp.lower() in a for exp in expected_any_of)


def check_retrieval(hits, expected_source: str | None) -> bool:
    if expected_source is None:
        return True
    return any(h.source == expected_source for h in hits)


def main() -> None:
    items = json.loads(EVAL_PATH.read_text())
    console.print(f"[bold cyan]Running eval on {len(items)} questions[/bold cyan]\n")

    results = []
    t0 = time.time()

    for item in items:
        q_start = time.time()
        answer, hits = ask(item["question"], topic=item.get("topic"))
        elapsed = time.time() - q_start

        answer_ok = check_answer(answer, item["expected_any_of"])
        retrieval_ok = check_retrieval(hits, item.get("expected_source"))

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": answer,
                "answer_ok": answer_ok,
                "retrieval_ok": retrieval_ok,
                "expected": item["expected_any_of"],
                "elapsed": elapsed,
            }
        )

        mark_a = "[green]OK[/green]" if answer_ok else "[red]FAIL[/red]"
        mark_r = "[green]OK[/green]" if retrieval_ok else "[red]FAIL[/red]"
        console.print(
            f"[bold]{item['id']}[/bold] ({elapsed:.1f}s) "
            f"answer={mark_a} retrieval={mark_r}"
        )
        first_line = answer.split("\n\n")[0][:200]
        console.print(f"  -> {first_line}\n")

    total = time.time() - t0

    table = Table(title="Eval Summary", show_lines=False)
    table.add_column("id")
    table.add_column("answer", justify="center")
    table.add_column("retrieval", justify="center")
    table.add_column("expected")
    table.add_column("sec", justify="right")
    for r in results:
        table.add_row(
            r["id"],
            "OK" if r["answer_ok"] else "FAIL",
            "OK" if r["retrieval_ok"] else "FAIL",
            ", ".join(r["expected"]),
            f"{r['elapsed']:.1f}",
        )
    console.print(table)

    n = len(results)
    a_pass = sum(1 for r in results if r["answer_ok"])
    r_pass = sum(1 for r in results if r["retrieval_ok"])
    both = sum(1 for r in results if r["answer_ok"] and r["retrieval_ok"])
    console.print(
        f"\n[bold]Answer accuracy:[/bold] {a_pass}/{n} ({a_pass / n:.0%})"
    )
    console.print(
        f"[bold]Retrieval accuracy:[/bold] {r_pass}/{n} ({r_pass / n:.0%})"
    )
    console.print(f"[bold]End-to-end pass:[/bold] {both}/{n} ({both / n:.0%})")
    console.print(
        f"[bold]Total wall time:[/bold] {total:.1f}s "
        f"(avg {total / n:.1f}s / question)"
    )

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    console.print(f"\nFull results -> {out}")


if __name__ == "__main__":
    main()
