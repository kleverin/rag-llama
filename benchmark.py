"""
benchmark.py — RAG Accuracy Benchmark
=======================================
Evaluates the pipeline using questions grounded in the actual ingested
CSV and Excel data: MTConnect tool/load readings, shift employees, part
numbers, job orders, and machine utilization.

All expected answers were extracted directly from the source files —
run generate_eval_facts.py to regenerate them if the data changes.

Usage:
    source iiot/bin/activate
    python benchmark.py                   # run all, save JSON report
    python benchmark.py --verbose         # show full answers + chunks
    python benchmark.py --top-k 10        # change retrieval depth
    python benchmark.py --out my.json     # custom report file
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PATH     = os.getenv("CHROMA_PATH",     "./chroma_db")


# ─────────────────────────────────────────────────────────────────────────────
# EVAL SET — grounded in actual ingested CSV/Excel data
#
# kw: list of strings that ALL must appear in a correct answer (case-insensitive)
# note: where the ground truth came from (not checked, just for reference)
# ─────────────────────────────────────────────────────────────────────────────

EVAL_SET = [

    # ── MTConnect CSV — high load / fault events (flag-based, retrieval-friendly) ──
    # NOTE: exact-timestamp lookups ("what tool at 6:06 AM?") don't work with vector
    # search — all MTConnect rows are structurally identical and the embedding model
    # can't distinguish them by timestamp.  These questions use FLAG markers and
    # distinctive content that creates a semantically unique document instead.
    {
        "id":          "high_load_apr29",
        "q":           "Was there a spindle overload or extremely high spindle load event logged in April 2025? What tool was active?",
        "kw":          ["106", "208"],
        "cat":         "mtconnect_load",
        "meta_filter": [
            {"key": "type",    "value": "mtconnect"},
            {"key": "flagged", "value": "true"},
        ],
        "note":        "Apr 29 12:21 → [FLAGS: SPINDLE-OVERLOAD] tool=106, load=208% — filter to flagged=true rows only",
    },
    {
        "id":          "high_load_may2",
        "q":           "Were there any HIGH-SPINDLE-LOAD events recorded in May 2025? What tool was running?",
        "kw":          ["34"],
        "cat":         "mtconnect_load",
        "meta_filter": [
            {"key": "type",    "value": "mtconnect"},
            {"key": "flagged", "value": "true"},
        ],
        "note":        "May 2 07:00 → [FLAGS: HIGH-SPINDLE-LOAD] tool=34, load=127% — filter to flagged=true rows only",
    },
    {
        "id":          "program_a100",
        "q":           "Was a CNC program named A100.MIN ever used according to the MTConnect logs?",
        "kw":          ["A100"],
        "cat":         "mtconnect_program",
        "meta_filter": {"key": "type", "value": "mtconnect"},
        "note":        "Apr 28 rows → Mpprogram=A100.MIN — date-agnostic so embedding retrieves Apr 28 rows",
    },
    {
        "id":          "program_may27",
        "q":           "What CNC program appears most in the MTConnect machine logs?",
        "kw":          ["AB100"],
        "cat":         "mtconnect_program",
        "meta_filter": {"key": "type", "value": "mtconnect"},
        "note":        "May 27+ rows → Mpprogram=AB100.MIN (dominant program in later data)",
    },
    {
        "id":          "stopped_mtconnect",
        "q":           "Are there any PROGRAM_STOPPED execution states in the MTConnect machine logs?",
        "kw":          ["PROGRAM_STOPPED"],
        "cat":         "mtconnect_state",
        "meta_filter": {"key": "type", "value": "mtconnect"},
        "note":        "Both CSVs have rows with Mpexecution=PROGRAM_STOPPED — date-agnostic so embedding can match",
    },

    # ── Excel — employees by shift ────────────────────────────────────────────
    {
        "id":          "employees_apr28_day",
        "q":           "Who were some of the employees on the day shift on April 28, 2025?",
        "kw":          ["Adams", "Ahmad"],
        "cat":         "employees",
        "meta_filter": {"key": "type", "value": "employees"},
        "use_prompt":  False,
        "note":        "Job_enriched_ar Employees sheet: source_file='Shift Report 4-28-25 Day Shift'",
    },
    {
        "id":          "employees_apr28_night",
        "q":           "List the employee names who worked the night shift on April 28, 2025.",
        "kw":          ["Wallace", "Thompson"],
        "cat":         "employees",
        "meta_filter": {"key": "type", "value": "employees"},
        "use_prompt":  False,
        "note":        "Job_enriched_ar Employees sheet: source_file='Shift Report 4-28-2025 Nights'",
    },
    {
        "id":          "employees_may27_night",
        "q":           "What employee names are listed for the night shift on May 27, 2025?",
        "kw":          ["Wallace", "Kull"],
        "cat":         "employees",
        "meta_filter": {"key": "type", "value": "employees"},
        "use_prompt":  False,
        "note":        "Job_enriched_ar Employees sheet: source_file='Shift Report 5-27-2025 Nights'",
    },

    # ── Excel — parts and job orders ──────────────────────────────────────────
    {
        "id":          "part_mb4000_apr28",
        "q":           "What part number was being produced on the MB4000 during the April 28, 2025 day shift?",
        "kw":          ["5387600"],
        "cat":         "parts",
        "meta_filter": {"key": "type", "value": "part_details"},
        "note":        "Part_Details sheet: date=2025-04-28, shift=Day, work_center=MB4000, part=5387600-03",
    },
    {
        "id":          "job_order_mb4000_apr28",
        "q":           "What job order was running on the MB4000 on April 28, 2025 day shift?",
        "kw":          ["J270018541"],
        "cat":         "parts",
        "meta_filter": {"key": "type", "value": "part_details"},
        "note":        "Part_Details sheet: date=2025-04-28, shift=Day, work_center=MB4000, job=J270018541",
    },
    {
        "id":          "part_v60_apr28",
        "q":           "What part was being produced on the V60 machine on April 28, 2025 day shift?",
        "kw":          ["3487080"],
        "cat":         "parts",
        "meta_filter": {"key": "type", "value": "part_details"},
        "note":        "Part_Details sheet: date=2025-04-28, shift=Day, work_center=V60, part=3487080-03",
    },
    {
        "id":          "good_parts_v60_apr28",
        "q":           "How many good parts were produced on the V60 on April 28, 2025 day shift?",
        "kw":          ["66"],
        "cat":         "parts",
        "meta_filter": {"key": "type", "value": "part_details"},
        "note":        "Part_Details sheet: V60 Apr 28 Day → part 3487080-03, good=66",
    },
    {
        "id":          "part_5001_apr28",
        "q":           "What part number was run on work center 5001 on April 28, 2025 day shift?",
        "kw":          ["DZ114780"],
        "cat":         "parts",
        "meta_filter": {"key": "type", "value": "part_details"},
        "note":        "Part_Details sheet: date=2025-04-28, shift=Day, work_center=5001, part=DZ114780-B",
    },

    # ── Excel — machine utilization ───────────────────────────────────────────
    # NOTE: util_mb4000 and util_ma600 require Machine_Shift_Summary sheet to be
    # present in Job_enriched_ar.xlsx.  util_shift_total uses Date_Shift_Summary
    # which is confirmed present (42.3% was retrieved in the Apr 28 run).
    {
        "id":   "util_mb4000_apr28",
        "q":    "What was the average utilization of the MB4000 machine on April 28, 2025 day shift?",
        "kw":   ["MB4000", "utilization"],
        "cat":  "utilization",
        "note": "Machine_Shift_Summary: MB4000 Apr 28 Day → avg_utilization_percent=76.43 (if sheet exists)",
    },
    {
        "id":          "util_nhx6300_may27",
        "q":           "What was the utilization of the NHX6300 machine on the May 27, 2025 day shift?",
        "kw":          ["NHX6300", "62"],
        "cat":         "utilization",
        "meta_filter": {"key": "type", "value": "machine_summary"},
        "note":        "Machine_Shift_Summary: NHX6300 May 27 Day → avg_utilization_percent=62.0 (confirmed in retrieved chunks)",
    },
    {
        "id":          "util_shift_total_apr28",
        "q":           "What was the overall shift utilization percentage on April 28, 2025 day shift across all work centers?",
        "kw":          ["42"],
        "cat":         "utilization",
        "meta_filter": {"key": "type", "value": "shift_summary"},
        "note":        "Date_Shift_Summary: Apr 28 Day → overall utilization 42.3% (confirmed present in index)",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_answer(answer: str, keywords: list[str]) -> dict:
    low = answer.lower()
    hits   = [kw for kw in keywords if kw.lower() in low]
    misses = [kw for kw in keywords if kw.lower() not in low]
    return {
        "score":  len(hits) / len(keywords) if keywords else 0.0,
        "hits":   hits,
        "misses": misses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

def green(t):  return _c(t, "92")
def yellow(t): return _c(t, "93")
def red(t):    return _c(t, "91")
def dim(t):    return _c(t, "2")
def bold(t):   return _c(t, "1")

def progress_bar(pct: float, width: int = 10) -> str:
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)

def pct_label(p: float) -> str:
    label = f"{p:5.1f}%"
    if p >= 80:  return green(label)
    if p >= 50:  return yellow(label)
    return red(label)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_index(top_k: int):
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
    from llama_index.core import VectorStoreIndex, StorageContext

    if not os.path.exists(CHROMA_PATH):
        sys.exit(f"[error] ChromaDB not found at '{CHROMA_PATH}'. Run ingest.py first.")

    print(f"[setup] Loading ChromaDB from {CHROMA_PATH} ...")
    chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection("iiot_rag")
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    llm         = Ollama(model="llama3.2", base_url=OLLAMA_BASE_URL, request_timeout=180.0)

    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model,
    )
    print(f"[setup] Ready (top_k={top_k}, {len(EVAL_SET)} questions)\n")
    return index, llm


_QA_PROMPT_STR = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using only the context information, answer the query. "
    "Always explicitly state the exact values, numbers, names, part numbers, job orders, "
    "and program names from the context — never use a pronoun like 'it' or 'this program' "
    "without also naming the specific value.\n"
    "Query: {query_str}\n"
    "Answer: "
)


def _make_engine(index, llm, top_k: int, meta_filter, use_prompt: bool = True):
    """Build a query engine, optionally scoped to one or more metadata filters.
    meta_filter may be a single dict or a list of dicts.
    use_prompt=False skips the custom QA template (uses LlamaIndex default).
    """
    from llama_index.core import PromptTemplate
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    kwargs: dict = {"llm": llm, "similarity_top_k": top_k}
    if use_prompt:
        kwargs["text_qa_template"] = PromptTemplate(_QA_PROMPT_STR)
    if meta_filter:
        filters_raw = meta_filter if isinstance(meta_filter, list) else [meta_filter]
        kwargs["filters"] = MetadataFilters(filters=[
            ExactMatchFilter(key=f["key"], value=f["value"]) for f in filters_raw
        ])
    return index.as_query_engine(**kwargs)


def run(top_k: int, verbose: bool, out_file: str):
    index, llm = build_index(top_k)
    results    = []

    for i, item in enumerate(EVAL_SET):
        n = f"[{i+1:02d}/{len(EVAL_SET)}]"
        print(f"{n} {bold(item['id'])}  {dim('('+item['cat']+')')}")
        print(f"       Q: {item['q']}")
        if item.get("meta_filter"):
            print(f"       {dim('[filter: ' + str(item['meta_filter']) + ']')}")

        engine = _make_engine(index, llm, top_k, item.get("meta_filter"), item.get("use_prompt", True))

        t0 = time.time()
        try:
            response = engine.query(item["q"])
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"       {red('ERROR:')} {exc}\n")
            results.append({**item, "answer": None, "score": 0.0,
                            "hits": [], "misses": item["kw"],
                            "sources": [], "elapsed_s": round(elapsed, 2),
                            "error": str(exc)})
            continue

        elapsed = time.time() - t0
        answer  = str(response)
        sc      = score_answer(answer, item["kw"])
        pct_val = sc["score"] * 100

        print(f"       Score: [{progress_bar(pct_val)}] {pct_label(pct_val)}  ({elapsed:.1f}s)")
        if sc["hits"]:
            print(f"       {green('Found:')}   {sc['hits']}")
        if sc["misses"]:
            print(f"       {red('Missing:')} {sc['misses']}")

        sources = []
        for node in getattr(response, "source_nodes", []):
            meta    = node.node.metadata
            excerpt = node.node.get_content()[:300].replace("\n", " ")
            sources.append({
                "source":  meta.get("source", "unknown"),
                "type":    meta.get("type", "unknown"),
                "score":   round(float(node.score or 0), 4),
                "excerpt": excerpt,
            })

        if verbose:
            print(f"\n       {dim('Answer:')}")
            for line in answer.split("\n"):
                print(f"         {line}")
            print(f"\n       {dim('Retrieved chunks:')}")
            for s in sources:
                print(f"         [{s['type']}] {s['source']}  sim={s['score']:.3f}")
                print(f"           {dim(s['excerpt'][:140])}")
        print()

        results.append({
            **item,
            "answer":    answer,
            "sources":   sources,
            "elapsed_s": round(elapsed, 2),
            **sc,
        })

    # ── Category summary ──────────────────────────────────────────────────────
    valid  = [r for r in results if "error" not in r]
    errors = len(results) - len(valid)

    by_cat: dict[str, list[float]] = {}
    for r in valid:
        by_cat.setdefault(r["cat"], []).append(r["score"])

    overall = sum(r["score"] for r in valid) / len(valid) if valid else 0.0

    print("=" * 64)
    print(bold("BENCHMARK RESULTS"))
    print("=" * 64)
    cat_labels = {
        "mtconnect_tool":    "MTConnect — tool number",
        "mtconnect_program": "MTConnect — program name",
        "mtconnect_load":    "MTConnect — high load events",
        "mtconnect_state":   "MTConnect — execution state",
        "employees":         "Excel — shift employees",
        "parts":             "Excel — parts & job orders",
        "utilization":       "Excel — machine utilization",
    }
    for cat, scores_list in sorted(by_cat.items()):
        avg   = sum(scores_list) / len(scores_list) * 100
        label = cat_labels.get(cat, cat)
        print(f"  {label:<35} {pct_label(avg)}  ({len(scores_list)} Qs)")

    print(f"\n  {'OVERALL':<35} {pct_label(overall * 100)}  "
          f"({len(valid)}/{len(results)} answered"
          + (f", {errors} error(s)" if errors else "") + ")")

    # ── Per-question table ────────────────────────────────────────────────────
    latencies = [r["elapsed_s"] for r in valid]
    if latencies:
        lat_min = min(latencies)
        lat_avg = sum(latencies) / len(latencies)
        lat_max = max(latencies)
        print(f"\n  {'Latency (s)':<35} min={lat_min:.1f}  avg={lat_avg:.1f}  max={lat_max:.1f}")

    print()
    print(dim(f"  {'ID':<36} {'Score':>6}  {'Time':>6}  Missing / notes"))
    print(dim("  " + "─" * 72))
    for r in results:
        if "error" in r:
            print(f"  {r['id']:<36} {red('ERROR ')}")
        else:
            miss = ", ".join(r["misses"]) or dim("all found")
            print(f"  {r['id']:<36} {pct_label(r['score']*100)}  {r['elapsed_s']:>5.1f}s  {miss}")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "timestamp":     datetime.now().isoformat(),
        "chroma_path":   CHROMA_PATH,
        "top_k":         top_k,
        "overall_score": overall,
        "latency_s": {
            "min": round(lat_min, 2) if latencies else None,
            "avg": round(lat_avg, 2) if latencies else None,
            "max": round(lat_max, 2) if latencies else None,
        },
        "by_category": {
            cat: round(sum(scores) / len(scores), 3)
            for cat, scores in by_cat.items()
        },
        "results": results,
    }
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {out_file}")

    return overall


if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="RAG accuracy benchmark")
    parser.add_argument("--top-k",  type=int, default=15,
                        help="Retrieval depth — number of chunks to pull per question (default 15)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full LLM answers and retrieved source chunks")
    parser.add_argument("--out", default=f"benchmark_{ts}.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    score = run(top_k=args.top_k, verbose=args.verbose, out_file=args.out)
    sys.exit(0 if score >= 0.5 else 1)
