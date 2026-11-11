"""Command line: python -m finrag <command> [options]"""
from __future__ import annotations

import argparse
import json

from . import financebench as fb
from .config import Config
from .eval.dataset import load_dataset, sample, save_jsonl
from .eval.generate import propose_questions
from .eval.run import compare, to_markdown
from .pipeline import fetch, ingest, load_pipeline

VARIANTS = {
    "dense, shared": {"hybrid": False, "scope": "shared"},
    "hybrid, shared": {"hybrid": True, "scope": "shared"},
    "hybrid, per_doc": {"hybrid": True, "scope": "per_doc"},
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finrag")
    ap.add_argument("--config", default="configs/default.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download the FinanceBench questions and the reports they need")
    f.add_argument("--max-docs", type=int, default=None, help="only the N most-asked reports (default: config max_docs, 0 = all)")

    sub.add_parser("ingest", help="read the PDFs in data/pdf, chunk them page by page, embed and build the index")

    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question")
    a.add_argument("--mode", choices=["rag", "agent"], default="rag")
    a.add_argument("--doc", action="append", help="search only this report (repeatable), e.g. --doc 3M_2018_10K")

    m = sub.add_parser("make-gold", help="turn the FinanceBench questions on the indexed reports into data/eval/gold.jsonl")
    m.add_argument("--out", default="data/eval/gold.jsonl")

    g = sub.add_parser("eval-generate", help="draft extra questions from random chunks (you must review them)")
    g.add_argument("--n", type=int, default=20)
    g.add_argument("--out", default="data/eval/draft.jsonl")

    e = sub.add_parser("eval", help="evaluate variants on the gold dataset")
    e.add_argument("--dataset", default="data/eval/gold.jsonl")
    e.add_argument("--mode", choices=["rag", "agent"], default="rag")
    e.add_argument("--limit", type=int, default=0, help="N answerable questions at random (fixed seed) + all unanswerable; 0 = all")
    e.add_argument("--variant", action="append", choices=list(VARIANTS), help="default: all variants")
    e.add_argument("--out", default="data/eval/results.json")

    args = ap.parse_args(argv)
    cfg = Config.load(args.config, max_docs=getattr(args, "max_docs", None))

    if args.cmd == "fetch":
        fetch(cfg)
    elif args.cmd == "ingest":
        ingest(cfg)
    elif args.cmd == "ask":
        pipe = load_pipeline(cfg)
        if args.mode == "rag":
            res = pipe.rag().ask(args.question, doc_ids=args.doc).to_dict()
        else:
            q = args.question + (f"\n(The relevant report: {', '.join(args.doc)}.)" if args.doc else "")
            res = pipe.agent().run(q).to_dict()
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.cmd == "make-gold":
        questions = fb.read_jsonl(fb.download_benchmark(cfg.benchmark_dir)["questions"])
        indexed = {p.stem for p in cfg.pdf_dir.glob("*.pdf")}
        rows = fb.to_gold(questions, keep_docs=indexed)
        save_jsonl(rows, args.out)
        print(f"{len(rows)}/{len(questions)} FinanceBench questions have their reports indexed -> {args.out}. "
              "Add your unanswerable questions to the same file.")
    elif args.cmd == "eval-generate":
        pipe = load_pipeline(cfg)
        rows = propose_questions(pipe.index.chunks, pipe.llm, n=args.n)
        save_jsonl(rows, args.out)
        print(f"{len(rows)} draft questions in {args.out}. Review each one, fix it, set reviewed=true, then append to gold.jsonl.")
    elif args.cmd == "eval":
        pipe = load_pipeline(cfg)
        examples = sample(load_dataset(args.dataset), args.limit)
        variants = {k: v for k, v in VARIANTS.items() if not args.variant or k in args.variant}
        results = compare(examples, pipe, variants, args.mode)
        print(to_markdown(results))
        save_jsonl([{"variant": k, **v} for k, v in results.items()], args.out)


if __name__ == "__main__":
    main()
