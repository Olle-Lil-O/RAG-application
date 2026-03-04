from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
PREPROCESS_SCRIPT = ROOT / "scripts" / "preprocess.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-platform local population pipeline (no .sh/.ps1 required)."
    )
    parser.add_argument("--env-file", default=".env", help="Env file to load last (default: .env)")
    parser.add_argument("--pdf-path", default=None, help="PDF path override")
    parser.add_argument("--source", default=None, help="Source name override")
    parser.add_argument("--skip-empty", action="store_true", help="Skip truncating target tables")
    parser.add_argument("--dry-run", action="store_true", help="Run preprocess in dry-run mode")
    parser.add_argument("--mini", action="store_true", help="Populate only `knowledge_base_mini`")
    parser.add_argument("--small", action="store_true", help="Populate only `knowledge_base_sm`")
    parser.add_argument("--medium", action="store_true", help="Populate only `knowledge_base_md`")
    parser.add_argument("--all", action="store_true", help="Populate mini + small + medium tables")
    return parser.parse_args()


def load_environment(env_file: str) -> None:
    load_dotenv(ROOT / "project.env", override=False)
    load_dotenv(ROOT / ".env", override=True)
    load_dotenv(ROOT / env_file, override=True)


def env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def build_pg_dsn() -> str:
    host = env_or_default("PGHOST", "localhost")
    port = env_or_default("PGPORT", "5431")
    database = env_or_default("PGDATABASE", "postgres")
    user = env_or_default("PGUSER", "postgres")
    password = env_or_default("PGPASSWORD", "password")
    sslmode = env_or_default("PGSSLMODE", "disable")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode={sslmode}"


def truncate_tables() -> None:
    dsn = build_pg_dsn()
    sql = "TRUNCATE TABLE knowledge_base_mini, knowledge_base_sm, knowledge_base_md RESTART IDENTITY;"
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("Tables truncated successfully.")


def base_preprocess_args() -> List[str]:
    pdf_path = env_or_default("PDF_PATH", "data/euaiact.pdf")
    source = os.getenv("SOURCE_NAME") or Path(pdf_path).name

    args = [
        "--pdf-path",
        pdf_path,
        "--source",
        source,
        "--spacy-model",
        env_or_default("SPACY_MODEL", "en_core_web_sm"),
        "--max-sentences",
        env_or_default("MAX_SENTENCES", "5"),
        "--chunker",
        env_or_default("CHUNKER", "spacy"),
        "--chunking-provider",
        env_or_default("CHUNKING_PROVIDER", "local"),
        "--chunking-local-model",
        env_or_default("CHUNKING_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "--chunking-deployment",
        os.getenv("CHUNKING_DEPLOYMENT") or os.getenv("DEPLOY_MEDIUM") or "",
        "--breakpoint-threshold-type",
        env_or_default("BREAKPOINT_THRESHOLD_TYPE", "percentile"),
        "--max-embed-tokens",
        env_or_default("MAX_EMBED_TOKENS", "2000"),
        "--split-overlap-tokens",
        env_or_default("SPLIT_OVERLAP_TOKENS", "80"),
        "--pg-host",
        env_or_default("PGHOST", "localhost"),
        "--pg-port",
        env_or_default("PGPORT", "5431"),
        "--pg-database",
        env_or_default("PGDATABASE", "postgres"),
        "--pg-user",
        env_or_default("PGUSER", "postgres"),
        "--pg-password",
        env_or_default("PGPASSWORD", "password"),
    ]

    return args


def run_step(name: str, args: List[str]) -> None:
    cmd = [sys.executable, str(PREPROCESS_SCRIPT), *args]
    print(f"> Running step: {name}")
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(f"Step '{name}' failed with exit code {result.returncode}: {joined}")


def _mini_step() -> Dict[str, List[str]]:
    return {
        "name": "knowledge_base_mini",
        "args": [
            "--table",
            "knowledge_base_mini",
            "--provider",
            "local",
            "--local-model",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
    }


def _small_step() -> Dict[str, List[str]]:
    return {
        "name": "knowledge_base_sm",
        "args": [
            "--table",
            "knowledge_base_sm",
            "--provider",
            "local",
            "--local-model",
            "BAAI/bge-large-en-v1.5",
        ],
    }


def _medium_step() -> Dict[str, List[str]]:
    deployment = os.getenv("DEPLOY_MEDIUM")
    endpoint = os.getenv("AZURE_ENDPOINT")
    api_key = os.getenv("AZURE_API_KEY")
    missing = [
        name
        for name, value in {
            "DEPLOY_MEDIUM": deployment,
            "AZURE_ENDPOINT": endpoint,
            "AZURE_API_KEY": api_key,
        }.items()
        if not value
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Cannot run Azure md step; missing env vars: {joined}. "
            "Set them in .env or run without medium step."
        )

    return {
        "name": "knowledge_base_md",
        "args": [
            "--table",
            "knowledge_base_md",
            "--provider",
            "azure",
            "--deployment",
            deployment,
            "--azure-endpoint",
            endpoint,
            "--azure-api-key",
            api_key,
        ],
    }


def build_steps(args: argparse.Namespace) -> List[Dict[str, List[str]]]:
    if args.all and (args.mini or args.small or args.medium):
        raise ValueError("`--all` cannot be combined with `--mini`, `--small`, or `--medium`.")

    selected: List[Dict[str, List[str]]] = []

    if args.all:
        selected.extend([_mini_step(), _small_step(), _medium_step()])
        return selected

    any_specific = args.mini or args.small or args.medium
    if any_specific:
        if args.mini:
            selected.append(_mini_step())
        if args.small:
            selected.append(_small_step())
        if args.medium:
            selected.append(_medium_step())
        return selected

    steps = [
        {
            "name": "knowledge_base_mini",
            "args": [
                "--table",
                "knowledge_base_mini",
                "--provider",
                "local",
                "--local-model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        },
        {
            "name": "knowledge_base_sm",
            "args": [
                "--table",
                "knowledge_base_sm",
                "--provider",
                "local",
                "--local-model",
                "BAAI/bge-large-en-v1.5",
            ],
        },
    ]

    return steps


def main() -> None:
    args = parse_args()
    load_environment(args.env_file)

    if not PREPROCESS_SCRIPT.exists():
        raise FileNotFoundError(f"Preprocess script not found: {PREPROCESS_SCRIPT}")

    if not args.skip_empty and not args.dry_run:
        truncate_tables()
    elif args.dry_run:
        print("Dry run enabled; skipping table truncation.")

    base_args = base_preprocess_args()
    if args.pdf_path:
        i = base_args.index("--pdf-path")
        base_args[i + 1] = args.pdf_path
    if args.source:
        i = base_args.index("--source")
        base_args[i + 1] = args.source

    if args.dry_run:
        base_args.append("--dry-run")

    steps = build_steps(args)
    for step in steps:
        run_step(step["name"], [*base_args, *step["args"]])

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
