from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import psycopg2
from dotenv import load_dotenv

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
PREPROCESS_SCRIPT = PROJECT_ROOT / "scripts" / "preprocess.py"
DEFAULT_ENV_FILE = "remote/remote.env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-platform remote population pipeline for PostgreSQL."
    )
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help=f"Env file loaded last (default: {DEFAULT_ENV_FILE})",
    )
    parser.add_argument("--pdf-path", default=None, help="PDF path override")
    parser.add_argument("--source", default=None, help="Source name override")
    parser.add_argument("--skip-empty", action="store_true", help="Skip truncating target tables")
    parser.add_argument("--dry-run", action="store_true", help="Run preprocess in dry-run mode")
    parser.add_argument(
        "--include-azure-md",
        action="store_true",
        help="Include `knowledge_base_md` Azure embedding stage",
    )
    return parser.parse_args()


def load_environment(env_file: str) -> None:
    load_dotenv(PROJECT_ROOT / "project.env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / DEFAULT_ENV_FILE, override=True)
    load_dotenv(PROJECT_ROOT / env_file, override=True)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value in (None, ""):
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def build_pg_dsn() -> str:
    host = require_env("PGHOST")
    port = require_env("PGPORT")
    database = require_env("PGDATABASE")
    user = require_env("PGUSER")
    password = require_env("PGPASSWORD")
    sslmode = env_or_default("PGSSLMODE", "require")
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
        require_env("PGHOST"),
        "--pg-port",
        require_env("PGPORT"),
        "--pg-database",
        require_env("PGDATABASE"),
        "--pg-user",
        require_env("PGUSER"),
        "--pg-password",
        require_env("PGPASSWORD"),
    ]

    return args


def run_step(name: str, args: List[str]) -> None:
    cmd = [sys.executable, str(PREPROCESS_SCRIPT), *args]
    print(f"> Running step: {name}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if result.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(f"Step '{name}' failed with exit code {result.returncode}: {joined}")


def build_steps(include_azure_md: bool) -> List[Dict[str, List[str]]]:
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

    if include_azure_md:
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
                "Set them in .env or run without --include-azure-md."
            )

        steps.append(
            {
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
        )

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

    steps = build_steps(include_azure_md=args.include_azure_md)
    for step in steps:
        run_step(step["name"], [*base_args, *step["args"]])

    print("Remote pipeline completed successfully.")


if __name__ == "__main__":
    main()
