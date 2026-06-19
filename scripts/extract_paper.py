from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_DIR = REPO_ROOT / "references"
PAPERS_DIR = REF_DIR / "papers"
TEXT_DIR = REF_DIR / "text"
TAGS_FILE = REF_DIR / "tags.yaml"

FRONTMATTER_FACETS = ("primary", "role", "status", "reward_term", "domain", "tags")


def load_tags() -> dict:
    with TAGS_FILE.open() as fh:
        data = yaml.safe_load(fh)
    papers = (data or {}).get("papers") or {}
    if not papers:
        sys.exit(f"No papers found in {TAGS_FILE}")
    return papers


def build_frontmatter(key: str, entry: dict) -> str:
    fm: dict = {"key": key, "title": entry.get("title", ""), "year": entry.get("year")}
    for facet in FRONTMATTER_FACETS:
        if facet in entry:
            fm[facet] = entry[facet]
    if "one_liner" in entry:
        fm["summary"] = entry["one_liner"]
    body = yaml.safe_dump(
        fm, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    return f"---\n{body}---\n\n"


def run_marker(pdf: Path, args: argparse.Namespace) -> str:
    """Run marker_single into a temp dir and return the extracted markdown text."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cmd = [
            args.marker_bin,
            str(pdf),
            "--output_dir",
            str(tmp_dir),
            "--output_format",
            "markdown",
        ]
        if args.force_ocr:
            cmd.append("--force_ocr")
        if args.use_llm:
            # With an LLM service, images become inline text descriptions, which
            # are useful in the text layer; so we keep extraction on in that mode.
            cmd.append("--use_llm")
        else:
            # Text-only layer: don't spend time extracting images we discard, and
            # avoid dangling image links in the markdown. Captions are separate
            # text blocks and are unaffected.
            cmd.append("--disable_image_extraction")
        if args.extra:
            cmd += args.extra.split()

        print(f"  $ {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            sys.exit(
                f"'{args.marker_bin}' not found on PATH. Install marker: "
                "`uv tool install marker-pdf`."
            )
        except subprocess.CalledProcessError as exc:
            sys.exit(f"marker failed for {pdf.name} (exit {exc.returncode}).")

        # marker writes <tmp>/<pdf_stem>/<pdf_stem>.md
        md_files = list(tmp_dir.rglob("*.md"))
        if not md_files:
            sys.exit(f"marker produced no markdown for {pdf.name}.")
        return md_files[0].read_text(encoding="utf-8")


def extract_one(key: str, entry: dict, args: argparse.Namespace) -> bool:
    pdf = PAPERS_DIR / f"{key}.pdf"
    out = TEXT_DIR / f"{key}.md"
    if not pdf.exists():
        print(f"  [skip] {key}: no PDF at {pdf}")
        return False
    if out.exists() and not args.force:
        print(f"  [have] {key}: {out.name} exists (use --force to redo)")
        return False

    print(f"  [run ] {key}")
    text = run_marker(pdf, args)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(build_frontmatter(key, entry) + text, encoding="utf-8")
    print(f"  [done] {out}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("key", nargs="?", help="bibkey to extract (omit with --all)")
    p.add_argument("--all", action="store_true", help="extract all papers in tags.yaml")
    p.add_argument(
        "--force", action="store_true", help="re-extract even if output exists"
    )
    p.add_argument("--force-ocr", action="store_true", dest="force_ocr")
    p.add_argument("--use-llm", action="store_true", dest="use_llm")
    p.add_argument("--extra", default="", help="raw args appended to marker_single")
    p.add_argument("--marker-bin", default="marker_single", dest="marker_bin")
    args = p.parse_args()

    if not args.all and not args.key:
        p.error("provide a <key> or --all")

    papers = load_tags()

    if args.all:
        keys = list(papers)
    else:
        if args.key not in papers:
            sys.exit(
                f"'{args.key}' not in {TAGS_FILE}. Known keys:\n  "
                + "\n  ".join(papers)
            )
        keys = [args.key]

    print(f"Extracting {len(keys)} paper(s) -> {TEXT_DIR}")
    done = 0
    for key in keys:
        done += extract_one(key, papers[key], args)
    print(f"\nExtracted {done} new file(s); {len(keys) - done} skipped.")


if __name__ == "__main__":
    main()
