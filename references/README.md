
# references/

Reference corpus for `synth-q-mat-generator`, structured so an AI agent (Claude Code, or a Claude Project) can find the right paper without reading every PDF.

## Layout

```
references/
  papers/        source PDFs, named <bibkey>.pdf (citation key = filename stem)
  text/          marker-pdf extractions, <bibkey>.md, with YAML frontmatter (grep target)
  papers.bib     BibTeX for every paper (doi.org URLs)
  tags.yaml      single source of truth for facets + one-liners (the extractor reads this)
  INDEX.md       human-readable index: one-liner + facets + filename, grouped by category
  annotated-bibliography.md   full annotations with key numbers and project relevance
```

## How to find a paper (retrieval path)

1. Grep `tags.yaml` or `INDEX.md` by facet — e.g. all topology-reward papers:
   `grep -B1 -A6 "reward_term:.*topology" references/tags.yaml`
   or by status: `grep "\[adopted" references/INDEX.md`
2. Read `text/<bibkey>.md` for the full extracted text (frontmatter carries the tags).
3. Open `papers/<bibkey>.pdf` only when you need figures, equations, or exact layout that the text extraction drops.

Do not brute-read the PDFs; they are binary and expensive. The text layer exists so grep/RAG can do the work first.

## Facets (in `tags.yaml` and frontmatter)

- `primary` — browsing bucket(s): generators, rl, predictors, physics, synthesis, data, tooling
- `role` — function in the pipeline (generator, rl-algorithm, reward-model, data-source, validation-tool, ...)
- `status` — adopted | rejected | candidate | context (with -lineage / -concept / -cautionary / -template / -baseline qualifiers)
- `reward_term` — topology | superconductivity | stability | composition | multi
- `domain` — ml | physics | chem | software
- `tags` — free-text keywords

## Generating the text layer

Extraction uses [marker-pdf](https://pypi.org/project/marker-pdf/) (v1.10.x). Install it isolated so its torch pin does not collide with the main env:

```bash
uv tool install marker-pdf      # provides the `marker_single` CLI on PATH
# GPU: install a CUDA torch build into the tool env, or set TORCH_DEVICE=cuda at runtime
```

Then, from the repo root:

```bash
make extract-papers                 # extract every paper missing a text/ output (incremental)
make extract-paper KEY=zeni2025mattergen   # extract exactly one
make extract-papers ARGS="--force"  # re-extract everything
```

LLM hybrid mode (`--use_llm`) is optional and needs its own inference API (Gemini, Ollama, or an Anthropic API key — NOT a Claude Code subscription). Base marker is already strong on scientific papers; only escalate per-paper:

```bash
# force OCR (re-OCR all pages, convert inline math to LaTeX) — GPU-accelerated
make extract-paper KEY=bradlyn2017tqc ARGS="--force-ocr --force"

# LLM cleanup via Anthropic API (needs ANTHROPIC_API_KEY in env)
make extract-paper KEY=szymanski2023alab ARGS="--use-llm --force \
  --extra '--llm_service marker.services.claude.ClaudeService --claude_api_key $ANTHROPIC_API_KEY'"
```

## Adding a new paper later

1. Drop the PDF in `papers/<bibkey>.pdf` (use the `authorYEARkeyword` convention).
2. Add a block to `tags.yaml` under `papers:` (copy an existing one).
3. Add the BibTeX entry to `papers.bib` and a line to `INDEX.md`.
4. `make extract-paper KEY=<bibkey>` — frontmatter is generated from `tags.yaml` automatically.

## For the Claude Project (web)

Upload `papers/*.pdf` plus `INDEX.md` and `annotated-bibliography.md`. The index and annotations give the RAG retriever clean anchors that map a question to the right filename. Keep the author+year filenames; they cite cleanly.
