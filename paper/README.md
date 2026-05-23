# Paper Workspace

This directory contains the bilingual LaTeX manuscript for:

- English: **KCH-MedRank: Knowledge-Constrained Learning-to-Rank for Evidence-Grounded Medical Retrieval-Augmented Generation**
- Chinese: **KCH-MedRank：面向循证医学检索增强生成的知识约束学习排序方法**

The template is intentionally generic. Do not switch to Elsevier, Springer, IEEE, MDPI, or another journal-specific template until the target venue is selected.

## Structure

```text
paper/
├── main_en.tex
├── main_zh.tex
├── references.bib
├── verified_sources.md
├── sections_en/
├── sections_zh/
├── tables/
├── figures/
└── build/
```

## Bilingual Synchronization Rules

English and Chinese sources must be updated together.

- `main_en.tex` and `main_zh.tex` are the two manuscript entry points.
- `sections_en/` and `sections_zh/` must stay one-to-one.
- If one version adds or changes a claim, result, limitation, table, citation, or conclusion, update the corresponding location in the other version in the same edit.
- If exact translation is temporarily blocked, add an explicit `% TODO:` note in both language versions.
- Before submission, compare both versions section by section and confirm they are synchronized.

## Citation Rules

- Add entries to `references.bib` only after verifying metadata online from reliable official sources.
- Reliable sources include arXiv, PubMed/NCBI, ACL Anthology, ACM, IEEE, Springer, Elsevier, Nature, MDPI, official Hugging Face pages, and official GitHub repositories for software or dataset metadata.
- Do not cite papers from memory.
- Do not invent authors, titles, years, venues, DOI, arXiv IDs, PMID, or BibTeX fields.
- If a relevant work cannot be verified, keep a `% TODO: verify citation ...` marker in the manuscript and do not add it to `references.bib`.
- Record verified source links in `verified_sources.md`.

## Compile Commands

From the repository root:

```powershell
xelatex -output-directory=paper/build paper/main_en.tex
bibtex paper/build/main_en
xelatex -output-directory=paper/build paper/main_en.tex
xelatex -output-directory=paper/build paper/main_en.tex

xelatex -output-directory=paper/build paper/main_zh.tex
bibtex paper/build/main_zh
xelatex -output-directory=paper/build paper/main_zh.tex
xelatex -output-directory=paper/build paper/main_zh.tex
```

If using the bundled Codex LaTeX helper, compile each entry point with the output directory set to `paper/build`.
