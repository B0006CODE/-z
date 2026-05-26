# 最新图表论文

This folder is an isolated, reproducible figure-building copy for the English KCH-MedRank manuscript. It reads the English manuscript tables and experiment logs from the original repository, writes clean CSV files under `paper_figures/data/processed/`, and generates publication figures under `paper_figures/figures/`.

The original `paper/`, `paper_overleaf_en/`, and `加数据图的论文/` folders are not modified.

Run:

```powershell
python -m pip install -r .\最新图表论文\paper_figures\requirements.txt
python .\最新图表论文\paper_figures\scripts\make_all.py
```

