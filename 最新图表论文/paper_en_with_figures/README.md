# English Manuscript With Publication Figures

This is the completed English manuscript copy with the code-generated result figures integrated into the paper body. The original `paper/` and `paper_overleaf_en/` sources are not modified.

Main PDF:

```text
KCH-MedRank_with_publication_figures.pdf
```

Source entry point:

```text
main.tex
```

Integrated data figures:

- Figure 2: `figures/data_figures/fig2_main_results.pdf`
- Figure 3: `figures/data_figures/fig3_recall_at_k.pdf`
- Figure 4: `figures/data_figures/fig4_structural_ablation.pdf`
- Figure 5: `figures/data_figures/fig5_subgroup_forest.pdf`
- Figure 6: `figures/data_figures/fig6_efficiency_pareto.pdf`
- Figure 7: `figures/data_figures/fig7_interpretability.pdf`
- Figure 8: `figures/data_figures/fig8_synonym_normalization.pdf`

Build command used:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
bibtex build\main
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

The LaTeX build completed successfully with no undefined references or citations in the final log. MiKTeX printed its routine update reminder, and the log contains minor underfull hbox warnings from narrow table cells.

