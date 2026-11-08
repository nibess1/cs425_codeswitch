# Paper source (ACL style, CS425 final report)

Compile with XeLaTeX (needed for the Chinese characters):

    xelatex main && bibtex main && xelatex main && xelatex main

(or `latexmk -xelatex main.tex` where latexmk is installed).

- Course limit: at most 15 A4 pages at 11 pt, including figures, references and appendices. The current PDF is exactly 15 pages, so any addition needs an equal cut.
- `main.tex`: preamble and section order. Uses `[preprint]` (named authors, page numbers).
- `sections/`: one file per section. `06b_running.tex` is the how-to-run section; `09c_examples_table.tex` is generated from `results/tables/28_qualitative_examples.csv`.
- `figures/`: copies of `results/figures/*.png`; the paper uses `fig2_role_errors.png` and `fig8_human_quant_delta.png`.
- `refs.bib`: every entry checked against OpenAlex/arXiv (26 Oct 2026). `pratap2024mms`: JMLR volume/pages not confirmed (OpenAlex lists only the arXiv record).
- `acl.sty`, `acl_natbib.bst`: from github.com/acl-org/acl-style-files.
- Number check: `python scripts/check_paper_numbers.py paper results/tables` (from the code repository).

Overleaf: upload this folder and set the compiler to XeLaTeX.
