# Paper source (ACL style, CS425 final report)

Compile with XeLaTeX (needed for the Chinese characters):

    xelatex main && bibtex main && xelatex main && xelatex main

(or `latexmk -xelatex main.tex` where latexmk is installed).

- Course limit: at most 15 A4 pages at 11 pt, including figures, references and appendices. Check the page count after every change.
- `main.tex`: preamble and section order. Uses `[preprint]` (named authors, page numbers).
- `sections/`: one file per section. RQ1 and RQ3 include the phone-recording results; the how-to-run commands are in the appendix (`app:run`); `14_contributions.tex` has each member's contribution; `09c_examples_table.tex` is generated from `results/tables/28_qualitative_examples.csv`.
- `figures/`: copies of `results/figures/*.png`; the paper uses `fig2_role_errors.png`. Figure 1 is drawn in TikZ in `03_task_metrics.tex`.
- `refs.bib`: every entry checked against OpenAlex/arXiv (26 Oct 2026). `pratap2024mms`: JMLR 25(97):1–52 (confirmed on jmlr.org, 8 Nov). `ugan2025pier` added 8 Nov.
- `acl.sty`, `acl_natbib.bst`: from github.com/acl-org/acl-style-files.
- Number check: `python scripts/check_paper_numbers.py paper results/tables` (from the code repository).

Overleaf: upload this folder and set the compiler to XeLaTeX.
