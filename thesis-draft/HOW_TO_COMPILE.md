# How to Compile the Thesis

## Option 1 — Overleaf (Recommended, no install needed)

1. Go to https://overleaf.com and create a free account.
2. Click **New Project → Upload Project**.
3. Zip the entire `thesis-draft/` folder and upload the zip.
4. Set the main document to `main.tex`.
5. Set the compiler to **pdfLaTeX** and bibliography backend to **BibTeX**.
6. Click **Compile** (green button).

The thesis will compile to PDF in ~30 seconds.

## Option 2 — Local (macOS)

Install MacTeX (large but complete):

```bash
brew install --cask mactex
```

Then compile:

```bash
cd thesis-draft
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
open main.pdf
```

## Notes

- The `figures/lnu_logo.png` is a placeholder white rectangle.
  Replace it with the actual Linnaeus University logo before submission.
- If you do not have a university logo, remove the `\includegraphics` line
  in `chapters/titlepage.tex`.
- Fill in the real supervisor and examiner names in `chapters/titlepage.tex`.
