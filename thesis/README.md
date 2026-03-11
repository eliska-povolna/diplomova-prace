# Thesis source

LaTeX source for the diploma thesis **"Interpretable and Controllable POI
Recommender System Using Sparse Autoencoders"**.

## Building

Requirements: a full TeX Live / MiKTeX distribution with `latexmk` and `biber`.

```bash
cd thesis
latexmk -pdf main.tex        # full build (also runs biber automatically)
latexmk -pdf -pvc main.tex   # continuous preview mode
latexmk -C                   # clean all generated files
```

The compiled PDF is **not** tracked in git (see `.gitignore`).
The GitHub Actions workflow (`.github/workflows/thesis.yml`) builds and
uploads the PDF as a workflow artefact on every push.

## Structure

| File / folder | Purpose |
|---|---|
| `main.tex` | Master document — preamble, front matter, includes |
| `bibliography.bib` | BibTeX references |
| `chapters/` | One `.tex` file per chapter |
| `figures/` | All figures (PDF, PNG, SVG, …) |
