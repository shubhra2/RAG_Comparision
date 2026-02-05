# WILP Project Report

This directory contains the source files for the MTech project report on "Evaluating Graph-Based Retrieval-Augmented Generation: Comparative Analysis and Quality Improvements over Standard RAG Systems".

## Files

- `report.md` - Main report document in Markdown format
- `references.bib` - BibTeX bibliography file with academic citations
- `build.sh` - Shell script to build PDF
- `Makefile` - Makefile for building PDF (alternative to build.sh)
- `main.tex` - LaTeX template for title page (included in report.md)

## Prerequisites

1. **Pandoc** (v2.0+)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install pandoc

   # macOS
   brew install pandoc
   ```

2. **XeLaTeX** (for PDF generation)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install texlive-xetex texlive-fonts-recommended

   # macOS
   brew install --cask mactex
   ```

3. **bc** (for size checking in build script)
   ```bash
   # Usually pre-installed on Linux/macOS
   ```

## Building the PDF

### Option 1: Using the build script (Recommended)

```bash
./build.sh
```

This will:
- Generate `WILP_Project_Report.pdf`
- Check file size (must be ≤ 10MB)
- Check page count (must be ≤ 400 pages)

### Option 2: Using Make

```bash
make build
```

Or just:
```bash
make
```

### Option 3: Manual Pandoc command

```bash
pandoc report.md \
  --pdf-engine=xelatex \
  --toc \
  --number-sections \
  --citeproc \
  --bibliography=references.bib \
  --from markdown+raw_tex \
  --to pdf \
  -o WILP_Project_Report.pdf
```

## Report Structure

The report follows the WILP guidelines:

### Front Matter (Roman numerals)
1. Cover
2. Title Page
3. Acknowledgements
4. Abstract (≤ 200 words)
5. Table of Contents

### Main Content (Arabic numerals)
1. Introduction
2. Main Text
   - System Architecture
   - Standard RAG Implementation
   - Graph-Based RAG Implementation
   - Evaluation Framework
   - Experimental Results
3. Conclusions and Recommendations

### Back Matter
- References
- Appendices
- Glossary

## Formatting Requirements

The report is configured to meet WILP requirements:

- **Page Size**: 9 × 11 inches (thesis size)
- **Margins**: 1 inch on all sides
- **Line Spacing**: Double (linestretch: 2)
- **Font**: Times New Roman, 12pt
- **Page Limit**: ≤ 400 pages
- **File Size Limit**: ≤ 10MB

## Customization

### Adding References

Edit `references.bib` and add entries in BibTeX format. Then cite them in `report.md` using:

```markdown
According to previous work [@lewis2020rag], ...
```

### Updating Results

Replace `[TBD]` placeholders in the Results section (Section 6) with actual evaluation results.

### Adding Diagrams

Place diagram images in `../docs/diagrams/` and reference them in the report:

```markdown
![Figure Caption](docs/diagrams/figure.png){#fig:label width=90%}
```

## Troubleshooting

### XeLaTeX not found

If XeLaTeX is not available, the build script will fall back to pdflatex. However, XeLaTeX is recommended for better font handling.

### Bibliography not working

Ensure `references.bib` exists and contains valid BibTeX entries. The `--citeproc` flag enables citation processing.

### Page numbering issues

Roman numerals are used for front matter (cover, title, abstract, TOC) and Arabic numerals for main content. This is handled automatically by Pandoc.

### File size too large

If the PDF exceeds 10MB:
- Compress images before including them
- Reduce image resolution
- Remove unnecessary appendices
- Use PDF compression tools

## Notes

- The title page uses LaTeX formatting embedded in Markdown (raw_tex extension)
- Signature pages should be scanned and embedded as images
- All text must be selectable (not scanned) except signature pages
- The report must pass plagiarism checks

## License

This report is part of the RAG Comparison Project (MIT License).
