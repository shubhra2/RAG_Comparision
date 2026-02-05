# Installation Guide for Building PDF Report

## Required Packages

To build the PDF report, you need to install the following packages:

### Arch Linux

```bash
# Install Pandoc
sudo pacman -S pandoc

# Install LaTeX and required packages (RECOMMENDED - full TeX Live distribution)
sudo pacman -S texlive-most texlive-lang

# OR install minimal TeX Live (smaller, but may need additional packages)
sudo pacman -S texlive-core texlive-bin texlive-latex texlive-latexextra

# If you get "lmodern.sty not found" error, install:
sudo pacman -S texlive-latex
```

### Linux (Ubuntu/Debian)

```bash
# Install Pandoc
sudo apt-get update
sudo apt-get install pandoc

# Install LaTeX and required packages
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended

# Optional: For better font support
sudo apt-get install texlive-xetex
```

### macOS

```bash
# Install Pandoc
brew install pandoc

# Install MacTeX (full LaTeX distribution)
brew install --cask mactex

# Or install BasicTeX (smaller, but may need additional packages)
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install collection-latexextra
```

### Minimal Installation (if you have issues)

If you're getting errors about missing packages, you can install just the essentials:

**Arch Linux:**
```bash
sudo pacman -S pandoc texlive-core texlive-bin texlive-latex texlive-latexextra
```

**Ubuntu/Debian:**
```bash
sudo apt-get install pandoc texlive-latex-base texlive-latex-extra
```

**macOS:**
```bash
brew install pandoc
brew install --cask basictex
sudo tlmgr install xcolor setspace graphicx times
```

## Verifying Installation

Check if required tools are installed:

```bash
# Check Pandoc
pandoc --version

# Check LaTeX
pdflatex --version

# Check for specific packages (Linux)
kpsewhich xcolor.sty
kpsewhich setspace.sty
kpsewhich graphicx.sty
```

## Troubleshooting

### Error: `xcolor.sty not found`

**Solution:** Install the required LaTeX packages:
```bash
sudo pacman -S texlive-latexextra  # Arch Linux
sudo apt-get install texlive-latex-extra  # Ubuntu/Debian
sudo tlmgr install xcolor  # macOS with BasicTeX
```

### Error: Font issues

**Solution:** The report uses the `times` package which should be available by default. If you get font errors:
- Linux: `sudo apt-get install texlive-fonts-recommended`
- macOS: Usually included with MacTeX/BasicTeX

### Error: Images not found

**Solution:** Make sure you're running the build script from the `report/` directory:
```bash
cd report
./build.sh
```

The script automatically sets the correct paths for images.

### Error: Bibliography not working

**Solution:** Make sure `references.bib` exists in the `report/` directory. The citation style (CSL) is currently commented out to avoid dependency issues. You can uncomment it in `report.md` after installing all packages.

## Quick Test

After installation, test the build:

```bash
cd report
./build.sh
```

If successful, you should see:
- ✓ PDF generated successfully
- File size information
- Page count information
