#!/bin/bash
# Build script for WILP Project Report PDF

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPORT="report.md"
BIBLIOGRAPHY="references.bib"
OUTPUT="WILP_Project_Report.pdf"
PDF_ENGINE="pdflatex"

echo -e "${GREEN}Building WILP Project Report PDF...${NC}"

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo -e "${RED}Error: pandoc is not installed.${NC}"
    echo "Install it with: sudo apt-get install pandoc (Linux) or brew install pandoc (macOS)"
    exit 1
fi

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    if command -v xelatex &> /dev/null; then
        echo -e "${YELLOW}Warning: pdflatex not found. Trying xelatex instead...${NC}"
        PDF_ENGINE="xelatex"
    else
        echo -e "${RED}Error: Neither pdflatex nor xelatex found!${NC}"
        echo "Install with: sudo pacman -S texlive-most (Arch) or sudo apt-get install texlive-latex-base (Ubuntu/Debian) or brew install --cask mactex (macOS)"
        exit 1
    fi
fi

# Check if files exist
if [ ! -f "$REPORT" ]; then
    echo -e "${RED}Error: $REPORT not found!${NC}"
    exit 1
fi

if [ ! -f "$BIBLIOGRAPHY" ]; then
    echo -e "${YELLOW}Warning: $BIBLIOGRAPHY not found. Building without bibliography...${NC}"
    BIBLIOGRAPHY=""
fi

# Build PDF
echo -e "${GREEN}Generating PDF...${NC}"

# Get script directory and change to it
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Build pandoc command
PANDOC_CMD="pandoc \"$REPORT\" \
    --pdf-engine=\"$PDF_ENGINE\" \
    --toc \
    --number-sections \
    --listings \
    --from markdown+raw_tex \
    --to pdf \
    -o \"$OUTPUT\" \
    --resource-path=.:.."

# Add bibliography if available
if [ -n "$BIBLIOGRAPHY" ]; then
    PANDOC_CMD="$PANDOC_CMD --citeproc --bibliography=\"$BIBLIOGRAPHY\""
fi

# Execute pandoc command
eval $PANDOC_CMD

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PDF generated successfully: $OUTPUT${NC}"

    # Check file size
    if command -v stat &> /dev/null; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            size=$(stat -f%z "$OUTPUT")
        else
            # Linux
            size=$(stat -c%s "$OUTPUT")
        fi
        size_mb=$(echo "scale=2; $size / 1024 / 1024" | bc)
        echo -e "${GREEN}File size: ${size_mb} MB${NC}"

        if (( $(echo "$size_mb > 10" | bc -l) )); then
            echo -e "${RED}⚠ WARNING: PDF exceeds 10MB limit!${NC}"
        else
            echo -e "${GREEN}✓ File size is within 10MB limit${NC}"
        fi
    fi

    # Check page count (if pdfinfo is available)
    if command -v pdfinfo &> /dev/null; then
        pages=$(pdfinfo "$OUTPUT" 2>/dev/null | grep Pages | awk '{print $2}')
        if [ -n "$pages" ]; then
            echo -e "${GREEN}Number of pages: $pages${NC}"
            if [ "$pages" -gt 400 ]; then
                echo -e "${RED}⚠ WARNING: PDF exceeds 400 page limit!${NC}"
            else
                echo -e "${GREEN}✓ Page count is within 400 page limit${NC}"
            fi
        fi
    fi

    echo -e "${GREEN}Build complete!${NC}"
else
    echo -e "${RED}✗ PDF generation failed!${NC}"
    exit 1
fi
