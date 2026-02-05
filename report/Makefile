# Makefile for building WILP Project Report PDF

# Variables
REPORT = report.md
BIBLIOGRAPHY = references.bib
OUTPUT = WILP_Project_Report.pdf
PDF_ENGINE = pdflatex

# Default target
all: $(OUTPUT)

# Build PDF from Markdown
$(OUTPUT): $(REPORT) $(BIBLIOGRAPHY)
	pandoc $(REPORT) \
		--pdf-engine=$(PDF_ENGINE) \
		--toc \
		--number-sections \
		--citeproc \
		--bibliography=$(BIBLIOGRAPHY) \
		--from markdown+raw_tex \
		--to pdf \
		-o $(OUTPUT)
	@echo "PDF generated: $(OUTPUT)"
	@echo "File size: $$(du -h $(OUTPUT) | cut -f1)"

# Clean generated files
clean:
	rm -f $(OUTPUT)
	rm -f *.aux *.log *.out *.toc

# Check file size (must be <= 10MB)
check-size: $(OUTPUT)
	@size=$$(stat -f%z $(OUTPUT) 2>/dev/null || stat -c%s $(OUTPUT) 2>/dev/null); \
	size_mb=$$(echo "scale=2; $$size / 1024 / 1024" | bc); \
	echo "PDF size: $$size_mb MB"; \
	if [ $$(echo "$$size_mb > 10" | bc) -eq 1 ]; then \
		echo "WARNING: PDF exceeds 10MB limit!"; \
	else \
		echo "PDF size is within 10MB limit."; \
	fi

# Count pages (must be <= 400)
check-pages: $(OUTPUT)
	@pages=$$(pdfinfo $(OUTPUT) 2>/dev/null | grep Pages | awk '{print $$2}' || echo "pdfinfo not available"); \
	echo "Number of pages: $$pages"; \
	if [ -n "$$pages" ] && [ $$pages -gt 400 ]; then \
		echo "WARNING: PDF exceeds 400 page limit!"; \
	else \
		echo "Page count is within 400 page limit."; \
	fi

# Full build with checks
build: $(OUTPUT) check-size check-pages

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build PDF (default)"
	@echo "  build        - Build PDF and run checks"
	@echo "  clean        - Remove generated files"
	@echo "  check-size   - Check if PDF is <= 10MB"
	@echo "  check-pages  - Check if PDF is <= 400 pages"
	@echo "  help         - Show this help message"

.PHONY: all clean check-size check-pages build help
