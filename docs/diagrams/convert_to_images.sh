#!/bin/bash
# Convert PlantUML diagrams to PNG images for PDF export
# Usage: ./convert_to_images.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGRAMS_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$SCRIPT_DIR"

echo "Converting PlantUML diagrams to PNG images..."
echo "Diagrams directory: $DIAGRAMS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if plantuml is installed
if ! command -v plantuml &> /dev/null; then
    echo "Error: PlantUML is not installed."
    echo ""
    echo "Installation options:"
    echo "1. Linux: sudo apt-get install plantuml"
    echo "2. macOS: brew install plantuml"
    echo "3. Or download plantuml.jar from https://plantuml.com/download"
    echo ""
    echo "If using JAR file, run:"
    echo "  java -jar plantuml.jar -tpng $DIAGRAMS_DIR/*.puml"
    exit 1
fi

# Convert all .puml files to PNG
for puml_file in "$DIAGRAMS_DIR"/*.puml; do
    if [ -f "$puml_file" ]; then
        filename=$(basename "$puml_file" .puml)
        echo "Converting: $filename.puml -> $filename.png"
        plantuml -tpng -o "$OUTPUT_DIR" "$puml_file"
    fi
done

echo ""
echo "Conversion complete!"
echo ""
echo "To use in markdown, replace PlantUML code blocks with:"
echo "  ![Diagram Name](docs/diagrams/diagram_name.png)"
echo ""
echo "For PDF export with pandoc, use image references instead of PlantUML blocks."
