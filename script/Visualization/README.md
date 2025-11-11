# Visualization Workflow for YOLO Segmentation Annotations

## Overview

This directory contains visualization tools for quality assurance and interpretability of instance segmentation annotations in the context of citizen science tree species identification. The visualization workflow enables researchers to validate YOLO-format polygon annotations by overlaying them on source imagery, facilitating manual inspection of automated segmentation outputs.

## Scientific Context

In computer vision pipelines for ecological applications, particularly those involving citizen science imagery, visual validation of machine-generated annotations is critical for ensuring data quality and model reliability. This workflow addresses the need for human-interpretable representations of instance segmentation masks, which are essential for:

1. **Quality Control**: Validating automated segmentation outputs against ground truth imagery
2. **Error Analysis**: Identifying systematic segmentation errors or edge cases
3. **Model Interpretability**: Understanding model behavior across different species and image conditions
4. **Annotation Refinement**: Guiding iterative improvements to segmentation algorithms

## Primary Script: `overlay_yolo_polygons.py`

### Methodology

The core visualization tool implements a polygon overlay algorithm that renders YOLO-format instance segmentation annotations onto source imagery using semi-transparent color overlays. The workflow follows these computational steps:

#### 1. Data Ingestion

The script accepts three primary inputs:
- **Source Image**: Original RGB imagery in standard formats (JPEG, PNG)
- **YOLO Label File**: Text file containing normalized polygon coordinates
- **Class Names** (optional): Taxonomic or functional class labels

#### 2. Polygon Parsing Algorithm

YOLO segmentation labels follow the format:
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Where:
- `class_id`: Integer class identifier (0-indexed)
- `xi, yi`: Normalized coordinates in [0, 1] range representing polygon vertices

The parser (`parse_yolo_seg_txt`) performs:
- Line-by-line tokenization of annotation files
- Validation of minimum vertex count (≥3 for valid polygons)
- Coordinate normalization clamping to ensure [0, 1] bounds
- Robust error handling for malformed annotations

#### 3. Coordinate Transformation

Normalized polygon coordinates undergo affine transformation to pixel space:

```python
pixel_x = normalized_x × image_width
pixel_y = normalized_y × image_height
```

This transformation preserves aspect ratio and enables accurate overlay on arbitrary image dimensions.

#### 4. Color Assignment Strategy

A deterministic color assignment function (`color_for_class`) generates visually distinct colors for each class using HSV color space:

```python
H = (class_id × 37) mod 180
S = 200 (fixed saturation)
V = 255 (maximum value/brightness)
```

This approach ensures:
- Perceptual distinctiveness between adjacent class IDs
- Consistent color assignment across multiple visualizations
- Conversion to BGR color space for OpenCV compatibility

#### 5. Rendering Pipeline

The overlay generation employs a two-layer composition strategy:

**Layer 1 - Filled Polygons (Semi-transparent)**:
- Polygons rendered as filled regions on separate overlay image
- Alpha blending factor (default: 0.4) controls transparency
- Enables simultaneous visualization of multiple overlapping instances

**Layer 2 - Polygon Boundaries**:
- High-contrast outlines (thickness: 2px) with anti-aliasing
- Rendered directly on source image for crisp edge definition

**Layer 3 - Class Labels**:
- Text annotations positioned near polygon centroids
- Font: Hershey Simplex (0.5 scale)
- Color-matched to polygon for visual association

#### 6. Image Composition

Final output generated via weighted alpha blending:

```python
output = overlay × α + image × (1 - α)
```

Where α is the user-specified transparency parameter.

### Usage Patterns

#### Command-Line Interface

```bash
python script/Visualization/overlay_yolo_polygons.py \
    --image /path/to/image.jpg \
    --labels /path/to/labels/image.txt \
    --names /path/to/class_names.txt \
    --alpha 0.4 \
    --save /path/to/output.jpg \
    [--no_show]
```

**Parameters**:
- `--image`: Path to source imagery (required)
- `--labels`: Path to YOLO annotation file (required)
- `--names`: Optional class name mapping file
- `--alpha`: Transparency factor [0.0, 1.0], default 0.4
- `--save`: Output path for rendered overlay
- `--no_show`: Suppress interactive display (batch processing mode)

#### Batch Processing Workflow

For systematic validation across datasets:

```bash
# Example batch processing script
for img in /dataset/images/*.jpg; do
    base=$(basename "$img" .jpg)
    python script/Visualization/overlay_yolo_polygons.py \
        --image "$img" \
        --labels "/dataset/labels/${base}.txt" \
        --names /dataset/names.txt \
        --save "/output/overlays/${base}_overlay.jpg" \
        --no_show
done
```

## Technical Specifications

### Dependencies

- **OpenCV (cv2)**: Image I/O, geometric transformations, rendering
- **NumPy**: Array operations, coordinate transformations
- **Python Standard Library**: argparse, os, typing

### Performance Characteristics

- **Time Complexity**: O(n × m) where n = number of instances, m = average vertices per polygon
- **Memory Complexity**: O(w × h × 3) for image buffers (width × height × channels)
- **Typical Execution Time**: <100ms for standard images (1024×768) with <50 instances

### Input Validation

The script implements comprehensive input validation:
- File existence checking for all input paths
- Image readability verification
- Polygon vertex count validation (minimum 3 points)
- Coordinate normalization enforcement ([0, 1] clamping)
- Graceful handling of empty label files

## Integration with Research Pipeline

This visualization tool integrates with the broader segmentation workflow:

```
Data Collection → SAM Segmentation → YOLO Format Conversion → [Visualization QA] → Model Training
```

The visualization step serves as a quality gate, enabling researchers to:
1. Verify segmentation accuracy before model training
2. Identify systematic errors in automated annotation
3. Make informed decisions about data filtering and augmentation
4. Document annotation quality for publication

## Output Specifications

### Visual Output

Generated overlays include:
- **Semi-transparent filled polygons**: Enable assessment of segmentation boundaries
- **High-contrast outlines**: Facilitate precise boundary inspection
- **Class labels**: Support taxonomic verification
- **Color-coding**: Enable rapid multi-class differentiation

### File Formats

- **Input**: JPEG, PNG (RGB or grayscale)
- **Output**: JPEG (default), PNG (lossless)
- **Recommended**: PNG for quality control, JPEG for web visualization

## Quality Assurance Applications

### Use Cases

1. **Post-SAM Segmentation Validation**: Verify Segment Anything Model outputs
2. **Pre-Training Annotation Review**: Ensure label quality before YOLO training
3. **Model Prediction Inspection**: Visualize inference outputs for error analysis
4. **Inter-Annotator Agreement**: Compare human and automated annotations
5. **Publication Figures**: Generate publication-quality visualization outputs

### Validation Checklist

When using visualization for QA:
- ✓ Verify polygon boundaries align with object edges
- ✓ Check for over-segmentation (excessive fragmentation)
- ✓ Identify under-segmentation (merged instances)
- ✓ Validate class label accuracy
- ✓ Detect coordinate normalization errors
- ✓ Assess segmentation consistency across similar specimens

## Limitations and Considerations

1. **Display Resolution**: Interactive display limited by screen resolution; use `--save` for full-resolution inspection
2. **Color Distinctiveness**: With >50 classes, color differentiation may become challenging
3. **Overlapping Instances**: Heavy overlap may obscure individual polygons; adjust `--alpha` parameter
4. **Memory Constraints**: Very large images (>10MP) may require memory optimization

## Related Visualization Tools

This directory is part of a comprehensive visualization ecosystem:

- **`../util/Mask_check_plot.py`**: Binary mask visualization for segmentation QA
- **`../util/training_validation_loss_plot.py`**: Model training diagnostics
- **`../visualize_val_predictions.py`**: Batch validation set visualization
- **`../Labelling/quick_visual_sanity_check.py`**: Rapid annotation sanity checking

## Best Practices

### For Research Workflows

1. **Systematic Sampling**: Visualize random samples from each class/species
2. **Edge Case Focus**: Prioritize challenging images (occlusion, low contrast)
3. **Version Control**: Save overlays with version tags for longitudinal QA
4. **Documentation**: Maintain logs of visualization-identified issues

### For Publication

1. **High Resolution**: Use PNG format with `--alpha 0.3` for clarity
2. **Representative Selection**: Choose exemplar images across phenotypic variation
3. **Annotation**: Add scale bars and taxonomic labels in post-processing
4. **Accessibility**: Consider colorblind-friendly palettes for multi-class visualization

## Future Enhancements

Potential extensions to the visualization workflow:
- Interactive polygon editing interface
- Confidence score visualization (for model predictions)
- Side-by-side ground truth comparison mode
- Quantitative metrics overlay (IoU, Dice coefficient)
- 3D visualization for multi-view datasets

## Citation

If using this visualization workflow in published research, please acknowledge:
```
Visualization tools from the Understanding Citizen Science Species Segmentation project
Repository: https://github.com/santoshbhatt21/Understanding_citizenscience_species_segmentation
```

## Contact and Support

For issues, enhancements, or methodological questions:
- Open an issue in the repository
- Consult the main README for project context
- Review example outputs in the documentation

---

**Last Updated**: November 2025  
**Maintainer**: Santosh Bhatt  
**License**: As per repository root
