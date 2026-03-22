# Asset Layout

This folder is the single source of media assets for the offline homepage.

## Logos
- Path: `asset/logos/`
- Current files are loaded by `site/assets/js/data/benchmark.js`

## Slider Results
- Path: `asset/slider/<expression>/<intensity>.jpg`
- Example: `asset/slider/happy/1.0.jpg`
- Expression names are lowercase:
  - `happy`, `sad`, `angry`, `fear`, `surprise`, `disgust`
  - `confused`, `contempt`, `confident`, `shy`, `sleepy`, `anxious`

You can change file naming rules later by editing:
- `site/assets/js/data/expressions.js`
