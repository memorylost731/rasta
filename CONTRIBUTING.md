# Contributing to Rasta

## Development Setup

```bash
git clone https://github.com/memorylost731/rasta.git
cd rasta
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check rasta/ tests/
ruff format rasta/ tests/
```

## Adding a New Material

1. Add entry to `MATERIALS` dict in `rasta/texture_identify.py`
2. Add OpenCV signature to `OPENCV_SIGNATURES` in same file
3. Add Three.js defaults to `THREEJS_DEFAULTS` in `rasta/texture_to_planner.py`
4. Add react-planner texture to `PLANNER_TEXTURE_MAP` in same file
5. Add tile size to `MATERIAL_TILE_SIZES` in `rasta/texture_extract.py`
6. Add test in `tests/test_materials.py`
7. Update `docs/materials.md`

## Pull Requests

- Keep PRs focused on a single change
- Include tests for new functionality
- Update docs if adding/changing API endpoints
- Run `ruff check` and `pytest` before submitting

## Architecture

See [README.md](README.md#architecture) for the pipeline diagram.

Key design decisions:
- **Ollama-first classification**: Better accuracy than OpenCV, but OpenCV fallback ensures offline operation
- **Seamless textures**: Mirror-blend technique, no content-aware fill dependency
- **ITU-R P.2040 RF data**: Industry-standard attenuation values, not empirical guesses
- **react-planner compatibility**: Output matches the exact schema react-planner expects
