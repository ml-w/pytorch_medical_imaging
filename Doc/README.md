# Building the Documentation

The docs use [Sphinx](https://www.sphinx-doc.org/) with the [Furo](https://pradyunsg.me/furo/) theme.

## 1. Install dependencies

From the repo root, install the package with the `doc` extras:

```bash
pip install -e ".[doc]"
```

This pulls in Sphinx, Furo, and all required extensions (`sphinx-copybutton`, `sphinxext-opengraph`, `sphinxcontrib-mermaid`, `m2r2`).

## 2. Build the HTML docs

```bash
cd Doc
make html
```

Output lands in `Doc/html/`.  Open `Doc/html/index.html` in a browser to review it.

## 3. Live-reload during writing

`sphinx-autobuild` watches the source tree and refreshes a local browser tab automatically:

```bash
cd Doc
sphinx-autobuild source html
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## 4. Clean a stale build

```bash
cd Doc
make clean
```

This removes the `build/` and `html/` directories so the next `make html` starts from scratch.  Useful when renaming or deleting RST files.

## 5. Windows (no `make`)

Use `make.bat` instead:

```bat
cd Doc
make.bat html
```

## Source layout

```
Doc/
├── README.md            # this file
├── Makefile             # Linux / macOS build entry point
├── make.bat             # Windows build entry point
└── source/
    ├── conf.py          # Sphinx configuration (theme, extensions, intersphinx)
    ├── index.rst        # root toctree
    ├── Introduction.rst
    ├── Tutorial/
    ├── PMIData/
    ├── PMIDataLoader/
    ├── Solvers/
    ├── Inferencers/
    ├── Networks/        # UNet, UNet_p, VNet, Layers
    ├── Loss/
    ├── Metrics/
    ├── Utils/
    ├── controller.rst
    ├── loggers.rst
    └── dev_notes.rst
```

## Docstring style

All docstrings in this project follow **Google style** with hanging-indent arguments:

```python
def fn(x, spacing):
    """One-line summary.

    Args:
        x (np.ndarray):
            Description of x.
        spacing (tuple of float):
            Voxel spacing in mm.

    Returns:
        float: The computed metric.
    """
```

The Napoleon extension parses both `Args` / `Returns` and the custom `Keys` / `Class Attributes` sections defined in `conf.py`.
