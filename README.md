# DS701 Fall 2026 — Student Notebooks

Notebooks for **DS701: Tools for Data Science** (Boston University, Fall 2026).

- **Course site:** https://tools4ds.github.io/ds701/
- `jupyter_notebooks/` — one notebook per lecture, the target of the "Open in
  Colab" badge at the top of each lecture.
- `class_activity_notebooks/<activity>/` — in-class group activities. Each
  activity directory holds the notebook and its `tests/` directory, which the
  notebook downloads automatically when it runs on Colab so that
  `grader.check(...)` works.

## Running the notebooks

**On Colab:** use the badge on the course site, or open a notebook from the
Colab link above. Each notebook fetches the data files, helper modules and images
it needs from this repository, so nothing else has to be uploaded.

**On your own machine (VS Code or JupyterLab):** clone this repository and open a
notebook in place. The notebooks find `data/`, `scripts/` and `laUtilities.py`
in the clone, so keep the folder layout as it is. Create an environment first:

```sh
git clone https://github.com/tools4ds/DS701-Materials-FA26.git
cd DS701-Materials-FA26
python3 -m venv .venv && source .venv/bin/activate
pip install jupyter pandas numpy scipy matplotlib seaborn scikit-learn statsmodels networkx
```

`requirements.txt` lists every package the course uses, grouped by lecture
(some, such as `torch` and `transformers`, are large). Install what a lecture
needs when it asks for it. Pictures in the notebooks load from this repository
over the internet.

## For students

On Colab, save your own copy (**File → Save a copy in Drive**) before you start working —
edits to the opened copy are not persisted.

## Note

**This repository is generated.** Its contents are built from the course-notes
repository and overwritten on every sync, so changes made here are lost. File
issues about notebook content through the course's Piazza instead.
