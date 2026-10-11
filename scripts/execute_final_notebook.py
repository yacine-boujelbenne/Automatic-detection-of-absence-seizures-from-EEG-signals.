"""Exécute toutes les cellules avec l'interpréteur courant et exporte une présentation HTML."""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/final"


def main():
    """Écrit le notebook exécuté seulement après réussite de toutes les cellules."""
    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="eeg-notebook-", dir=OUT) as runtime:
        execute(Path(runtime))


def execute(runtime):
    """Garde les fichiers du noyau et les caches dans un dossier temporaire."""
    notebook_path = ROOT / "notebooks/mini_projet_eeg_final.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    kernel_root = runtime / "kernels"
    kernel_path = kernel_root / "dm-project-local"
    kernel_path.mkdir(parents=True, exist_ok=True)
    specification = {
        "argv": [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Python (DM-project)",
        "language": "python",
        "env": {
            "MPLCONFIGDIR": str(runtime / "mplconfig"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }
    (kernel_path / "kernel.json").write_text(json.dumps(specification), encoding="utf-8")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime / "jupyter_runtime")
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    manager = KernelManager(
        kernel_name="dm-project-local",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
    )

    def progress(cell, cell_index, **_):
        if cell.cell_type == "code":
            print(f"Cellule exécutée : {cell_index + 1}/{len(notebook.cells)}", flush=True)

    client = NotebookClient(
        notebook,
        km=manager,
        timeout=1200,
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_executed=progress,
    )
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, OUT / "execution_incomplete.ipynb")
        raise
    finally:
        # Le gestionnaire fourni appartient à ce script : fermer le noyau avant
        # de retirer son dossier temporaire, notamment l'historique SQLite.
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        if client.kc is not None:
            client.kc.stop_channels()
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert all(output.output_type != "error" for output in cell.outputs)
    nbformat.write(notebook, notebook_path)
    exporter = HTMLExporter(
        template_name="lab", exclude_input_prompt=True, exclude_output_prompt=True
    )
    body, _ = exporter.from_notebook_node(notebook)
    html_path = ROOT / "notebooks/mini_projet_eeg_final.html"
    html_path.write_text(body, encoding="utf-8")
    print("Notebook intégralement exécuté, aucune erreur ; présentation HTML enregistrée.")


if __name__ == "__main__":
    main()
