"""
Script to Convert Jupyter Notebooks to RST for Documentation

This script facilitates the conversion of Jupyter tutorials into RST format without execution. The converted RST files
are intended for direct integration into Sphinx documentation. This approach ensures that the tutorials are integrated
into the documentation using their saved state, eliminating the necessity for further execution during the documentation build.

Procedure:
1. Ensure the Jupyter tutorials you want to include in the documentation are fully executed and saved with their outputs.
2. Run this script to convert these tutorials into RST format. (automatically in conf.py)
3. Include the generated RST files in the Sphinx documentation's toctree.

Before running this script, ensure the project is in 'editable' mode to maintain consistency across documentation:
- If using `poetry`:
    poetry install
- Alternatively, for traditional projects:
    pip install -e .

This ensures that when developers run their Jupyter tutorials, they reference the local package version.
"""
import re
import os
import nbconvert
import nbformat
from pathlib import Path
from nbconvert.writers import FilesWriter
from nbconvert.preprocessors import Preprocessor


# Folder and File Constants
SEP = os.sep
FOLDER_PROJECT = str(Path(__file__).parent.parent.parent) + SEP
FOLDER_SOURCE = os.path.dirname(os.path.abspath(__file__)) + SEP
FOLDER_NOTEBOOKS = FOLDER_PROJECT + "tutorials" + SEP
FOLDER_GENERATED_RST = FOLDER_SOURCE + "generated" + SEP  # Saving .rst directly in 'generated'
LIST_EXCLUDE = []


# Helper functions
class CustomPreprocessor(Preprocessor):
    def preprocess(self, nb, resources):
        """
        Rename image resources and update the notebook cells accordingly.
        """
        output_items = list(resources['outputs'].items())
        for idx, (output_name, output_content) in enumerate(output_items):
            # Formulate the new name
            new_name = f"NOTEBOOK_{idx + 1}_{output_name}"
            # Modify resources['outputs']
            resources['outputs'][new_name] = resources['outputs'].pop(output_name)
            # Update the notebook cell outputs
            for cell in nb.cells:
                if cell.cell_type == 'code' and 'outputs' in cell:
                    for output in cell['outputs']:
                        if "metadata" in output:
                            metadata = output["metadata"]
                            if "filenames" in metadata:
                                filename = metadata["filenames"]
                                if 'image/png' in filename:
                                    # Prevent overriding
                                    old_name = filename["image/png"]
                                    if old_name.split("output")[1] == new_name.split("output")[1]:
                                        filename['image/png'] = str(new_name)
                                        break
        return nb, resources


# Main function
def export_notebooks_to_rst():
    """Export Jupyter tutorials to RST without execution."""
    for filename in os.listdir(FOLDER_NOTEBOOKS):
        if filename.endswith('.ipynb') and filename not in LIST_EXCLUDE:
            full_path = os.path.join(FOLDER_NOTEBOOKS, filename)
            # Load the notebook
            with open(full_path, 'r') as f:
                notebook = nbformat.read(f, as_version=4)
            # Convert to notebook to RST
            custom_preprocessor = CustomPreprocessor()
            rst_exporter = nbconvert.RSTExporter(preprocessors=[custom_preprocessor])
            output, resources = rst_exporter.from_notebook_node(notebook)
            # Write the RST and any accompanying files (like images)
            writer = FilesWriter(build_directory=FOLDER_GENERATED_RST)
            writer.write(output, resources, notebook_name=filename.replace('.ipynb', ''))


