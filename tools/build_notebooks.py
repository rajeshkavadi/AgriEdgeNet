"""Build the Colab notebooks from the percent-format sources in notebooks/src.

    python tools/build_notebooks.py

notebooks/src/*.py are the source of truth (diffable, testable with plain python).
Outputs:
    notebooks/01_rgb_segmentation_gli.ipynb
    notebooks/02_sequoia_alignment_indices.ipynb
    notebooks/03_combined_segmentation_indices.ipynb   (01 followed by 02, plus a closing cell)
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / 'notebooks' / 'src', ROOT / 'notebooks'


def parse(path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = '\n'.join(buf).strip('\n')
        if kind == 'markdown':
            text = '\n'.join(re.sub(r'^# ?', '', l) for l in text.splitlines())
        if text.strip():
            cells.append((kind, text))

    for line in Path(path).read_text().splitlines():
        if line.startswith('# %%'):
            flush()
            kind, buf = ('markdown' if '[markdown]' in line else 'code'), []
        else:
            buf.append(line)
    flush()
    return cells


def notebook(cells):
    def cell(kind, text):
        src = [l + '\n' for l in text.splitlines()]
        src[-1] = src[-1].rstrip('\n')
        c = {'cell_type': kind, 'metadata': {}, 'source': src}
        if kind == 'code':
            c.update(execution_count=None, outputs=[])
        return c

    return {'cells': [cell(k, t) for k, t in cells], 'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                         'language_info': {'name': 'python'}, 'colab': {'provenance': []},
                         'accelerator': 'GPU'}}


COMBINED_HEAD = """# Consolidated pipeline: RGB crop segmentation + Sequoia NDVI / NDRE

Part 1 is `01_rgb_segmentation_gli` and part 2 is `02_sequoia_alignment_indices`, unchanged and run back to back.

**What this notebook does not do:** it does not compute NDVI or NDRE per segmented crop patch. That needs the
RGB masks and the Sequoia bands on one shared pixel grid, which the current data cannot provide:

* the RGB frames come from a different camera (IMX219 parameters in the old notebook) than the Sequoia, with no
  shared georeference, and
* the 12 Sequoia frames (`IMG_260707_*`, 7 July 2026) show tree canopy over a road, not the five crops.

To get per-crop indices, either (a) fly the Sequoia over the crop plots and segment its own RGB or registered
bands, or (b) build orthomosaics of both sensors on a shared CRS and rasterise the masks onto the index grid."""

COMBINED_TAIL_MD = """## Package results"""
COMBINED_TAIL = """import shutil
bundle = Path('/content/Combined_Results') if IN_COLAB else Path('Combined_Results')
bundle.mkdir(exist_ok=True)
for src, name in ((RESULTS, 'Segmentation'), (SEQ_OUT_ROOT, 'Sequoia_Indices')):
    if Path(src).exists():
        shutil.copytree(src, bundle / name, dirs_exist_ok=True)
archive = shutil.make_archive(str(bundle), 'zip', bundle)
print('archive:', archive)
if IN_COLAB:
    from google.colab import files
    files.download(archive)"""


def main():
    seg = parse(SRC / '01_rgb_segmentation_gli.py')
    seq = parse(SRC / '02_sequoia_alignment_indices.py')
    outs = {
        '01_rgb_segmentation_gli.ipynb': seg,
        '02_sequoia_alignment_indices.ipynb': seq,
        '03_combined_segmentation_indices.ipynb':
            [('markdown', COMBINED_HEAD), ('markdown', '# Part 1: RGB crop segmentation')] + seg
            + [('markdown', '# Part 2: Sequoia calibration, alignment, NDVI and NDRE')] + seq
            + [('markdown', COMBINED_TAIL_MD), ('code', COMBINED_TAIL)],
    }
    for name, cells in outs.items():
        (OUT / name).write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + '\n')
        print(f'{name}: {len(cells)} cells')


if __name__ == '__main__':
    main()
