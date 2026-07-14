"""Bake a ``gene`` column into every bundled benchmark dataset (right after ``entry``).

Rationale: the bundled datasets carried only ``entry`` (a UniProt accession or a synthetic id),
so any readable label or gene-based ``sample`` selection had to be reconstructed by hand. This
one-off data-prep tool adds a first-class ``gene`` column so ``load_dataset`` is self-describing.

Sources:
- **Domain datasets** (``DOM_*``, whose ``entry`` is a UniProt accession): ``gene`` is the primary
  gene symbol fetched from the UniProt REST API (``gene_primary``). An accession UniProt cannot
  resolve falls back to the positional ``name_<row>`` placeholder.
- **Amino-acid / sequence datasets** (``AA_*`` / ``SEQ_*``, whose ``entry`` is a synthetic id like
  ``CASPASE3_1``): no gene exists, so ``gene`` is the positional ``name_<row>`` placeholder
  (``row`` is the 1-based position within the dataset).

Every other column and value is left byte-identical; only the ``gene`` column is inserted after
``entry``. Re-runnable (idempotent): re-inserting is skipped when ``gene`` is already present.

Usage: ``python .github/scripts/add_gene_names_to_datasets.py`` (run from the repo root).
"""
import os
import sys
import time
import urllib.request

FOLDER = os.path.join("aaanalysis", "_data", "benchmarks")
GENE_COL = "gene"
ACCESSION_PREFIX = "DOM_"          # datasets whose entries are UniProt accessions
UNIPROT_URL = ("https://rest.uniprot.org/uniprotkb/accessions"
               "?accessions={accs}&fields=accession,gene_primary&format=tsv")
BATCH = 100


def _accession_base(acc):
    """UniProt query key: drop an isoform suffix (``P05067-2`` -> ``P05067``)."""
    return acc.split("-")[0]


def fetch_gene_map(accessions):
    """Return ``{accession: gene_symbol}`` from UniProt for the given accessions (non-empty only)."""
    uniq = sorted(set(accessions))
    by_base = {}
    for i in range(0, len(uniq), BATCH):
        chunk = [_accession_base(a) for a in uniq[i:i + BATCH]]
        url = UNIPROT_URL.format(accs=",".join(chunk))
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = resp.read().decode()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        for line in data.strip().split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip():
                by_base[parts[0]] = parts[1].strip()
        time.sleep(0.3)
    return {a: by_base[_accession_base(a)] for a in uniq if by_base.get(_accession_base(a))}


def _read_rows(path):
    """Return ``(header, rows, trailing_newline)`` split on tab, preserving every field verbatim."""
    with open(path, newline="") as fh:
        text = fh.read()
    trailing = text.endswith("\n")
    lines = text.split("\n")
    if trailing:
        lines = lines[:-1]
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:]]
    return header, rows, trailing


def _write_rows(path, header, rows, trailing):
    out = "\n".join(["\t".join(header)] + ["\t".join(r) for r in rows])
    if trailing:
        out += "\n"
    with open(path, "w", newline="") as fh:
        fh.write(out)


def add_gene_column(path, gene_map):
    """Insert a ``gene`` column after ``entry`` in one dataset TSV (byte-identical otherwise)."""
    header, rows, trailing = _read_rows(path)
    if GENE_COL in header:
        return "skipped (already present)"
    entry_idx = header.index("entry")
    header.insert(entry_idx + 1, GENE_COL)
    n_gene = 0
    for i, row in enumerate(rows):
        entry = row[entry_idx]
        gene = gene_map.get(entry)
        if gene:
            n_gene += 1
        else:
            gene = f"name_{i + 1}"
        row.insert(entry_idx + 1, gene)
    _write_rows(path, header, rows, trailing)
    return f"{n_gene}/{len(rows)} real gene symbols, rest name_<row>"


def main():
    datasets = sorted(f for f in os.listdir(FOLDER)
                      if f.endswith(".tsv") and f != "Overview.tsv")
    # Collect accessions from the accession-based (domain) datasets and fetch their genes once.
    accessions = []
    for f in datasets:
        if f.startswith(ACCESSION_PREFIX):
            header, rows, _ = _read_rows(os.path.join(FOLDER, f))
            ei = header.index("entry")
            accessions += [r[ei] for r in rows]
    gene_map = fetch_gene_map(accessions) if accessions else {}
    print(f"UniProt resolved {len(gene_map)} gene symbols for {len(set(accessions))} accessions.")
    for f in datasets:
        status = add_gene_column(os.path.join(FOLDER, f), gene_map)
        print(f"  {f:22s} {status}")


if __name__ == "__main__":
    sys.exit(main())
