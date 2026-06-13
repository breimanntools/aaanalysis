# Code-security hardening audit (issue #88)

This document records the audit of AAanalysis's code-security surface —
subprocess invocations, network fetches, and untrusted file parsing — and the
small, behaviour-preserving fixes applied. It is **not** a vulnerability-
reporting policy: per `.claude/rules/sharp-edges.md`, no `SECURITY.md` is added
and none is intended.

Most of the surface lives in the `*_pro` subpackages (gated behind the `pro`
extra: biopython, requests, the MEME suite for FIMO, external CLI binaries).

## Trust boundary (summary)

- **Backend trusts the frontend** for argument validation (house rule). But
  **external / remote input is untrusted**: anything fetched over the network
  (UniProt records, AlphaFold model/PAE files, an MSA from UniProt) and any
  user-supplied file parsed off disk (FASTA, PDB/mmCIF, the TSV/`.clstr` output
  of external clustering tools) is treated as untrusted and validated/bounded at
  the boundary.
- **No user input reaches a shell.** Every subprocess call uses an argument list
  (no `shell=True`, no f-string-built command), so shell metacharacters in a
  path or sequence id cannot be interpreted by a shell.

## 1. Subprocess call sites

`grep -rn "shell=True" aaanalysis/` returns **0**. All call sites use list-arg
form and cast numeric/option values explicitly.

| Call site | Binary | Disposition |
|---|---|---|
| `seq_analysis_pro/_scan_motif.py::_run_fimo` | FIMO (MEME) | **OK** — `subprocess.run(check=True)` with a list arg; all flag values cast (`str(float(...))`, `str(int(...))`); FASTA + MEME files come from `tempfile.TemporaryDirectory()`; `bg_file` validated as an existing file in the frontend. |
| `seq_analysis_pro/_backend/_utils.py::run_command` | CD-HIT / MMseqs2 shared runner | **OK** — `subprocess.Popen(cmd, …)` with a list arg (no `shell=True`); temp dirs via `tempfile.mkdtemp`. Note: the broad `except Exception` here re-raises as `RuntimeError` (it does not silence), so it is acceptable, though a narrower catch would be cleaner (left as a finding, see below). |
| `seq_analysis_pro/_backend/cd_hit.py::run_cd_hit` | cd-hit | **OK** — list arg; threshold/word-size/coverage/threads all `str(...)`-cast; in/out files under a `tempfile` dir. |
| `seq_analysis_pro/_backend/mmseq2.py::run_mmseqs2` | mmseqs | **OK** — three list-arg commands; `--min-seq-id` / `--threads` / `-k` / `-c` all `str(...)`-cast; db/cluster/tsv paths under a `tempfile` dir. |
| `data_handling_pro/_backend/struct_preproc/_chainsaw.py::run_chainsaw_on_entry` | ChainSaw (`get_predictions.py`) | **OK** — list arg `[sys.executable, str(script), "--structure_file", str(pdb_path)]`; `chainsaw_path` validated by `resolve_chainsaw_path` (must be a dir containing `get_predictions.py`); has `timeout=300`; `cwd` confined to the validated clone root. |

Planned MSA backends (#65: BLAST / MMseqs2 / Clustal / MUSCLE) inherit the same
rule: list-arg form, explicit `str(...)` casts, `tempfile` for scratch files.

## 2. Network fetches

| Call site | Endpoint | Disposition |
|---|---|---|
| `data_handling_pro/_backend/_fetch.py::http_get_` | shared transport seam | **OK** — single `_get_session().get(url, timeout=timeout)` with `timeout` defaulting to `30.0`; all pro fetchers route through it. |
| `data_handling_pro/_backend/annot_preproc/_uniprot.py::fetch_uniprot_json` | UniProtKB REST JSON | **OK** — goes through `http_get_` (timeout); accession interpolated into a fixed `…/uniprotkb/{acc}.json` template; non-200 → `RuntimeError`; transport error caught as `requests.RequestException` → `RuntimeError`. |
| `data_handling_pro/_backend/struct_preproc/_alphafold.py` (`_af_resolve_urls`, `fetch_af_file`) | AlphaFold-DB API + files | **OK** — through `http_get_` (timeout); 400/404 → soft "not in DB"; other non-200 → `RuntimeError`; download written via an atomic `.part` → `replace` to a caller-provided `out_folder`. |
| `seq_analysis_pro/_comp_seq_cons.py::get_msa` | UniProt `.fasta` | **HARDENED** — added the missing `timeout=` (was a bare `requests.get(url)`, the one gap the issue flagged). Now `requests.get(url, timeout=timeout)` with `timeout: float = 30.0`. See note below — this module is an orphan, unreachable from the public API. |

A grep/AST meta-test (`tests/unit/api_tests/test_network_timeout_hygiene.py`)
now asserts **100 %** of `requests.get` / session `.get` / `http_get_` calls in
`aaanalysis/` pass an explicit `timeout=`, so a future untimed fetch fails CI.

### `_comp_seq_cons.py` is an orphan module

`seq_analysis_pro/_comp_seq_cons.py` (`get_msa`, `comp_seq_cons`,
`map_known_mutations`) is **not imported anywhere** — not in any `__init__.py`,
not reachable from the public API — and is excluded from coverage in
`.coveragerc` (see `tests/README.md`). The missing-timeout fix was still applied
because the issue explicitly flagged it, but it cannot be exercised by a public-
API unit test. Whether to wire it up or remove it is a separate decision
(finding F3 below). It does **not** validate `uniprot_id`, so it is on the
"wire-it-up" backlog rather than a shipped surface.

## 3. File parsing

| Parser | Input | Disposition |
|---|---|---|
| `data_handling/_backend/parse_fasta.py::get_entries_from_fasta` | user FASTA | **OK (bounded)** — line-by-line read, no `eval`/`exec`, no archive expansion; a malformed header just yields odd columns rather than executing anything. |
| `seq_analysis_pro/_backend/cd_hit.py::_get_df_clust_cd_hit` | cd-hit `.clstr` (tool output, under `tempfile`) | **OK** — text parse of a file the package itself just produced in a temp dir; not user-supplied. |
| `seq_analysis_pro/_backend/mmseq2.py::_get_df_mmseq` | mmseqs `.tsv` (tool output) | **OK** — `pd.read_csv(sep="\t")` on a temp-dir file the package produced. |
| `_chainsaw.py::_parse_chainsaw_tsv` | ChainSaw stdout | **OK** — splits the captured stdout; raises `RuntimeError` on a missing `chopping` column / missing data row rather than mis-parsing. |
| `data_handling/_load_*.py`, `utils.read_csv_cached` | bundled `_data/` TSVs | **OK** — read only from the package's own `_data/` directory (trusted, shipped assets). |
| AlphaFold download → `fetch_af_file` | remote `.pdb`/`.cif`/`.json` | **OK** — written to a caller-provided `out_folder` under the canonical `<entry>.<fmt>` / `AF-<entry>-…json` names; the entry id originates from the user's `df_seq`, not from the remote response, so the saved filename is not attacker-controlled. |

## Findings left as decisions (not fixed here)

These would change behaviour or need a design decision, so per the issue they
are recorded rather than applied:

- **F1 — UniProt/AlphaFold accession pattern validation.** The issue suggests
  validating accessions against an expected pattern (e.g. `^[A-Z0-9]+$`) before
  interpolating into the URL. Today accessions come from the user's `df_seq`
  `entry` column (frontend-trusted), are URL-path-interpolated (not shell), and
  a bad one yields a clean 400/404 → soft "not in DB". Adding a hard regex
  reject is a small behavioural change (it would turn some currently-soft misses
  into hard `ValueError`s) and belongs in the frontend `# Validate` block of the
  StructurePreprocessor / AnnotationPreprocessor methods — deferred so this audit
  stays behaviour-preserving.
- **F2 — `run_command` broad `except Exception`.** `seq_analysis_pro/_backend/
  _utils.py::run_command` catches `except Exception` to guarantee temp-dir
  cleanup, then **re-raises** as `RuntimeError` (it does not silence). It is not
  a violation of the "no bare except to silence" rule, but a `try/finally` for
  the cleanup plus a narrower catch would be cleaner. Left as a finding to avoid
  touching the CD-HIT/MMseqs2 error path under this audit.
- **F3 — `_comp_seq_cons.py` wire-up-or-remove.** Orphan module (see above);
  needs an owner decision. If wired up, `get_msa` should validate `uniprot_id`
  and `comp_seq_cons` should fail closed on a malformed/empty MSA.
- **F4 — Response-size bounding.** No fetch streams or caps response size; an
  adversarial endpoint could return an unbounded body. UniProt/AlphaFold-EBI are
  trusted first-party endpoints and `requests` buffers into memory, so this is
  low-risk; streaming-with-a-cap is a larger change deferred to the #65 MSA work.

## KPIs (issue #88 acceptance)

- `grep -rn "shell=True" aaanalysis/` → **0**. ✔
- All 6 call-site groups audited (FIMO, CD-HIT, MMseqs2, ChainSaw, UniProt,
  AlphaFold + conservation HTTP), each marked OK/hardened above. ✔
- **100 %** of `requests.get` calls pass an explicit `timeout` — enforced by
  `tests/unit/api_tests/test_network_timeout_hygiene.py`. ✔
- No `SECURITY.md` added. ✔
- Accession/URL-pattern validation + path-traversal hard-reject left as findings
  F1/F4 (behaviour-changing; out of scope for a behaviour-preserving audit).
