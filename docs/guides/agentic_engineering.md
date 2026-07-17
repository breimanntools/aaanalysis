# Agentic engineering protocol

How AAanalysis is developed with AI-assisted ("agentic") workflows — and, just as
importantly, the **automated gates every change must pass before a human merges it**.
The durable artifacts of this process (ADRs, `CONTEXT.md`, `CLAUDE.md` / `.claude/rules`,
and the code) are the single sources of truth; per-issue planning notes are ephemeral and
not committed.

## When to use this protocol

This is the discipline for **autonomously driving an entire issue or a complete new feature from
spec to merge** — a substantial bug-fix issue counts. It is deliberately heavy: it spins up a
worktree, opens a draft PR, and runs the full automated-review + ripple machinery. **Do not** run
it for typos, one-line fixes, or trivial local edits that won't reach a PR — make those directly.
The `agentic-engineering` skill executes this protocol and carries the same scope guard.

## Delegated skills (orchestration model)

The protocol is run by an **orchestrator** that delegates each specialist gate to a local
sub-skill — `/grill-with-docs` (spec-vs-code reality check), `/review` (PR-diff review),
`/security-review` (vulnerability scan), `/code-review high` / `ultra` (deeper quality review),
`/simplify` (complexity cleanup), `/docstrings` (public-API & docstring gate), and the issue-side
helpers (`/triage`, `/to-issues`, `/github-issues`) and `/schedule` (background sync). These are
sub-skills, not shell or GitHub commands. **If a required sub-skill is unavailable, fails, or
returns an inconclusive result, stop and surface the missing gate** rather than continuing as
though it passed — otherwise a real gate is skipped under the appearance of green. Continue only
after an explicit human decision to run a manual fallback or skip that gate. The human permission
gates (push, PR, merge, deletion, CONFIRM-FIRST files) override every delegated result.

## Workflow

Eight steps in three phases. Phases group the work; within the Build phase you iterate, so the
order there is natural rather than rigid. The full rationale lives in this section; the
`agentic-engineering` skill carries a one-line-per-step checklist that points back here.

### Prepare

1. **Pick & sharpen the issue.** No skill required to start. Optionally clean up or generate the
   wording with `/triage` or `/to-issues` (house style: `docs/guides/issue_style_guide.md`); for
   "what next?", `/github-issues` produces a prioritized, parallelization-aware plan.
2. **`/grill-with-docs` — the highest-leverage step.** Sharpen the spec against the *live*
   codebase and refresh `CONTEXT.md` / ADRs **before any code is written**. The closest built-in,
   `/init`, only (re)generates codebase docs — it does not do the adversarial spec-vs-reality
   pass; use `/init` only as a one-time bootstrap when `CONTEXT.md` does not exist yet.
3. **Branch + isolated worktree.** Use the explicit canonical command so the branch is created
   *directly in a fresh worktree* off `master` (no ambiguity about which checkout it lands in):

   ```bash
   git fetch origin
   git worktree add ../wt-<slug> -b <type>/<slug> origin/master   # fix/ feat/ doc/ refactor/
   cd ../wt-<slug>
   ```

   The worktree is what keeps concurrent streams from sharing a checkout (see Process notes →
   *Isolated worktrees*); isolation comes from `git worktree add`, never from `git stash`.

### Build

4. **Implement + open a draft PR early.** Use plan mode for multi-file or architectural changes
   (approve the approach before commits land); drop to plain edits for trivial diffs. Honor the
   path-scoped rules in `.claude/rules/` that auto-load for the files you touch. A PR needs ≥1
   commit, so push a scaffolding commit and open a **draft PR** early — CI and the Read the Docs
   (RTD) preview then run while you keep pushing to refine. **Push and open the PR *before* the
   human-review gate (step 6), not after** — opening the PR is what starts CI/RTD, so the checks run
   *while* the user reviews or decides to skip, instead of the user waiting on a cold start.
   **Before a substantive push, run the fast local unit gate**
   (`pytest tests -m "not regression" -x -n auto -c tests/pytest.ini`) — a local run catches the
   obvious break far faster than a red-CI round-trip. **No change is done until you have walked the
   ripple checklist below** — a code edit almost always has to land alongside its docstring, example,
   test, and other mirrors *in the same PR*.
5. **Automated review gate (machine-enforced, not eyeballed).** Run `/review` (reviews the PR
   diff) and `/security-review` (scans the pending diff for vulnerabilities); for a substantial
   diff reach for `/code-review high` (or `ultra` for the deep cloud review) and `/simplify`, and
   when public API or docstrings change run `/docstrings`. The quality gates below must be green.
   **Never merge red** — automated checks *gate* the human review, they do not replace it.
   *Meanwhile, as a background concern:* periodically merge `master` → branch to
   stay current (a good fit for `/schedule`: auto-**sync** each morning so you wake to a synced
   branch or an early conflict warning). **A scheduled job syncs only — it must never resolve
   conflicts or merge a branch to `master` unattended; it just flags you.**

### Review, merge & clean up

6. **Human review gate — the PR is already up; the user picks how to review.** You pushed and opened
   the PR back in step 4, so the GitHub Actions + RTD preview are **already running while the user
   decides** (confirm with `gh pr checks <n>` / `gh run list --branch <branch>`). This is a deliberate
   checkpoint for human judgement on the *content* of the change — **not** a decision about whether to
   push (that already happened), and distinct from / on top of the automated gates in step 5 and CI,
   which independently enforce *never merge red*. Do **not** advance to merge on your own; surface the
   fork explicitly and **wait for the user's decision**:

   - **(a) Manual PR review — iterate.** The user reviews the PR diff on GitHub and leaves comments.
     Address **each** comment by refactoring **forward on the same branch** (honor the *Propagate
     every change* ripple checklist and the auto-loaded `.claude/rules/`), re-run the fast local
     gate, push (*each re-push is a publish action → §0 go-ahead*), and report back what changed per
     comment. Then **loop**: wait for the next round of comments and repeat the *review → refactor →
     push* cycle. Stay in the loop until the user explicitly signals the review is complete (e.g.
     "merge it" / "looks good" / "skip further review") — only then proceed to step 7. Each re-push
     re-triggers CI, and any armed auto-merge simply waits for the new green, so iterating never
     races the merge.
   - **(b) Skip review — approve + auto-merge.** The user opts out of a manual pass: post a short
     **approving review comment** (e.g. "Skipping manual review — automated gates green, all good") so
     the skip is recorded on the PR, then proceed to step 7 to arm auto-merge. The PR merges and
     closes itself once CI is green.

   Recommend (a) for substantial or architectural diffs and (b) for trivial ones, but **never
   assume the answer — the user picks.** Holding here is also why the draft PR + CI run early
   (step 4): the user can review against a green, RTD-previewed PR rather than a moving target.
7. **Arm auto-merge; fix-forward on red.** Once the user has cleared the review gate (step 6) — on the
   skip path, after you've posted the approving review comment — and you've read the RTD preview + PR
   diff, enable GitHub-native auto-merge: **`gh pr merge --auto --merge`** — a **merge commit, never
   `--squash`** (squash rewrites the branch into a new SHA, which loses the individual commits *and*
   blinds the step-8 cleanup detection; merge commits keep `git branch --merged` / `-d` honest). The
   merge **method** is its own explicit decision — never fold `--squash`/`--merge` into the step-6
   skip-review option (a real incident: a PR was squash-merged because the flag rode along in an
   auto-merge option the user picked for its review-skip meaning). GitHub merges the moment every
   required check passes and the branch is conflict-free,
   so **"never merge red" still holds** — a red check blocks the merge instead of completing it. If
   CI goes red, pull the failing logs (`gh run view --log-failed`, or `gh run watch` to follow
   live), reproduce locally, and fix **forward on the same branch**; armed auto-merge re-arms itself
   and completes on the green re-run. Disarm with `gh pr merge --disable-auto` to hold the PR, or
   skip `--auto` for a hard human gate. Arming auto-merge is a publish action → needs the per-action
   go-ahead in root `CLAUDE.md` §0.
8. **Clean up — gated on merge + a green `master`.** Key cleanup off the **merge state, never a
   single CI run**: once `gh pr view <n> --json state,mergedAt` shows `MERGED`, let the
   push-triggered workflows on `master` run and **wait for them to pass** — that confirms the
   merge didn't break anything the branch CI couldn't see (master may have moved under it). An
   intervening push, from this session or another, just re-runs checks and armed auto-merge waits
   for the *new* green, so the trigger survives the race.

   **First refresh, then verify by reachability.** Run `git fetch origin --prune` to drop
   remote-tracking refs another session's merge+delete left behind. Because PRs land as **merge
   commits** (never squash), the branch's commits are reachable from `master`, so
   `git branch --merged master` lists it and a plain **`git branch -d <branch>`** deletes it safely —
   no `-D` force, no `git diff master...<branch>` workaround. (The `MERGED` PR state is the
   confirming cross-check.) A branch with **no PR and a non-empty `git diff master...<branch>`** is
   *forgotten work* — never delete it; surface it for a human.

   Then, **with permission (root `CLAUDE.md` §0, per-action)** and after that verification —
   `git switch master` → `git worktree remove <path>` (a tree with uncommitted work needs
   `--force`, which also needs permission) → `git worktree prune` → `git branch -d <branch>`. The
   remote branch is auto-deleted by GitHub on merge only when the repo's "automatically delete
   head branches" setting is on; otherwise `git push origin --delete <branch>` (a push → §0).

   **Canonical tool — `python .github/scripts/prune_merged_branches.py`.** PR-state-driven (not
   reachability), idempotent, report-only by default; `--apply` deletes the merged/contained local
   branches (removing a clean worktree first), `--remote --apply` also prunes stale remote branches,
   and it **never** deletes a FORGOTTEN no-PR branch. Prefer it over ad-hoc deletes and run it from
   *any* session or as a periodic reconcile — because parallel auto-merges routinely land *after* the
   session that opened the PR has ended, so per-session end-of-run cleanup alone can't keep up. A
   scheduled/unattended job may *detect and flag* "ready to clean up" but must never delete on its
   own (§0) — the same sync-only discipline as the step-5 background sync.

## Propagate every change — the ripple checklist

**No code change is complete until every surface that mirrors it is updated in the same PR**
(tutorials may trail in a separate PR, but never a different release). The package is heavily
cross-referenced — docstrings include example notebooks, the cheat sheet and tables are generated
from the public API, meta-tests assert consistency across files — so "I only touched code" is
almost never true. When you change code, walk this list and update what applies:

- **Docstrings** — numpydoc on every changed class / method / function; new `[Key]_` citations go
  in `docs/source/index/references.rst`, new conventions/abbreviations in `docstring_guide.rst`.
  *Enforced:* the `/docstrings` checkers (now blocking CI).
- **Public API** — `aaanalysis/__init__.py` `__all__` for any symbol added / removed / renamed
  (**CONFIRM-FIRST**); the API reference (`docs/source/api.rst` + autosummary `generated/`) flows
  from `__all__` + docstrings, so it updates with them.
- **Examples** — `examples/<abbr>_<method>.ipynb` (one per public method, included into the
  docstring via `.. include:: examples/<name>.rst`): cover every public parameter and re-run with
  executed outputs (`aa.display_df(...)`, `plt.show()`).
- **Tutorials** — `tutorials/*.ipynb` when the change alters a taught workflow.
- **Protocols** — `protocols/protocol<N>_*.ipynb` when an end-to-end workflow changes
  (`docs/guides/protocol_style_guide.md`); they render under *Examples : Protocols* on RTD.
- **Tests** — unit tests for the change itself, plus the cross-file meta-tests that catch drift:
  parameter coverage (`tests/unit/api_tests/test_param_coverage.py`), class-abbreviation registry
  (`test_class_abbreviation_registry.py`), backend import hygiene, and the extras / missing-stub
  parity tests.
- **Cheat sheet** — `docs/_cheatsheet/content.py` (single source of truth → regenerate the html /
  pdf); every snippet must use only public `__all__` symbols with real signatures.
- **Tables** — `docs/source/index/tables*.rst` (generated by `docs/source/create_tables_doc.py`)
  when scales / datasets / overview rows change.
- **Release notes** — `docs/source/index/release_notes.rst` (the changelog): add an entry under
  the current *Unreleased* section.
- **Contributing** — `CONTRIBUTING.rst` **and** its RST port
  `docs/source/index/CONTRIBUTING_COPY.rst` when the dev process changes.
- **Glossary / ADRs** — `CONTEXT.md` when terminology shifts; a new `docs/adr/NNNN-*.md` for an
  architectural decision (ideally settled in step 2 via `/grill-with-docs`). Draft it number-less
  and assign the number last — see Process notes → *ADR numbering*.
- **Conventions** — `CLAUDE.md` / `.claude/rules/*` when the change establishes or alters a rule.
- **Build / deps** — `pyproject.toml` (**CONFIRM-FIRST**) for dependency, extras, or version bumps;
  `aaanalysis/config.py` (**CONFIRM-FIRST**) for a new global option.

Most of these surface late — a stale cheat sheet, a red meta-test, a wrong RTD render — not in the
fast unit job. Check them at implement time (step 4), not after CI goes red.

## Quality gates — how AAanalysis checks each

> **Rule: never merge red.** These automated checks gate the merge. With
> `gh pr merge --auto` GitHub enforces it for you — it completes the merge only once every
> required check is green and the branch is conflict-free.

| Gate | "Green" means | AAanalysis mechanism |
|---|---|---|
| **Tests** | full matrix passes | `.github/workflows/main.yml` ("Unit Tests"): `pytest tests -m "not regression and not integration and not e2e" -x -n auto` on **Ubuntu py3.10–3.14** + **Windows py3.10 & 3.14** (Windows brackets min+max; the full Windows range and the `-m regression` exact-value CPP anchor run in the nightly — ADR-0015). |
| **Integration & E2E** | cross-component seams + protocol workflows pass | `.github/workflows/integration_e2e.yml` ("Integration & E2E Tests"): `pytest tests/integration tests/e2e --ignore=…/test_online_fetches.py -m "not regression" -n auto` on **Ubuntu py3.10 + 3.14** (core-only, offline). The fourth master-gating workflow (push + PR to master); excluded from the Unit Tests matrix so it runs once — ADR-0031. |
| **Coverage** | **≥ 88 %** line coverage, package-only | `.github/workflows/test_coverage.yml`: `pytest … --cov=aaanalysis --cov-fail-under=88` (+ Codecov `patch` / `project` / `project/cpp_core`). Measured on the package only (`--cov=aaanalysis`, never `--cov=./`) — ADR-0016. |
| **Packaging** | built wheel + sdist install clean; full public API imports; bundled `_data` loads | `.github/workflows/packaging.yml` ("Packaging", push + PR to master): builds sdist + wheel with the default `python -m build` (the sdist ships the Cython `.pyx` via `MANIFEST.in`, so the wheel-from-sdist chain works), installs **both** the wheel and the sdist into separate fresh base-deps-only venvs, and runs `tests/_check_public_api_packaged.py` against each — every `__all__` symbol imports, `pro`/`dev` symbols degrade to `missing_feature_stub`, `load_*` reads `_data`. **Ubuntu py3.10 + 3.14**, one wheel built per version (ABI-specific). Catches missing package-data, broken re-exports, or a broken source install the editable dev matrix cannot — ADR-0060. |
| **Docs** | RTD builds; API + examples render | `docs/readthedocs.org` check: Sphinx + nbsphinx. `docs/source/conf.py` runs `export_example_notebooks_to_rst` with `nbsphinx_execute='never'` — it **renders committed notebook outputs, it does not execute them**. |
| **Docstrings** | numpydoc shape, named `Returns`, per-method `Examples` include, no doc-vs-signature drift | the `/docstrings` skill: `check_docstrings.py`, `doc_signature_drift.py`, `check_example_notebooks.py` under `.claude/skills/docstrings/scripts/`. `check_docstrings` + `doc_signature_drift` are **blocking CI** in the `codeql_analysis.yml` "code-quality" job; `check_example_notebooks` runs there **advisory (non-blocking)** until the remaining notebook param-coverage gaps are cleared. All three also run locally via the skill. |
| **Notebooks execute** | every `examples/` + `tutorials/` notebook runs clean with embedded outputs | `pytest --nbmake --nbmake-timeout=120 examples/ tutorials/`. **Local gate only — NOT in blocking CI.** Re-run and re-commit outputs before every push (see Process notes). |
| **Architecture** | matches `CONTEXT.md` / ADRs; no cross-class backend imports or layering violations | machine: `tests/unit/api_tests/test_backend_import_hygiene.py` (runs inside `pytest tests`). Spec / ADR conformance is human + `/grill-with-docs`. |
| **ADR hygiene** | no duplicate ADR numbers (the parallel-session collision), cross-refs resolve, status lines valid, `INDEX.md` current | `.github/workflows/adr_hygiene.yml` (`paths: ['docs/adr/**']`, push + PR) runs `check_adrs.py`. Number-last drafting is the prevention (Process notes → *ADR numbering*); this is the backstop. Gaps are not flagged (ADRs legitimately live on unmerged branches). |
| **Numerical equivalence** | an output-affecting optimization declares + satisfies a tolerance tier | human at review against the **numerical-equivalence tolerance policy** (ADR-0032): T1 byte-identical (default) / T2 `allclose(atol=1e-10, rtol=0)` + identical discrete decisions / T3 quality-metric within a documented band. Reviewer checklist (tier named, validation harness linked, tolerance/band stated numerically, ADR-0015-style regression anchor committed, one optimization per PR) lives in `CONTRIBUTING.rst`; anchors run in the non-gating nightly. |
| **Parameter coverage** | every public parameter is exercised by name in tests | `tests/unit/api_tests/test_param_coverage.py` — runs in the "Unit Tests" job (it is an ordinary test under `tests/`, picked up by `pytest tests`). |
| **Lint (errors)** | no syntax errors / undefined names | `.github/workflows/codeql_analysis.yml` ("code-quality" job): `flake8 . --select=E9,F63,F7,F82`. |
| **Types (pyright ratchet)** | pyright diagnostic count does not rise above the committed baseline (`.github/pyright_baseline.txt`) | `.github/workflows/pyright.yml` runs `.github/scripts/check_pyright_budget.py`, which **gates** on a count *above* the baseline (blocks new diagnostics; merging at or below is fine). Public-API-first (`pyrightconfig.json` excludes `_backend`); pyright version pinned so the baseline is reproducible. Not a clean/strict gate. mypy is not used. |
| **Style (full)** | black (88) / isort / flake8 (88) | **manual at review** — no pre-commit or ruff in CI (v2 target; `.claude/rules/sharp-edges.md`). |
| **Security** | CodeQL clean | `.github/workflows/codeql_analysis.yml` ("Analyze" job). No separate dependency-scan PR gate yet — worth adding. |
| **Perf A/B (gating)** | no covered hot path slower than its band vs the latest PyPI release (`1.3x` meaty / `2.0x` sub-2ms) | `.github/workflows/perf_nightly.yml` runs the benchmark suite (`tests/benchmarks/`, opt-in `[bench]` extra) **twice on the same runner** — current working tree and latest stable release installed `--no-deps` on the same deps — and `.github/scripts/check_perf_regression.py current_run.json --baseline-run released_run.json` compares them. Same-runner A/B cancels hardware/dependency variance, so it **gates merges** on PRs + master push (ADR-0037; supersedes the old non-gating static-baseline). No committed baseline. Locally: see the A/B recipe in the `tests/benchmarks/test_perf_hot_paths.py` module docstring (issue #187). |
| **Issue linkage** | the PR's lifecycle keyword is set per policy | `Closes #NN` in the **PR body** (see Process notes → *Issue lifecycle*). |

## Process notes (hard-won)

- **Isolated worktrees per task.** Create a `git worktree` per branch so two concurrent
  streams never share one working tree / `HEAD`. Sharing a single checkout caused commits to
  land on the wrong branch and uncommitted work to bleed across tasks. A worktree also lets you
  build and verify one branch without disturbing another in-flight branch — do the edits in the
  worktree, commit, push, then `git worktree remove`. **If you run parallel agents the per-task
  worktree is mandatory:** in a shared checkout a concurrent agent's commit/push can land
  *between* your `git status` and your commit (observed: a cheat-sheet refactor was committed by
  another process mid-task, right after this guide's own redundancy-cleanup edits were
  inspected). Mitigate even in a shared tree — re-check `git status` immediately before staging,
  commit **explicit pathspecs only** (never a blind `git add -A` / `git commit -a`), and never
  commit, revert, or discard changes you did not make; stop and surface unexpected edits instead.
  The requirement is really *one working tree + `HEAD` per concurrent stream* — a `git worktree`
  (lighter) or a separate clone both satisfy it; a single shared checkout is safe only for strictly
  serial work. And **commit early**: a stray `git switch` / `git reset --hard` can only destroy
  *uncommitted* work (it once wiped a session's edits when the tree was reset to `origin/master`),
  so frequent commits are the cheapest safeguard. To see what other streams are in flight, derive
  it on demand from `git worktree list` + `gh pr list` — there is no committed board to maintain.
- **ADR numbering — allocate last, not while drafting.** Parallel branches each pick `max + 1`
  from a stale checkout, so two ADRs collide on a number or leave a gap (a real incident left
  `0034`–`0036` unindexed). Draft number-less (`# ADR-XXXX —`, `docs/adr/XXXX-<slug>.md`); de-dup
  the *decision* up front via `/grill-with-docs` + a scan of open PRs; then assign the number only
  as the ADR's PR is about to merge — one past the max across committed ADRs **and** open PRs on a
  freshly-fetched `origin/master` — and regenerate `INDEX.md`. The index row and sequential
  filename make a genuine duplicate surface as a merge conflict. Convention home:
  `docs/adr/README.md` → *Conventions*.
- **Issue lifecycle — `Closes #NN`.** GitHub auto-closes a referenced issue on merge to the
  default branch when a closing keyword (`Closes` / `Fixes` / `Resolves #NN`) appears in **either
  the PR body or the merge commit message**. To **keep an issue open through a merge,
  remove the keyword from the PR body before merging** — fixing only the commit-message text is
  not enough. To auto-close, keep `Closes #NN` in the PR body.
- **Auto-merge + auto-fix loop.** `gh pr merge --auto --merge` (a **merge commit, never `--squash`**)
  is the default finish: it is
  safe because GitHub merges only on all-green + conflict-free, preserving *never merge red*. When
  a check goes red, **fix forward on the same branch** (`gh run view --log-failed` → reproduce
  locally → push) — the armed auto-merge needs no re-issuing and completes on the green re-run.
  Use `gh pr merge --disable-auto` to hold a PR. Arming auto-merge is still a publish action, so
  it needs the per-action go-ahead in root `CLAUDE.md` §0, exactly like a manual merge.
- **Notebooks are a local-only gate.** Because nbmake is not in blocking CI, a broken example
  surfaces only on RTD (as wrong/un-rendered output) or in a local run. Always run
  `pytest --nbmake examples/ tutorials/` and commit fresh outputs before pushing.
