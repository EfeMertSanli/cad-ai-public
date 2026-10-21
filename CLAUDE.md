# CLAUDE.md

Project: cad-ai. Learns the path from an architect's raw DWG to the electrical project from the office's archive. Read `README.md`, then `docs/architecture.md` and `docs/decisions.md`. The user writes Turkish; answer in Turkish. Code, identifiers and file names are English; Every text a person reads is written in Turkish and English where it is made (ADR-033): `say(tr, en)` in Python, `T(tr, en)` / `data-en` in the pages; `tests/test_i18n.py` fails on Turkish without its English.

## Hard rules
1. **The archive is read-only.** Never write into `CADAI_PROJECTS` (Desktop\proje\Proje). Converted files, caches and models go to `CADAI_DATA` (D:\cad-ai-data).
2. **No client drawings or anything derived from them in git** (DXF, scenes, labels, features, models, PNGs). They all live under `CADAI_DATA`. Only exception: the anonymised projects in `demo/`, built by `python -m cadai.demo.build` (ADR-032).
3. **C: has ~2 GB free.** Never put large files on C:.
4. **Every non-obvious decision → an ADR** in `docs/decisions.md`; reference it in code comments.
5. **Every bug fix comes with a test** that fails before the fix. Run `D:\cad-ai-data\venv\Scripts\python -m pytest -q` before finishing.
6. **Readable code over clever code.** Small modules with a docstring that says what and why. No dead code, no stubs that pretend to work.
7. **Never invent regulation values.** Unknown values are marked `TODO-ENGINEER`.
8. Long work runs as a job (`cadai/jobs.py`) so it is visible on the debug panel; don't block the web server.
9. Before killing a process, check its command line (other Python programs run on this machine).
