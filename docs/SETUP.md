# Setup

Once assembled, the folder is self-contained: it carries its own CPython
3.8.10 (`python38\`), its own packages (`venv\`) and its own model weights
(`models\`). Nothing in it reads from `C:\`, from a package index, or from the
internet.

It arrives in **two pieces**, because the environment is 1.8 GB of binaries and
two files inside it are over GitHub's 100 MB hard limit:

| Piece | Where | What |
|---|---|---|
| Source | this git repository | `src\`, `scripts\`, `tests\`, `config\`, `docs\`, `models\` |
| Environment | Google Drive archive | `venv\` and `python38\` |

---

## 1. Assemble the folder

Everything must end up at exactly this path:

```
E:\Project\voice-indentification
```

Note the spelling: **indentification**, with the `n`. It is a typo in the
original directory name and it is now load-bearing — see §4.

**1a. Source.** Clone, or copy the working tree from another machine:

```powershell
git clone git@github.com:anderson-venture/voice-indentification.git E:\Project\voice-indentification
```

**1b. Environment.** Download the archive and extract it so that `venv\` and
`python38\` land directly inside the folder:

```
Drive link: https://drive.google.com/file/d/1MHoy-q3j0NVKxb3QmDo_vkuAaWR5iXsf/view
File:       voice-indentification-env-py38-win64.zip   (433 MB, 42,821 files)
```

Download it **in a browser** — Drive serves an HTML page instead of the zip to
`curl` and `wget` at this size. Then:

```powershell
cd E:\Project\voice-indentification
Expand-Archive .\voice-indentification-env-py38-win64.zip -DestinationPath . -Force
Remove-Item .\voice-indentification-env-py38-win64.zip
```

Afterwards `venv\Scripts\python.exe` and `python38\python.exe` must both
exist. §3 confirms the rest.

The download needs a network, so it happens on a connected machine. To reach a
fully offline machine, assemble the folder first and carry the whole thing on a
USB drive.

**1c. Leave `data\` behind.** It holds the encrypted voiceprint database and
the PBKDF2 salt, and a new site starts clean. If you deliberately migrate an
existing database you must copy `data\.salt` with it **and** use the same
passphrase, or every embedding fails to decrypt — and moving voiceprints
between machines carries legal weight (`docs/DPIA.md`).

## 2. Activate

```powershell
cd E:\Project\voice-indentification
venv\Scripts\Activate.ps1
```

If PowerShell refuses with an execution-policy error, either use `cmd`:

```
venv\Scripts\activate.bat
```

or allow scripts for that one shell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 3. Confirm the copy survived

```powershell
python -c "import sys; print(sys.version); print(sys.prefix)"
speakerid verify-models
pytest
```

Expected:

* `3.8.10`, prefix `E:\Project\voice-indentification\venv`
* three `ok` lines from `verify-models` (SHA-256 against `models\MANIFEST.json`)
* **255 passed, 2 skipped** in about 2.5 minutes

The two skips are deliberate and `pytest -rs` names them: a POSIX
file-permission test that cannot apply on Windows, and the PyTorch↔ONNX
parity test, which needs `SPEAKERID_REFERENCE_CHECKPOINT` pointing at a
reference checkpoint that is not shipped.

If those three commands pass, the system is installed.

---

## 4. Why the path must match exactly

`venv\` is not relocatable. Two things inside it hold absolute paths:

* `venv\pyvenv.cfg` records `home = E:\Project\voice-indentification\python38`,
  which is where the interpreter and the **entire standard library** are
  loaded from. A Windows venv contains no stdlib of its own; `venv\Lib\` holds
  only `site-packages`.
* The generated `.exe` launchers in `venv\Scripts\` (`pip.exe`, `pytest.exe`,
  `black.exe`, …) each embed the absolute path of `venv\Scripts\python.exe`.

Copy the tree somewhere else and both break. If you must move it, edit the
`home =` line in `venv\pyvenv.cfg` to the new location; the `.exe` launchers
then still need regenerating, so prefer `python -m pip`, `python -m pytest`
and `python -m speakerid.cli` over them until you do.

Two things are deliberately relocation-proof and need no fixing:

* `venv\Lib\site-packages\_speakerid_src.pth` contains the relative path
  `..\..\..\src`, which is how `import speakerid` works without a `pip install`
  and without setting `PYTHONPATH`.
* `venv\Scripts\speakerid.bat` calls `%~dp0python.exe`, relative to itself.

## 5. Why `python38\` is in the folder

Because the alternative is fragile. Without it, `venv\pyvenv.cfg` points at
whatever Python 3.8 is installed on the machine — originally
`C:\Users\Administrator\AppData\Local\Programs\Python\Python38`. That requires
the offline machine to have the same Windows username, the same per-user
install location and the same patch release. Any mismatch and the venv will
not start.

`python38\` is a verbatim copy of a CPython 3.8.10 install, 62 MB against
venv's 1.7 GB. It makes the tree depend on nothing outside itself.

This was verified, not assumed: with the machine's own Python 3.8 renamed out
of the way, the full suite still passes and every dependency still imports.

The one external requirement left is the Microsoft Visual C++ runtime, which
`numpy`, `onnxruntime` and `torch` link against. `vcruntime140.dll` and
`vcruntime140_1.dll` are inside `python38\`, and the rest (the Universal CRT)
ships with Windows 10 / Server 2016 and later.

---

## 6. What is in the tree

| Folder | Size | What |
|---|---|---|
| `venv\` | 1.7 GB | Packages. `requirements.lock.txt` records exactly what. |
| `python38\` | 62 MB | CPython 3.8.10, stdlib, `python38.dll`, VC runtime. |
| `models\` | 25 MB | The three ONNX files + `MANIFEST.json`. |
| `src\`, `scripts\`, `tests\`, `config\`, `docs\` | small | The project. |

`venv\` is large because PyTorch is in it (1.1 GB of the 1.7). PyTorch never runs on a
deployed device — inference is ONNX Runtime only — but it is the only way to
re-export a model or to fine-tune AASIST-L on replay recordings, which
`docs/FIRST_RUN.md` shows is necessary before the countermeasure protects
anything. On an offline machine it cannot be added later, so it is included.

`venv\` and `python38\` are `.gitignore`d and distributed as the archive in
§1b. Do not uncomment those ignore rules: `torch\lib\dnnl.lib` (617 MB) and
`torch\lib\torch_cpu.dll` (232 MB) are both over GitHub's 100 MB hard limit,
and a push containing them is rejected by the remote.

### Rebuilding the archive

After changing what is installed, re-cut it on the connected machine and
replace the file on Drive:

```powershell
cd E:\Project\voice-indentification
venv\Scripts\python -m scripts.make_env_archive
venv\Scripts\python -m pip freeze --all | Sort-Object | Out-File -Encoding utf8 requirements.lock.txt
```

The archive lands beside the project folder, never inside it. `--store` skips
compression — much faster, about 1.8 GB out.

Do not reach for `Compress-Archive` here: PowerShell 5.1's implementation
buffers and gives up near 2 GB, which is close enough to this payload to bite.
`scripts/make_env_archive.py` streams through `zipfile` with zip64 instead.

`requirements.lock.txt` is committed, so the repo always records what the
current archive contains. If they disagree, the archive is the truth.

## 7. You cannot install anything on the target

There is no wheelhouse and no package index. `pip install` will fail, by
design. If you need a package that is not in `requirements.lock.txt`, it has
to be installed on a connected machine, the archive re-cut, and the folder
re-assembled.

For the same reason: **keep the archive.** It is the only way back if `venv\`
is damaged.

---

## 8. Everyday commands

With the venv activated:

```powershell
pytest                         # 255 passed, 2 skipped
python -m ruff check .         # lint  (python38\ and venv\ are excluded)
python -m black .              # format
python -m mypy                 # type check, strict, on src\
speakerid --help               # the CLI
speakerid status               # models, thresholds, calibration state
```

## 9. It runs, but it will not identify anyone

`config\thresholds.yaml` ships empty and the loader refuses to default it.
That is deliberate (CLAUDE.md §8), and it is the last install step:

```powershell
speakerid status                  # thresholds: NOT CALIBRATED
speakerid calibrate --trials trials.csv --cm-trials cm_trials.csv `
    --identification-trials ident.csv `
    --target-far 0.001 --cm-target-far 0.05 --site "<this room>"
```

Calibrate on recordings made on that hardware, in that room. Never copy a
calibrated `thresholds.yaml` between sites: the numbers describe one
microphone, one acoustic environment, one enrolled population.

## 10. Windows note: the salt file

Windows ignores POSIX mode bits, so `data\.salt` inherits the directory ACL
instead of being created `0600`. The code logs a warning on creation. The salt
is not itself a secret — the passphrase carries the strength — but restrict
`data\` with an ACL and prefer full-disk encryption.
