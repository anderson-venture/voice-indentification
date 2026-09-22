# Setup

There is one supported way to install this system:

> Copy the folder `E:\Project\voice-indentification` to the offline machine,
> to **the same absolute path**. Nothing else. No Python installer, no `pip`,
> no network.

The folder is self-contained. It carries its own CPython 3.8.10 (`python38\`),
its own packages (`venv\`), and its own model weights (`models\`). Nothing in
it reads from `C:\`, from a package index, or from the internet.

---

## 1. Copy

On the offline machine, put the folder at exactly:

```
E:\Project\voice-indentification
```

Note the spelling: **indentification**, with the `n`. It is a typo in the
original directory name and it is now load-bearing — see §4.

Use a copy that preserves the whole tree. `robocopy` from the USB drive:

```powershell
robocopy D:\voice-indentification E:\Project\voice-indentification /E
```

Leave `data\` behind if the source machine has one. It holds the encrypted
voiceprint database and the PBKDF2 salt, and a new site starts clean. If you
deliberately migrate an existing database you must copy `data\.salt` with it
**and** use the same passphrase, or every embedding fails to decrypt — and
moving voiceprints between machines carries legal weight (`docs/DPIA.md`).

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
venv's 3.2 GB. It makes the tree depend on nothing outside itself.

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
| `venv\` | 3.2 GB | Packages. `requirements.lock.txt` records exactly what. |
| `python38\` | 62 MB | CPython 3.8.10, stdlib, `python38.dll`, VC runtime. |
| `models\` | 25 MB | The three ONNX files + `MANIFEST.json`. |
| `src\`, `scripts\`, `tests\`, `config\`, `docs\` | small | The project. |

`venv\` is large because PyTorch is in it (~2.5 GB). PyTorch never runs on a
deployed device — inference is ONNX Runtime only — but it is the only way to
re-export a model or to fine-tune AASIST-L on replay recordings, which
`docs/FIRST_RUN.md` shows is necessary before the countermeasure protects
anything. On an offline machine it cannot be added later, so it is included.

`venv\` and `python38\` are `.gitignore`d. They are binaries, not source, and
they travel by USB.

## 7. You cannot install anything on the target

There is no wheelhouse and no package index. `pip install` will fail, by
design. If you need a package that is not in `requirements.lock.txt`, it has
to be installed on a connected machine and the tree re-copied.

For the same reason: **keep the USB copy.** It is the only way back if `venv\`
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
