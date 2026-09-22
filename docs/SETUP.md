# Setup

Two machines:

* **BUILD** — has a network. Installs everything once.
* **TARGET** — fully offline. Receives a copy and runs it.

The plan is to copy the whole project folder, `.venv` included, so the target
needs no installation step at all. That works, but only if three things match
on both machines. Getting any of them wrong produces a venv that will not
start, so they are worth reading before you begin.

| Must match | Why |
|---|---|
| **Operating system and architecture** | The venv holds compiled binaries. Windows needs `.pyd`; Linux `.so`. A venv never survives crossing operating systems. |
| **Absolute project path** | Launchers in `.venv\Scripts\` hardcode their own location. Use the same path on both, e.g. `E:\project\voice-indentification`. |
| **Python 3.8 install path** | `.venv\pyvenv.cfg` records a `home = ...` pointing at the base Python, and the **standard library is loaded from there** — the venv contains only `site-packages`. If the base Python is missing or elsewhere on the target, the venv cannot start. |

The third one catches people out. Install Python **for all users**, so it
lands in `C:\Program Files\Python38` on both machines rather than
`C:\Users\<name>\AppData\...`, which differs whenever the usernames differ.

---

## 1. BUILD machine (with network)

Install **Python 3.8.10** — the last version python.org published a Windows
installer for, so the patch version is forced:

<https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe>

In the installer: tick **"Add Python 3.8 to PATH"**, choose **"Customize
installation"**, then on the second page tick **"Install for all users"** so
the target path is `C:\Program Files\Python38`.

```powershell
git clone <your-repo-url> E:\project\voice-indentification
cd E:\project\voice-indentification

py -3.8 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
```

Verify before copying anything:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python -m speakerid.cli verify-models
.venv\Scripts\python -m pytest
```

`verify-models` checksums each `.onnx` against `models/MANIFEST.json`.
`pytest` should report **255 passed, 1 skipped**.

### Optional insurance

If the copied venv ever misbehaves on the target, being able to rebuild it
offline saves a trip. One command, ~140 MB:

```powershell
.venv\Scripts\python -m pip download -r requirements.txt -r requirements-dev.txt -d wheels
```

Carry `wheels\` with the project. It is already gitignored.

---

## 2. TARGET machine (offline)

1. Install **the same Python 3.8.10**, same installer, **same "for all users"
   option**, so it lands at `C:\Program Files\Python38` again.
2. Copy the whole project folder to **the identical path**,
   `E:\project\voice-indentification`, `.venv` included.

Then confirm it survived the move:

```powershell
cd E:\project\voice-indentification
$env:PYTHONPATH = "src"
.venv\Scripts\python -c "import numpy, onnxruntime, soundfile; print('ok')"
.venv\Scripts\python -m speakerid.cli verify-models
.venv\Scripts\python -m pytest
```

If those three pass, the system is installed.

### If the venv does not start

Almost always the base Python is missing or at a different path. Check what
the venv expects:

```powershell
type .venv\pyvenv.cfg
```

and confirm that `home = ...` exists on this machine. If it does not, either
install Python there, or rebuild the venv offline from the `wheels\` folder:

```powershell
rmdir /s /q .venv
py -3.8 -m venv .venv
.venv\Scripts\python -m pip install --no-index --find-links=wheels -r requirements.txt -r requirements-dev.txt
```

### What not to copy

Leave `data\` behind. It holds the encrypted voiceprint database and the
PBKDF2 salt, and the target should start clean. If you deliberately migrate an
existing enrolment database you must copy `data\.salt` with it **and** use the
same passphrase, or every embedding fails to decrypt. Moving voiceprints
between machines also carries legal weight — see `docs/DPIA.md`.

---

## 3. It still will not identify anyone

`config\thresholds.yaml` ships empty and the loader refuses to default it.
That is deliberate (CLAUDE.md §8) and it is the last step:

```powershell
.venv\Scripts\python -m speakerid.cli status      # shows: thresholds NOT CALIBRATED
.venv\Scripts\python -m speakerid.cli calibrate --trials trials.csv ^
    --cm-trials cm_trials.csv --identification-trials ident.csv ^
    --target-far 0.001 --cm-target-far 0.05 --site "<this room>"
```

Calibrate on recordings made on that hardware, in that room. Never copy a
calibrated `thresholds.yaml` between sites: the numbers describe one
microphone, one acoustic environment, one enrolled population.

---

## Everyday commands

Set `PYTHONPATH` once per shell, or permanently with `setx PYTHONPATH src`
(reopen the terminal afterwards).

```powershell
.venv\Scripts\python -m pytest                              # tests
.venv\Scripts\python -m ruff check src scripts tests        # lint
.venv\Scripts\python -m black src scripts tests             # format
.venv\Scripts\python -m mypy                                # type check
.venv\Scripts\python -m speakerid.cli --help                # the CLI
```

## The three requirements files

| File | When |
|---|---|
| `requirements.txt` | always — running the system |
| `requirements-dev.txt` | tests, lint, format, type check |
| `requirements-export.txt` | **only** to re-export or fine-tune a model (~2 GB, PyTorch) |

`requirements-export.txt` is separate on purpose. Inference runs through ONNX
Runtime; PyTorch is a development-only tool and never belongs on a deployed
device (CLAUDE.md §4). The three models are already exported and committed, so
you need it only when changing a model — most likely fine-tuning AASIST-L on
replay data, which `docs/FIRST_RUN.md` explains is necessary before the
countermeasure protects anything.

The optional localhost API additionally needs `fastapi` and `uvicorn`; the CLI
does not.

## Windows notes

**The PBKDF2 salt cannot be permission-restricted.** Windows ignores POSIX
mode bits, so `data\.salt` inherits the directory ACL instead of being created
`0600`. The code logs a warning, and the POSIX-only test is skipped on Windows
while a separate test still checks the salt is created correctly. The salt is
not itself a secret — the passphrase carries the strength — but restrict
`data\` with an ACL and prefer full-disk encryption.

**Verified from Linux, not on Windows.** Every pin has a genuine
`cp38`/`win_amd64` wheel, both requirements files resolve for that platform,
and path handling is separator-safe (`MANIFEST.json` keys are normalised, and
SQLAlchemy accepts `sqlite:///C:\...` URLs). The test suite has never actually
been run on Windows — `pytest` on the BUILD machine will tell you in about 40
seconds.
