# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for whisper-wrapper.
#
# Produces a single-file binary that bundles the faster-whisper runtime
# (CTranslate2 native libs), soundfile, ffmpeg (via imageio-ffmpeg), and
# silero-vad. Model files are NOT bundled — they are downloaded on first
# use to the user data dir.

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None

hiddenimports = []
hiddenimports += collect_submodules("ctranslate2")
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("tokenizers")
hiddenimports += collect_submodules("huggingface_hub")
hiddenimports += collect_submodules("silero_vad")
hiddenimports += collect_submodules("soundfile")
hiddenimports += ["uvicorn.protocols.http.h11_impl", "uvicorn.protocols.websockets.wsproto_impl"]

binaries = []
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("soundfile")
binaries += collect_dynamic_libs("torch")

datas = []
datas += collect_data_files("faster_whisper")
datas += collect_data_files("silero_vad")
datas += collect_data_files("imageio_ffmpeg")
datas += collect_data_files("tokenizers")


a = Analysis(
    ["src/whisper_wrapper/__main__.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="whisper-wrapper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
