# PyInstaller : pyinstaller signallab.spec
from PyInstaller.utils.hooks import collect_data_files

datas = [("signallab/data", "signallab/data")] + collect_data_files("matplotlib")

a = Analysis(["run.py"], datas=datas, hiddenimports=["matplotlib.backends.backend_qtagg"],
             excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.Qt3DCore"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SignalLab", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="SignalLab")
app = BUNDLE(coll, name="SignalLab.app", bundle_identifier="ca.umontreal.signallab")
