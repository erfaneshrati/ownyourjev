import sys, shutil
import fast_export as fe
from pathlib import Path
d = Path(sys.argv[1])
for mode in ["slice", "matmul"]:
    fe.ROPE_MODE = mode
    fe.lean_export(d, d / f"rope_{mode}.onnx", check=False, smooth=0.8)
