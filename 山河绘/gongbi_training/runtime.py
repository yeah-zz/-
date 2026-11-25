"""Reuse the isolated CUDA environment. All new data/cache/checkpoints on E."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
LAB = Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022')
GPU = Path(r'E:\Codex_Workspace_Data\CLIPascene_20261021\recovered_from_C\Temp\pip-target-m7b0uqsu\lib\python')
_dll_handles = []


def configure():
    sys.dont_write_bytecode = True
    for key, suffix in {'TEMP':'temp', 'TMP':'temp', 'PIP_CACHE_DIR':'cache/pip',
                        'TORCH_HOME':'cache/torch', 'HF_HOME':'cache/huggingface',
                        'XDG_CACHE_HOME':'cache', 'CUDA_CACHE_PATH':'cache/cuda',
                        'TORCH_EXTENSIONS_DIR':'cache/extensions','TORCHINDUCTOR_CACHE_DIR':'cache/inductor',
                        'TRITON_CACHE_DIR':'cache/triton','PYTORCH_KERNEL_CACHE_PATH':'cache/kernels',
                        'MPLCONFIGDIR':'cache/matplotlib'}.items():
        os.environ[key] = str(LAB/suffix)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for folder in ('temp','cache/torch','data/raw','data/processed','datasets','runs','previews'):
        (LAB/folder).mkdir(parents=True, exist_ok=True)
    if str(GPU) not in sys.path:
        sys.path.insert(0,str(GPU))
    if hasattr(os,'add_dll_directory'):
        _dll_handles.append(os.add_dll_directory(str(GPU/'torch'/'lib')))
    return LAB
