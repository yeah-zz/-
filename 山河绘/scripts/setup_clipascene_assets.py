"""Download official sources and CLIP weights into this experiment only."""
from pathlib import Path
from urllib.request import urlopen
import concurrent.futures
import hashlib
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(r"E:\Codex_Workspace_Data\CLIPascene_20261021")
EXP = STORAGE / "source"
ASSETS = EXP / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
SHA = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
FILES = {
    "pybind11.zip": "https://codeload.github.com/pybind/pybind11/zip/refs/tags/v2.11.1",
    "thrust.zip": "https://codeload.github.com/NVIDIA/thrust/zip/refs/tags/1.9.10",
    "CLIP.zip": "https://codeload.github.com/openai/CLIP/zip/refs/heads/main",
    "ViT-B-32.pt": f"https://openaipublic.azureedge.net/clip/models/{SHA}/ViT-B-32.pt",
}


def fetch(item):
    name, url = item
    dest = ASSETS / name
    if not dest.exists():
        part = dest.with_suffix(dest.suffix + ".part")
        with urlopen(url, timeout=90) as response, part.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024*1024)
        part.replace(dest)
    if name.endswith(".zip"):
        with zipfile.ZipFile(dest) as archive:
            archive.extractall(ASSETS)
    else:
        actual = hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual != SHA:
            raise RuntimeError(f"Invalid CLIP checksum: {actual}")
    print(name, dest.stat().st_size, "OK", flush=True)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, FILES.items()))
    diff = EXP / "diffvg-master"
    shutil.copytree(ASSETS / "pybind11-2.11.1", diff / "pybind11", dirs_exist_ok=True)
    shutil.copytree(ASSETS / "thrust-1.9.10", diff / "thrust", dirs_exist_ok=True)
    shutil.copytree(ASSETS / "CLIP-main" / "clip", STORAGE / "environment" / "deps" / "clip", dirs_exist_ok=True)
    shutil.copytree(diff / "pydiffvg", STORAGE / "environment" / "deps" / "pydiffvg", dirs_exist_ok=True)
    print("Official source assets prepared", flush=True)
