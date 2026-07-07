"""Auto-download and update llama.cpp binaries from GitHub releases."""

import argparse
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
INSTALL_DIR = APP_DIR / "llama_cpp"
VERSION_FILE = INSTALL_DIR / "version.json"
GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"

# V100 Volta-optimized binaries from andrewleech/v100-llm-kit
V100_WIN_URL = "https://github.com/andrewleech/v100-llm-kit/releases/download/v1.0/llama.cpp-gemma4-win-sm70.zip"
V100_LINUX_URL = "https://github.com/andrewleech/v100-llm-kit/releases/download/v1.0/llama.cpp-gemma4-linux-sm70.zip"


def kill_llama_processes() -> None:
    """Terminate any running llama-server.exe process to release file locks."""
    print("[setup] Завершение работающих процессов llama-server...")
    if sys.platform == "win32":
        try:
            creationflags = subprocess.CREATE_NO_WINDOW
            subprocess.run(
                ["taskkill", "/f", "/im", "llama-server.exe"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=creationflags
            )
        except Exception:
            pass
    else:
        try:
            subprocess.run(["killall", "-9", "llama-server"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def detect_gpus() -> tuple[bool, bool, list[str]]:
    """Detect NVIDIA GPUs. Returns (has_gpu, is_v100, gpu_names)."""
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, creationflags=creationflags
        )
        if res.returncode == 0:
            names = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
            is_v100 = any("v100" in name.lower() for name in names)
            return True, is_v100, names
    except Exception:
        pass
    return False, False, []


def get_installed_version() -> str:
    """Return currently installed tag or empty string."""
    try:
        if VERSION_FILE.exists():
            data = json.loads(VERSION_FILE.read_text(encoding="utf-8"))
            return data.get("tag", "")
    except Exception:
        pass
    return ""


def fetch_recent_releases() -> list[dict]:
    req = urllib.request.Request(GITHUB_API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def find_assets(release: dict, cpu_only: bool) -> tuple[str, list]:
    tag = release["tag_name"]
    assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}

    if cpu_only or sys.platform != "win32":
        key = "llama-" + tag.lstrip("b") + "-bin-win-avx2-x64.zip" if sys.platform == "win32" else ""
        avx_asset = next((k for k in assets if "bin-win-avx" in k and "x64" in k), None)
        if avx_asset:
            return tag, [(avx_asset, assets[avx_asset])]
        raise RuntimeError("CPU-версия не найдена в релизе")

    # CUDA build for Windows (exclude cudart standalone zip)
    cuda_asset = next(
        (k for k in assets if "bin-win-cuda" in k and "x64" in k and k.endswith(".zip") and not k.startswith("cudart-")), None
    )
    if not cuda_asset:
        raise RuntimeError("CUDA-версия не найдена в релизе")

    selected = [(cuda_asset, assets[cuda_asset])]

    # Add matching cudart
    cuda_ver = None
    for part in cuda_asset.split("-"):
        if part.startswith("12") or part.startswith("11"):
            cuda_ver = part
            break
    if cuda_ver:
        cudart = next((k for k in assets if k.startswith("cudart-") and cuda_ver in k), None)
        if cudart:
            selected.append((cudart, assets[cudart]))

    return tag, selected


def download(url: str, dest: Path) -> None:
    print(f"[setup] Скачиваю {url.split('/')[-1]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    print(f"[setup] Распаковываю {zip_path.name}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        root_dirs = {m.split("/")[0] for m in members if "/" in m and m.split("/")[0]}

        if len(root_dirs) == 1:
            prefix = root_dirs.pop() + "/"
            for member in members:
                if member == prefix or member.endswith("/"):
                    continue
                rel = member[len(prefix):]
                if not rel:
                    continue
                out = target_dir / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            for member in members:
                if member.endswith("/"):
                    continue
                out = target_dir / member
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def save_version(tag: str, assets_installed: list[str]) -> None:
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(
        json.dumps({
            "tag": tag,
            "assets": assets_installed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def setup(cpu_only: bool = False, force: bool = False) -> bool:
    installed = get_installed_version()
    has_gpu, is_v100, gpu_names = detect_gpus()

    if not cpu_only and has_gpu and is_v100:
        tag = "v100-llm-kit-v1.0"
        print(f"[setup] Обнаружен GPU Tesla V100 ({', '.join(gpu_names)}).")
        print("[setup] Выбираем сборку SM_70 (Volta) из andrewleech/v100-llm-kit.")

        if installed == tag and not force:
            print(f"[setup] llama.cpp {tag} (V100-optimized) - уже актуален")
            return True

        if sys.platform == "win32":
            url, asset_name = V100_WIN_URL, "llama.cpp-gemma4-win-sm70.zip"
        else:
            url, asset_name = V100_LINUX_URL, "llama.cpp-gemma4-linux-sm70.zip"

        selected_assets = [(asset_name, url)]
    else:
        try:
            releases = fetch_recent_releases()
            tag = None
            selected_assets = []
            for r_entry in releases:
                try:
                    tag, selected_assets = find_assets(r_entry, cpu_only)
                    break
                except Exception:
                    continue
            if not selected_assets:
                raise RuntimeError("Не удалось найти подходящий релиз с Windows-бинарниками в последних 10 релизах")
        except Exception as e:
            if installed:
                print(f"[setup] Не удалось проверить обновления: {e}")
                print(f"[setup] Используем установленную версию {installed}")
                return True
            print(f"[setup] Ошибка загрузки информации о релизах: {e}")
            return False

        if installed == tag and not force:
            print(f"[setup] llama.cpp {tag} - уже актуален")
            return True

    action = "Обновление" if installed else "Установка"
    asset_names = [a[0] for a in selected_assets]
    print(f"[setup] {action} llama.cpp: {installed or 'нет'} -> {tag} ({', '.join(asset_names)})")

    # Kill any running llama-server to release file locks
    kill_llama_processes()
    if INSTALL_DIR.exists():
        try:
            shutil.rmtree(INSTALL_DIR)
        except Exception as e:
            print(f"[setup] Внимание: не удалось очистить старую папку: {e}")
            print("[setup] Попытка продолжить установку поверх существующих файлов...")

    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for asset_name, url in selected_assets:
            zip_path = Path(tmp) / asset_name
            try:
                download(url, zip_path)
            except Exception as e:
                print(f"[setup] Ошибка скачивания {asset_name}: {e}")
                return False
            try:
                extract_zip(zip_path, INSTALL_DIR)
            except Exception as e:
                print(f"[setup] Ошибка распаковки {asset_name}: {e}")
                return False

    # Flatten bin/ subfolder if present (andrewleech V100 packs use it)
    bin_dir = INSTALL_DIR / "bin"
    if bin_dir.exists() and bin_dir.is_dir():
        print("[setup] Папка 'bin' обнаружена. Перемещаю файлы в корень...")
        for item in bin_dir.iterdir():
            dest_item = INSTALL_DIR / item.name
            if dest_item.exists():
                shutil.rmtree(dest_item) if dest_item.is_dir() else dest_item.unlink()
            shutil.move(str(item), str(INSTALL_DIR))
        try:
            bin_dir.rmdir()
        except Exception:
            pass

    save_version(tag, asset_names)
    print(f"[setup] llama.cpp {tag} успешно установлен в {INSTALL_DIR}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup llama.cpp")
    parser.add_argument("--cpu", action="store_true", help="Принудительно скачать CPU-версию")
    parser.add_argument("--force", action="store_true", help="Принудительно переустановить")
    args = parser.parse_args()

    cpu_only = args.cpu
    if not cpu_only:
        has_gpu, is_v100, gpu_names = detect_gpus()
        if has_gpu:
            if is_v100:
                print(f"[setup] Обнаружен Tesla V100 GPU ({', '.join(gpu_names)}).")
            else:
                print(f"[setup] Обнаружен NVIDIA GPU ({', '.join(gpu_names)}). Выбираем CUDA сборку.")
        else:
            print("[setup] NVIDIA GPU не обнаружен. Выбираем CPU сборку.")
            cpu_only = True

    ok = setup(cpu_only=cpu_only, force=args.force)
    if not ok:
        print("[setup] ВНИМАНИЕ: llama.cpp не установлен. Скачайте вручную.")
        sys.exit(1)


if __name__ == "__main__":
    main()
