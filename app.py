from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

APP_DIR = Path(__file__).resolve().parent
LLAMA_ROOT = APP_DIR.parent
CONFIG_FILE = APP_DIR / "runner_config.json"
CHATS_DIR = APP_DIR / "chats"
CHATS_DIR.mkdir(exist_ok=True)

DEFAULT_SETTINGS = {
    "host": "127.0.0.1",
    "port": 8080,
    "ctx_size": 8192,
    "gpu_layers": "auto",
    "threads": -1,
    "batch_size": 2048,
    "ubatch_size": 512,
    "flash_attn": "auto",
    "cache_type_k": "q4_0",
    "cache_type_v": "q4_0",
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.05,
    "repeat_penalty": 1.08,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "max_tokens": -1,
    "seed": -1,
    "reasoning": "auto",
    "reasoning_budget": -1,
    "mmap": True,
    "mlock": False,
    "kv_offload": True,
    "webui": True,
    "system_prompt": "Ты полезный локальный ассистент. Отвечай точно и по делу.",
    "extra_args": "",
    "mmproj": "",
}

PRESETS = {
    "Сбалансированный": {
        "temperature": 0.7, "top_p": 0.9, "top_k": 40,
        "min_p": 0.05, "repeat_penalty": 1.08, "max_tokens": -1,
    },
    "Точный код": {
        "temperature": 0.25, "top_p": 0.85, "top_k": 30,
        "min_p": 0.03, "repeat_penalty": 1.05, "max_tokens": -1,
    },
    "Креативный": {
        "temperature": 0.95, "top_p": 0.95, "top_k": 80,
        "min_p": 0.02, "repeat_penalty": 1.02, "max_tokens": -1,
    },
    "Длинный контекст": {
        "ctx_size": 16384, "temperature": 0.55, "top_p": 0.9,
        "repeat_penalty": 1.12, "max_tokens": -1,
    },
}

LOAD_PROFILES = {
    "single": {
        "name": "Одна модель",
        "description": "Одна модель на порту 8080",
        "slots": [
            {"id": "primary", "port": 8080}
        ]
    },
    "dual": {
        "name": "Две модели (быстрая + качественная)",
        "description": "Лёгкая на 8080, тяжёлая на 8081",
        "slots": [
            {"id": "fast", "port": 8080},
            {"id": "quality", "port": 8081}
        ]
    },
    "triple": {
        "name": "Три модели",
        "description": "Быстрая + средняя + тяжёлая",
        "slots": [
            {"id": "fast", "port": 8080},
            {"id": "medium", "port": 8081},
            {"id": "heavy", "port": 8082}
        ]
    }
}


@dataclass
class AppConfig:
    llama_root: str = str(LLAMA_ROOT)
    models_dir: str = str(LLAMA_ROOT / "models")
    models_dirs: list[str] = field(default_factory=lambda: [str(LLAMA_ROOT / "models")])
    custom_models: list[str] = field(default_factory=list)
    server_exe: str = str(LLAMA_ROOT / "llama-server.exe")
    selected_model: str = ""
    profiles: dict[str, dict] = field(default_factory=dict)
    settings: dict = field(default_factory=lambda: DEFAULT_SETTINGS.copy())
    load_profile: str = "single"
    model_slots: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> AppConfig:
        if not CONFIG_FILE.exists():
            return cls()
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            config = cls()
            config.llama_root = data.get("llama_root", config.llama_root)
            config.models_dir = data.get("models_dir", config.models_dir)
            config.models_dirs = data.get("models_dirs", [config.models_dir])
            config.custom_models = data.get("custom_models", [])
            config.server_exe = data.get("server_exe", config.server_exe)
            config.selected_model = data.get("selected_model", config.selected_model)
            config.profiles = data.get("profiles", {})
            config.settings.update(data.get("settings", {}))
            config.load_profile = data.get("load_profile", "single")
            config.model_slots = data.get("model_slots", {})
            return config
        except (OSError, json.JSONDecodeError):
            return cls()

    def save(self) -> None:
        CONFIG_FILE.write_text(
            json.dumps(self.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )



class LlamaServer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.process: subprocess.Popen | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.logs: list[str] = []

    def start(self) -> dict:
        if self.process and self.process.poll() is None:
            return {"ok": False, "error": "Сервер уже запущен"}

        model = Path(self.config.selected_model)
        server = Path(self.config.server_exe)

        if not server.exists():
            return {"ok": False, "error": "Не найден llama-server.exe"}
        if not model.exists():
            return {"ok": False, "error": "Выберите существующий .gguf файл"}

        command = self._build_command(server, model)
        self._log("Запуск: " + subprocess.list2cmdline(command))

        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(server.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

        threading.Thread(target=self._read_output, daemon=True).start()
        return {"ok": True, "command": subprocess.list2cmdline(command)}

    def stop(self) -> dict:
        if not self.process or self.process.poll() is not None:
            return {"ok": True, "message": "Сервер уже остановлен"}
        self._log("Остановка сервера")
        self.process.terminate()
        try:
            self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.process.kill()
        return {"ok": True, "message": "Сервер остановлен"}

    def status(self) -> dict:
        running = self.process is not None and self.process.poll() is None
        if running and self._health_check():
            base = f"http://{self.config.settings['host']}:{self.config.settings['port']}"
            return {"running": True, "ready": True, "url": base}
        elif running:
            return {"running": True, "ready": False, "message": "Сервер запускается"}
            
        # Check if the process exited with an error
        if self.process is not None:
            code = self.process.poll()
            if code is not None and code != 0:
                err_msg = f"Процесс завершился с кодом {code}"
                # Search for error details in last 15 log entries
                for log in reversed(self.logs[-15:]):
                    clean = log
                    if log.startswith('[') and ']' in log:
                        parts = log.split(']', 1)
                        if len(parts) > 1:
                            clean = parts[1].strip()
                    clean_lower = clean.lower()
                    if "error" in clean_lower or "failed" in clean_lower or "exception" in clean_lower:
                        err_msg = clean
                        break
                return {"running": False, "ready": False, "crashed": True, "error": err_msg}
                
        return {"running": False, "ready": False}


    def chat(self, messages: list[dict], settings: dict) -> dict:
        base = f"http://{self.config.settings['host']}:{self.config.settings['port']}"
        payload = {
            "messages": messages,
            "temperature": settings.get("temperature", 0.7),
            "top_k": settings.get("top_k", 40),
            "top_p": settings.get("top_p", 0.9),
            "min_p": settings.get("min_p", 0.05),
            "repeat_penalty": settings.get("repeat_penalty", 1.08),
            "presence_penalty": settings.get("presence_penalty", 0.0),
            "frequency_penalty": settings.get("frequency_penalty", 0.0),
            "max_tokens": settings.get("max_tokens", -1),
            "seed": settings.get("seed", -1),
            "stream": False,
        }
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=900) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "content": data["choices"][0]["message"]["content"].strip()}

    def get_logs(self) -> list[str]:
        collected = []
        try:
            while True:
                line = self.output_queue.get_nowait()
                ts = time.strftime("%H:%M:%S")
                entry = f"[{ts}] {line}"
                self.logs.append(entry)
                collected.append(entry)
        except queue.Empty:
            pass
        return collected

    def _build_command(self, server: Path, model: Path) -> list[str]:
        s = self.config.settings
        command = [
            str(server), "-m", str(model),
            "--host", s["host"], "--port", str(s["port"]),
            "-c", str(s["ctx_size"]),
            "-ngl", str(s["gpu_layers"]),
            "-t", str(s["threads"]),
            "-b", str(s["batch_size"]),
            "-ub", str(s["ubatch_size"]),
            "-fa", (
                "on" if s["flash_attn"] in (True, "true", "True", "on", "ON")
                else ("off" if s["flash_attn"] in (False, "false", "False", "off", "OFF")
                else "auto")
            ),
            "--cache-type-k", s["cache_type_k"],
            "--cache-type-v", s["cache_type_v"],
            "--reasoning", s["reasoning"],
            "--reasoning-budget", str(s["reasoning_budget"]),
        ]
        if s["mlock"]:
            command.append("--mlock")
        if not s["mmap"]:
            command.append("--no-mmap")
        if not s["kv_offload"]:
            command.append("--no-kv-offload")
        if not s["webui"]:
            command.append("--no-webui")
        mmproj = s.get("mmproj", "").strip()
        if mmproj:
            command.extend(["--mmproj", mmproj])
        extra = s.get("extra_args", "").strip()
        if extra:
            command.extend(extra.split())
        return command

    def _read_output(self) -> None:
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            self.output_queue.put(line.rstrip())

    def _health_check(self) -> bool:
        base = f"http://{self.config.settings['host']}:{self.config.settings['port']}"
        try:
            with urllib.request.urlopen(base + "/health", timeout=0.8) as resp:
                return resp.status < 500
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def _log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {text}")


@dataclass
class ModelSlot:
    """Один слот модели в пуле."""
    id: str
    model_path: str = ""
    port: int = 8080
    ctx_size: int = 8192
    gpu_layers: str = "auto"
    process: subprocess.Popen | None = None
    output_queue: queue.Queue = field(default_factory=queue.Queue)
    logs: list[str] = field(default_factory=list)

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {text}")


class ModelPool:
    """Пул запущенных моделей на разных портах."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.slots: dict[str, ModelSlot] = {}
        self.last_used_slot: str | None = None
        self._sync_slots()

    def _sync_slots(self) -> None:
        """Синхронизировать слоты из конфига."""
        profile = LOAD_PROFILES.get(self.config.load_profile, LOAD_PROFILES["single"])
        for slot_def in profile["slots"]:
            sid = slot_def["id"]
            if sid not in self.slots:
                saved = self.config.model_slots.get(sid, {})
                self.slots[sid] = ModelSlot(
                    id=sid,
                    model_path=saved.get("model_path", ""),
                    port=slot_def["port"],
                    ctx_size=saved.get("ctx_size", 8192),
                    gpu_layers=saved.get("gpu_layers", "auto"),
                )

    def _build_command(self, slot: ModelSlot) -> list[str]:
        s = self.config.settings
        server = Path(self.config.server_exe)
        model = Path(slot.model_path)
        command = [
            str(server), "-m", str(model),
            "--host", s["host"], "--port", str(slot.port),
            "-c", str(slot.ctx_size),
            "-ngl", str(slot.gpu_layers),
            "-t", str(s["threads"]),
            "-b", str(s["batch_size"]),
            "-ub", str(s["ubatch_size"]),
            "-fa", (
                "on" if s["flash_attn"] in (True, "true", "True", "on", "ON")
                else ("off" if s["flash_attn"] in (False, "false", "False", "off", "OFF")
                else "auto")
            ),
            "--cache-type-k", s["cache_type_k"],
            "--cache-type-v", s["cache_type_v"],
            "--reasoning", s["reasoning"],
            "--reasoning-budget", str(s["reasoning_budget"]),
        ]
        if s["mlock"]:
            command.append("--mlock")
        if not s["mmap"]:
            command.append("--no-mmap")
        if not s["kv_offload"]:
            command.append("--no-kv-offload")
        if not s["webui"]:
            command.append("--no-webui")
        mmproj = s.get("mmproj", "").strip()
        if mmproj:
            command.extend(["--mmproj", mmproj])
        extra = s.get("extra_args", "").strip()
        if extra:
            command.extend(extra.split())
        return command

    def start_slot(self, slot_id: str) -> dict:
        if slot_id not in self.slots:
            return {"ok": False, "error": f"Слот '{slot_id}' не найден"}
        slot = self.slots[slot_id]
        if slot.is_running():
            return {"ok": False, "error": f"Слот '{slot_id}' уже запущен"}
        if not slot.model_path:
            return {"ok": False, "error": f"Слот '{slot_id}': модель не назначена"}
        # Check port conflict with old singleton server
        if slot.port == self.config.settings.get("port"):
            try:
                from app import server as _singleton
                if _singleton.process and _singleton.process.poll() is None:
                    return {"ok": False, "error": f"Порт {slot.port} занят старым сервером. Остановите его через 'Остановить' в основных настройках."}
            except Exception:
                pass
        model = Path(slot.model_path)
        server = Path(self.config.server_exe)
        if not server.exists():
            return {"ok": False, "error": "Не найден llama-server.exe"}
        if not model.exists():
            return {"ok": False, "error": f"Модель не найдена: {slot.model_path}"}
        command = self._build_command(slot)
        slot._log("Запуск: " + subprocess.list2cmdline(command))
        try:
            slot.process = subprocess.Popen(
                command,
                cwd=str(server.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        threading.Thread(target=self._read_output, args=(slot,), daemon=True).start()
        return {"ok": True, "slot": slot_id, "port": slot.port}

    def stop_slot(self, slot_id: str) -> dict:
        if slot_id not in self.slots:
            return {"ok": False, "error": f"Слот '{slot_id}' не найден"}
        slot = self.slots[slot_id]
        if not slot.is_running():
            return {"ok": True, "message": f"Слот '{slot_id}' уже остановлен"}
        slot._log("Остановка")
        slot.process.terminate()
        try:
            slot.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            slot.process.kill()
            slot.process.wait(timeout=3)
        # Clean up: drain queue, clear logs, release process handle
        while not slot.output_queue.empty():
            try:
                slot.output_queue.get_nowait()
            except queue.Empty:
                break
        slot.logs.clear()
        slot.process = None
        return {"ok": True, "message": f"Слот '{slot_id}' остановлен"}

    def start_all(self) -> dict:
        results = {}
        for sid in self.slots:
            results[sid] = self.start_slot(sid)
        return results

    def stop_all(self) -> dict:
        results = {}
        for sid in self.slots:
            results[sid] = self.stop_slot(sid)
        return results

    def status_all(self) -> dict:
        result = {"last_used_slot": self.last_used_slot, "slots": {}}
        for sid, slot in self.slots.items():
            running = slot.is_running()
            ready = running and self._health_check(slot.port)
            result["slots"][sid] = {
                "running": running,
                "ready": ready,
                "port": slot.port,
                "model": slot.model_path,
                "ctx_size": slot.ctx_size,
                "gpu_layers": slot.gpu_layers,
            }
        return result

    def get_slot_for_complexity(self, complexity: float) -> ModelSlot:
        """Выбрать слот по сложности запроса (0.0-1.0)."""
        available = {sid: s for sid, s in self.slots.items() if s.is_running()}
        if not available:
            return list(self.slots.values())[0] if self.slots else None
        slot = None
        if "fast" in available and "quality" in available:
            if complexity < 0.5:
                slot = available["fast"]
            else:
                slot = available["quality"]
        elif "fast" in available and "medium" in available and "heavy" in available:
            if complexity < 0.3:
                slot = available["fast"]
            elif complexity < 0.7:
                slot = available["medium"]
            else:
                slot = available["heavy"]
        else:
            slot = list(available.values())[0]
        if slot:
            self.last_used_slot = slot.id
        return slot

    def _health_check(self, port: int) -> bool:
        host = self.config.settings["host"]
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=0.8) as resp:
                return resp.status < 500
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def _read_output(self, slot: ModelSlot) -> None:
        if not slot.process or not slot.process.stdout:
            return
        for line in slot.process.stdout:
            slot.output_queue.put(line.rstrip())

    def save_slot_config(self) -> None:
        """Сохранить конфигурацию слотов в config."""
        for sid, slot in self.slots.items():
            self.config.model_slots[sid] = {
                "model_path": slot.model_path,
                "ctx_size": slot.ctx_size,
                "gpu_layers": slot.gpu_layers,
            }
        self.config.load_profile = self.config.load_profile
        self.config.save()


def select_folder_via_ps() -> str:
    cmd = [
        "powershell", "-NoProfile", "-Command",
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
        "$f.Description = 'Выберите папку с моделями'; "
        "if ($f.ShowDialog() -eq 'OK') { $f.SelectedPath }"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        return res.stdout.strip()
    except Exception:
        return ""


def select_file_via_ps() -> str:
    cmd = [
        "powershell", "-NoProfile", "-Command",
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$f = New-Object System.Windows.Forms.OpenFileDialog; "
        "$f.Filter = 'GGUF Models (*.gguf)|*.gguf'; "
        "$f.Title = 'Выберите файл модели GGUF'; "
        "if ($f.ShowDialog() -eq 'OK') { $f.FileName }"
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        return res.stdout.strip()
    except Exception:
        return ""


app = Flask(__name__)
config = AppConfig.load()
server = LlamaServer(config)
pool = ModelPool(config)

from orchestrator_api import orchestrator_bp
app.register_blueprint(orchestrator_bp)



@app.route("/")
def index():
    return render_template("index.html")


def parse_params(filename: str) -> tuple[str, float]:
    match = re.search(r'(?i)\b(\d+(?:\.\d+)?)\s*([bm])\b', filename)
    if not match:
        match = re.search(r'(?i)(\d+(?:\.\d+)?)\s*([bm])', filename)
    if match:
        num = float(match.group(1))
        unit = match.group(2).upper()
        if unit == 'B':
            return f"{num}B", num
        else:
            return f"{num}M", num / 1000.0
    return "", 0.0


@app.route("/api/models")
def api_models():
    models = []
    seen = set()
    
    # 1. Scan folders in models_dirs
    for d_str in config.models_dirs:
        d = Path(d_str)
        if d.exists() and d.is_dir():
            try:
                for m in sorted(d.rglob("*.gguf"), key=lambda p: p.name.lower()):
                    m_str = str(m.resolve())
                    if m_str in seen:
                        continue
                    seen.add(m_str)
                    size_gb = m.stat().st_size / (1024 ** 3)
                    p_str, p_num = parse_params(m.name)
                    models.append({
                        "path": m_str,
                        "name": m.name,
                        "folder": str(d),
                        "size_gb": round(size_gb, 2),
                        "selected": m_str == config.selected_model,
                        "custom": False,
                        "params": p_str,
                        "params_num": p_num
                    })
            except Exception:
                pass
                
    # 2. Add custom files
    valid_customs = []
    for c_str in config.custom_models:
        c = Path(c_str)
        if c.exists() and c.is_file():
            m_str = str(c.resolve())
            valid_customs.append(m_str)
            if m_str in seen:
                continue
            seen.add(m_str)
            size_gb = c.stat().st_size / (1024 ** 3)
            p_str, p_num = parse_params(c.name)
            models.append({
                "path": m_str,
                "name": c.name,
                "folder": "Индивидуальные файлы",
                "size_gb": round(size_gb, 2),
                "selected": m_str == config.selected_model,
                "custom": True,
                "params": p_str,
                "params_num": p_num
            })
            
    if len(valid_customs) != len(config.custom_models):
        config.custom_models = valid_customs
        config.save()
        
    return jsonify(models)



@app.route("/api/models/dirs", methods=["GET"])
def api_get_models_dirs():
    return jsonify({"dirs": config.models_dirs})


@app.route("/api/models/dirs/add", methods=["POST"])
def api_add_models_dir():
    data = request.get_json() or {}
    path = data.get("path", "").strip()
    if path and path not in config.models_dirs:
        config.models_dirs.append(path)
        config.save()
    return jsonify({"ok": True, "dirs": config.models_dirs})


@app.route("/api/models/dirs/remove", methods=["POST"])
def api_remove_models_dir():
    data = request.get_json() or {}
    path = data.get("path", "").strip()
    if path in config.models_dirs:
        config.models_dirs.remove(path)
        config.save()
    return jsonify({"ok": True, "dirs": config.models_dirs})


@app.route("/api/models/custom/add", methods=["POST"])
def api_add_custom_model():
    data = request.get_json() or {}
    path = data.get("path", "").strip()
    if path and path not in config.custom_models:
        config.custom_models.append(path)
        config.save()
    return jsonify({"ok": True, "custom_models": config.custom_models})


@app.route("/api/models/custom/remove", methods=["POST"])
def api_remove_custom_model():
    data = request.get_json() or {}
    path = data.get("path", "").strip()
    if path in config.custom_models:
        config.custom_models.remove(path)
        config.save()
    return jsonify({"ok": True, "custom_models": config.custom_models})


@app.route("/api/utils/select-folder", methods=["POST"])
def api_select_folder():
    path = select_folder_via_ps()
    return jsonify({"path": path})


@app.route("/api/utils/select-file", methods=["POST"])
def api_select_file():
    path = select_file_via_ps()
    return jsonify({"path": path})


@app.route("/api/utils/open-explorer", methods=["POST"])
def api_open_explorer():
    data = request.get_json() or {}
    path_str = data.get("path", "")
    if not path_str:
        return jsonify({"ok": False, "error": "Путь не указан"})
    path = Path(path_str)
    if not path.exists():
        return jsonify({"ok": False, "error": "Путь не существует"})
    try:
        if path.is_file():
            # Open explorer and select file
            subprocess.run(f'explorer /select,"{path}"', creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        else:
            # Open folder
            os.startfile(path)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/config", methods=["GET"])
def api_get_config():
    return jsonify({
        "models_dir": config.models_dir,
        "models_dirs": config.models_dirs,
        "custom_models": config.custom_models,
        "server_exe": config.server_exe,
        "selected_model": config.selected_model,
        "settings": config.settings,
        "profiles": config.profiles,
        "presets": PRESETS,
        "load_profile": config.load_profile,
        "model_slots": config.model_slots,
        "load_profiles": LOAD_PROFILES,
    })


@app.route("/api/config", methods=["POST"])
def api_save_config():
    data = request.get_json()
    if "models_dir" in data:
        config.models_dir = data["models_dir"]
    if "models_dirs" in data:
        config.models_dirs = data["models_dirs"]
    if "custom_models" in data:
        config.custom_models = data["custom_models"]
    if "server_exe" in data:
        config.server_exe = data["server_exe"]
    if "selected_model" in data:
        config.selected_model = data["selected_model"]
    if "settings" in data:
        config.settings.update(data["settings"])
    config.save()
    return jsonify({"ok": True})


@app.route("/api/profiles", methods=["POST"])
def api_save_profile():
    data = request.get_json()
    name = data.get("name", "Мой профиль").strip() or "Мой профиль"
    config.profiles[name] = config.settings.copy()
    config.save()
    return jsonify({"ok": True, "profiles": config.profiles})


@app.route("/api/profiles/<name>", methods=["POST"])
def api_load_profile(name):
    profile = config.profiles.get(name)
    if not profile:
        return jsonify({"ok": False, "error": "Профиль не найден"}), 404
    config.settings.update(profile)
    config.save()
    return jsonify({"ok": True, "settings": config.settings})


@app.route("/api/server/start", methods=["POST"])
def api_start_server():
    data = request.get_json() or {}
    if "settings" in data:
        config.settings.update(data["settings"])
    if "selected_model" in data:
        config.selected_model = data["selected_model"]
    config.save()
    return jsonify(server.start())


@app.route("/api/server/stop", methods=["POST"])
def api_stop_server():
    return jsonify(server.stop())


@app.route("/api/server/status")
def api_server_status():
    return jsonify(server.status())


@app.route("/api/server/logs")
def api_server_logs():
    return jsonify({"logs": server.get_logs(), "all_logs": server.logs[-200:]})


@app.route("/api/pool/profiles", methods=["GET"])
def api_pool_profiles():
    profiles = {}
    for k, v in LOAD_PROFILES.items():
        profiles[k] = {"name": v["name"], "description": v["description"],
                        "slots": [{"id": s["id"], "port": s["port"]} for s in v["slots"]]}
    return jsonify({"profiles": profiles, "current": config.load_profile})


@app.route("/api/pool/profile", methods=["POST"])
def api_pool_set_profile():
    data = request.get_json() or {}
    profile_id = data.get("profile", "single")
    if profile_id not in LOAD_PROFILES:
        return jsonify({"ok": False, "error": f"Профиль '{profile_id}' не найден"}), 400
    config.load_profile = profile_id
    pool._sync_slots()
    config.save()
    return jsonify({"ok": True, "profile": profile_id, "slots": list(pool.slots.keys())})


@app.route("/api/pool/status", methods=["GET"])
def api_pool_status():
    return jsonify(pool.status_all())


@app.route("/api/pool/start", methods=["POST"])
def api_pool_start_all():
    results = pool.start_all()
    return jsonify(results)


@app.route("/api/pool/stop", methods=["POST"])
def api_pool_stop_all():
    results = pool.stop_all()
    return jsonify(results)


@app.route("/api/pool/slot/start", methods=["POST"])
def api_pool_slot_start():
    data = request.get_json() or {}
    slot_id = data.get("slot_id")
    if not slot_id:
        return jsonify({"ok": False, "error": "slot_id required"}), 400
    return jsonify(pool.start_slot(slot_id))


@app.route("/api/pool/slot/stop", methods=["POST"])
def api_pool_slot_stop():
    data = request.get_json() or {}
    slot_id = data.get("slot_id")
    if not slot_id:
        return jsonify({"ok": False, "error": "slot_id required"}), 400
    return jsonify(pool.stop_slot(slot_id))


@app.route("/api/pool/slot/assign", methods=["POST"])
def api_pool_slot_assign():
    data = request.get_json() or {}
    slot_id = data.get("slot_id")
    model_path = data.get("model_path", "")
    if not slot_id or slot_id not in pool.slots:
        return jsonify({"ok": False, "error": f"Слот '{slot_id}' не найден"}), 400
    if not model_path or not Path(model_path).exists():
        return jsonify({"ok": False, "error": f"Модель не найдена: {model_path}"}), 400
    slot = pool.slots[slot_id]
    slot.model_path = model_path
    slot.ctx_size = data.get("ctx_size", slot.ctx_size)
    slot.gpu_layers = data.get("gpu_layers", slot.gpu_layers)
    pool.save_slot_config()
    return jsonify({"ok": True, "slot": slot_id, "model": model_path, "port": slot.port})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    messages = data.get("messages", [])
    settings = data.get("settings", config.settings)
    stream = data.get("stream", False)
    
    if not stream:
        try:
            result = server.chat(messages, settings)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
            
    base = f"http://{config.settings['host']}:{config.settings['port']}"
    payload = {
        "messages": messages,
        "temperature": settings.get("temperature", 0.7),
        "top_k": settings.get("top_k", 40),
        "top_p": settings.get("top_p", 0.9),
        "min_p": settings.get("min_p", 0.05),
        "repeat_penalty": settings.get("repeat_penalty", 1.08),
        "presence_penalty": settings.get("presence_penalty", 0.0),
        "frequency_penalty": settings.get("frequency_penalty", 0.0),
        "max_tokens": settings.get("max_tokens", -1),
        "seed": settings.get("seed", -1),
        "stream": True,
    }
    
    def generate():
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    yield line
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode("utf-8")
            
    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/chats", methods=["GET"])
def api_get_chats():
    chats = []
    if CHATS_DIR.exists():
        for f in sorted(CHATS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                chats.append({
                    "id": data.get("id"),
                    "title": data.get("title", "Без названия"),
                    "created_at": data.get("created_at", f.stat().st_mtime),
                    "msg_count": len(data.get("messages", []))
                })
            except Exception:
                pass
    return jsonify(chats)


@app.route("/api/chats/<chat_id>", methods=["GET"])
def api_get_chat(chat_id):
    f = CHATS_DIR / f"{chat_id}.json"
    if not f.exists():
        return jsonify({"ok": False, "error": "Диалог не найден"}), 404
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return jsonify(data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chats/<chat_id>", methods=["POST"])
def api_save_chat(chat_id):
    data = request.get_json() or {}
    messages = data.get("messages", [])
    title = data.get("title", "Новый диалог").strip()
    created_at = data.get("created_at", time.time())
    
    # Save file
    f = CHATS_DIR / f"{chat_id}.json"
    chat_data = {
        "id": chat_id,
        "title": title,
        "created_at": created_at,
        "messages": messages
    }
    try:
        f.write_text(json.dumps(chat_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def api_delete_chat(chat_id):
    f = CHATS_DIR / f"{chat_id}.json"
    if f.exists():
        try:
            f.unlink()
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})
    return jsonify({"ok": False, "error": "Файл не найден"}), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)


