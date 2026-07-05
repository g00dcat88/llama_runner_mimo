"""Account Manager — управление аккаунтами пользователей."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Account:
    username: str
    password_hash: str = ""
    display_name: str = ""
    role: str = "user"           # "admin", "user", "readonly"
    enabled: bool = True
    created_at: float = 0.0
    last_login: float = 0.0
    tools: list[str] = field(default_factory=list)  # разрешённые инструменты

    def to_dict(self) -> dict:
        return {
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "tools": self.tools,
            "has_password": bool(self.password_hash),
        }


class AccountManager:
    """Менеджер аккаунтов. Хранит данные в accounts.json"""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._file = data_dir / "accounts.json"
        self._accounts: dict[str, Account] = {}
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            # Создаём аккаунт admin по умолчанию
            default_pw = os.environ.get("ORCHESTRATOR_API_KEY", "admin")
            self._accounts["admin"] = Account(
                username="admin",
                password_hash=self._hash_password(default_pw),
                display_name="Administrator",
                role="admin",
                enabled=True,
                created_at=time.time(),
            )
            self._save()
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            for username, info in data.get("accounts", {}).items():
                self._accounts[username] = Account(
                    username=username,
                    password_hash=info.get("password_hash", ""),
                    display_name=info.get("display_name", ""),
                    role=info.get("role", "user"),
                    enabled=info.get("enabled", True),
                    created_at=info.get("created_at", 0),
                    last_login=info.get("last_login", 0),
                    tools=info.get("tools", []),
                )
        except (json.JSONDecodeError, KeyError):
            pass

    def _save(self) -> None:
        data = {"accounts": {}}
        for username, acc in self._accounts.items():
            data["accounts"][username] = {
                "password_hash": acc.password_hash,
                "display_name": acc.display_name,
                "role": acc.role,
                "enabled": acc.enabled,
                "created_at": acc.created_at,
                "last_login": acc.last_login,
                "tools": acc.tools,
            }
        self._file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def authenticate(self, username: str, password: str) -> Account | None:
        acc = self._accounts.get(username)
        if not acc or not acc.enabled:
            return None
        if acc.password_hash != self._hash_password(password):
            return None
        acc.last_login = time.time()
        self._save()
        return acc

    def get(self, username: str) -> Account | None:
        return self._accounts.get(username)

    def list_all(self) -> list[Account]:
        return list(self._accounts.values())

    def create(self, username: str, password: str, display_name: str = "", role: str = "user") -> Account | None:
        if username in self._accounts:
            return None
        acc = Account(
            username=username,
            password_hash=self._hash_password(password),
            display_name=display_name or username,
            role=role,
            enabled=True,
            created_at=time.time(),
        )
        self._accounts[username] = acc
        self._save()
        return acc

    def update(self, username: str, **kwargs) -> Account | None:
        acc = self._accounts.get(username)
        if not acc:
            return None
        if "password" in kwargs and kwargs["password"]:
            acc.password_hash = self._hash_password(kwargs["password"])
        if "display_name" in kwargs:
            acc.display_name = kwargs["display_name"]
        if "role" in kwargs:
            acc.role = kwargs["role"]
        if "enabled" in kwargs:
            acc.enabled = kwargs["enabled"]
        if "tools" in kwargs:
            acc.tools = kwargs["tools"]
        self._save()
        return acc

    def delete(self, username: str) -> bool:
        if username not in self._accounts:
            return False
        if username == "admin":
            return False  # нельзя удалить admin
        del self._accounts[username]
        self._save()
        return True
