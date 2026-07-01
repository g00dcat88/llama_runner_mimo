import sys
import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from flask import Blueprint, jsonify, request


def is_server_online(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=1) as resp:
            return resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False

# Add llm_orchestrator to path so we can import gateway, tools, skills, etc.
APP_DIR = Path(__file__).resolve().parent
ORCHESTRATOR_DIR = APP_DIR.parent / "llm_orchestrator"
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.append(str(ORCHESTRATOR_DIR))

from gateway import BaseLLM, create_llm_from_config
from providers import LlamaServerLLM, MockLLM, OpenAICompatibleLLM, AnthropicLLM, PROVIDER_TYPES
from tools import Tool, ToolRegistry, PythonSandbox, WebMonitorTool, ERPIntegrationTools
from skills import SkillsManager
from config import Config, ProviderConfig, RouterConfig
from router import RouterLLM, classify_complexity
from main import run_agentic_loop
from dispatcher import QueryDispatcher
from conversation import ConversationBuffer
from rag import BM25SearchEngine
from cache import ResponseCache
from rate_limiter import DualRateLimiter
from metrics import MetricsCollector
from session_store import SessionStore
from self_learning import SelfLearner
from guardrails import InputGuardrails, OutputGuardrails
from token_manager import TokenManager

orchestrator_bp = Blueprint("orchestrator", __name__)

# Persistent session storage and self-learning
_session_store = SessionStore(APP_DIR / "orchestrator_sessions.db")
_learner = SelfLearner(_session_store)



@orchestrator_bp.before_request
def check_api_key():
    # Exclude health-check and localhost requests from API key requirement
    if request.path == "/api/orchestrator/status":
        return
    if request.remote_addr in ("127.0.0.1", "::1"):
        return
    from app import config as runner_config
    expected_key = runner_config.settings.get("orchestrator_api_key")
    if not expected_key:
        return
    api_key = request.headers.get("X-Orchestrator-API-Key")
    if api_key != expected_key:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

# Initialize dependencies
sandbox_dir = ORCHESTRATOR_DIR / "sandbox"
sandbox = PythonSandbox(sandbox_dir)
registry = ToolRegistry()

# Register execute_python tool
execute_python_tool = Tool(
    name="execute_python",
    description="Выполняет код на Python в изолированной папке песочницы и возвращает stdout/stderr. Используйте для вычислений или обработки данных.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Исходный код программы на Python для выполнения."
            }
        },
        "required": ["code"]
    },
    func=sandbox.execute_code,
    category="python_sandbox"
)
registry.register(execute_python_tool)

# Register web monitor tool
monitor_log_path = sandbox_dir / "monitoring_log.json"
web_monitor = WebMonitorTool(monitor_log_path)
web_monitor_tool = Tool(
    name="monitor_web_resource",
    description="Проверяет состояние указанного веб-ресурса (URL), получает preview-данные и записывает проверку в журнал логов.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Полный URL-адрес веб-ресурса для проверки (например: https://httpbin.org/status/200)."
            }
        },
        "required": ["url"]
    },
    func=web_monitor.monitor,
    category="web_monitor"
)
registry.register(web_monitor_tool)

# Register ERP Integration tools
from tools import ERPIntegrationTools
erp_tools = ERPIntegrationTools()

registry.register(Tool(
    name="get_project_card",
    description="Получает карточку проекта по его коду (заказчик, комментарии, допуски, оборудование).",
    parameters={
        "type": "object",
        "properties": {
            "project_code": {
                "type": "string",
                "description": "Код проекта (например: PROJ-001)."
            }
        },
        "required": ["project_code"]
    },
    func=erp_tools.get_project_card,
    category="projects"
))

registry.register(Tool(
    name="get_trip_details",
    description="Получает детали командировки по ее ID (имя сотрудника, даты, цель поездки).",
    parameters={
        "type": "object",
        "properties": {
            "schedule_id": {
                "type": "integer",
                "description": "ID командировки (графика)."
            }
        },
        "required": ["schedule_id"]
    },
    func=erp_tools.get_trip_details,
    category="hr"
))

registry.register(Tool(
    name="append_task_details",
    description="Дополняет лог выполнения наряда (накапливаемый отчет) новой записью.",
    parameters={
        "type": "object",
        "properties": {
            "work_order_id": {
                "type": "integer",
                "description": "ID наряда (задачи)."
            },
            "text": {
                "type": "string",
                "description": "Текст нового отчета по работе."
            },
            "author_name": {
                "type": "string",
                "description": "Имя сотрудника, вносящего изменения."
            }
        },
        "required": ["work_order_id", "text", "author_name"]
    },
    func=erp_tools.append_task_details,
    category="projects"
))

registry.register(Tool(
    name="consolidate_to_project",
    description="Переносит/сохраняет отчет в карточку проекта (comments).",
    parameters={
        "type": "object",
        "properties": {
            "project_id": {
                "type": "integer",
                "description": "ID проекта."
            },
            "summary_text": {
                "type": "string",
                "description": "Сводный текст отчетов для карточки проекта."
            }
        },
        "required": ["project_id", "summary_text"]
    },
    func=erp_tools.consolidate_to_project,
    category="projects"
))

registry.register(Tool(
    name="get_task_comments",
    description="Получает историю переписки и комментариев в чате задачи.",
    parameters={
        "type": "object",
        "properties": {
            "work_order_id": {
                "type": "integer",
                "description": "ID задачи (work order)."
            }
        },
        "required": ["work_order_id"]
    },
    func=erp_tools.get_task_comments,
    category="projects"
))

registry.register(Tool(
    name="update_task_summary",
    description="Заменяет официальный сводный лог/отчет выполнения задачи (history_log).",
    parameters={
        "type": "object",
        "properties": {
            "work_order_id": {
                "type": "integer",
                "description": "ID задачи (work order)."
            },
            "summary_text": {
                "type": "string",
                "description": "Новая сводка/отчет хода работ."
            }
        },
        "required": ["work_order_id", "summary_text"]
    },
    func=erp_tools.update_task_summary,
    category="projects"
))

registry.register(Tool(
    name="list_upcoming_trips",
    description="Получает список всех запланированных и активных командировок сотрудников (включая ID графиков, имена и даты).",
    parameters={
        "type": "object",
        "properties": {}
    },
    func=erp_tools.list_upcoming_trips,
    category="hr"
))

registry.register(Tool(
    name="search_knowledge_base",
    description="Ищет информацию в регламентах, инструкциях и документации компании ООО 'Л-Старт' по ключевым словам.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Ключевые слова для поиска (например, 'инструкция ПНР', 'регламент отпуска', 'контакты ТЭЦ')."
            }
        },
        "required": ["query"]
    },
    func=erp_tools.search_knowledge_base
))

# Skills manager
skills_dir = ORCHESTRATOR_DIR / "skills"
skills_manager = SkillsManager(skills_dir)

@orchestrator_bp.route("/api/orchestrator/status", methods=["GET"])
def get_status():
    from app import config as runner_config
    server_url = f"http://{runner_config.settings['host']}:{runner_config.settings['port']}"
    online = is_server_online(server_url)
    return jsonify({
        "online": online,
        "server_url": server_url,
        "sandbox_path": str(sandbox_dir),
        "tools": [t.to_schema() for t in registry.tools.values()]
    })

@orchestrator_bp.route("/api/orchestrator/skills", methods=["GET"])
def get_skills():
    return jsonify(skills_manager.list_skills())

@orchestrator_bp.route("/api/orchestrator/skills/<skill_id>", methods=["POST"])
def save_skill(skill_id):
    data = request.get_json() or {}
    name = data.get("name", "")
    description = data.get("description", "")
    system_prompt = data.get("system_prompt", "")
    if not name or not system_prompt:
        return jsonify({"ok": False, "error": "Имя и системный промпт обязательны"}), 400
    saved = skills_manager.save_skill(skill_id, name, description, system_prompt)
    return jsonify({"ok": True, "skill": saved})

@orchestrator_bp.route("/api/orchestrator/logs", methods=["GET"])
def get_logs():
    try:
        if monitor_log_path.exists():
            return jsonify(json.loads(monitor_log_path.read_text(encoding="utf-8")))
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/api/orchestrator/providers", methods=["GET"])
def get_providers():
    try:
        config = Config.from_env()
        providers = []
        for p in config.providers:
            providers.append({
                "name": p.name,
                "type": p.type,
                "enabled": p.enabled,
                "base_url": p.base_url,
                "model": p.model,
                "temperature": p.temperature,
                "max_tokens": p.max_tokens,
                "role": p.role,
                "has_key": bool(p.api_key),
            })
        return jsonify({"providers": providers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/api/orchestrator/providers/<name>/test", methods=["POST"])
def test_provider(name):
    try:
        config = Config.from_env()
        provider_cfg = None
        for p in config.providers:
            if p.name == name:
                provider_cfg = p
                break
        if not provider_cfg:
            return jsonify({"ok": False, "error": f"Provider '{name}' not found"}), 404

        provider = OpenAICompatibleLLM(
            base_url=provider_cfg.base_url,
            api_key=provider_cfg.api_key,
            model=provider_cfg.model,
        ) if provider_cfg.type == "openai-compatible" else AnthropicLLM(
            api_key=provider_cfg.api_key,
            model=provider_cfg.model,
        ) if provider_cfg.type == "anthropic" else LlamaServerLLM(
            base_url=provider_cfg.base_url,
        )

        import time
        start = time.time()
        result = provider.generate(prompt="Hello", max_tokens=10)
        duration = time.time() - start

        return jsonify({
            "ok": result.get("ok", False),
            "provider": name,
            "duration_ms": round(duration * 1000, 1),
            "error": result.get("error"),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@orchestrator_bp.route("/api/orchestrator/providers/<name>/toggle", methods=["POST"])
def toggle_provider(name):
    try:
        config = Config.from_env()
        env_key = f"LLM_{name.upper()}_ENABLED"
        current = os.getenv(env_key, "true").lower()
        new_value = "false" if current == "true" else "true"
        os.environ[env_key] = new_value
        return jsonify({"name": name, "enabled": new_value == "true"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/api/orchestrator/providers/<name>/update", methods=["POST"])
def update_provider(name):
    try:
        data = request.get_json() or {}
        env_prefix = f"LLM_{name.upper()}"

        if "api_key" in data:
            os.environ[f"{env_prefix}_KEY"] = data["api_key"]
        if "base_url" in data:
            os.environ[f"{env_prefix}_URL"] = data["base_url"]
        if "model" in data:
            os.environ[f"{env_prefix}_MODEL"] = data["model"]
        if "temperature" in data:
            os.environ[f"{env_prefix}_TEMPERATURE"] = str(data["temperature"])
        if "max_tokens" in data:
            os.environ[f"{env_prefix}_MAX_TOKENS"] = str(data["max_tokens"])
        if "enabled" in data:
            os.environ[f"{env_prefix}_ENABLED"] = "true" if data["enabled"] else "false"

        return jsonify({"ok": True, "name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/api/orchestrator/router/status", methods=["GET"])
def router_status():
    try:
        config = Config.from_env()
        return jsonify({
            "strategy": config.router.strategy,
            "fallback_chain": config.router.fallback_chain,
            "classification_provider": config.router.classification_provider,
            "tool_call_provider": config.router.tool_call_provider,
            "complexity_threshold": config.router.complexity_threshold,
            "providers_count": len([p for p in config.providers if p.enabled]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/api/orchestrator/router/strategy", methods=["POST"])
def set_router_strategy():
    try:
        data = request.get_json() or {}
        strategy = data.get("strategy", "hybrid")
        if strategy not in ("hybrid", "local-first", "api-first"):
            return jsonify({"error": "Invalid strategy"}), 400
        os.environ["ROUTE_STRATEGY"] = strategy
        return jsonify({"strategy": strategy})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/api/orchestrator/complexity", methods=["POST"])
def check_complexity():
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "")
        score = classify_complexity(prompt)
        return jsonify({"prompt": prompt[:100], "complexity": score, "threshold": 0.7})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/api/orchestrator/learning/patterns", methods=["GET"])
def get_learning_patterns():
    try:
        user_id = request.args.get("user_id")
        scope = request.args.get("scope")
        limit = int(request.args.get("limit", 50))
        patterns = _session_store.get_patterns(user_id=user_id, scope=scope, limit=limit)
        return jsonify({"patterns": patterns, "total": len(patterns)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/api/orchestrator/learning/stats", methods=["GET"])
def get_learning_stats():
    try:
        user_id = request.args.get("user_id", "anonymous")
        stats = _session_store.get_user_stats(user_id)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/api/orchestrator/learning/feedback", methods=["POST"])
def learning_feedback():
    try:
        data = request.get_json() or {}
        pattern_id = data.get("pattern_id")
        rating = data.get("rating", 1.0)
        if not pattern_id:
            return jsonify({"error": "pattern_id required"}), 400
        _learner.record_feedback(pattern_id, rating)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@orchestrator_bp.route("/api/orchestrator/session/clear", methods=["POST"])
def clear_session():
    data = request.get_json() or {}
    user_id = data.get("user_id", "anonymous")
    session_id = data.get("session_id", "default")
    _session_store.clear_session(user_id, session_id)
    return jsonify({"ok": True})

@orchestrator_bp.route("/api/orchestrator/session/history", methods=["POST"])
def get_session_history():
    data = request.get_json() or {}
    user_id = data.get("user_id", "anonymous")
    session_id = data.get("session_id", "default")
    messages = _session_store.get_history(user_id, session_id)
    return jsonify({"messages": messages})

@orchestrator_bp.route("/api/orchestrator/run", methods=["POST"])
def run_orchestrator():
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "")
        user_id = data.get("user_id", "anonymous")
        session_id = data.get("session_id", "default")

        # Apply secure dynamic configurations for calling back the ERP
        erp_url = data.get("erp_url")
        erp_service_token = data.get("erp_service_token")
        if erp_url:
            erp_tools.base_url = erp_url.rstrip("/")
        if erp_service_token:
            erp_tools.service_token = erp_service_token

        if not prompt:
            return jsonify({"ok": False, "error": "Промпт пуст"}), 400

        # Pool-aware routing: use complexity to select the right model slot
        try:
            from app import pool
            from router import classify_complexity
            complexity = classify_complexity(prompt)
            slot = pool.get_slot_for_complexity(complexity)
            if slot and slot.is_running():
                target_port = slot.port
                provider_name = f"local:{slot.id}"
                llm = LlamaServerLLM(base_url=f"http://127.0.0.1:{target_port}")
            else:
                config = Config.from_env()
                llm = create_llm_from_config(config)
                provider_name = getattr(llm, 'provider', 'local')
        except Exception:
            config = Config.from_env()
            llm = create_llm_from_config(config)
            provider_name = getattr(llm, 'provider', 'local')

        # Initialize all required dependencies
        project_dir = ORCHESTRATOR_DIR
        cache = ResponseCache(str(project_dir / "cache.db"))
        rate_limiter = DualRateLimiter(llm_rate=5, llm_burst=10)
        metrics = MetricsCollector(str(project_dir / "metrics.db"))
        input_guard = InputGuardrails()
        output_guard = OutputGuardrails()
        token_mgr = TokenManager(max_context=config.conversation_max_messages or 4096)

        rag_engine = BM25SearchEngine()
        knowledge_dir = project_dir / "knowledge_base"
        if knowledge_dir.exists():
            rag_engine.index_directory(knowledge_dir)

        # Load conversation history from persistent store
        history = _session_store.get_history(user_id, session_id, limit=config.conversation_max_messages)
        conversation = ConversationBuffer(max_messages=config.conversation_max_messages, user_id=user_id)
        for msg in history:
            if msg["role"] == "user":
                conversation.add_user_message(msg["content"])
            else:
                conversation.add_assistant_message(msg["content"])

        dispatcher = QueryDispatcher(llm)

        # Disable self-critique by default — it adds an extra LLM call
        config.self_critique_enabled = data.get("self_critique", False)

        # Skill_id from ERP frontend takes priority over dispatcher scope
        skill_id = data.get("skill_id")
        skills = skills_manager.list_skills()
        if skill_id and skill_id in skills:
            # Override dispatcher scope-based prompt with skill system_prompt
            config.self_critique_enabled = False  # skip extra LLM call for module agents

        result = run_agentic_loop(
            llm=llm,
            registry=registry,
            user_prompt=prompt,
            dispatcher=dispatcher,
            conversation=conversation,
            rag_engine=rag_engine,
            config=config,
            cache=cache,
            rate_limiter=rate_limiter,
            metrics=metrics,
            input_guard=input_guard,
            output_guard=output_guard,
            token_mgr=token_mgr,
            self_learner=_learner,
            skill_id=skill_id,
            skills_manager=skills_manager,
        )

        # Save conversation to persistent store
        scope = result.get("scope", "general")
        tools_used = result.get("tool_calls", [])
        _session_store.save_message(user_id, session_id, "user", prompt, scope=scope)
        if result.get("ok"):
            _session_store.save_message(user_id, session_id, "assistant",
                                        result.get("content", ""),
                                        scope=scope, tool_calls=tools_used)

        # Get updated history
        history = _session_store.get_history(user_id, session_id, limit=config.conversation_max_messages)

        return jsonify({
            "ok": result.get("ok", False),
            "response": result.get("content", ""),
            "user_id": user_id,
            "provider": provider_name,
            "scope": scope,
            "duration_ms": result.get("duration_ms", 0),
            "tools_used": tools_used,
            "history": history,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
