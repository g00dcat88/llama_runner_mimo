import sys
import json
from pathlib import Path
from flask import Blueprint, jsonify, request

# Add llm_orchestrator to path so we can import gateway, tools, skills, etc.
APP_DIR = Path(__file__).resolve().parent
ORCHESTRATOR_DIR = APP_DIR.parent / "llm_orchestrator"
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.append(str(ORCHESTRATOR_DIR))

from gateway import LlamaServerLLM
from tools import Tool, ToolRegistry, PythonSandbox, WebMonitorTool
from skills import SkillsManager
from main import run_agentic_loop, is_server_online, MockLLM

orchestrator_bp = Blueprint("orchestrator", __name__)

@orchestrator_bp.before_request
def check_api_key():
    # Exclude health-check endpoint from API key requirement
    if request.path == "/api/orchestrator/status":
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

@orchestrator_bp.route("/api/orchestrator/run", methods=["POST"])
def run_orchestrator():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    max_retries = int(data.get("max_retries", 3))

    # Apply secure dynamic configurations for calling back the ERP
    erp_url = data.get("erp_url")
    erp_service_token = data.get("erp_service_token")
    if erp_url:
        erp_tools.base_url = erp_url.rstrip("/")
    if erp_service_token:
        erp_tools.service_token = erp_service_token
    
    if not prompt:
        return jsonify({"ok": False, "error": "Промпт пуст"}), 400
        
    from app import config as runner_config
    server_url = f"http://{runner_config.settings['host']}:{runner_config.settings['port']}"
    
    if is_server_online(server_url):
        llm = LlamaServerLLM(base_url=server_url)
    else:
        llm = MockLLM()
        
    steps_log = []
    def log_callback(msg):
        steps_log.append(msg)

    # Инициализируем классификатор и определяем скоуп и навык
    from dispatcher import QueryDispatcher
    dispatcher = QueryDispatcher(llm)
    classification = dispatcher.classify(prompt)
    scope_raw = classification.get("scope", "general")
    reason = classification.get("reason", "По умолчанию")
    
    # Сопоставляем категорию классификатора с конфигурацией скоупов/навыков
    if scope_raw in ["hr_single", "hr_summary"]:
        scope = "hr"
        skill_id = "hr_assistant"
    elif scope_raw in ["fsm_single", "fsm_summary"]:
        scope = "projects"
        skill_id = "projects_assistant"
    elif scope_raw == "python_sandbox":
        scope = "python_sandbox"
        skill_id = "python_coder"
    elif scope_raw == "web_monitor":
        scope = "web_monitor"
        skill_id = "web_monitoring"
    elif scope_raw == "task_constructor":
        scope = "general"
        skill_id = "task_constructor_assistant"
    else:
        scope = "general"
        skill_id = "core_agent"

    log_callback(f"🤖 [Маршрутизатор] Анализ запроса... Направление: {scope_raw} -> Скоуп: '{scope}', Навык: '{skill_id}' (Обоснование: {reason})")
        
    # Загружаем соответствующий системный промпт
    skills = skills_manager.list_skills()
    selected_skill = skills.get(skill_id, skills.get("core_agent"))
    system_prompt = selected_skill["system_prompt"] if selected_skill else None
    
    try:
        result = run_agentic_loop(
            llm=llm,
            registry=registry,
            user_prompt=prompt,
            max_retries=max_retries,
            system_prompt=system_prompt,
            log_callback=log_callback,
            scope=scope
        )
        return jsonify({
            "ok": True,
            "steps": steps_log,
            "response": result.get("content", "") if result else ""
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "steps": steps_log}), 500
