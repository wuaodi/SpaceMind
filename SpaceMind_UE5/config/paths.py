from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
EVALUATION_DIR = CONFIG_DIR / "evaluation"
SKILLS_DIR = PROJECT_ROOT / "skills"
RUNTIME_LOGS_DIR = PROJECT_ROOT / "runtime_logs"
RUNTIME_LOG_DIR = RUNTIME_LOGS_DIR / "log"
RUNTIME_TRACE_DIR = RUNTIME_LOGS_DIR / "trace"
TRACE_VIEWER_PATH = RUNTIME_LOGS_DIR / "viewer" / "trace_viewer.html"
FRAMEWORK_MANIFEST_PATH = CONFIG_DIR / "framework_manifest.json"
ROOT_DOTENV_PATH = PROJECT_ROOT / ".env"
TOOLS_SERVER_MODULE = "tools.mcp_server"
