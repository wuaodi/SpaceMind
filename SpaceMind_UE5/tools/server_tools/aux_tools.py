import os
import subprocess
import sys
import tempfile

from mcp.server.fastmcp import FastMCP

from .common import logger


def _truncate_log_text(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _format_execution_result(runtime: str, returncode: int, stdout: str, stderr: str) -> str:
    lines = [
        f"runtime={runtime}",
        f"exit_code={returncode}",
        "stdout:",
        stdout.strip() or "(empty)",
    ]
    if stderr.strip():
        lines.extend(["stderr:", stderr.strip()])
    return "\n".join(lines)


def _build_python_wrapper(code: str) -> str:
    return f"""# -*- coding: utf-8 -*-
import ast
import contextlib
import io
import sys

_source = {code!r}
_globals = {{"__name__": "__main__"}}
_stdout_buffer = io.StringIO()
_spacemind_last_expr = None

_tree = ast.parse(_source, filename="<execute_code>", mode="exec")
if _tree.body and isinstance(_tree.body[-1], ast.Expr):
    _tree.body[-1] = ast.Assign(
        targets=[ast.Name(id="_spacemind_last_expr", ctx=ast.Store())],
        value=_tree.body[-1].value,
    )
ast.fix_missing_locations(_tree)

with contextlib.redirect_stdout(_stdout_buffer):
    exec(compile(_tree, "<execute_code>", "exec"), _globals, _globals)

_captured = _stdout_buffer.getvalue()
if _captured:
    sys.stdout.write(_captured)
elif "result" in _globals:
    print(repr(_globals["result"]))
elif _globals.get("_spacemind_last_expr") is not None:
    print(repr(_globals["_spacemind_last_expr"]))
"""


def _run_code_unrestricted(code: str, runtime: str) -> str:
    runtime = (runtime or "python").strip().lower()
    if runtime not in {"python", "bash"}:
        return f"Error: Unsupported runtime '{runtime}'. Use 'python' or 'bash'."

    temp_path = None
    if runtime == "python":
        wrapper = _build_python_wrapper(code)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".py",
            delete=False,
        ) as handle:
            handle.write(wrapper)
            temp_path = handle.name
        command = [sys.executable, "-X", "utf8", temp_path]
    else:
        command = ["bash", "-lc", code]

    logger.info("execute_code request runtime=%s\n%s", runtime, _truncate_log_text(code))
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"

    try:
        completed = subprocess.run(
            command,
            cwd=os.getcwd(),
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=20,
            shell=False,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
    except FileNotFoundError as e:
        logger.error("execute_code failed to start runtime=%s: %s", runtime, e)
        return f"Error: Failed to start runtime '{runtime}': {e}"
    except subprocess.TimeoutExpired:
        logger.error("execute_code timeout runtime=%s", runtime)
        return f"Error: Execution timeout (20s) for runtime={runtime}"
    except Exception as e:
        logger.exception("execute_code unexpected failure runtime=%s", runtime)
        return f"Error: {str(e)}"
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("execute_code could not remove temp file: %s", temp_path)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    logger.info(
        "execute_code result runtime=%s exit_code=%s\nstdout:\n%s\nstderr:\n%s",
        runtime,
        completed.returncode,
        _truncate_log_text(stdout),
        _truncate_log_text(stderr),
    )
    return _format_execution_result(runtime, completed.returncode, stdout, stderr)


def register_aux_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def knowledge_base() -> str:
        """Return target-specific knowledge if one has been loaded.

        Returns:
            A text block containing target-specific prior knowledge.
        """
        return (
            "Knowledge base is currently empty for this target.\n"
            "Do not assume spacecraft type, size, or components from prior knowledge.\n"
            "Rely on the currently observed evidence instead."
        )

    @mcp.tool()
    def execute_code(code: str, runtime: str = "python") -> str:
        """Execute code or shell commands in an unrestricted runtime.

        This tool is a general-purpose computation / execution helper.
        The caller must explicitly choose the runtime.

        Supported runtimes:
        - python: executes `code` with the current Python interpreter
        - bash: executes `code` via `bash -lc`

        Usage guidance:
        - Use this tool when you need to calculate geometry, transform coordinates,
          inspect files, or run CLI commands that help decide the next action.
        - This tool itself does not move the spacecraft. If you need to act on the
          result, do that in a later step with set_position() or set_attitude().

        Args:
            code: Source code or shell command string to execute
            runtime: Runtime selector, one of "python" or "bash"

        Returns:
            A text block containing runtime, exit_code, stdout, and optional stderr
        """
        try:
            return _run_code_unrestricted(code, runtime)
        except Exception as e:
            logger.exception("execute_code wrapper failure runtime=%s", runtime)
            return f"Error: {str(e)}"
