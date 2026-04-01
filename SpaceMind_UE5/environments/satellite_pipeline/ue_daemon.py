r"""
UE 编辑器内守护脚本，实现卫星自动切换 + PIE 自动启停。

参考: D:/project/mm-Space-Bench/data_collect/ue_daemon.py

使用方式（UE -> Output Log -> Python）:
import sys
sys.path.append(r"D:\project\SpaceMind\SpaceMind_UE5\environments\satellite_pipeline")
import ue_daemon as d
d.start()

外部脚本 (run_phase_b.py) 通过文件 IPC 下发命令:
- 写 _tmp/ue_cmd.json  -> daemon 读取后自动执行换星+导出映射+启动PIE
- 读 _tmp/ready.flag   -> PIE 已启动，可以开始实验
- 写 _tmp/done.flag    -> 实验结束，daemon 停止PIE
- 读 _tmp/idle.flag    -> daemon 空闲，可接受下一条命令

所有 UE 操作在主线程执行（slate_post_tick_callback），避免 IsInGameThread() 断言。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import unreal

import ue_place_satellite as sat_tool

_tick_handle = None
_state = "idle"
_current_cmd = None
_base_dir = None
_wait_until = 0.0


def _log(msg: str) -> None:
    try:
        unreal.log(msg)
    except Exception:
        pass


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ok\n", encoding="utf-8")


def _rm(p: Path) -> None:
    try:
        p.unlink()
    except Exception:
        pass


def _main_thread_tick(delta_seconds: float) -> None:
    global _state, _current_cmd, _base_dir, _wait_until

    if _base_dir is None:
        return

    cmd_path = _base_dir / "ue_cmd.json"
    ready_flag = _base_dir / "ready.flag"
    done_flag = _base_dir / "done.flag"
    idle_flag = _base_dir / "idle.flag"

    if _state == "idle":
        if cmd_path.exists():
            try:
                _current_cmd = json.loads(cmd_path.read_text(encoding="utf-8"))
                _rm(cmd_path)
                _rm(idle_flag)
                _rm(ready_flag)
                _rm(done_flag)

                sat = str(_current_cmd.get("sat_name", "")).strip()
                if not sat:
                    _touch(idle_flag)
                    return

                _log(f"[ue_daemon] received cmd: sat={sat}")
                _state = "place_satellite"
            except Exception as e:
                _log(f"[ue_daemon] invalid cmd json: {e}")
                _rm(cmd_path)

    elif _state == "place_satellite":
        sat = str(_current_cmd.get("sat_name", "")).strip()
        _log(f"[ue_daemon] placing satellite: {sat}")
        try:
            sat_tool.place_selected_satellite(sat)
            _log(f"[ue_daemon] place done, waiting 5s...")
            _wait_until = time.time() + 5.0
            _state = "wait_after_place"
        except Exception as e:
            _log(f"[ue_daemon] place failed: {e}")
            _touch(idle_flag)
            _state = "idle"
            _current_cmd = None

    elif _state == "wait_after_place":
        if time.time() >= _wait_until:
            _state = "export_mapping"

    elif _state == "export_mapping":
        mapping_out = str(_current_cmd.get("mapping_out", "")).strip()
        if not mapping_out:
            mapping_out = str((_base_dir / "name_label_mapping.txt").resolve())
        mapping_dir = Path(mapping_out).parent
        mapping_dir.mkdir(parents=True, exist_ok=True)
        try:
            sat_tool.dump_name_label_mapping(mapping_out, only_static_mesh_actor=True)
            _log(f"[ue_daemon] mapping exported -> {mapping_out}, waiting 3s...")
            _wait_until = time.time() + 3.0
            _state = "wait_after_mapping"
        except Exception as e:
            _log(f"[ue_daemon] export mapping failed: {e}")
            _touch(idle_flag)
            _state = "idle"
            _current_cmd = None

    elif _state == "wait_after_mapping":
        if time.time() >= _wait_until:
            _log("[ue_daemon] starting PIE...")
            _state = "start_pie"

    elif _state == "start_pie":
        ok = False
        try:
            ok = bool(unreal.PieControlBPLibrary.start_pie())
        except Exception as e:
            _log(f"[ue_daemon] start_pie failed: {e}")

        _log(f"[ue_daemon] start_pie ok={ok}")
        _touch(_base_dir / "ready.flag")
        _state = "wait_done"

    elif _state == "wait_done":
        if done_flag.exists():
            _log("[ue_daemon] done flag detected, stopping PIE...")
            _state = "stop_pie"

    elif _state == "stop_pie":
        try:
            unreal.PieControlBPLibrary.stop_pie()
            _log("[ue_daemon] PIE stopped, waiting 5s...")
            _wait_until = time.time() + 5.0
            _state = "wait_after_stop"
        except Exception as e:
            _log(f"[ue_daemon] stop_pie failed: {e}")
            _rm(ready_flag)
            _rm(done_flag)
            _touch(idle_flag)
            _state = "idle"
            _current_cmd = None

    elif _state == "wait_after_stop":
        if time.time() >= _wait_until:
            _rm(ready_flag)
            _rm(done_flag)
            _touch(idle_flag)

            sat = str(_current_cmd.get("sat_name", "")) if _current_cmd else "?"
            _log(f"[ue_daemon] sat={sat} finished, back to idle")
            _state = "idle"
            _current_cmd = None


def start(tmp_dir: str | None = None) -> None:
    """
    启动守护进程。在 UE Python 控制台执行一次：
    import ue_daemon as d; d.start()
    """
    global _tick_handle, _base_dir, _state

    if _tick_handle is not None:
        _log("[ue_daemon] already running")
        return

    _base_dir = Path(tmp_dir) if tmp_dir else (Path(__file__).parent / "_tmp")
    _base_dir.mkdir(parents=True, exist_ok=True)

    _rm(_base_dir / "ready.flag")
    _rm(_base_dir / "done.flag")
    _touch(_base_dir / "idle.flag")
    _state = "idle"

    _tick_handle = unreal.register_slate_post_tick_callback(_main_thread_tick)

    _log(f"[ue_daemon] started (main thread tick), tmp_dir={_base_dir.as_posix()}")
    _log("[ue_daemon] waiting for ue_cmd.json ...")


def stop() -> None:
    """停止守护进程。"""
    global _tick_handle, _state

    if _tick_handle is not None:
        unreal.unregister_slate_post_tick_callback(_tick_handle)
        _tick_handle = None
        _state = "idle"
        _log("[ue_daemon] stopped")
