# 启动:
#   python launch_env.py
#   python launch_env.py --skip-build
#   python launch_env.py --no-open
#   python launch_env.py --satellite IBEX                          # 切换目标星
#   python launch_env.py --init_x -11 --init_y 0 --init_z 0        # 初始位形（对齐 C1 标称位）
#   python launch_env.py --spin_deg_s 0.5                          # E1 旋转目标
#   python launch_env.py --noise                                   # E3N 执行噪声
#   python launch_env.py --fault_axis dy --fault_scale 0.5         # E3F 推力故障
#   python launch_env.py --lidar_dropout 0.3                       # E4L LiDAR 间歇失效
#   python launch_env.py --exposure_disturb_step 3 --exposure_disturb_value -3  # E4E 曝光扰动
#
# 作用:
#   1) 检查 Redis
#   2) 确保 npm 依赖存在
#   3) 构建前端 app/dist
#   4) 启动 bridge（转发环境参数）和本地静态页面服务
#   5) 自动打开浏览器
#
# 停止:
#   在当前终端按 Ctrl+C，会自动结束 bridge 和 http server

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_DIST = ROOT / "app" / "dist"
LOG_DIR = ROOT / "logs"
NODE_MODULES = ROOT / "node_modules"


def print_line(message: str) -> None:
    print(f"[threejs_env] {message}", flush=True)


def run_checked(command: list[str], cwd: Path, desc: str) -> None:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    print_line(desc)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{desc} failed with exit code {completed.returncode}")


def ensure_dependencies() -> None:
    if NODE_MODULES.exists():
        return
    run_checked(
        ["cmd", "/c", "npm install --ignore-scripts --cache .npm-cache"],
        ROOT,
        "node_modules 缺失，正在安装 npm 依赖",
    )


def build_frontend(skip_build: bool) -> None:
    if skip_build:
        if not APP_DIST.exists():
            raise RuntimeError("已指定 --skip-build，但 app/dist 不存在。")
        return
    run_checked(["cmd", "/c", "npm run build:app"], ROOT, "正在构建 three.js 前端")
    if not APP_DIST.exists():
        raise RuntimeError("构建完成后未找到 app/dist。")


def redis_port_open(host: str = "127.0.0.1", port: int = 6379) -> bool:
    sock = socket.socket()
    sock.settimeout(1.0)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def wait_port(port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        sock = socket.socket()
        sock.settimeout(0.5)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            time.sleep(0.2)
        finally:
            sock.close()
    return False


def start_logged_process(command: list[str], cwd: Path, stdout_name: str, stderr_name: str) -> tuple[subprocess.Popen[str], object, object]:
    LOG_DIR.mkdir(exist_ok=True)
    stdout_path = LOG_DIR / stdout_name
    stderr_path = LOG_DIR / stderr_name
    stdout_handle = open(stdout_path, "w", encoding="utf-8")
    stderr_handle = open(stderr_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return process, stdout_handle, stderr_handle


def terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Three.js SpaceMind environment.")
    parser.add_argument("--skip-build", action="store_true", help="Do not rebuild app/dist.")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open the browser.")
    parser.add_argument("--http-port", type=int, default=4173, help="Static HTTP server port.")
    parser.add_argument("--ws-port", type=int, default=8765, help="Expected bridge WebSocket port.")
    # 环境参数，与 fly_redis.py 对齐，原样转发给 bridge
    parser.add_argument("--satellite", default=None,
                        choices=["BioSentinel", "CAPSTONE", "Huygens", "IBEX", "New_Horizons"],
                        help="Target satellite model.")
    parser.add_argument("--init_x", type=float, default=None, help="Servicer initial x (m).")
    parser.add_argument("--init_y", type=float, default=None, help="Servicer initial y (m).")
    parser.add_argument("--init_z", type=float, default=None, help="Servicer initial z (m).")
    parser.add_argument("--init_yaw", type=float, default=None, help="Servicer initial yaw (rad).")
    parser.add_argument("--spin_deg_s", type=float, default=None, help="E1 target spin rate (deg/s).")
    parser.add_argument("--noise", action="store_true", help="E3N actuation noise.")
    parser.add_argument("--noise_pos", type=float, default=None, help="E3N position noise sigma (m).")
    parser.add_argument("--noise_att", type=float, default=None, help="E3N attitude noise sigma (rad).")
    parser.add_argument("--fault_axis", default=None, choices=["dx", "dy", "dz"], help="E3F fault axis.")
    parser.add_argument("--fault_scale", type=float, default=None, help="E3F fault execution scale.")
    parser.add_argument("--lidar_dropout", type=float, default=None, help="E4L LiDAR dropout probability.")
    parser.add_argument("--exposure_disturb_step", type=int, default=None,
                        help="E4E inject exposure disturbance after Nth pose_change.")
    parser.add_argument("--exposure_disturb_value", type=float, default=None,
                        help="E4E disturbed exposure value.")
    return parser.parse_args()


def build_bridge_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    value_options = [
        "satellite", "init_x", "init_y", "init_z", "init_yaw", "spin_deg_s",
        "noise_pos", "noise_att", "fault_axis", "fault_scale", "lidar_dropout",
        "exposure_disturb_step", "exposure_disturb_value",
    ]
    for name in value_options:
        value = getattr(args, name)
        if value is not None:
            forwarded.extend([f"--{name}", str(value)])
    if args.noise:
        forwarded.append("--noise")
    return forwarded


def main() -> int:
    args = parse_args()

    if not redis_port_open():
        print_line("未检测到 Redis(127.0.0.1:6379)。请先启动 Redis。")
        return 1

    ensure_dependencies()
    build_frontend(skip_build=args.skip_build)

    print_line("正在启动 bridge ...")
    bridge_proc, bridge_out, bridge_err = start_logged_process(
        ["node", "--experimental-strip-types", "bridge/index.ts", *build_bridge_args(args)],
        ROOT,
        "launcher_bridge_stdout.log",
        "launcher_bridge_stderr.log",
    )

    print_line("正在启动静态页面服务 ...")
    http_proc, http_out, http_err = start_logged_process(
        [sys.executable, "-m", "http.server", str(args.http_port)],
        APP_DIST,
        "launcher_http_stdout.log",
        "launcher_http_stderr.log",
    )

    try:
        if not wait_port(args.ws_port, 10.0):
            raise RuntimeError(f"bridge 端口 {args.ws_port} 未在预期时间内就绪。")
        if not wait_port(args.http_port, 10.0):
            raise RuntimeError(f"http 端口 {args.http_port} 未在预期时间内就绪。")

        url = f"http://127.0.0.1:{args.http_port}"
        print_line(f"环境已启动: {url}")
        print_line(f"bridge 日志: {LOG_DIR / 'launcher_bridge_stdout.log'}")
        print_line(f"http 日志: {LOG_DIR / 'launcher_http_stdout.log'}")
        print_line("建议下一步:")
        print_line("  1) 先运行: python self_test.py --scenario smoke")
        print_line("  2) 再运行 host.py 做真实 agent 测试")
        print_line("  3) 按 Ctrl+C 停止环境")

        if not args.no_open:
            webbrowser.open(url)

        while True:
            if bridge_proc.poll() is not None:
                raise RuntimeError("bridge 已退出，请检查 launcher_bridge_stderr.log")
            if http_proc.poll() is not None:
                raise RuntimeError("http server 已退出，请检查 launcher_http_stderr.log")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print_line("收到 Ctrl+C，正在停止环境 ...")
    finally:
        terminate_process(bridge_proc)
        terminate_process(http_proc)
        bridge_out.close()
        bridge_err.close()
        http_out.close()
        http_err.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
