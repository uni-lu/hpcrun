#!/usr/bin/env python
"""
Telerun: a small, portable tool to build, run, and evaluate code
on hpc machines.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import subprocess as sp
import sys
from datetime import datetime, timezone
import re
import tempfile
import textwrap
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import yaml
import paramiko

LOG_DIR = Path(".telerun_logs")
LOG_DIR.mkdir(exist_ok=True)
ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")  # strip terminal color codes
DEBUG = False

CLOUDFARE_URL = "https://telerun-results.huangweiurob.workers.dev"

# ------------------------------
# Utilities
# ------------------------------
class ShellError(RuntimeError):
    def __init__(self, cmd: str, returncode: int, where: str = "local"):
        super().__init__(f"Command failed ({where}): {cmd}\nreturn code: {returncode}")
        self.cmd = cmd
        self.returncode = returncode
        self.where = where

def normalize(s: str) -> str:
    # remove ANSI, normalize line endings, trim trailing spaces per line
    s = ANSI_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    return "\n".join(lines).rstrip() + "\n"  # keep a single trailing newline

def _now_ms() -> int:
    return int(time.time() * 1000)


def log_path(prefix: str) -> Path:
    return LOG_DIR / f"{prefix}-{_now_ms()}.log"


def read_yaml(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".json"}:
        return json.loads(path.read_text())
    if yaml is None:
        raise SystemExit(
            "PyYAML is not installed. Install it with 'pip install pyyaml' or use a JSON config."
        )
    return yaml.safe_load(path.read_text()) or {}


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    if path.suffix.lower() in {".json"}:
        path.write_text(json.dumps(data, indent=2))
    else:
        if yaml is None:
            raise SystemExit(
                "PyYAML is not installed. Install it with 'pip install pyyaml' to write YAML."
            )
        path.write_text(yaml.safe_dump(data, sort_keys=False))

def _pretty_ts(ts: str) -> str:
    """Convert ISO timestamp to 'YYYY-MM-DD HH:MM:SS' (UTC)."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts[:19] 

class KVSubmitError(RuntimeError):
    pass

def submit_result(
    submit_url: str,
    *,
    lab: str,
    user: str,
    runtime_ms: float,
    token: Optional[str] = None,
    timeout: float = 10.0,
    # you can pass through extra fields if your Worker supports them
    **extra: Any,
) -> Dict[str, Any]:
    """
    Submit a result to Cloudflare Worker.
    - submit_url: e.g. "https://telerun-results.<zone>.workers.dev/submit"
    - lab, user, runtime_ms: minimal payload
    - token: optional Bearer token
    - timeout: request timeout (seconds)
    - **extra: any additional JSON fields (e.g. test, correct, perf_ok, details, commit, ts)

    Returns: parsed JSON response (dict). Raises KVSubmitError on failure.
    """
    payload = {"lab": lab, "user": user, "runtime_ms": float(runtime_ms)}
    payload.update(extra)

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(submit_url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(data) if data else {}
            except json.JSONDecodeError:
                raise KVSubmitError(f"Non-JSON response: HTTP {resp.status} — {data[:300]}")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
        except json.JSONDecodeError:
            err_json = {"error": err_body.strip() or e.reason}
        raise KVSubmitError(f"HTTP {e.code}: {err_json}") from None
    except urllib.error.URLError as e:
        raise KVSubmitError(f"Request failed: {e.reason}") from None



def print_box(title: str) -> None:
    bar = "=" * len(title)
    print(f"{bar}\n{title}\n{bar}")

# --- .env helpers ---
def load_dotenv(path: Path = Path(".env")) -> None:
    """Load .env. Uses python-dotenv"""
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(path, override=True)
    return

def _expand(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return os.path.expanduser(os.path.expandvars(p))


# ------------------------------
# Config dataclasses
# ------------------------------
@dataclasses.dataclass
class Thresholds:
    correctness: str = "diff"  # one of: "diff" or a custom shell command with {expected} {actual}
    perf_ms: Optional[int] = None  # max allowed runtime (ms) per test
    perf_reference: Optional[str] = None  # path to reference binary to compare against (optional)


@dataclasses.dataclass
class SSHConfig:
    host: Optional[str] = None
    user: Optional[str] = None
    port: int = 22
    key_path: Optional[str] = None
    remote_dir: Optional[str] = ".telerun"
    rsync: bool = True
    # --- normalize fields right after creation ---
    def __post_init__(self):
        # expand ~ and $VARS
        if self.key_path:
            self.key_path = os.path.expanduser(os.path.expandvars(self.key_path))
        if self.remote_dir:
            self.remote_dir = os.path.expanduser(os.path.expandvars(self.remote_dir))
        # coerce port if it came in as a string
        if isinstance(self.port, str):
            try:
                self.port = int(self.port)
            except ValueError:
                self.port = 22
        # normalize empties
        if self.user == "":
            self.user = None
        if self.host == "":
            self.host = None

    @classmethod
    def from_env(cls, base: "SSHConfig" | None = None) -> "SSHConfig":
        data: Dict = dataclasses.asdict(base) if base else {}
        env = os.environ
        if env.get("TELERUN_SSH_HOST"):      data["host"] = env["TELERUN_SSH_HOST"]
        if env.get("TELERUN_SSH_USER"):      data["user"] = env["TELERUN_SSH_USER"]
        if env.get("TELERUN_SSH_PORT"):      data["port"] = env["TELERUN_SSH_PORT"]
        if env.get("TELERUN_SSH_KEY_PATH"):  data["key_path"] = env["TELERUN_SSH_KEY_PATH"]
        if env.get("TELERUN_SSH_RSYNC"):
            data["rsync"] = env["TELERUN_SSH_RSYNC"].strip().lower() in {"1", "true", "yes", "on"}
        return cls(**data)
    

@dataclasses.dataclass
class SlurmConfig:
    enabled: bool = False
    partition: Optional[str] = None
    time: Optional[str] = None            # e.g. "00:05:00"
    nodes: Optional[int] = None
    ntasks: Optional[int] = 1
    cpus_per_task: Optional[int] = None
    gpus: Optional[int] = None
    mem: Optional[str] = None            # e.g. "4G"
    account: Optional[str] = None
    qos: Optional[str] = None
    exclusive: bool = False
    extra_args: List[str] = dataclasses.field(default_factory=list)
    

@dataclasses.dataclass
class BuildConfig:
    build_script: Optional[str] = None  # path to a script to run (bash, make, etc.)
    binary: Optional[str] = None
    eval_binary: Optional[str] = None
    compile_commands: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TestCase:
    name: str
    args: List[str] = dataclasses.field(default_factory=list)
    stdin: Optional[str] = None  # inline input (string)
    input_file: Optional[str] = None  # or a file path
    expected_output: Optional[str] = None  # inline expected
    expected_file: Optional[str] = None  # or file path
    leaderboard: bool = False

@dataclasses.dataclass
class PipeStep:
    name: str
    kind: str                    # "sync" | "build" | "run" | "eval" | "shell"
    test: Optional[str] = None   # for kind=="run"
    env: Dict[str, str] = dataclasses.field(default_factory=dict)
    continue_on_error: bool = False
    submit: bool = False

@dataclasses.dataclass
class Pipeline:
    name: str
    steps: List[PipeStep]

@dataclasses.dataclass
class TelerunConfig:
    lab_name: str = ""
    source_code_loc: str = "."
    run_command: Optional[str] = None  # fallback if no binary
    module_script: Optional[str] = None
    build: BuildConfig = dataclasses.field(default_factory=BuildConfig)
    ssh: SSHConfig = dataclasses.field(default_factory=SSHConfig)
    slurm: SlurmConfig = dataclasses.field(default_factory=SlurmConfig)
    thresholds: Thresholds = dataclasses.field(default_factory=Thresholds)
    tests: List[TestCase] = dataclasses.field(default_factory=list)
    env: Dict[str, str] = dataclasses.field(default_factory=dict)
    pipelines: Dict[str, Pipeline] = dataclasses.field(default_factory=dict)
    pokemon: str = ""

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TelerunConfig":
        env = os.environ
        build = BuildConfig(**(d.get("build", {}) or {}))
        ssh = SSHConfig.from_env()
        thr = Thresholds(**(d.get("thresholds", {}) or {}))
        tests = [TestCase(**x) for x in d.get("tests", [])]
        pipes: Dict[str, Pipeline] = {}

        for pname, raw_steps in (d.get("pipelines") or {}).items():
            steps = [PipeStep(**s) for s in raw_steps]
            pipes[pname] = Pipeline(name=pname, steps=steps)
        return TelerunConfig(
            lab_name=d.get("lab_name", ""),
            source_code_loc=d.get("source_code_loc", "."),
            run_command=d.get("run_command"),
            module_script=d.get("module_script"),
            build=build,
            ssh=ssh,
            slurm=SlurmConfig(**(d.get("slurm", {}) or {})),
            thresholds=thr,
            tests=tests,
            pipelines=pipes,
            env=d.get("env", {}),
            pokemon=env['TELERUN_POKEMON_NAME'] if env.get("TELERUN_POKEMON_NAME") else ""
        )


# ------------------------------
# Local & remote command execution
# ------------------------------
class Streamer:
    """Run a command on hpc and stream stdout/stderr live."""

    def __init__(self, where: str = "local"):
        self.where = where

    def run_local(self, cmd: str, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, float]:
        start = time.perf_counter()
        proc = sp.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **(env or {})},
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        log_file = log_path("local")
        with log_file.open("w", encoding="utf-8", errors="ignore") as f:
            try:
                for line in proc.stdout:  # type: ignore
                    sys.stdout.write(line)
                    f.write(line)
            finally:
                proc.wait()
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            raise ShellError(cmd, proc.returncode, where="local")
        return proc.returncode, elapsed

    def run_remote(self, ssh: SSHConfig, cmd: str) -> Tuple[int, float, str]:
        start = time.perf_counter()
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=ssh.host,
            port=ssh.port,
            username=ssh.user,
            key_filename=ssh.key_path,
        )
        buf = []
        try:
            chan = client.get_transport().open_session() # type: ignore
            chan.get_pty()
            chan.exec_command(cmd)
            while True:
                if chan.recv_ready():
                    data = chan.recv(4096).decode(errors="ignore")
                    buf.append(data)
                if chan.recv_stderr_ready():
                    data = chan.recv_stderr(4096).decode(errors="ignore")
                    buf.append(data)
                if chan.exit_status_ready():
                    break
                time.sleep(0.02)
            rc = chan.recv_exit_status()
        finally:
            client.close()
        elapsed = time.perf_counter() - start
        output = "".join(buf)
        if rc != 0:
            raise ShellError(cmd, rc, where=f"ssh:{ssh.host}")
        return rc, elapsed, output


# ------------------------------
# Core operations
# ------------------------------
class Telerun:
    def __init__(self, cfg: TelerunConfig):
        self.cfg = cfg
        self.streamer = Streamer()

    # --- helpers ---
    def _slurm_prefix(self) -> Optional[str]:
        s = self.cfg.slurm
        if not s.enabled:
            return None
        parts: List[str] = ["srun"]
        if s.partition:
            parts += [f"--partition={shlex.quote(s.partition)}"]
        if s.time:
            parts += [f"--time={shlex.quote(s.time)}"]
        if s.nodes:
            parts += [f"--nodes={s.nodes}"]
        if s.ntasks:
            parts += [f"--ntasks={s.ntasks}"]
        if s.cpus_per_task:
            parts += [f"--cpus-per-task={s.cpus_per_task}"]
        if s.gpus:
            parts += [f"--gpus={s.gpus}"]
        if s.mem:
            parts += [f"--mem={shlex.quote(s.mem)}"]
        if s.account:
            parts += [f"--account={shlex.quote(s.account)}"]
        if s.qos:
            parts += [f"--qos={shlex.quote(s.qos)}"]
        if s.exclusive:
            parts += [f"--exclusive"]
        parts += s.extra_args
        return " ".join(parts)

    def rsync(self) -> None:
        ssh = self.cfg.ssh
        if not ssh.host or not ssh.remote_dir:
            print("SSH host/remote_dir not configured")
            return
        if not ssh.rsync:
            print("rsync disabled in config.")
            return
        src = Path(self.cfg.source_code_loc).resolve()
        if not src.exists():
            raise SystemExit(f"source_code_loc not found: {src}")
        dest = f"{ssh.user+'@' if ssh.user else ''}{ssh.host}:{os.path.join('/home/users/', ssh.user, ssh.remote_dir.rstrip('/'))}"  # user@host:/dir
        key_opt = f"-e \"ssh -p {ssh.port} -i {shlex.quote(ssh.key_path)}\"" if ssh.key_path else f"-e \"ssh -p {ssh.port}\""
        cmd = f"rsync -az --exclude-from '.rsyncignore' {key_opt} {shlex.quote(str(src))}/ {shlex.quote(dest)}/"
        if DEBUG: print_box(f"command: {cmd}") 
        print_box(f"rsync -> {dest}")
        self.streamer.run_local(cmd)

    # --- build ---
    def build(self) -> None:
        b = self.cfg.build
        env = self.cfg.env
        cwd = Path(self.cfg.source_code_loc)
        cmd: Optional[str] = None
        if b.build_script:
            cmd = f"bash {shlex.quote(b.build_script)}"
        elif b.compile_commands:
            cmd = " && ".join(b.compile_commands)
        else:
            print("No build steps configured; skipping build.")
            return
        ssh = self.cfg.ssh
        if not ssh.host or not ssh.remote_dir:
            raise SystemExit("Remote build requested but ssh.host/remote_dir not set.")
        slurm_prefix = self._slurm_prefix()
        remote_cmd = f"cd {shlex.quote(ssh.remote_dir)} && {cmd}"
        if slurm_prefix:
            remote_cmd = f"cd {shlex.quote(ssh.remote_dir)} && {slurm_prefix} bash -lc {shlex.quote(cmd)}"
        print_box(f"REMOTE build on {ssh.host}{' via SLURM' if slurm_prefix else ''}")
        _, _, output = self.streamer.run_remote(ssh, remote_cmd)
        print(output)

    # --- run ---
    def _compose_run_cmd(self, which: str, test: TestCase, perf: bool) -> Tuple[str, Optional[bytes]]:
        b = self.cfg.build
        stdin_bytes: Optional[bytes] = None
        bin_path: Optional[str] = None
        if which == "implementation" and b.binary:
            bin_path = b.binary
        elif which == "evaluation" and b.eval_binary:
            bin_path = b.eval_binary
        elif self.cfg.run_command and which == "implementation":
            bin_path = self.cfg.run_command
        else:
            raise SystemExit(f"No run command/binary configured for '{which}'.")

        argv = [bin_path] + test.args
        if test.stdin is not None:
            stdin_bytes = test.stdin.encode()
        elif test.input_file:
            stdin_bytes = Path(test.input_file).read_bytes()

        warmup = 3
        runs = 10
        cmd = " ".join(shlex.quote(a) for a in argv)
        cmd = (
            (f"source {shlex.quote(self.cfg.module_script)} && " if self.cfg.module_script else "")
            + (
                f"./bin/hyperfine -N --warmup {warmup} --runs {runs} "
                f"--export-json result.json -- '{cmd}' > /dev/null"
                if perf
                else f"{cmd}"
            )
            + (f" && cat result.json" if perf else "")
        )
        ssh = self.cfg.ssh
        if not ssh.host or not ssh.remote_dir:
            raise SystemExit("Remote run requested but ssh.host/remote_dir not set.")
        slurm_prefix = self._slurm_prefix()
        if slurm_prefix:
            cmd = f"cd {shlex.quote(ssh.remote_dir)} && {slurm_prefix} bash -lc {shlex.quote(cmd)}"
        else:
            cmd = f"cd {shlex.quote(ssh.remote_dir)} && {cmd}"
        return cmd, stdin_bytes

    def run_once(self, which: str, test: TestCase, perf: bool = False) -> Tuple[str, float]:
        cmd, stdin_bytes = self._compose_run_cmd(which, test, perf)
        print_box(f"{'RUN' if not perf else 'PERF'} {which} :: {test.name}")
        start = time.perf_counter()
        # Remote streaming does not handle stdin easily here; simple fallback via echo/heredoc
        if stdin_bytes is not None:
            tmp_name = f".telerun_stdin_{int(start)}.txt"
            Path(tmp_name).write_bytes(stdin_bytes)
            try:
                # push via scp
                ssh = self.cfg.ssh
                if not ssh.host:
                    raise SystemExit("ssh.host missing")
                user = f"{ssh.user+'@' if ssh.user else ''}{ssh.host}"
                port = ssh.port
                scp = ["scp", "-P", str(port)]
                if ssh.key_path:
                    scp += ["-i", ssh.key_path]
                scp += [tmp_name, f"{user}:{ssh.remote_dir}/{tmp_name}"]
                sp.run(scp, check=True)
                cmd = f"cat {shlex.quote(tmp_name)} | {cmd}; rm -f {shlex.quote(tmp_name)}"
            finally:
                Path(tmp_name).unlink(missing_ok=True)
        rc, elapsed, output = self.streamer.run_remote(self.cfg.ssh, cmd)
        return output, elapsed

    # --- evaluation ---
    def _expected_text(self, t: TestCase) -> Optional[str]:
        if t.expected_output is not None:
            return t.expected_output
        if t.expected_file:
            return Path(t.expected_file).read_text(encoding="utf-8", errors="ignore")
        return None

    def eval_test(self, t: TestCase) -> Dict[str, Any]:
        impl_out, _ = self.run_once("implementation", t)
        expected = self._expected_text(t)
        correctness = None
        details = ""
        if expected is None and self.cfg.build.eval_binary:
            eval_out, _ = self.run_once("evaluation", t)
            expected = eval_out
        if expected is None:
            correctness = True  # no oracle provided
            details = "no expected/evaluation provided; skipping correctness"
        else:
            # Simple diff style compare
            if self.cfg.thresholds.correctness == "diff":
                correctness = (normalize(impl_out) == normalize(expected))

                if not correctness:
                    details = "output differs from expected"
            else:
                # Custom command: provide two temp files {expected} {actual}
                with tempfile.TemporaryDirectory() as td:
                    p_exp = Path(td) / "exp.txt"
                    p_act = Path(td) / "act.txt"
                    p_exp.write_text(expected)
                    p_act.write_text(impl_out)
                    cmd = self.cfg.thresholds.correctness.format(expected=str(p_exp), actual=str(p_act))
                    try:
                        sp.run(cmd, shell=True, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
                        correctness = True
                    except sp.CalledProcessError:
                        correctness = False
                        details = f"checker failed: {cmd}"
        benchmark_result, _  = self.run_once("implementation", t, perf=True)
        result_json_data = json.loads(benchmark_result)
        impl_time = result_json_data["results"][0]["mean"]
        perf_ok = True
        perf_limit = self.cfg.thresholds.perf_ms
        if perf_limit is not None:
            perf_ok = (impl_time * 1000.0) <= perf_limit
        return {
            "test": t.name,
            "correct": bool(correctness),
            "perf_ok": bool(perf_ok),
            "runtime_ms": round(impl_time * 1000.0, 6),
            "details": details,
        }
    
    def run_pipeline(self, name: str) -> None:
        if name not in self.cfg.pipelines:
            raise SystemExit(f"Unknown pipeline: {name}")
        pl = self.cfg.pipelines[name]
        print_box(f"PIPELINE :: {pl.name}")

        for i, step in enumerate(pl.steps, 1):
            print_box(f"[{i}/{len(pl.steps)}] {step.name} ({step.kind})")
            try:
                # merge step env into cfg for this step
                saved_env = dict(self.cfg.env)
                self.cfg.env.update(step.env or {})

                if step.kind == "sync":
                    self.rsync()

                elif step.kind == "build":
                    self.build()

                elif step.kind == "run":
                    # choose test or default 1st
                    t = next((t for t in self.cfg.tests if t.name == step.test), None)
                    if t is None:
                        if step.test: raise SystemExit(f"Unknown test: {step.test}")
                        if not self.cfg.tests: raise SystemExit("No tests configured")
                        t = self.cfg.tests[0]
                    out, secs = self.run_once("implementation", t)
                    print_box("OUTPUT")
                    print(out)

                elif step.kind == "eval":
                    for t in self.cfg.tests:
                        res = self.eval_test(t)
                        print(f"{t.name}: correct={res['correct']} perf_ok={res['perf_ok']} time={res['runtime_ms']} ms")
                        if res["correct"] and step.submit and t.leaderboard:
                            try:
                                res = submit_result(
                                    f"{CLOUDFARE_URL}/submit",
                                    lab=self.cfg.lab_name,
                                    user=self.cfg.pokemon,
                                    runtime_ms=res['runtime_ms'],
                                )
                                print_box("SUBMIT SUCCESS")
                            except KVSubmitError as e:
                                print("Submit failed:", e)
                else:
                    raise SystemExit(f"Unsupported step kind: {step.kind}")

            except Exception as e:
                print(f"[pipeline] step failed: {e}")
                if not step.continue_on_error:
                    raise
            finally:
                self.cfg.env = saved_env  # restore

def load_cfg(path: Path) -> TelerunConfig:
    data = read_yaml(path)
    return TelerunConfig.from_dict(data)

def cmd_sync(args: argparse.Namespace) -> None:
    cfg = load_cfg(Path(args.config))
    tr = Telerun(cfg)
    tr.rsync()


def cmd_build(args: argparse.Namespace) -> None:
    cfg = load_cfg(Path(args.config))
    tr = Telerun(cfg)
    tr.build()


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_cfg(Path(args.config))
    tr = Telerun(cfg)
    # pick the named test or the first
    if args.test:
        ts = [t for t in cfg.tests if t.name == args.test]
        if not ts:
            raise SystemExit(f"No such test: {args.test}")
        test = ts[0]
    else:
        if not cfg.tests:
            raise SystemExit("No tests configured.")
        test = cfg.tests[0]
    output, elapsed = tr.run_once("implementation", test)
    print_box("RESULT")
    print(output)


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = load_cfg(Path(args.config))
    tr = Telerun(cfg)
    results = []
    for t in cfg.tests:
        try:
            res = tr.eval_test(t)
        except ShellError as e:
            res = {"test": t.name, "correct": False, "perf_ok": False, "runtime_ms": None, "details": str(e)}
        results.append(res)
        print_box(f"{t.name}: {'OK' if res['correct'] else 'WRONG'} | {'FAST' if res['perf_ok'] else 'SLOW'} | {res['runtime_ms']} ms")
        if res["details"]:
            print(res["details"])
        print()
        if res['correct'] and args.submit and t.leaderboard:
            try:
                res = submit_result(
                    f"{CLOUDFARE_URL}/submit",
                    lab=cfg.lab_name,
                    user=cfg.pokemon,
                    runtime_ms=res['runtime_ms'],
                )
                print_box("SUBMIT SUCCESS")
            except KVSubmitError as e:
                print("Submit failed:", e)
    # Summary
    ok = sum(1 for r in results if r["correct"]) 
    fast = sum(1 for r in results if r["perf_ok"]) 
    print_box("SUMMARY")
    print(json.dumps(results, indent=2))
    print(f"\nCorrect: {ok}/{len(results)} | Perf OK: {fast}/{len(results)}")

def cmd_pipe(args: argparse.Namespace) -> None:
    cfg = load_cfg(Path(args.config))
    tr = Telerun(cfg)
    tr.run_pipeline(args.name)

def cmd_leaderboard(lab: str, limit: int = 100) -> None:
    """
    Leaderboard
    """
    lab_encoded = urllib.parse.quote(lab, safe="")
    url = f"{CLOUDFARE_URL}/lab?lab={lab_encoded}&limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} while fetching leaderboard: {e.reason}")
        return
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}")
        return
    except json.JSONDecodeError:
        print("Invalid JSON response from worker")
        return
    

    results: List[Dict[str, Any]] = data.get("telerun") or []

    if not results:
        print(f"No results found for lab '{lab}'.")
        return

    # Sort ascending by runtime_ms
    results.sort(key=lambda r: r.get("runtime_ms", float("inf")))

    # Print nicely formatted table
    print("=" * 60)
    print(f"🏆 Leaderboard for lab: {lab}")
    print("=" * 60)
    print(f"{'Rank':<5} {'User':<15} {'Runtime (ms)':>15} {'Timestamp':>20}")
    print("-" * 60)
    for i, r in enumerate(results, start=1):
        user = r.get("user", "?")
        runtime = r.get("runtime_ms", 0)
        ts = _pretty_ts(r.get("ts", ""))
        print(f"{i:<5} {user:<15} {runtime:>15.3f} {ts:>20}")
    print("=" * 60)

# ------------------------------
# Entry point
# ------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="telerun", description="Build, run, and evaluate labs locally or via SSH.")
    p.add_argument("-c", "--config", default="telerun.yml", help="Path to YAML/JSON config (default: telerun.yml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_sync = sub.add_parser("sync", help="rsync source_code_loc to remote host")
    sp_sync.set_defaults(func=cmd_sync)

    sp_build = sub.add_parser("build", help="Run build steps")
    sp_build.set_defaults(func=cmd_build)

    sp_run = sub.add_parser("run", help="Run the implemented binary for a single test (default: first)")
    sp_run.add_argument("--test", help="Test name to run")
    sp_run.set_defaults(func=cmd_run)

    sp_eval = sub.add_parser("eval", help="Evaluate all tests: correctness + performance")
    sp_eval.add_argument("--submit", action="store_true", help="Submit benchmark result to the leaderboard")
    sp_eval.set_defaults(func=cmd_eval)

    sp_pipe = sub.add_parser("pipe", help="Run a named pipeline from the config")
    sp_pipe.add_argument("name", help="Pipeline name. e.g. benchmark")
    sp_pipe.set_defaults(func=cmd_pipe)

    sp_leader = sub.add_parser("leaderboard", help="Show leaderboard for a given lab")
    sp_leader.add_argument("lab", help="Lab name to query")
    sp_leader.set_defaults(func=lambda args: cmd_leaderboard(args.lab))


    args = p.parse_args(argv)
    try:
        args.func(args)
    except ShellError as e:
        print(str(e), file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
