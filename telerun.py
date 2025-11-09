#!/usr/bin/env python
"""
Telerun: a small, portable tool to build, run, and evaluate code
on local or hpc (SSH) machines.

High-level features (mirrors the diagram):
- Reads a YAML config that declares: lab name, source dir, build steps,
  compile/run commands, thresholds, and (optional) SSH/Slurm settings.
- Optional rsync to a remote host, then build + run there via SSH.
- Streams stdout/stderr live to the terminal; writes full logs to disk.
- Evaluates correctness (diff or custom command) and performance (runtime).
- Clean, defensive error handling with clear messages.

Usage
-----
$ telerun.py init                   # write an example YAML to ./telerun.yml
$ telerun.py build -c telerun.yml   # build code
$ telerun.py run   -c telerun.yml   # run a single test
$ telerun.py eval  -c telerun.yml   # run all tests, check correctness + perf
$ telerun.py sync  -c telerun.yml   # rsync to remote (if SSH configured)

The config format is documented below in ExampleConfig.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import signal
import subprocess as sp
import sys
import tarfile
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import yaml
import paramiko

LOG_DIR = Path(".telerun_logs")
LOG_DIR.mkdir(exist_ok=True)
DEBUG = True


# ------------------------------
# Utilities
# ------------------------------
class ShellError(RuntimeError):
    def __init__(self, cmd: str, returncode: int, where: str = "local"):
        super().__init__(f"Command failed ({where}): {cmd}\nreturn code: {returncode}")
        self.cmd = cmd
        self.returncode = returncode
        self.where = where


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


def print_box(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")

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


@dataclasses.dataclass
class TelerunConfig:
    lab_name: str = ""
    source_code_loc: str = "."
    run_command: Optional[str] = None  # fallback if no binary
    build: BuildConfig = dataclasses.field(default_factory=BuildConfig)
    ssh: SSHConfig = dataclasses.field(default_factory=SSHConfig)
    slurm: SlurmConfig = dataclasses.field(default_factory=SlurmConfig)
    thresholds: Thresholds = dataclasses.field(default_factory=Thresholds)
    tests: List[TestCase] = dataclasses.field(default_factory=list)
    env: Dict[str, str] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TelerunConfig":
        build = BuildConfig(**(d.get("build", {}) or {}))
        ssh = SSHConfig.from_env()
        thr = Thresholds(**(d.get("thresholds", {}) or {}))
        tests = [TestCase(**x) for x in d.get("tests", [])]
        return TelerunConfig(
            lab_name=d.get("lab_name", ""),
            source_code_loc=d.get("source_code_loc", "."),
            run_command=d.get("run_command"),
            build=build,
            ssh=ssh,
            slurm=SlurmConfig(**(d.get("slurm", {}) or {})),
            thresholds=thr,
            tests=tests,
            env=d.get("env", {}),
        )


# ------------------------------
# Local & remote command execution
# ------------------------------
class Streamer:
    """Run a command (local or remote) and stream stdout/stderr live."""

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
    def run_remote(self, ssh: SSHConfig, cmd: str) -> Tuple[int, float]:
        start = time.perf_counter()
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=ssh.host,
            port=ssh.port,
            username=ssh.user,
            key_filename=ssh.key_path,
        )
        try:
            chan = client.get_transport().open_session() # type: ignore
            chan.get_pty()
            chan.exec_command(cmd)
            log_file = log_path("remote")
            with log_file.open("w", encoding="utf-8", errors="ignore") as f:
                while True:
                    if chan.recv_ready():
                        data = chan.recv(4096).decode(errors="ignore")
                        sys.stdout.write(data)
                        f.write(data)
                    if chan.recv_stderr_ready():
                        data = chan.recv_stderr(4096).decode(errors="ignore")
                        sys.stdout.write(data)
                        f.write(data)
                    if chan.exit_status_ready():
                        break
                    time.sleep(0.02)
            rc = chan.recv_exit_status()
        finally:
            client.close()
        elapsed = time.perf_counter() - start
        if rc != 0:
            raise ShellError(cmd, rc, where=f"ssh:{ssh.host}")
        return rc, elapsed


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
        dest = f"{ssh.user+'@' if ssh.user else ''}{ssh.host}:{os.path.join("/home/users/", ssh.user, ssh.remote_dir.rstrip('/'))}"  # user@host:/dir
        key_opt = f"-e \"ssh -p {ssh.port} -i {shlex.quote(ssh.key_path)}\"" if ssh.key_path else f"-e \"ssh -p {ssh.port}\""
        cmd = f"rsync -az --exclude-from '.rsyncignore' {key_opt} {shlex.quote(str(src))}/ {shlex.quote(dest)}/"
        if DEBUG: print_box(f"command: {cmd}") 
        print_box(f"rsync -> {dest}")
        self.streamer.run_local(cmd)

    # --- build ---
    def build(self, remote: bool = False) -> None:
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
        if remote:
            ssh = self.cfg.ssh
            if not ssh.host or not ssh.remote_dir:
                raise SystemExit("Remote build requested but ssh.host/remote_dir not set.")
            slurm_prefix = self._slurm_prefix()
            remote_cmd = f"cd {shlex.quote(ssh.remote_dir)} && {cmd}"
            if slurm_prefix:
                remote_cmd = f"cd {shlex.quote(ssh.remote_dir)} && {slurm_prefix} bash -lc {shlex.quote(cmd)}"
            print_box(f"REMOTE build on {ssh.host}{' via SLURM' if slurm_prefix else ''}")
            self.streamer.run_remote(ssh, remote_cmd)
        else:
            print_box("LOCAL build")
            self.streamer.run_local(cmd, cwd=cwd, env=env)

    # --- run ---
    def _compose_run_cmd(self, which: str, test: TestCase, remote: bool) -> Tuple[str, Optional[bytes]]:
        b = self.cfg.build
        stdin_bytes: Optional[bytes] = None
        bin_path: Optional[str] = None
        if DEBUG:
            print(self.cfg.run_command, which)
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

        cmd = " ".join(shlex.quote(a) for a in argv)
        cmd = f"source {shlex.quote("scripts/module.sh")}"+ " && time -p " + cmd
        if DEBUG: print(argv)
        if remote:
            ssh = self.cfg.ssh
            if not ssh.host or not ssh.remote_dir:
                raise SystemExit("Remote run requested but ssh.host/remote_dir not set.")
            slurm_prefix = self._slurm_prefix()
            if slurm_prefix:
                cmd = f"cd {shlex.quote(ssh.remote_dir)} && {slurm_prefix} bash -lc {shlex.quote(cmd)}"
            else:
                cmd = f"cd {shlex.quote(ssh.remote_dir)} && {cmd}"
        return cmd, stdin_bytes

    def run_once(self, which: str, test: TestCase, remote: bool = False) -> Tuple[str, float]:
        cmd, stdin_bytes = self._compose_run_cmd(which, test, remote)
        print_box(f"RUN {which} :: {test.name}")
        start = time.perf_counter()
        if remote:
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
            rc, elapsed = self.streamer.run_remote(self.cfg.ssh, cmd)
        else:
            # Local with streaming + optional stdin
            env = {**os.environ, **(self.cfg.env or {})}
            proc = sp.Popen(
                cmd,
                shell=True,
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )
            out_lines: List[str] = []
            log_file = log_path(f"run-{which}")
            with log_file.open("w", encoding="utf-8", errors="ignore") as f:
                try:
                    if stdin_bytes is not None:
                        proc.stdin.write(stdin_bytes.decode())  # type: ignore
                        proc.stdin.close()  # type: ignore
                    for line in proc.stdout:  # type: ignore
                        out_lines.append(line)
                        sys.stdout.write(line)
                        f.write(line)
                finally:
                    proc.wait()
            elapsed = time.perf_counter() - start
            if proc.returncode != 0:
                raise ShellError(cmd, proc.returncode)
            return "".join(out_lines), elapsed
        # If we ran remote, we only have timing; correctness evaluation will re-run via implementation/evaluation collection.
        return "", elapsed

    # --- evaluation ---
    def _expected_text(self, t: TestCase) -> Optional[str]:
        if t.expected_output is not None:
            return t.expected_output
        if t.expected_file:
            return Path(t.expected_file).read_text(encoding="utf-8", errors="ignore")
        return None

    def eval_test(self, t: TestCase, remote: bool = False) -> Dict[str, Any]:
        impl_out, impl_time = self.run_once("implementation", t, remote=remote)
        expected = self._expected_text(t)
        correctness = None
        details = ""
        if expected is None and self.cfg.build.eval_binary:
            eval_out, _ = self.run_once("evaluation", t, remote=remote)
            expected = eval_out
        if expected is None:
            correctness = True  # no oracle provided
            details = "no expected/evaluation provided; skipping correctness"
        else:
            # Simple diff style compare
            if self.cfg.thresholds.correctness == "diff":
                correctness = (eval_out == expected)
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
        perf_ok = True
        perf_limit = self.cfg.thresholds.perf_ms
        if perf_limit is not None:
            perf_ok = (impl_time * 1000.0) <= perf_limit
        return {
            "test": t.name,
            "correct": bool(correctness),
            "perf_ok": bool(perf_ok),
            "runtime_ms": round(impl_time * 1000.0, 2),
            "details": details,
        }


# ------------------------------
# CLI
# ------------------------------
EXAMPLE_CONFIG_YAML = textwrap.dedent(
    """
    # telerun.yaml — example configuration (Python or C/C++)
    lab_name: "Example Lab"
    source_code_loc: "."  # path where build + run happen

    build:
      # Either provide a build script (recommended)…
      # build_script: scripts/build.sh

      # …or an explicit list of compile commands to be joined with &&
      compile_commands:
        - make clean
        - make all



    thresholds:
      correctness: diff      # or e.g. "python3 compare.py {expected} {actual}"
      perf_ms: 2000          # fail if runtime exceeds 2s per test

    # Optional SSH config if you want to build/run remotely
    ssh:
      host: null             # e.g. hpc.mycluster.edu
      user: null
      port: 22
      key_path: null         # e.g. ~/.ssh/id_ed25519
      remote_dir: null       # e.g. /home/me/labs/example-lab
      rsync: true

    # Optional SLURM settings (used only when running remotely)
    slurm:
      enabled: false
      partition: null        # e.g. "short"
      time: "00:05:00"
      nodes: 1
      ntasks: 1
      cpus_per_task: 1
      gpus: null
      mem: "2G"
      account: null
      qos: null
      extra_args: []         # e.g. ["--exclusive"]

    # Environment variables to set during local runs/builds
    env:
      OMP_NUM_THREADS: "1"

    tests:
      - name: small
        args: ["--n", "1000"]
        stdin: null
        expected_output: "42 "   # or use expected_file: path/to/file

      - name: bigger
        args: ["--n", "200000"]
        input_file: null
        expected_output: "84"
    """
)

# A ready-to-use C++ lab example
EXAMPLE_CPP_YAML = textwrap.dedent(
    """
    lab_name: "C++ Sorting Lab"
    source_code_loc: "."

    build:
      compile_commands:
        - mkdir -p bin
        - g++ -O2 -std=c++17 -o bin/student src/student_sort.cpp
        - g++ -O2 -std=c++17 -o bin/teacher src/teacher_sort.cpp

    thresholds:
      correctness: diff
      perf_ms: 1000

    ssh:
      host: hpc.example.edu
      user: myuser
      port: 22
      key_path: ~/.ssh/id_ed25519
      remote_dir: /home/myuser/sorting-lab
      rsync: true

    slurm:
      enabled: true
      partition: short
      time: "00:02:00"
      ntasks: 1
      cpus_per_task: 1
      mem: "1G"

    tests:
      - name: small
        args: ["--size", "1000"]
        expected_output: "OK
      - name: medium
        args: ["--size", "500000"]
        expected_output: "OK


    thresholds:
      correctness: diff      # or e.g. "python3 compare.py {expected} {actual}"
      perf_ms: 2000          # fail if runtime exceeds 2s per test

    # Optional SSH config if you want to build/run remotely
    ssh:
      host: null             # e.g. hpc.mycluster.edu
      user: null
      port: 22
      key_path: null         # e.g. ~/.ssh/id_ed25519
      remote_dir: null       # e.g. /home/me/labs/example-lab
      rsync: true

    # Environment variables to set during local runs/builds
    env:
      OMP_NUM_THREADS: "1"

    tests:
      - name: small
        args: ["--n", "1000"]
        stdin: null
        expected_output: "42\n"   # or use expected_file: path/to/file

      - name: bigger
        args: ["--n", "200000"]
        input_file: null
        expected_output: "84\n"
    """
)


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
    tr.build(remote=args.remote)


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
    output, elapsed = tr.run_once("implementation", test, remote=args.remote)
    print_box("RESULT")
    print(output)
    print(f"\nRuntime: {round(elapsed*1000,2)} ms")


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = load_cfg(Path(args.config))
    tr = Telerun(cfg)
    results = []
    for t in cfg.tests:
        try:
            res = tr.eval_test(t, remote=args.remote)
        except ShellError as e:
            res = {"test": t.name, "correct": False, "perf_ok": False, "runtime_ms": None, "details": str(e)}
        results.append(res)
        print_box(f"{t.name}: {'OK' if res['correct'] else 'WRONG'} | {'FAST' if res['perf_ok'] else 'SLOW'} | {res['runtime_ms']} ms")
        if res["details"]:
            print(res["details"])
    # Summary
    ok = sum(1 for r in results if r["correct"]) 
    fast = sum(1 for r in results if r["perf_ok"]) 
    print_box("SUMMARY")
    print(json.dumps(results, indent=2))
    print(f"\nCorrect: {ok}/{len(results)} | Perf OK: {fast}/{len(results)}")


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
    sp_build.add_argument("--remote", action="store_true", help="Execute on remote host defined in config")
    sp_build.set_defaults(func=cmd_build)

    sp_run = sub.add_parser("run", help="Run the implemented binary for a single test (default: first)")
    sp_run.add_argument("--test", help="Test name to run")
    sp_run.add_argument("--remote", action="store_true", help="Execute on remote host defined in config")
    sp_run.set_defaults(func=cmd_run)

    sp_eval = sub.add_parser("eval", help="Evaluate all tests: correctness + performance")
    sp_eval.add_argument("--remote", action="store_true", help="Execute on remote host defined in config")
    sp_eval.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except ShellError as e:
        print(str(e), file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
