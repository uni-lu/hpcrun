# Telerun — easy build, run, benchmark on HPC

`telerun.py` is a small, portable tool to build, run, and evaluate labs on a remote HPC cluster via SSH/SLURM. It streams logs live, checks correctness, and benchmarks runs using `hyperfine`.

```
├── bin/
│   └── hyperfine
├── hello_world
├── hello_world.cpp
├── README.md
├── scripts/
│   ├── build.sh
│   └── module.sh
├── telerun.py
├── telerun.yml
├── temp.txt
```

## 1) Requirements

* Python 3.9+ with:
  * `pyyaml`, `paramiko`, `dotenv`
* (Remote builds/runs) SSH access to your cluster

Install Python deps (locally or in a venv):

```bash
pip install PyYAML paramiko python-dotenv
```

> Tip: you can also use `uv` to manage the environment (`uv add pyyaml paramiko python-dotenv`).

## 2) Configuration

First of all, you need a `.env`, below is an example of mine, replace `TELERUN_SSH_USER` and `TELERUN_SSH_KEY_PATH` based on your profile:

```dotenv
TELERUN_SSH_HOST=access-aion.uni.lu
TELERUN_SSH_USER=whuang
TELERUN_SSH_PORT=8022
TELERUN_SSH_KEY_PATH=~/.ssh/id_ed25519
TELERUN_SSH_RSYNC=true
```

Then you need to configure `scripts/build.sh` and `script/module.sh` to manage your build process. For example in order to benchmark a `hello_world.cpp`, we have:

```sh
# build.sh
# For build process, we load the module then compile
ml compiler/GCC
g++ -O3 -std=c++20 -o hello_world hello_world.cpp
```
```sh
# module.sh
# This script will be called before each run.
ml compiler/GCC
```

Here is a example `yml` configuration of `C++ Hello World` lab:

```yaml
lab_name: "C++ Hello World Lab"
source_code_loc: "."

build:
  build_script: scripts/build.sh
  eval_binary: ./hello_world_eval     # optional: teacher/oracle; omit if none

module_script: scripts/module.sh
run_command: ./hello_world            # used when build.binary is unset

thresholds:
  correctness: diff                   # string compare after normalization
  perf_ms: 1000                       # pass if mean runtime <= this (ms)

slurm:
  enabled: true
  partition: batch
  time: "00:10:00"
  nodes: 1
  mem: 0
  exclusive: true

tests:
  - name: small
    args: ["test"]
    expected_output: "hello world1\n"
  - name: medium
    args: ["--size", "500000"]
    # expected_file: "./temp.txt"
    leaderboard: true                   # specify this test case as leaderboard submission test

pipelines:
  benchmark:
    - name: sync to cluster
      kind: sync
      remote: false
      continue_on_error: false
    - name: remote build
      kind: build
      remote: true
    - name: run small
      kind: run
      remote: true
      test: small
    - name: evaluate all
      kind: eval
      remote: true
      submit: true                       # submit the performance result after evalution phase.
```

## 3) Commands

### Sync

```bash
./telerun.py sync --remote
```

### Build

```bash
./telerun.py build --remote
```

### Run

```bash
./telerun.py run --remote
```

### Evaluate all tests (correctness + performance)

```bash
./telerun.py eval --remote
```

The `eval` step:

* runs the “implementation” (your binary) to collect output,
* compares it to `expected_output`/`expected_file` (or to `eval_binary` output if provided),
* then runs a **perf pass** using `hyperfine` and checks `mean <= perf_ms`.


## 4) Pipelines

Use the YAML `pipelines:` to compose steps. The example `benchmark` pipeline will:

1. Sync to the cluster
2. Build via SLURM
3. Run you program
4. Evaluate all tests

Run it with:

```bash
./telerun.py pipe benchmark
```

You can add more pipelines, e.g. a quick run:

```yaml
pipelines:
  quick:
    - name: build
      kind: build
    - name: run small
      kind: run
      test: small
```

## 5) Leaderboard

In order to submit you solution to leaderboard, add `--submit` when submit:
```sh
./telerun.py eval --remote --submit
```

You can check the result you've submitted afterward, by using this command:
```sh
./telerun.py leaderboard <labname>
```
For example:

```sh
./telerun.py leaderboard "C++ Hello World Lab"
```
