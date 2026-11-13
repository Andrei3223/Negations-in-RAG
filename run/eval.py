import socket
import subprocess
import signal
import time
import sys

import hydra
import requests
from omegaconf import DictConfig, OmegaConf

from verl.trainer.main_ppo import main as ppo_main


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server(rs, port: int) -> subprocess.Popen:
    retriever_python = "/home/16207041/miniconda3/envs/retriever/bin/python"
    return subprocess.Popen(
        [retriever_python, "-u", "subsearch/search/retrieval_server.py",
         "--index_path",      rs.index_path,
         "--corpus_path",     rs.corpus_path,
         "--topk",            str(rs.get("topk", 3)),
         "--retriever_name",  rs.name,
         "--retriever_model", rs.model_path,
         "--port",            str(port)],
    )


def _wait_for_server(port: int, proc: subprocess.Popen, timeout: int = 1800):
    url = f"http://127.0.0.1:{port}/retrieve"
    for i in range(timeout // 5):
        try:
            requests.post(url, json={"queries": ["test"], "topk": 1}, timeout=30)
            print(f"Retrieval server ready (~{i * 5}s).")
            return
        except Exception:
            pass
        if proc.poll() is not None:
            sys.exit("ERROR: retrieval server process died")
        time.sleep(5)
    sys.exit("ERROR: retrieval server did not start in time")


def _read_running_url(rs) -> str:
    """Endpoint from rs.url, else from the file retrieval.job publishes; "" if none yet."""
    url = rs.get("url")
    if not url:
        try:
            with open(rs.get("url_file")) as f:
                url = f.read().strip()
        except FileNotFoundError:
            url = ""
    if not url:
        return ""
    if not url.startswith("http"):
        url = f"http://{url}"
    url = url.rstrip("/")
    if url.endswith("/retrieve"):
        url = url[: -len("/retrieve")]
    return url


def _resolve_running_url(rs) -> str:
    """Endpoint of an already-running retriever, waiting up to rs.wait_minutes for it.

    url_file is re-read on every attempt: until a freshly started retrieval.job has
    loaded its index (15-30 min for wiki-18) the file still names the previous,
    dead server, and the new server only overwrites it once its port is bound.
    """
    deadline = time.time() + 60 * float(rs.get("wait_minutes", 0))
    attempt, last_err, url = 0, "no endpoint published yet", ""
    while True:
        attempt += 1
        url = _read_running_url(rs)
        if url:
            try:
                requests.post(f"{url}/retrieve", json={"queries": ["test"], "topk": 1},
                              timeout=30).raise_for_status()
                if attempt > 1:
                    print(f"Running retriever at {url} answered after {attempt} attempts.")
                return url
            except Exception as e:
                last_err = f"{url} did not answer ({e})"
        if time.time() >= deadline:
            sys.exit(f"ERROR: running retriever: {last_err}. Is retrieval.job up? "
                     f"(endpoint file: {rs.get('url_file')})")
        if attempt == 1:
            print(f"Waiting up to {rs.get('wait_minutes')} min for the running retriever ({last_err})...")
        time.sleep(30)


@hydra.main(config_path="../verl/trainer/config", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    # Golden modes (eval/retriever=golden|golden_hn) read docs from the parquet:
    # no server to start or probe, and golden groups define no retriever_server.
    mode = cfg.retriever.get("mode", "live")
    if mode != "live":
        print(f"Dataset: {cfg.dataset_id}  Model: {cfg.model_id}  "
              f"Retriever: {mode} (docs from the parquet, no server)")
        ppo_main.__wrapped__(cfg)
        return

    if cfg.retriever_server.get("use_running", False):
        url = _resolve_running_url(cfg.retriever_server)
        print(f"Dataset: {cfg.dataset_id}  Model: {cfg.model_id}  Running retriever: {url}")
        OmegaConf.update(cfg, "retriever.url", f"{url}/retrieve", merge=False)
        ppo_main.__wrapped__(cfg)
        return

    port = _free_port()
    print(f"Dataset: {cfg.dataset_id}  Model: {cfg.model_id}  Port: {port}")

    server = _start_server(cfg.retriever_server, port)

    def _cleanup(*_):
        print(f"Stopping retrieval server (PID {server.pid})")
        server.terminate()

    signal.signal(signal.SIGTERM, _cleanup)

    try:
        _wait_for_server(port, server)
        OmegaConf.update(cfg, "retriever.url", f"http://127.0.0.1:{port}/retrieve", merge=False)
        ppo_main.__wrapped__(cfg)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
