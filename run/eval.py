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


def _resolve_running_url(rs) -> str:
    """Endpoint of an already-running retriever, from rs.url or the file retrieval.job publishes."""
    url = rs.get("url")
    if not url:
        with open(rs.url_file) as f:
            url = f.read().strip()
    if not url:
        sys.exit(f"ERROR: no retriever endpoint in {rs.url_file}")
    if not url.startswith("http"):
        url = f"http://{url}"
    url = url.rstrip("/")
    if url.endswith("/retrieve"):
        url = url[: -len("/retrieve")]
    try:
        requests.post(f"{url}/retrieve", json={"queries": ["test"], "topk": 1}, timeout=30).raise_for_status()
    except Exception as e:
        sys.exit(f"ERROR: running retriever at {url} did not answer ({e}). Is retrieval.job still up?")
    return url


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
