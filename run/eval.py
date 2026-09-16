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
        [retriever_python, "-u", "search_r1/search/retrieval_server.py",
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


@hydra.main(config_path="../verl/trainer/config", config_name="eval", version_base=None)
def main(cfg: DictConfig):
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
