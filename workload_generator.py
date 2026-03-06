# #!/usr/bin/env python3
# """
# Workload generator for TorchServe inference.

# Responsibilities:
# - Generate Poisson arrivals
# - Send HTTP inference requests
# - Log completion timestamps

# Design notes:
# - CPU-only process
# - No energy measurement
# - No NVML usage
# """

# import argparse
# import json
# import queue
# import random
# import threading
# import time
# from typing import List, Optional
# from dataclasses import dataclass

# import requests
# from PIL import Image
# from torchvision.datasets import CIFAR10


# @dataclass
# class Completion:
#     timestamp: float


# def img_to_bytes(img: Image.Image) -> bytes:
#     import io
#     buf = io.BytesIO()
#     img.save(buf, format="JPEG")
#     return buf.getvalue()


# def wait_for_torchserve(url: str, timeout: float = 180.0):
#     start = time.time()
#     while True:
#         try:
#             r = requests.get(f"{url}/ping", timeout=1.0)
#             if r.status_code == 200:
#                 return
#         except Exception:
#             pass
#         if time.time() - start > timeout:
#             raise RuntimeError("TorchServe did not become ready")
#         time.sleep(0.5)


# def send_request(url: str, model_name: str, sample: bytes) -> Completion:
#     requests.post(
#         f"{url}/predictions/{model_name}",
#         data=sample,
#         timeout=30.0,
#     )
#     return Completion(time.perf_counter())


# def worker_loop(
#     url: str,
#     model_name: str,
#     samples: List[bytes],
#     task_queue: "queue.Queue[Optional[int]]",
#     completion_log,
# ):
#     while True:
#         token = task_queue.get()
#         try:
#             if token is None:
#                 return
#             sample = random.choice(samples)
#             c = send_request(url, model_name, sample)
#             completion_log.write(f"{c.timestamp}\n")
#             completion_log.flush()
#         finally:
#             task_queue.task_done()


# def poisson_scheduler(deadline: float, rps: int, q: queue.Queue):
#     if rps <= 0:
#         return
#     while time.perf_counter() < deadline:
#         time.sleep(random.expovariate(rps))
#         if time.perf_counter() < deadline:
#             q.put(1)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--url", required=True)
#     parser.add_argument("--model-name", required=True)
#     parser.add_argument("--phases-json", required=True)
#     parser.add_argument("--concurrency", type=int, default=16)
#     parser.add_argument("--log-file", required=True)
#     args = parser.parse_args()

#     wait_for_torchserve(args.url)

#     ds = CIFAR10(root="./data", train=False, download=True)
#     samples = [img_to_bytes(ds[i][0]) for i in range(100)]

#     q = queue.Queue()

#     with open(args.log_file, "w") as log:
#         for _ in range(args.concurrency):
#             threading.Thread(
#                 target=worker_loop,
#                 args=(args.url, args.model_name, samples, q, log),
#                 daemon=True,
#             ).start()

#         with open(args.phases_json) as f:
#             phases = json.load(f)

#         for phase in phases:
#             deadline = time.perf_counter() + float(phase["duration"])
#             poisson_scheduler(deadline, int(phase["rps"]), q)


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Workload generator for TorchServe inference.

Now logs:
- request SEND timestamps
- request COMPLETION timestamps
"""

import argparse
import json
import queue
import random
import threading
import time
from typing import Optional
from dataclasses import dataclass

import requests
from PIL import Image
from torchvision.datasets import CIFAR10


@dataclass
class Completion:
    timestamp: float


def img_to_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def wait_for_torchserve(url: str, timeout: float = 180.0):
    start = time.time()
    while True:
        try:
            r = requests.get(f"{url}/ping", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        if time.time() - start > timeout:
            raise RuntimeError("TorchServe did not become ready")
        time.sleep(0.5)


def send_request(url: str, model_name: str, sample: bytes) -> Completion:
    requests.post(
        f"{url}/predictions/{model_name}",
        data=sample,
        timeout=30.0,
    )
    return Completion(time.perf_counter())


def worker_loop(
    url: str,
    model_name: str,
    sample: bytes,
    task_queue: "queue.Queue[Optional[int]]",
    sent_log,
    completion_log,
):
    while True:
        token = task_queue.get()
        try:
            if token is None:
                return

            # SEND timestamp
            send_ts = time.perf_counter()
            sent_log.write(f"{send_ts}\n")
            sent_log.flush()

            # SEND request
            c = send_request(url, model_name, sample)

            # COMPLETION timestamp
            completion_log.write(f"{c.timestamp}\n")
            completion_log.flush()

        finally:
            task_queue.task_done()


def poisson_scheduler(deadline: float, rps: int, q: queue.Queue):
    if rps <= 0:
        return
    while time.perf_counter() < deadline:
        time.sleep(random.expovariate(rps))
        if time.perf_counter() < deadline:
            q.put(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--phases-json", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--completion-log", required=True)
    parser.add_argument("--sent-log", required=True)
    args = parser.parse_args()

    wait_for_torchserve(args.url)

    ds = CIFAR10(root="./data", train=False, download=True)
    fixed_sample = img_to_bytes(ds[0][0])

    q = queue.Queue()

    with open(args.completion_log, "w") as completion_log, \
         open(args.sent_log, "w") as sent_log:

        for _ in range(args.concurrency):
            threading.Thread(
                target=worker_loop,
                args=(args.url, args.model_name, fixed_sample, q, sent_log, completion_log),
                daemon=True,
            ).start()

        with open(args.phases_json) as f:
            phases = json.load(f)

        for phase in phases:
            deadline = time.perf_counter() + float(phase["duration"])
            poisson_scheduler(deadline, int(phase["rps"]), q)


if __name__ == "__main__":
    main()
