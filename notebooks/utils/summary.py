from typing import Dict
import os
import shutil
import json
from time import time
from collections import defaultdict, OrderedDict

from torch.utils.tensorboard.writer import SummaryWriter as _SummaryWriter


Scalars = Dict[str, Dict[str, Dict[int, float]]]

SCALARS = "scalars"
JSON_EXT = ".json"


class SummaryWriter:
    def __init__(self, log_dir: str, max_queue: int = 100, flush_secs: int = 120):
        """Creates a `SummaryWriter` that will write out events and summaries
        to the event file.

        Args:
            log_dir (string): Save directory location.
            max_queue (int): Size of the queue for pending events and
              summaries before one of the 'add' calls forces a flush to disk.
              Default is 100 items.
            flush_secs (int): How often, in seconds, to flush the
              pending events and summaries to disk. Default is every two minutes.
        """
        self.log_dir = log_dir
        self.max_queue = max_queue
        self.flush_secs = flush_secs

        self.scalars_path = os.path.join(log_dir, SCALARS + JSON_EXT)
        self.steps = 0
        self.timer = time()

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        self.scalars: Scalars = defaultdict(lambda: defaultdict(OrderedDict))

    def _auto_flush(self):
        self.steps += 1
        if (self.steps > self.max_queue) or (time() - self.timer > self.flush_secs):
            self.flush()
            self.steps = 0
            self.timer = time()

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        global_step: int,
    ) -> None:
        main_dict = self.scalars[main_tag]

        for tag, scalar in tag_scalar_dict.items():
            main_dict[tag][global_step] = scalar

        self._auto_flush()

    def flush(self):
        """Flushes the event file to disk.
        Call this method to make sure that all pending events have been written to
        disk.
        """
        with open(self.scalars_path, "w") as fn:
            json.dump(self.scalars, fn)

    def to_tensorboard(self):
        "Save the content as tensorboard events"
        path = os.path.join(self.log_dir, "tb")

        if os.path.isdir(path):
            shutil.rmtree(path)

        with _SummaryWriter(path, max_queue=10000) as writer:

            for main_tag, df in self.scalars.items():
                for tag, ds in df.items():
                    for step, scalar in ds.items():
                        writer.add_scalars(main_tag, {tag: scalar}, step, step)

        self.tb_path = path

    def close(self):
        self.flush()
        self.to_tensorboard()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def launch(self):
        try:
            os.system("tensorboard --logdir=" + self.tb_path)
        except KeyboardInterrupt:
            print("Closing session")
