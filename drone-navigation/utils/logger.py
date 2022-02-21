from termcolor import colored
import collections
import datetime
import dateutil.tz
import json
import logging
import os
import os.path as osp
import sys
import tempfile
import subprocess
import shutil
from torch.utils.tensorboard import SummaryWriter
import uuid
import time

try:
    import mlflow
    from mlflow.entities import Metric
except ImportError:
    print("MLFlow not installed")
    mlflow = None

__all__ = ["setup", "scoped_setup"]


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = "%-8.6g" % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)),)
            )

        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        return s[:30] + "..." if len(s) > 33 else s

    def writeseq(self, seq):
        for arg in seq:
            self.file.write(arg)
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                v = v.tolist()
                kvs[k] = float(v)
                self.file.write(json.dumps(kvs) + "\n")
                self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = kvs.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TensorboardOutputFormat(KVWriter):
    def __init__(self, dirname):
        self._writer = SummaryWriter(dirname)
        self.step = 0

    def writekvs(self, kvs):
        for k, v in kvs.items():
            self._writer.add_scalar(k, v, self.step)
        self.step += 1

    def close(self):
        self._writer.close()


class MlflowOutputFormat(KVWriter):
    def __init__(self) -> None:
        if mlflow is not None:
            self._client = mlflow.tracking.MlflowClient()
            exp_id = os.environ.get("MLFLOW_EXPERIMENT_ID", None)
            if exp_id is None:
                exp_id = self._client.create_experiment(
                    "lgs_{}".format(str(uuid.uuid4()))
                )

            self._exp_id = exp_id
            run_id = os.environ.get("MLFLOW_RUN_ID", None)
            if run_id is None:
                self._run = self._client.create_run(self._exp_id)
            else:
                self._run = self._client.get_run(run_id)
        else:
            self._client = None
        self.step = 0

    def writekvs(self, kvs):
        if self._client is not None:
            timestamp = int(time.time() * 1000)
            metrics_arr = [Metric(key, value, timestamp, self.step or 0) for key, value in kvs.items()]            
            self._client.log_batch(run_id=self._run.info.run_id, metrics=metrics_arr, params=[], tags=[])
            
        self.step += 1

    def close(self):
        if self._client is not None:
            self._client.set_terminated(self._run.info.run_id)


class _MyFormatter(logging.Formatter):
    def format(self, record):
        date = colored("[%(asctime)s @%(filename)s:%(lineno)d]", "green")
        msg = "%(message)s"
        if record.levelno == logging.WARNING:
            fmt = date + " " + colored("WRN", "red", attrs=["blink"]) + " " + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = (
                date
                + " "
                + colored("ERR", "red", attrs=["blink", "underline"])
                + " "
                + msg
            )
        else:
            fmt = date + " " + msg
        if hasattr(self, "_style"):
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)


class Logger:
    CURRENT = None

    def __init__(self, dirname, output_formats):
        self.name2val = collections.defaultdict(float)
        self.name2cnt = collections.defaultdict(int)
        self.dirname = dirname
        self.output_formats = output_formats
        self._logger = logging.getLogger("lgs")
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)
        self._handler = logging.StreamHandler(sys.stdout)
        self._handler.setFormatter(_MyFormatter(datefmt="%m%d %H:%M:%S"))
        self._logger.addHandler(self._handler)

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        if val is None:
            self.name2val[key] = None
            return
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()

    def add_image(self, key, img):
        for fmt in self.output_formats:
            if isinstance(fmt, TensorboardOutputFormat):
                fmt._writer.add_image(key, img, fmt.step)

            if isinstance(fmt, MlflowOutputFormat):
                if fmt._client is not None:
                    fmt._client.log_image(
                        fmt._run.info.run_id,
                        img,
                        "{}_epoch={}.png".format(key, fmt.step),
                    )

    def add_figure(self, key, fig, copy=False):
        for fmt in self.output_formats:
            if isinstance(fmt, TensorboardOutputFormat):
                fmt._writer.add_figure(key, fig, fmt.step)
                if copy and self.dirname is not None:
                    fig.savefig(
                        osp.join(
                            self.dirname,
                            "figs",
                            "{}_epoch={}.png".format(key, fmt.step),
                        )
                    )

            if isinstance(fmt, MlflowOutputFormat):
                if fmt._client is not None:
                    fmt._client.log_figure(
                        fmt._run.info.run_id,
                        fig,
                        "{}_epoch={}.png".format(key, fmt.step),
                    )

    def add_text(self, key, txt):
        for fmt in self.output_formats:
            if isinstance(fmt, TensorboardOutputFormat):
                fmt._writer.add_text(key, txt, fmt.step)

    def add_hist(self, key, hist):
        for fmt in self.output_formats:
            if isinstance(fmt, TensorboardOutputFormat):
                fmt._writer.add_histogram(key, hist, fmt.step)

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    def get_dir(self):
        return self.dirname


def make_output_format(format, ev_dir):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress.json"))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress.csv"))
    elif format == "tensorboard":
        return TensorboardOutputFormat(osp.join(ev_dir, "tb"))
    elif format == "mlflow":
        return MlflowOutputFormat()
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


Logger.CURRENT = Logger(dirname=None, output_formats=[HumanOutputFormat(sys.stdout)])


def _get_time_str():
    return datetime.datetime.now().strftime("%m%d-%H%M%S")


def setup(
    dirname=None, format_strs=["stdout", "tensorboard", "csv", "mlflow"], action=None
):
    if dirname is None:
        dirname = os.getenv("LGDSN_LOGDIR")

    if dirname is None:
        dirname = osp.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("ceem-%Y-%m-%d-%H-%M-%S-%f"),
        )

    if os.path.isdir(dirname) and len(os.listdir(dirname)):
        if not action:
            warn(
                """\
Log directory {} exists! Please either backup/delete it, or use a new directory.""".format(
                    dirname
                )
            )
            warn(
                """\
If you're resuming from a previous run you can choose to keep it."""
            )
            info("Select Action: k (keep) / d (delete) / q (quit):")
        while not action:
            action = input().lower().strip()
        act = action
        if act == "b":
            backup_name = dirname + _get_time_str()
            shutil.move(dirname, backup_name)
            info(
                "Directory '{}' backuped to '{}'".format(dirname, backup_name)
            )  # noqa: F821
        elif act == "d":
            shutil.rmtree(dirname)
        elif act == "n":
            dirname = dirname + _get_time_str()
            info("Use a new log directory {}".format(dirname))  # noqa: F821
        elif act == "k":
            pass
        elif act == "q":
            raise OSError("Directory {} exits!".format(dirname))
        else:
            raise ValueError("Unknown action: {}".format(act))
    os.makedirs(dirname, exist_ok=True)
    os.makedirs(osp.join(dirname, "figs"), exist_ok=True)
    os.makedirs(osp.join(dirname, "ckpts"), exist_ok=True)
    output_formats = [make_output_format(f, dirname) for f in format_strs]
    Logger.CURRENT = Logger(dirname=dirname, output_formats=output_formats)
    hdl = logging.FileHandler(
        filename=osp.join(dirname, "log.log"), encoding="utf-8", mode="w"
    )
    hdl.setFormatter(_MyFormatter(datefmt="%m%d %H:%M:%S"))
    Logger.CURRENT._logger.removeHandler(Logger.CURRENT._handler)
    Logger.CURRENT._logger.addHandler(hdl)
    Logger.CURRENT._logger.info("Argv: " + " ".join(sys.argv))
    has_git = True
    timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime(
        "%Y_%m_%d_%H_%M_%S"
    )
    try:
        current_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        clean_state = (
            len(subprocess.check_output(["git", "status", "--porcelain"])) == 0
        )
    except subprocess.CalledProcessError as _:
        Logger.CURRENT._logger.warn("Warning: failed to execute git commands")
        has_git = False
    except FileNotFoundError as _:
        Logger.CURRENT._logger.warn("Warning: failed to execute git commands")
        has_git = False

    if has_git:
        if clean_state:
            Logger.CURRENT._logger.info("Commit: {}".format(current_commit))
        else:
            Logger.CURRENT._logger.info(
                "Commit: {}_dirty_{}".format(current_commit, timestamp)
            )


_LOGGING_METHOD = [
    "info",
    "warning",
    "error",
    "critical",
    "warn",
    "exception",
    "debug",
    "setLevel",
]

# export logger functions
for func in _LOGGING_METHOD:
    locals()[func] = getattr(Logger.CURRENT._logger, func)
    __all__.append(func)


def get_dir():
    return Logger.CURRENT.get_dir()


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    """
    Logger.CURRENT.logkv(key, val)


def logkv_mean(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    """
    Logger.CURRENT.logkv_mean(key, val)


def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    Logger.CURRENT.dumpkvs()


def add_image(key, img):
    Logger.CURRENT.add_image(key, img)


def add_text(key, txt):
    Logger.CURRENT.add_text(key, txt)


def add_hist(key, hist):
    Logger.CURRENT.add_hist(key, hist)


def add_figure(key, fig, copy=False):
    Logger.CURRENT.add_figure(key, fig, copy)


def set_step(stepval):
    for fmt in Logger.CURRENT.output_formats:
        if isinstance(fmt, TensorboardOutputFormat):
            fmt.step = stepval


class scoped_setup(object):
    def __init__(self, dirname=None, format_strs=None, action="b"):
        self.dirname = dirname
        self.format_strs = format_strs
        self.prevlogger = None
        self.action = action

    def __enter__(self):
        self.prevlogger = Logger.CURRENT
        setup(dirname=self.dirname, format_strs=self.format_strs, action=self.action)

    def __exit__(self, *args):
        Logger.CURRENT.close()
        Logger.CURRENT = self.prevlogger