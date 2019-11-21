import os
import sys
import os.path as osp
import tempfile
import json
import time
import datetime
from collections import OrderedDict
from contextlib import contextmanager


DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, 'wt')
            self.own_file = True
        else:
            fmtstr = "expected file or str, got {}".format(filename_or_file)
            assert hasattr(filename_or_file, 'read'), fmtstr
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in kvs.items():
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print('WARNING: tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in key2str.items():
            lines.append("| {}{} | {}{} |".format(key, ' ' * (keywidth - len(key)),
                                                  val, ' ' * (valwidth - len(val))))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        return s[:40] + '...' if len(s) > 43 else s

    def writeseq(self, seq):
        for arg in seq:
            self.file.write(arg)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, 'wt')

    def writekvs(self, kvs):
        for k, v in kvs.items():
            if hasattr(v, 'dtype'):
                v = v.tolist()
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, 'w+t')
        self.keys = []
        self.sep = ','

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
                    self.file.write(',')
                self.file.write(k)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = kvs.get(k)
            if v:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()


def make_output_format(format, ev_dir, suffix=''):
    os.makedirs(ev_dir, exist_ok=True)
    if format == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif format == 'log':
        return HumanOutputFormat(osp.join(ev_dir, "log{}.txt".format(suffix)))
    elif format == 'json':
        return JSONOutputFormat(osp.join(ev_dir, "progress{}.json".format(suffix)))
    elif format == 'csv':
        return CSVOutputFormat(osp.join(ev_dir, "progress{}.csv".format(suffix)))
    else:
        raise ValueError("unknown format specified: {}".format(format))


# Frontend

def logkv(key, val):
    """Log a key-value pair with the current logger. This method
    should be called every iteration for the quantities to monitor.
    """
    Logger.CURRENT.logkv(key, val)


def logkvs(d):
    """Log a dictionary of key-value pairs with the current logger"""
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """Write all the key-values pairs accumulated in the logger
    to the write ouput format(s), then flush the dictionary.
    """
    Logger.CURRENT.dumpkvs()


def getkvs():
    """Return the key-value pairs accumulated in the current logger"""
    return Logger.CURRENT.name2val


def log(*args, level=INFO):  # noqa
    """Write the sequence of args, with no separators, to the console
    and output files (if an output file has been configured).
    """
    Logger.CURRENT.log(*args, level=level)


# Create distinct functions fixed at all the values taken by `level`

def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """Set logging threshold on current logger"""
    Logger.CURRENT.set_level(level)


def get_dir():
    """Get directory to which log files are being written"""
    return Logger.CURRENT.get_dir()


# Define aliases for higher-level language
record_tabular = logkv
dump_tabular = dumpkvs


# Backend

class Logger(object):

    DEFAULT = None
    CURRENT = None

    def __init__(self, dir_, output_formats):
        self.name2val = OrderedDict()  # values this iteration
        self.level = INFO
        self.dir_ = dir_
        self.output_formats = output_formats

    def logkv(self, key, val):
        self.name2val.update({key: val})

    def dumpkvs(self):
        if self.level == DISABLED:
            return
        for output_format in self.output_formats:
            if isinstance(output_format, KVWriter):
                output_format.writekvs(self.name2val)
        self.name2val.clear()

    def log(self, *args, level=INFO):
        if self.level <= level:
            # If the current logger level is higher than
            # the `level` argument, don't print to stdout
            self._log(args)

    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir_

    def close(self):
        for output_format in self.output_formats:
            output_format.close()

    def _log(self, args):
        for output_format in self.output_formats:
            if isinstance(output_format, SeqWriter):
                output_format.writeseq(map(str, args))


def configure(dir_=None, format_strs=None):
    """Configure logger"""
    if dir_ is None:
        dir_ = osp.join(tempfile.gettempdir(),
                        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f_temp_log"))
    assert isinstance(dir_, str), "wrong type: {}".format(type(dir_))
    # Make sure the provided directory exists
    os.makedirs(dir_, exist_ok=True)
    if format_strs is None:
        format_strs = []
    # Setup the output formats
    output_formats = [make_output_format(f, dir_) for f in format_strs]
    Logger.CURRENT = Logger(dir_=dir_, output_formats=output_formats)


def configure_default_logger():
    """Configure default logger"""
    # Write to stdout by default
    format_strs = ['stdout']
    # Configure the current logger
    configure(format_strs=format_strs)
    # Logging successful configuration of default logger
    log("configuring default logger for each worker (logging to stdout only by default)")
    # Define the default logger with the current logger
    Logger.DEFAULT = Logger.CURRENT


def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log('resetting logger')


configure_default_logger()
