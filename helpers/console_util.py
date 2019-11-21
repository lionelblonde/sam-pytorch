import time
from contextlib import contextmanager

import numpy as np

from helpers.misc_util import zipsame, prettify_time


def cell(x, width):
    """Format a tabular cell to the specified width"""
    if isinstance(x, np.ndarray):
        x = x.item()
    rep = "{:G}".format(x) if isinstance(x, float) else str(x)
    return rep + (' ' * (width - len(rep)))


def columnize(names, tuples, widths, indent=2):
    """Generate and return the content of table
    (w/o logging or printing anything)

    Args:
        width (int): Width of each cell in the table
        indent (int): Indentation spacing prepended to every row in the table
    """
    indent_space = indent * ' '
    # Add row containing the names
    table = indent_space + " | ".join(cell(name, width) for name, width in zipsame(names, widths))
    table_width = len(table)
    # Add header hline
    table += '\n' + indent_space + ('-' * table_width)
    for tuple_ in tuples:
        # Add a new row
        table += '\n' + indent_space
        table += " | ".join(cell(value, width) for value, width in zipsame(tuple_, widths))
    # Add closing hline
    table += '\n' + indent_space + ('-' * table_width)
    return table


def colorize(string, color, bold=False, highlight=False):
    color2num = dict(gray=30, red=31, green=32, yellow=33, blue=34,
                     magenta=35, cyan=36, white=37, crimson=38)
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def log_module_info(logger, name, model):
    logger.info("logging {} specs".format(name))
    tuples = zip([x for x in model.state_dict()],
                 [model.state_dict()[x].size() for x in model.state_dict()],
                 [model.state_dict()[x].numel() for x in model.state_dict()])
    logger.info(columnize(['param', 'size', 'numel'], tuples, widths=[24, 24, 8]))
    logger.info("  total num params: {}".format(sum([model.state_dict()[x].numel()
                                                     for x in model.state_dict()])))


def timed_cm_wrapper(logger, color_message='magenta', color_elapsed_time='cyan'):
    """Wraps a context manager that records the time taken by encapsulated ops"""
    @contextmanager
    def _timed(message):
        """Display the time it took for the mpi master
        to perform the task within the context manager
        """
        logger.info(colorize(message, color=color_message))
        tstart = time.time()
        yield
        logger.info(colorize("  [done in {:.3f} seconds]".format(time.time() - tstart),
                             color=color_elapsed_time))
    return _timed


def pretty_iter(logger, i):
    """Display the current iteration with a colored decorator"""
    logger.info(colorize("I T E R A T I O N  {}".format(i), color='blue'))


def pretty_elapsed(logger, tstart):
    """Display the elapsed time with a colored decorator"""
    elapsed = prettify_time(time.time() - tstart)
    # logger.info('')
    logger.info(colorize("E L A P S E D  {}".format(elapsed), color='green'))
