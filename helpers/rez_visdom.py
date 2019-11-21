import os
import glob
import argparse

import visdom


parser = argparse.ArgumentParser(description="Revive visdom logs")
parser.add_argument('--visdom_dir', type=str, default="data/visdom_logs",
                    help="directory where the visdom logs to revive are stored")


def resurrect(viz, visdom_dir):
    """Resurrect the visdom logs present in the input dir `dir_`
    into the cirrently running visdom instance `viz`
    """
    for filename in glob.iglob("{}/*".format(visdom_dir), recursive=True):
        if os.path.isfile(filename):
            print("resurrecting visdom log: {}".format(filename))
            viz.replay_log(filename)


if __name__ == "__main__":
    _args = parser.parse_args()
    viz = visdom.Visdom()
    assert viz.check_connection(timeout_seconds=4), "viz co not great"
    resurrect(viz, _args.visdom_dir)
