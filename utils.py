"""Various non-pose-estimation-specific utilities."""

import sys


# See http://stackoverflow.com/a/242531
def pdb_excepthook(type, value, tb):
    """sys.excepthook callback which starts a pdb shell whenever an uncaught
    exception occurs."""
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback
        import pdb
        # we are NOT in interactive mode, print the exception and start a
        # debugger
        traceback.print_exception(type, value, tb)
        print()
        pdb.post_mortem(tb)


def register_pdb_hook():
    sys.excepthook = pdb_excepthook
