#!/usr/bin/env python3
"""GUI to produce a COCO-minus-people dataset (i.e. a set of labels for COCO
indicating whether each frame contains a human or not)."""

import sys

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib  # flake8: noqa


def null_handler(*args):
    print('Null handler called with {}'.format(args))


def quit_handler(*args):
    sys.exit(0)


handlers = {
    'on_next_button_clicked': null_handler,
    'on_prev_button_clicked': null_handler,
    'on_label_combo_changed': null_handler,
    'on_application_window_delete_event': quit_handler,
    'on_application_window_key_press_event': null_handler
}

if __name__ == '__main__':
    builder = Gtk.Builder()
    builder.add_from_file('coco_minus_people.glade')
    builder.connect_signals(handlers)
    win = builder.get_object('application_window')
    win.show_all()
    # This has the advantage (over Gtk.main()) of responding nicely to SIGINT
    GLib.MainLoop().run()
