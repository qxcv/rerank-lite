#!/usr/bin/env python3
"""GUI to produce a COCO-minus-people dataset (i.e. a set of labels for COCO
indicating whether each frame contains a human or not)."""

from argparse import ArgumentParser
import os
import json
import sys
import logging

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib, Gdk, GdkPixbuf  # flake8: noqa


class PersistentMapping:
    """A persistent dictionary structure which saves to a JSON file. Also keeps
    a log of all actions, just in case."""

    def __init__(self, db_path):
        self._db_path = db_path
        try:
            # try to load old DB first
            with open(db_path, 'r') as fp:
                old_data = json.load(fp)
            self._action_log = old_data['actions']
            self._db = old_data['data']
        except IOError:
            # otherwise, create a blank internal DB
            self._action_log = []
            self._db = {}

        # Sanity checks
        assert isinstance(self._db, dict)
        assert isinstance(self._action_log, list)

    def _save_db(self):
        out_obj = {'actions': self._action_log, 'data': self._db}
        with open(self._db_path, 'w') as fp:
            json.dump(out_obj, fp)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError('Keys must be strings')
        self._db[key] = value
        self._action_log.append(dict(action='__setitem__',
                                     key=key,
                                     value=value))
        self._save_db()

    def __getitem__(self, key):
        return self._db[key]

    def keys(self):
        return self._db.keys()

    def __delitem__(self, key):
        del self.db[key]
        self._action_log.append(dict(action='__delitem__', key=key))
        self._save_db()


class Application:
    """Main class for the program. Handles all GTK events."""
    # Maximum number of images to skip forward when looking for unlabelled
    # images. Ensures that search for unlabelled images terminates in a
    # reasonable amount of time.
    MAX_SKIP = 500

    def __init__(self, image_dir, db):
        builder = Gtk.Builder()
        builder.add_from_file('coco_minus_people.glade')
        builder.connect_signals(self)
        self.window = builder.get_object('application_window')
        self.image_widget = builder.get_object('current_frame')
        self.filename_label = builder.get_object('datum_name')
        self.status_label = builder.get_object('status_label')
        self.label_combo = builder.get_object('label_combo')
        self.skip_checkbutton = builder.get_object('skip_checkbutton')

        self.db = db

        # Populate image list with all .jpgs in image_dir
        image_names = []
        for name in os.listdir(image_dir):
            base = os.path.basename(name)
            if name.endswith('.jpg'):
                image_names.append(base)

        # we'll iterate through this list
        self.image_names = image_names
        self.image_dir = image_dir
        self.image_index = 0

        self.refresh_image()

    def skip_labelled(self):
        return self.skip_checkbutton.get_active()

    def redraw_image(self, *args):
        image_name = self.image_names[self.image_index]
        image_path = os.path.join(self.image_dir, image_name)

        max_width = self.image_widget.get_preferred_width().natural_width
        max_height = self.image_widget.get_preferred_height().natural_height

        pixbuf = GdkPixbuf.Pixbuf.new_from_file(image_path)

        image_width = pixbuf.get_width()
        image_height = pixbuf.get_height()

        scale_width = min(max_width,
                          int(max_height / float(image_height) * image_width))
        scale_height = min(max_height,
                           int(max_width / float(image_width) * image_height))

        pixbuf = pixbuf.scale_simple(scale_width, scale_height,
                                     GdkPixbuf.InterpType.BILINEAR)

        self.image_widget.set_from_pixbuf(pixbuf)

    def refresh_image(self):
        self.redraw_image()
        image_name = self.image_names[self.image_index]

        # Update labels at top of screen
        status_text = 'Image %i/%i' % (self.image_index + 1,
                                       len(self.image_names))
        self.status_label.set_text(status_text)
        self.filename_label.set_text(image_name)

        # Update selector
        try:
            gtk_label = self.db[image_name]

            if gtk_label not in ['unknown', 'no_people', 'has_people']:
                logging.warn('Invalid label %s, deleting' % gtk_label)
                del self.db[image_name]
                gtk_label = 'unknown'
        except KeyError:
            gtk_label = 'unknown'
        self.label_combo.set_active_id(gtk_label)

    def go_next(self):
        new_index = None
        if self.skip_labelled():
            # just keep skipping forward until we find something; I don't have
            # an intelligent way of doing this
            for index_offset in range(1, self.MAX_SKIP + 1):
                index = (self.image_index + index_offset) % len(self.image_names)
                image_name = self.image_names[index]
                try:
                    if self.db[image_name] not in ['no_people', 'has_people']:
                        new_index = index
                        break
                except KeyError:
                    # no label, since self.db[index] check failed
                    new_index = index
                    break
            if new_index is None:
                logging.error("Couldn't find unlabelled frame in {} frames".format(self.MAX_SKIP))
        if new_index is None:
            new_index = min(self.image_index + 1, len(self.image_names) - 1)
        self.image_index = new_index
        self.refresh_image()

    def go_prev(self):
        self.image_index = max(self.image_index - 1, 0)
        self.refresh_image()

    def on_next_button_clicked(self, *args):
        self.go_next()

    def on_prev_button_clicked(self, *args):
        self.go_prev()

    def set_label(self, new_label):
        assert new_label in ['unknown', 'has_people', 'no_people']
        image_name = self.image_names[self.image_index]
        self.db[image_name] = new_label
        logging.info('Setting label of %s to %s' % (image_name, new_label))
        self.refresh_image()

    def on_label_combo_changed(self, *args):
        new_label = self.label_combo.get_active_id()
        self.set_label(new_label)

    def on_application_window_key_press_event(self, widget, event, *args):
        if event.keyval == Gdk.KEY_Left:
            # go back on left key
            self.go_prev()
        elif event.keyval == Gdk.KEY_Right:
            # go forward on right key
            self.go_next()
        elif event.keyval in [Gdk.KEY_y, Gdk.KEY_Y]:
            # 'y' key marks the frame as having people and goes forward
            self.set_label('has_people')
            self.go_next()
        elif event.keyval in [Gdk.KEY_n, Gdk.KEY_N]:
            # 'n' key marks the frame as having not people and goes forward
            self.set_label('no_people')
            self.go_next()

    def on_application_window_delete_event(self, *args):
        sys.exit(0)


parser = ArgumentParser(
    description='Utility to label COCO images as containing people or not')
parser.add_argument('image_dir', type=str, help='COCO images directory')
parser.add_argument('db_path', type=str, help='path to database')

if __name__ == '__main__':
    args = parser.parse_args()
    db = PersistentMapping(args.db_path)
    app = Application(args.image_dir, db)
    app.window.show_all()
    # This has the advantage (over Gtk.main()) of responding nicely to SIGINT
    GLib.MainLoop().run()
