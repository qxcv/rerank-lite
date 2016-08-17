#!/usr/bin/env python3
"""GUI to produce a COCO-minus-people dataset (i.e. a set of labels for COCO
indicating whether each frame contains a human or not)."""

from argparse import ArgumentParser
from ast import literal_eval
import gzip
import json
import logging
import os
import sys
import time

from coco import COCO

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib, Gdk, GdkPixbuf  # flake8: noqa

# COCO supercategories that we ignore (I'm practising my horrible variable
# naming)
REPUGNANT_SUPERCATS = {'person'}
# Maximum number of images to skip forward when looking for unlabelled images.
# Ensures that search for unlabelled images terminates in a reasonable amount
# of time.
MAX_SKIP = 500
# Don't save constantly, since the required JSON encoding/decoding is really
# slow
SAVE_INTERVAL_S = 60


class PersistentMapping:
    """A persistent dictionary structure which saves to a JSON file. Also keeps
    a log of all actions, just in case."""

    def __init__(self, db_path):
        self._db_path = db_path
        try:
            # try to load old DB first
            with gzip.open(db_path, 'rt') as fp:
                old_data = json.load(fp)
            self._action_log = old_data['actions']
            self._db = old_data['data']
        except (IOError, json.decoder.JSONDecodeError):
            # otherwise, create a blank internal DB
            self._action_log = []
            self._db = {}

        # Sanity checks
        assert isinstance(self._db, dict)
        assert isinstance(self._action_log, list)

    def save_db(self):
        out_obj = {'actions': self._action_log, 'data': self._db}
        with gzip.open(self._db_path, 'wt') as fp:
            json.dump(out_obj, fp)

    def _encode_key(self, key_val):
        allowed_types = (int, str)
        if isinstance(key_val, (int, str)):
            return repr(key_val)
        raise ValueError('Keys must be one of {}, but type was'.format(
            allowed_types, type(key_val)))

    def _decode_key(self, key_str):
        return literal_eval(key_str)

    def __setitem__(self, key_val, value):
        self.set(key_val, value, should_save=True, save_action=True)

    def set(self, key_val, value, should_save=True, save_action=True):
        key = self._encode_key(key_val)
        self._db[key] = value
        if save_action:
            self._action_log.append(dict(action='set', key=key, value=value))
        if should_save:
            self.save_db()

    def __getitem__(self, key_val):
        key = self._encode_key(key_val)
        return self._db[key]

    def get(self, key_val, default=None):
        key = self._encode_key(key_val)
        return self._db.get(key, default)

    def keys(self):
        return map(self._decode_key, self._db.keys())

    def __delitem__(self, key_val):
        key = self._encode_key(key_val)
        del self._db[key]
        self._action_log.append(dict(action='del', key=key))
        self.save_db()


def pre_init_db(db, coco):
    """Run through the entire DB, setting "has_people" flag on any image which
    COCO thinks has people, and which isn't already set.

    :param db: ``PersistentMapping`` holding person labels.
    :param coco: ``COCO`` instance with COCO annotations."""
    bad_cat_ids = []
    for cat_id, cat_info in coco.cats.items():
        if cat_info['supercategory'] in REPUGNANT_SUPERCATS:
            bad_cat_ids.append(cat_id)
    bad_images = coco.getImgIds(catIds=bad_cat_ids)
    for image_id in bad_images:
        current_label = db.get(image_id, None)
        if current_label in ['no_people', 'unknown']:
            logging.warn('Image #%i labelled %s but COCO says it has people!',
                         image_id, current_label)
            # Skip, don't overwrite
            continue
        db.set(image_id, 'has_people', should_save=False, save_action=False)
    db.save_db()


class Application:
    """Main class for the program. Handles all GTK events."""

    def __init__(self, anno_path, image_dir, db):
        """Constructor. Will load the GUI from a Glade description, then read
        any existing labels."""
        builder = Gtk.Builder()
        builder.add_from_file('coco_minus_people.glade')
        builder.connect_signals(self)
        self.window = builder.get_object('application_window')
        self.image_widget = builder.get_object('current_frame')
        self.filename_label = builder.get_object('datum_name')
        self.status_label = builder.get_object('status_label')
        self.label_combo = builder.get_object('label_combo')
        self.skip_checkbutton = builder.get_object('skip_checkbutton')

        self.coco = COCO(anno_path)

        self.db = db
        self.last_save = self._time()

        # we'll iterate through this list
        self.image_ids = sorted(self.coco.imgs.keys())
        self.image_dir = image_dir
        self.image_index = 0

        logging.info('Pre-initialising DB')
        pre_init_db(self.db, self.coco)
        logging.info('Pre-initialisation done!')

        self.refresh_image()

    def skip_labelled(self):
        """Should labelled images be skipped?"""
        return self.skip_checkbutton.get_active()

    def redraw_image(self, *args):
        """Reload the image and update the GTK image view used to display it"""
        image_id = self.image_ids[self.image_index]
        image_name = self.coco.imgs[image_id]['file_name']
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
        """Re-display the current image and all related widgets (label
        selector, image name label, etc.)"""
        self.redraw_image()
        image_id = self.image_ids[self.image_index]

        # Update labels at top of screen
        status_text = 'Image %i/%i' % (self.image_index + 1,
                                       len(self.image_ids))
        self.status_label.set_text(status_text)
        self.filename_label.set_text('Image %i' % image_id)

        # Update selector
        try:
            gtk_label = self.db[image_id]

            if gtk_label not in ['unknown', 'no_people', 'has_people']:
                logging.warn('Invalid label %s, deleting' % gtk_label)
                del self.db[image_id]
                gtk_label = 'unknown'
        except KeyError:
            # Silently set to unknown
            gtk_label = 'unknown'
        self.label_combo.set_active_id(gtk_label)

    def go_next(self):
        """Advance to the next image and update the UI"""
        new_index = None
        if self.skip_labelled():
            # just keep skipping forward until we find something; I don't have
            # an intelligent way of doing this
            for index_offset in range(1, MAX_SKIP + 1):
                index = (self.image_index + index_offset) % len(self.image_ids)
                new_id = self.image_ids[index]
                if self.db.get(new_id) not in ['no_people', 'has_people']:
                    new_index = index
                    break
            if new_index is None:
                logging.error(
                    "Couldn't find unlabelled frame in {} frames".format(
                        MAX_SKIP))
        if new_index is None:
            new_index = min(self.image_index + 1, len(self.image_ids) - 1)
        self.image_index = new_index
        self.refresh_image()

    def go_prev(self):
        """Go back to the previous image and update the UI"""
        self.image_index = max(self.image_index - 1, 0)
        self.refresh_image()

    def _time(self):
        return time.clock_gettime(time.CLOCK_MONOTONIC)

    def set_label(self, new_label):
        """Set the label (person/no person) associated with the current
        frame."""
        assert new_label in ['unknown', 'has_people', 'no_people']
        image_id = self.image_ids[self.image_index]
        should_save = self._time() - self.last_save > SAVE_INTERVAL_S
        self.db.set(image_id, new_label, should_save=should_save)
        if should_save:
            self.db.save_db()
            self.last_save = self._time()
        logging.info('Setting label of #%i to %s' % (image_id, new_label))
        self.refresh_image()

    ######################
    # GTK event handlers #
    ######################

    def on_next_button_clicked(self, *args):
        self.go_next()

    def on_prev_button_clicked(self, *args):
        self.go_prev()

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
        self.db.save_db()
        sys.exit(0)


parser = ArgumentParser(
    description='Utility to label COCO images as containing people or not')
parser.add_argument('anno_path',
                    type=str,
                    help='COCO annotation path directory')
parser.add_argument('image_dir', type=str, help='COCO images directory')
parser.add_argument('db_path', type=str, help='path to database')

if __name__ == '__main__':
    args = parser.parse_args()
    db = PersistentMapping(args.db_path)
    app = Application(args.anno_path, args.image_dir, db)
    app.window.show_all()
    # This has the advantage (over Gtk.main()) of responding nicely to SIGINT
    GLib.MainLoop().run()
