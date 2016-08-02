#!/usr/bin/env python3
"""Main script to train model and run experiments."""

from argparse import ArgumentParser
from collections import namedtuple
from json import load as load_json
from logging import debug
from os import path
from typing import Tuple

import numpy as np  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.layers.core import Dense, Flatten, Dropout  # type: ignore
from keras.layers.convolutional import (Convolution2D,
                                        MaxPooling2D)  # type: ignore


def build_model(input_shape: Tuple[int, ...], num_classes: int) -> Sequential:
    """Build a simple classifier based on VGGNet, but with less pooling. You'll
    have to compile it yourself.

    I might make this a simple renset later (e.g. resnet 34)."""
    model = Sequential()
    model.add(Convolution2D(64,
                            3,
                            3,
                            activation='relu',
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    return model


def unobjectify(obj_array: np.ndarray) -> np.ndarray:
    """Turns a Numpy object array of arrays with some particular dtype into one
    homogeneous array."""
    # TODO: Do I need this anymore? Now that I'm not manipulating loaded .mat
    # files, it shouldn't matter a great deal
    assert obj_array.dtype.name == 'object'
    if obj_array.size == 0:
        return np.array([])

    # Define shape and data type for new array based on its first element
    chosen_elem = obj_array.flat[0]
    if isinstance(chosen_elem, np.ndarray):
        new_shape = obj_array.shape + chosen_elem.shape
        new_dtype = chosen_elem.dtype
    else:
        # We have an array of (homogeneous?) scalars
        new_shape = obj_array.shape
        new_dtype = type(chosen_elem)

    # Now construct the array and fill it out index-by-index
    return_value = np.ndarray(shape=new_shape, dtype=new_dtype)
    orig_iter = np.nditer(obj_array, flags=['refs_ok', 'multi_index'])
    while not orig_iter.finished:
        indices = orig_iter.multi_index
        return_value[indices, ...] = orig_iter[0]
        orig_iter.iternext()

    return return_value


def transpose_objects(objlist):
    """Turn a list of dictionaries into a dictionary with list values."""
    keys = objlist[0].keys()
    return {k: [d[k] for d in objlist] for k in keys}


_PersonRect = namedtuple('_PersonRect', ['scale', 'head_pos', 'point'])
_DatumAnnotation = namedtuple('_DatumAnnotation', ['image_name', 'people'])


class PoseDataset:
    """Encapsulating class for the entire MPII Human Pose dataset (for
    example). Each sample is identified by an integer corresponding to its
    position in the original data set."""

    def __init__(self, data_path: str) -> None:
        """Load dataset from path"""
        self._data_path = data_path
        json_path = path.join(data_path, 'mpii_human_pose_v1_u12_1.json')
        with open(json_path) as fp:
            loaded_json = load_json(fp)['RELEASE']

        # Mask telling us whether sample is used for training or not
        self._train_mask = np.array(loaded_json['img_train']) != 0

        # Not going to bother with this for now
        # flat_sp = loaded_json['single_person']
        # for idx in range(len(flat_sp)):
        #     flat_sp[idx] = flat_sp[idx].flatten()
        # self._single_person = flat_sp

        # Load the actual annotations for each datum
        processed_annos = []
        for datum_id, datum_anno in enumerate(loaded_json['annolist']):
            people = []
            annorect = datum_anno['annorect']
            # Work around JSONLab's intelligent guess at the desired shape (not
            # correct, in this case)
            if isinstance(annorect, dict):
                annorects = [annorect]
            elif annorect is None:
                annorects = []
            else:
                annorects = annorect
            for person_rect in annorects:
                if 'x1' in person_rect:
                    head_x1 = person_rect['x1']
                    head_y1 = person_rect['y1']
                    head_x2 = person_rect['x2']
                    head_y2 = person_rect['y2']
                    head_pos = np.array([head_x1, head_y1, head_x2, head_y2]),
                else:
                    head_pos = None

                if 'scale' in person_rect:
                    scale = person_rect['scale']
                else:
                    scale = None

                point_dtype = [('x', 'uint16'), ('y', 'uint16'),
                               ('is_visible', 'object'), ('id', 'uint8')]
                if person_rect.get('annopoints', None) is not None:
                    annopoints = person_rect['annopoints']
                    point_struct = annopoints['point']
                    if isinstance(point_struct, dict):
                        point_struct = [point_struct]
                    transposed = transpose_objects(point_struct)
                    if 'is_visible' not in transposed:
                        # IDK, do something sane?
                        fake_vis = [False] * len(transposed['x'])
                        transposed['is_visible'] = fake_vis

                    # Store as sane, flattened structure array
                    point_data = np.array(
                        list(zip(transposed['x'], transposed['y'], transposed[
                            'is_visible'], transposed['id'])),
                        dtype=point_dtype)
                else:
                    point_data = np.array([])

                rect = _PersonRect(scale=scale,
                                   head_pos=head_pos,
                                   point=point_data)

                people.append(rect)

            # I kid you not, this was the easiest way of getting at the string
            # contents
            image_name = datum_anno['image']['name']
            anno = _DatumAnnotation(image_name=image_name, people=people)
            processed_annos.append(anno)

        self._annotations = np.array(processed_annos)

        debug('MPII data load successful, got %i annotations',
              len(self._annotations))

    def __getitem__(self, key):
        return self._annotations[key]

    def train_indices(self) -> np.ndarray:
        return np.nonzero(self._train_mask)

    def test_indices(self) -> np.ndarray:
        return np.nonzero(~self._train_mask)


parser = ArgumentParser(
    description='Script to train re-ranking model and run experiments')
parser.add_argument('--data_path', default='data/', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = PoseDataset(args.data_path)
