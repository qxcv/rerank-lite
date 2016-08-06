#!/usr/bin/env python3
"""Main script to train model and run experiments."""

from argparse import ArgumentParser
from json import load as load_json
from logging import debug, info
from os import path
from random import shuffle
from typing import Any, Dict, List, TypeVar, Union, Iterator, Tuple, Optional

import numpy as np  # type: ignore
from scipy.ndimage import imread  # type: ignore
from scipy.misc import imresize  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.layers.core import Dense, Flatten, Dropout  # type: ignore
from keras.layers.convolutional import (Convolution2D,  # type: ignore
                                        MaxPooling2D)  # type: ignore

Any  # XXX: Getting rid of annoying unused var warning
K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')


def crop_patch(image: np.ndarray,
               center_x: float,
               center_y: float,
               width: float,
               height: float,
               mode: str='edge') -> np.ndarray:
    """Crops a patch out an image, padding with the boundary value if
    necessary. ``mode`` takes on the same values as ``numpy.pad``'s ``mode``
    keyword argument."""
    assert width >= 1 and height >= 1
    assert image.ndim >= 2

    top = int(center_y - height / 2)
    top_b = max(top, 0)
    top_pad = top_b - top
    assert top_pad >= 0

    bot = int(top + height)
    bot_b = min(bot, image.shape[0])  # type: int
    bot_pad = bot - bot_b
    assert bot_pad >= 0

    left = int(center_x - width / 2)
    left_b = max(left, 0)
    left_pad = left_b - left
    assert left_pad >= 0

    right = int(left + width)
    right_b = min(right, image.shape[1])  # type: int
    right_pad = right - right_b
    assert right_pad >= 0

    sample = image[top_b:bot_b, left_b:right_b]
    assert bot_b > top_b and right_b > left_b, (top_b, bot_b, left_b, right_b)
    pad_amounts = ((top_pad, bot_pad), (left_pad, right_pad))  # type: Any
    extra_dims = image.ndim - 2  # type: int
    pad_amounts += ((0, 0), ) * extra_dims
    rv = np.pad(sample, pad_amounts, mode=mode)
    assert (np.array([rv.shape[0] - height, rv.shape[1] - width]) <= 1).all()

    return rv


def build_model(num_classes: int) -> Sequential:
    """Build a simple classifier based on VGGNet, but with less pooling. You'll
    have to compile it yourself.

    I might make this a simple renset later (e.g. resnet 34)."""
    model = Sequential()
    model.add(Convolution2D(64,
                            3,
                            3,
                            activation='relu',
                            border_mode='same',
                            input_shape=(3, 224, 224)))
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


def transpose_objects(objlist: List[Dict[K, V]]) -> Dict[K, List[V]]:
    """Turn a list of dictionaries into a dictionary with list values."""
    keys = objlist[0].keys()
    return {k: [d[k] for d in objlist] for k in keys}


MPII_JOINT_INDICES = {
    'r_ankle': 0,
    'r_knee': 1,
    'r_hip': 2,
    'l_hip': 3,
    'l_knee': 4,
    'l_ankle': 5,
    'pelvis': 6,
    'thorax': 7,
    'upper_neck': 8,
    'head_top': 9,
    # Elbow entry is missing in README; see
    # https://github.com/anewell/pose-hg-train/blob/master/src/pypose/ref.py
    # for probable correct implementation
    'r_wrist': 10,
    'r_elbow': 11,
    'r_shoulder': 12,
    'l_shoulder': 13,
    'l_elbow': 14,
    'l_wrist': 15
}
MPII_JOINT_NAMES = {index: name for name, index in MPII_JOINT_INDICES.items()}


class PersonRect:
    """Class to represent a single person instance in an image."""

    def __init__(self, scale: Optional[float], head_pos: Optional[np.ndarray],
                 point: Optional[np.ndarray]) -> None:
        self.scale = scale
        self.head_pos = head_pos
        self.point = point
        assert {'x', 'y', 'id', 'is_visible'} == set(point.dtype.names)

    def _get_idx(self, joint: str) -> int:
        """Return an index into self.point for the given joint"""
        joint_id = MPII_JOINT_INDICES[joint]
        correct_id = int(np.nonzero(self.point['id'] == joint_id)[0])
        return correct_id

    def joint_xy(self, joint: str) -> np.ndarray:
        """Get [x, y] location of a named joint"""
        joint_idx = self._get_idx(joint)
        return np.array(
            [self.point['x'][joint_idx], self.point['y'][joint_idx]])

    def joint_vis(self, joint: str) -> Optional[bool]:
        """Return whether named joint is visible"""
        joint_idx = self._get_idx(joint)
        return self.point['is_visible'][joint_idx]

    def __repr__(self):
        return 'PersonRect({}, {}, {})'.format(self.scale, self.head_pos,
                                               self.point)


class DatumAnnotation:
    def __init__(self, image_name: str, people: List[PersonRect]) -> None:
        self.image_name = image_name
        self.people = people

    def __repr__(self):
        return 'DatumAnnotation({}, {})'.format(self.image_name, self.people)


class PoseDataset:
    """Encapsulating class for the entire MPII Human Pose dataset (for
    example). Each sample is identified by an integer corresponding to its
    position in the original data set."""

    def __init__(self, data_path: str, train_frac: float=1) -> None:
        """Load dataset from path"""
        self._data_path = data_path
        json_path = path.join(data_path, 'mpii_human_pose_v1_u12_1.json')
        with open(json_path) as fp:
            loaded_json = load_json(fp)['RELEASE']

        # Mask telling us whether sample is used for training or not
        self._train_mask = np.array(loaded_json['img_train']) != 0

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
                    point_data = np.array([], dtype=point_dtype)

                rect = PersonRect(scale=scale,
                                  head_pos=head_pos,
                                  point=point_data)

                people.append(rect)

            # I kid you not, this was the easiest way of getting at the string
            # contents
            image_name = datum_anno['image']['name']
            anno = DatumAnnotation(image_name=image_name, people=people)
            processed_annos.append(anno)

        self._annotations = np.array(processed_annos)

        # Make train/val split (not the same as MPII's train/test split)
        valid_inds = self.mpii_train_indices
        perm = np.random.permutation(valid_inds)
        num_train = int(train_frac * valid_inds.size)
        self.split_train_indices = perm[:num_train]
        self.split_val_indices = perm[num_train:]

        debug('MPII data load successful, got %i annotations',
              len(self._annotations))

    def __getitem__(
            self, key:
            Union[int, np.ndarray]) -> Union[np.ndarray, DatumAnnotation]:
        return self._annotations[key]

    @property
    def mpii_train_indices(self) -> np.ndarray:
        rv, = np.nonzero(self._train_mask)
        assert rv.ndim == 1
        return rv

    @property
    def mpii_test_indices(self) -> np.ndarray:
        rv, = np.nonzero(~self._train_mask)
        assert rv.ndim == 1
        return rv

    def load_image(self, datum: DatumAnnotation) -> np.ndarray:
        image_path = path.join(self._data_path, 'images', datum.image_name)
        return imread(image_path)


class MaxIterExceeded(Exception):
    pass


def random_box(shape: Tuple[int, int],
               box_side: int,
               people: List[PersonRect],
               max_iter=1000) -> List[float]:
    """Return a random box which doesn't (significantly) intersect any labelled
    person's joints. Returns ``[x, y, w, h]`` for the chosen box."""
    # Width and height of image, size of box to be returned
    height, width = shape

    points = np.concatenate([p.point for p in people])
    # reshape for broadcasting with chosen_y
    point_x = points['x'].reshape((1, -1))
    point_y = points['y'].reshape((1, -1))

    # try to find a randomly selected box which doesn't intersect any point in
    # either dimension
    chosen_y = np.random.uniform(0, height - box_side, size=(max_iter, 1))
    y_intersect = (point_y > chosen_y) & (point_y < chosen_y + box_side)
    chosen_x = np.random.uniform(0, width - box_side, size=(max_iter, 1))
    x_intersect = (point_x > chosen_x) & (point_x < chosen_x + box_side)
    # any_intersect[i] is nonzero iff there is a joint that intersects the box
    any_intersect = np.sum(x_intersect & y_intersect, axis=1)
    assert any_intersect.shape == (max_iter, )

    inds, = np.nonzero(~any_intersect)
    if inds.size == 0:
        raise MaxIterExceeded('no box found in %i iterations' % max_iter)
    return_x = chosen_x[inds[0]]
    return_y = chosen_y[inds[0]]
    return [return_x, return_y, box_side, box_side]


class DatumSpec:
    def __init__(self,
                 dataset_index: int,
                 is_foreground: bool,
                 person_index: Optional[int]=None,
                 joint_name: Optional[str]=None) -> None:
        assert isinstance(dataset_index, int)
        assert isinstance(is_foreground, bool)
        self.dataset_index = dataset_index
        self.is_foreground = is_foreground
        if self.is_foreground:
            assert isinstance(joint_name, str)
            assert isinstance(person_index, int)
        else:
            assert joint_name is None
            assert person_index is None
        self.person_index = person_index
        self.joint_name = joint_name


def fetch_datum(dataset: PoseDataset, spec:
                DatumSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve a single training sample from the dataset. Requires image
    loading, joint location, augmentation, etc."""
    datum = dataset[spec.dataset_index]
    image = dataset.load_image(datum)
    if spec.is_foreground:
        label = np.array([0, 1])
        person_rect = datum.people[spec.person_index]
        joint_xy = person_rect.joint_xy(spec.joint_name)
        # On a 200px tall person, the cropped boxes would be 25px to a side
        box_side = int(25 * person_rect.scale)
        crop_box = [int(joint_xy[0]), int(joint_xy[1]), box_side, box_side]
    else:
        # Select a background crop. TODO: more robust way of getting
        # person-free crops
        label = np.array([1, 0])
        box_side = int(25 * np.mean([rect.scale for rect in datum.people]))
        crop_box = random_box(image.shape[:2], box_side, datum.people)

    cropped_image = crop_patch(image, *crop_box)
    scaled_image = imresize(cropped_image, (224, 224))

    return (scaled_image, label)


def infinishuffle(data: List[T]) -> Iterator[T]:
    """Keep shuffling a data array forever, yielding each element in the array
    in seqeuence each time it is shuffled."""
    # Make copy so that we can shuffle in-place
    to_shuf = list(data)
    while True:
        shuffle(to_shuf)
        yield from to_shuf


def valid_frame(datum: DatumAnnotation) -> bool:
    """Check that a ``DatumAnnotation`` is for a frame that contains people,
    each of whom has a scale."""
    people = datum.people

    # Make sure there are people in the frame
    if not people:
        return False

    # Make sure that every person has a scale
    for person in people:
        if person.scale is None:
            break
    else:
        return True
    return False


def fg_spec_generator(dataset: PoseDataset, accepted_inds:
                      np.ndarray) -> Iterator[DatumSpec]:
    """Generate transformation specs for foreground patches."""
    ind_pairs = []  # type: List[Tuple[int, int, str]]
    for ds_idx in accepted_inds:
        if not valid_frame(dataset[ds_idx]):
            continue
        for person_idx, person_rect in enumerate(dataset[ds_idx].people):
            for joint_id in person_rect.point['id']:
                joint_name = MPII_JOINT_NAMES[joint_id]
                ind_pairs.append((int(ds_idx), int(person_idx), joint_name))

    random_inds = infinishuffle(ind_pairs)

    for ds_idx, person_idx, joint_name in random_inds:
        yield DatumSpec(dataset_index=ds_idx,
                        person_index=person_idx,
                        is_foreground=True,
                        joint_name=joint_name)


def bg_spec_generator(dataset: PoseDataset, accepted_inds:
                      np.ndarray) -> Iterator[DatumSpec]:
    """Generate transformation specs for background patches."""
    inds = infinishuffle(list(accepted_inds))
    for ind in inds:
        if not valid_frame(dataset[ind]):
            continue
        yield DatumSpec(dataset_index=int(ind),
                        person_index=None,
                        is_foreground=False,
                        joint_name=None)


def random_interleave(left: Iterator[T], right: Iterator[T], pleft:
                      float) -> Iterator[T]:
    """Yield from left or right forever, choosing a generator to yield from at
    random each time. ``pleft`` is the probability the left iterator will be
    chosen; otherwise, the right will be."""
    while True:
        if np.random.random() < pleft:
            yield next(left)
        else:
            yield next(right)


def data_generator(dataset: PoseDataset,  # yapf: disable
                   is_train: bool) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generates ``(input, label)`` pairs for Keras, forever."""
    if is_train:
        # train data generator
        accepted_inds = dataset.split_train_indices
    else:
        # validation data generator
        accepted_inds = dataset.split_val_indices
    pos_generator = fg_spec_generator(dataset, accepted_inds)
    neg_generator = bg_spec_generator(dataset, accepted_inds)
    all_specs = random_interleave(pos_generator, neg_generator, 0.5)
    for spec in all_specs:
        yield fetch_datum(dataset, spec)


parser = ArgumentParser(
    description='Script to train re-ranking model and run experiments')
parser.add_argument('--data_path',
                    default='data/',
                    type=str,
                    help='path to data folder')
parser.add_argument('--epoch_length',
                    default=1000,
                    type=int,
                    help='length of epochs (in batches)')
parser.add_argument('--num_epochs',
                    default=100,
                    type=int,
                    help='number of epochs to train for')
parser.add_argument(
    '--train_frac',
    default=0.8,
    type=float,
    help='fraction of data to use for training (rest validation)')

if __name__ == '__main__':
    np.random.seed(0)
    args = parser.parse_args()

    dataset = PoseDataset(args.data_path, train_frac=args.train_frac)
    train_gen = data_generator(dataset, True)
    # valid_gen = data_generator(dataset, False)
    # XXX: this is just for debugging the generator
    for thing in train_gen:
        print('Got something')

    model = build_model(num_classes=2)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    info('Fitting model')
    model.fit_generator(train_gen,
                        args.epoch_length,
                        args.num_epochs,
                        # validation_data=valid_gen,
                        nb_worker=1)
