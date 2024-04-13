"""
HICODet dataset under PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import numpy as np
import torch
from typing import Optional, List, Callable, Tuple
from pocket.data import ImageDataset, DataSubset
from .static_hico import HOI_IDX_TO_ACT_IDX, UA_HOI_IDX, SA_HOI_IDX


def get_ua():
    ua = [HOI_IDX_TO_ACT_IDX[idx] for idx in UA_HOI_IDX]
    return ua


class HICODetSubset(DataSubset):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def filename(self, idx: int) -> str:
        """Override: return the image file name in the subset"""
        return self._filenames[self._idx[self.pool[idx]]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Override: return the size (width, height) of an image in the subset"""
        return self._image_sizes[self._idx[self.pool[idx]]]

    @property
    def anno_interaction(self) -> List[int]:
        """Override: Number of annotated box pairs for each interaction class"""
        num_anno = [0 for _ in range(self.num_interation_cls)]
        intra_idx = [self._idx[i] for i in self.pool]
        for idx in intra_idx:
            for hoi in self._anno[idx]['hoi']:
                num_anno[hoi] += 1
        return num_anno

    @property
    def anno_object(self) -> List[int]:
        """Override: Number of annotated box pairs for each object class"""
        num_anno = [0 for _ in range(self.num_object_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[1]] += anno_interaction[corr[0]]
        return num_anno

    @property
    def anno_action(self) -> List[int]:
        """Override: Number of annotated box pairs for each action class"""
        num_anno = [0 for _ in range(self.num_action_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[2]] += anno_interaction[corr[0]]
        return num_anno


class HICODet(ImageDataset):
    """
    Arguments:
        root(str): Root directory where images are downloaded to
        anno_file(str): Path to json annotation file
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
    """

    def __init__(self, root: str, anno_file: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None) -> None:
        super(HICODet, self).__init__(root, transform, target_transform, transforms)
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.num_object_cls = 80
        self.num_interation_cls = 600
        self.num_action_cls = 117
        self._anno_file = anno_file

        # Load annotations
        is_train = False
        if 'train' in root:
            is_train = True

        self.ua = UA_HOI_IDX # delete unseen verb
        self.sa = SA_HOI_IDX
        self._load_annotation_and_metadata(anno, is_train=is_train)


    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._idx)

    def __getitem__(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image

        Returns:
            tuple[image, target]: By default, the tuple consists of a PIL image and a
                dict with the following keys:
                    "boxes_h": list[list[4]]
                    "boxes_o": list[list[4]]
                    "hoi":: list[N]
                    "verb": list[N]
                    "object": list[N]
        """
        intra_idx = self._idx[i]
        image, annotations = self._transforms(
            self.load_image(os.path.join(self._root, self._filenames[intra_idx])),
            self._anno[intra_idx])
        return (image, annotations), self.load_image(os.path.join(self._root, self._filenames[intra_idx])), intra_idx

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ', anno_file='
        reprstr += repr(self._anno_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._root)
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    @property
    def class_corr(self) -> List[Tuple[int, int, int]]:
        """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
        return self._class_corr.copy()

    @property
    def object_n_verb_to_interaction(self) -> List[list]:
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([self.num_object_cls, self.num_action_cls], None)
        for i, j, k in self._class_corr:
            lut[j, k] = i
        return lut.tolist()

    @property
    def object_to_interaction(self) -> List[list]:
        """
        The interaction classes that involve each object type

        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int

    @property
    def object_to_verb(self) -> List[list]:
        """
        The valid verbs for each object type

        Returns:
            list[list]
        """
        obj_to_verb = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_verb[corr[1]].append(corr[2])
        return obj_to_verb

    @property
    def anno_interaction(self) -> List[int]:
        """
        Number of annotated box pairs for each interaction class

        Returns:
            list[600]
        """
        return self._num_anno.copy()

    @property
    def anno_object(self) -> List[int]:
        """
        Number of annotated box pairs for each object class

        Returns:
            list[80]
        """
        num_anno = [0 for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            num_anno[corr[1]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def anno_action(self) -> List[int]:
        """
        Number of annotated box pairs for each action class

        Returns:
            list[117]
        """
        num_anno = [0 for _ in range(self.num_action_cls)]
        for corr in self._class_corr:
            num_anno[corr[2]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def objects(self) -> List[str]:
        """
        Object names

        Returns:
            list[str]
        """
        return self._objects.copy()

    @property
    def verbs(self) -> List[str]:
        """
        Verb (action) names

        Returns:
            list[str]
        """
        return self._verbs.copy()

    @property
    def interactions(self) -> List[str]:
        """
        Combination of verbs and objects

        Returns:
            list[str]
        """
        return [self._verbs[j] + ' ' + self.objects[i]
                for _, i, j in self._class_corr]

    @property
    def rare(self) -> List[int]:
        """
        List of rare class indices

        Returns:
            list[int]
        """
        return self._rare

    @property
    def non_rare(self) -> List[int]:
        """
        List of non-rare class indices

        Returns:
            list[int]
        """
        return self._non_rare

    def split(self, ratio: float) -> Tuple[HICODetSubset, HICODetSubset]:
        """
        Split the dataset according to given ratio

        Arguments:
            ratio(float): The percentage of training set between 0 and 1
        Returns:
            train(Dataset)
            val(Dataset)
        """
        perm = np.random.permutation(len(self._idx))
        n = int(len(perm) * ratio)
        return HICODetSubset(self, perm[:n]), HICODetSubset(self, perm[n:])

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._filenames[self._idx[idx]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self._image_sizes[self._idx[idx]]

    def obtain_seen_target(self, annotation, seen_idx):
        if isinstance(annotation, dict):
            result = dict()
            for key in annotation.keys():
                result[key] = self.obtain_seen_target(annotation[key], seen_idx)
            return result
        if isinstance(annotation, List):
            return [annotation[idx] for idx in seen_idx]

    def _load_annotation_and_metadata(self, f: dict, is_train=False) -> None:
        """
        Arguments:
            f(dict): Dictionary loaded from {anno_file}.json
        """
        idxs = list(range(len(f['filenames'])))
        for empty_idx in f['empty']:
            idxs.remove(empty_idx)

        if is_train:
            self.unseen_target = []
            remove_idx = []
            for idx in idxs:
                annotation = f['annotation'][idx]
                st = [] # 记录保留下来的hoi是原来的第几个hoi,每个图片有有一个st flag annotation
                for id, hoi in enumerate(annotation['hoi']):
                    if hoi in UA_HOI_IDX:
                        continue
                    else:
                        st.append(id)
                if len(st) == 0:    # 没有可见的instance了，直接剔除掉
                    remove_idx.append(idx)
                else:
                    annotation = self.obtain_seen_target(annotation, st)
                    annotation['seen_target'] = st  # 有的话加入到anno中去方便后面提取出tensor来
                    f['annotation'][idx] = annotation
            for idx in remove_idx:
                idxs.remove(idx)

        num_anno = [0 for _ in range(self.num_interation_cls)]
        for id, anno in enumerate(f['annotation']):
            if id not in idxs:
                continue
            for hoi in anno['hoi']:
                num_anno[hoi] += 1

        self._idx = idxs
        self._num_anno = num_anno

        self._anno = f['annotation']
        self._filenames = f['filenames']
        self._image_sizes = f['size']
        self._class_corr = f['correspondence']
        self._empty_idx = f['empty']
        self._objects = f['objects']
        self._verbs = f['verbs']
        self._rare = f['rare']
        self._non_rare = f['non_rare']