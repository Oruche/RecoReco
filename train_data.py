from io import BytesIO
import json
import os
import random
from typing import List

import cv2
import numpy as np
import requests
from PIL import Image

import config


def prepare(labels, count, image_size, color=True, shuffle=True, flatten=True):

    pairs = UrlLabelPairs.download(labels, count, shuffle=shuffle)
    factory = TrainImageFactory(image_size, color=color, flatten=flatten)

    train_images = []
    train_labels = []

    for p in pairs:
        try:
            img = factory.create(p["url"])
        except (CouldNotDownloadErr, NotMatchedChannelErr):
            continue
        except Exception as e:
            raise e
        else:
            train_images.append(img)
            train_labels.append(p["label"])

    return train_images, to_num_label(train_labels)


def to_num_label(labels):
    _dict = {str(label): i for i, label in enumerate(labels)}
    return [_dict[label] for label in labels]


class UrlLabelPairs:
    @classmethod
    def download(cls, labels: List[str], count: int, shuffle=True):
        url_label_pairs = []
        for label in labels:
            urls = cls._fetch_urls(label, count)
            for url in urls:
                url_label_pairs.append({"url": url, "label": label})
        if shuffle:
            random.shuffle(url_label_pairs)
        return url_label_pairs

    @classmethod
    def _fetch_urls(cls, label: str, count: int):
        req_data = json.dumps({"tags": [label], "counts": count})
        resp = requests.post(config.config.search_endpoint, data=req_data)
        if resp.status_code != 200:
            print(resp.status_code)
        items = resp.json()["items"]
        return [cls._create_url(r["path"]) for r in items]

    @classmethod
    def _create_url(cls, path):
        return os.path.join(config.config.service_domain, path)


class TrainImageFactory:

    def __init__(self, image_size: int, color: bool=True, flatten: bool=True) -> np.ndarray:
        """

        :rtype: object
        """
        self.image_size = image_size
        self.color = color
        self.flatten = flatten
        if color:
            self.fit_color = self.__class__._fit_3ch
        else:
            self.fit_color = self.__class__._fit_1ch

    def create(self, url: str):
        try:
            img = download_image(url)
        except Exception:
            raise CouldNotDownloadErr

        img = self.fit_color(img)

        img = cv2.resize(img, (self.image_size, self.image_size))

        if self.flatten:
            img = img.flatten().astype(np.float32) / 255.0

        return img

    @classmethod
    def _fit_3ch(cls, img):
        ch = get_num_channel(img)
        if ch == 1:
            raise NotMatchedChannelErr
        elif ch == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:  # ch==4
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    @classmethod
    def _fit_1ch(cls, img):
        ch = get_num_channel(img)
        if ch == 1:
            return img
        elif ch == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else: # ch==4
            return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)


class NotMatchedChannelErr(Exception):
        pass


class CouldNotDownloadErr(Exception):
        pass


def download_image(url: str):
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content))
    img = np.array(img)
    return img


def get_num_channel(img: np.ndarray):
    if len(img.shape) == 2:
        return 1
    if img.shape[2] == 3:
        return 3
    else:
        return 4


def save_mock_image(records):
    saved_path = "/Users/uejun/Dropbox/projects/Oruche/RecoReco/mock_images"
    for i, r in enumerate(records):
        filename = str(r["label"]) + "_" + str(i) + ".png"
        fullpath = os.path.join(saved_path, filename)
        cv2.imwrite(fullpath, r["image"])


def load_mock_image():
    saved_path = "/Users/uejun/Dropbox/projects/Oruche/RecoReco/mock_images"
    files = os.listdir(saved_path)
    images = []
    labels = []
    for file in files:
        if not file.endswith('png'):
            continue
        filepath = os.path.join(saved_path, file)
        label = int(file.split('_')[0])
        img = cv2.imread(filepath)

        images.append(img.flatten().astype(np.float32) / 255.0)

        # ラベルを1-of-k方式で用意する
        num_classes = 2
        tmp = np.zeros(num_classes)
        tmp[label] = 1
        labels.append(tmp)
    return images, labels


def _create_one_hot_labels(labels):
    # ラベルを1-of-k方式で用意する
    num_classes = len(labels)
    indices = {str(label): i for i, label in enumerate(labels)}

    one_hot_labels = []
    for label in labels:
        one_hot = np.zeros(num_classes).astype(np.float32)
        index = indices[label]
        one_hot[index] = np.float32(1.0)
        one_hot_labels.append(one_hot)

    return one_hot_labels


