from unittest import TestCase
from testfixtures import compare

import cv2

import config
import train_data
from train_data import TrainImageFactory, NotMatchedChannelErr


class TestPrepare(TestCase):
    def setUp(self):
        config.load("dev", "../config.yml")

    def test_prepare(self):
        labels = ["rice", "lemon"]
        count = 4
        image_size = 56
        shuffle = True
        flatten = False

        train_images, train_labels = train_data.prepare(labels=labels,
                                                        count=count,
                                                        image_size=image_size,
                                                        shuffle=shuffle,
                                                        flatten=flatten)

        self.assertEqual(len(train_images), len(train_labels))

class TestNumLabelConverter(TestCase):
    def test_to_numlabels(self):
        labels = ["banana", "apple", "grape"]
        strlabels = ["apple", "banana", "banana", "grape"]
        converter = train_data.NumLabelConverter(labels)
        numlabels = converter.to_numlabels(strlabels)
        compare(numlabels, [1, 0, 0, 2])


class TestTrainImageFactory(TestCase):
    def setUp(self):
        self.url1 = "https://s3-ap-northeast-1.amazonaws.com/aigoimg/02/e638ef11-cb45-4b7f-a487-23a1fd64b15b_1_1_161229152921899107.jpg"
        self.img_1ch = cv2.imread("../mock_images/0_1.png", 0)
        self.img_3ch = cv2.imread("../mock_images/0_1.png", 1)
        self.img_4ch = cv2.cvtColor(self.img_3ch, cv2.COLOR_BGR2BGRA)

    def test_create(self):
        image_size = 56
        factory = TrainImageFactory(image_size, color=True, flatten=False)
        img = factory.create(self.url1)



    def test_fit_color(self):
        image_size = 56

        f1 = TrainImageFactory(image_size, color=False, flatten=True)
        img1 = f1.fit_color(self.img_3ch)
        self.assertEqual(len(img1.shape), 2)

        f2 = TrainImageFactory(image_size, color=True, flatten=True)
        img2 = f2.fit_color(self.img_4ch)
        self.assertEqual(img2.shape[2], 3)

        with self.assertRaises(NotMatchedChannelErr):
            f3 = TrainImageFactory(image_size, color=True, flatten=True)
            f3.fit_color(self.img_1ch)

    def test_flatten(self):
        image_size = 56
        color = True

        f1 = TrainImageFactory(image_size, color, flatten=True)
        img = f1.create(self.url1)
        self.assertEqual(len(img.shape), 1)

        f2 = TrainImageFactory(image_size, color, flatten=False)
        img = f2.create(self.url1)
        self.assertEqual(len(img.shape), 3)
