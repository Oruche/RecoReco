from unittest import TestCase

import config
import train_data


class TestData(TestCase):
    def setUp(self):
        config.load("dev", filename="../config.yml")

    def test_prepare(self):
        labels = ["rice", "lemon"]
        data_count = 4
        image_size = 56

        train_image, train_label = train_data.prepare(labels, data_count, image_size)

