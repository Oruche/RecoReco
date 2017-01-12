import uuid

import arrow
from marshmallow import Schema, fields, post_load


class ItemSchema(Schema):
    intelligence_id = fields.String()
    modelkey = fields.String()
    bucket = fields.String()
    data_counts = fields.Int()
    labels = fields.List(fields.String())
    image_size = fields.Int()
    max_steps = fields.Int()
    batch_size = fields.Int()
    learning_rate = fields.Float()
    color = fields.Bool()

    @post_load
    def make_item(self, data):
        return Item(**data)


class Item(object):

    def __init__(self, **entries):
        self.intelligence_id = None
        self.modelkey = None
        self.bucket = None
        self.labels = None
        self.data_counts = None
        self.image_size = None
        self.max_steps = None
        self.batch_size = None
        self.learning_rate = None
        self.color = None
        self.__dict__.update(entries)

    def __str__(self):
        return ItemSchema().dumps(self).data


class IntelligenceID:
    uuidhex = uuid.uuid4().hex

    @classmethod
    def create(cls):
        now = arrow.now('Asia/Tokyo')
        now_str = now.format('YYMMDDHHmmss') + str(now.microsecond)
        id = cls.uuidhex + "_" + now_str
        return id



