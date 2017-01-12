import sys
import yaml

from marshmallow import Schema, fields, post_load

config = None


class Config:
    def __init__(self, ipersistence_endpoint, service_domain):
        self.ipersistence_endopoint = ipersistence_endpoint
        self.service_domain = service_domain
        self.search_endpoint = ipersistence_endpoint + "/images/_search"


class ConfigSchema(Schema):
    ipersistence_endpoint = fields.Url()
    service_domain = fields.Url()

    @post_load
    def make_config(self, data):
        return Config(**data)


def load(env, filename='config.yml'):
    try:
        with open(filename, 'r') as f:
            data = yaml.load(f)
            result = ConfigSchema().load(data[env])
            global config
            config = result.data
    except Exception as e:
        print(e)
        sys.exit(1)






