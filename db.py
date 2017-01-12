import decimal
import boto3

from entity import ItemSchema

session = boto3.Session(profile_name="ynu_ueda")
dynamodb = session.resource("dynamodb", region_name='ap-northeast-1')
table = dynamodb.Table('Recognitions')


def put_item(item):
    item.learning_rate = decimal.Decimal(str(item.learning_rate))
    table.put_item(
        Item=item.__dict__
    )


def get_item(intelligence_id):
    resp = table.get_item(
        Key={
            'intelligence_id': intelligence_id
        }
    )
    return ItemSchema().load(resp["Item"]).data


