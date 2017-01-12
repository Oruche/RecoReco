import boto3

session = boto3.Session(profile_name="ynu_ueda")
s3 = session.resource('s3')
bucket_name = "aigomodel"


def save_model(filename: str, objectkey: str):
    s3.Bucket(bucket_name).upload_file(filename, objectkey)


def download_model(filename: str, objectkey: str):
    s3.Bucket(bucket_name).download_file(objectkey, filename)



