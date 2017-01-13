import json

import config
import db
import entity
from entity import IntelligenceID
import storage
import train_data
import training


def train(intelligence_id):
    # intelligence_idに対応するモデルをDBから検索
    item = db.get_item(intelligence_id)

    # モデルをストレージからダウンロード
    filename = './model_tmp/downloads/' + item.modelkey
    storage.download_model(filename, item.modelkey)


    # 学習ハイパーパラメータをロード
    # 画像をダウンロード
    #train_image, train_label = data.load_mock_image()
    train_image, train_label = train_data.prepare(labels=item.labels,
                                                  count=item.data_counts,
                                                  image_size=item.image_size,
                                                  color=item.color,
                                                  shuffle=True,
                                                  flatten=True)

    # train
    savename = './model_tmp/saved/' + item.modelkey
    training.execute_train(train_image, train_label, item, loadname=filename, savename=savename)

    # ストレージに保存
    storage.save_model(savename, item.modelkey)

    # DBに保存
    db.put_item(item)


def first_train(labels):
    intelligence_id = IntelligenceID.create()
    image_size = 56
    data_counts = 4
    num_classes = len(labels)

    train_image, train_label = train_data.load_mock_image()

    max_steps = 2
    batch_size = 1
    learning_rate = 0.0001

    filename = "./model_tmp/saved/" + intelligence_id + "_model.ckpt"
    model_objectkey = intelligence_id + "_model.ckpt"
    bucket = "aigomodel"

    item = entity.Item(**{
        "intelligence_id": intelligence_id,
        "modelkey":  model_objectkey,
        "bucket": bucket,
        "labels": labels,
        "data_counts": data_counts,
        "image_size": image_size,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "color": True
    })

    training.execute_train(train_image, train_label, item, savename=filename)

    storage.save_model(filename, model_objectkey)

    db.put_item(item)


def train_test():
    intelligence_id = IntelligenceID.create()

    with open('model_sample.json', 'r') as f:
        model_blueprint = json.load(f)


    #train_image, train_label = train_data.load_mock_image()
    train_image, train_label = train_data.prepare(model_blueprint["labels"],
                                                  count=4,
                                                  image_size=model_blueprint["image_size"][0],
                                                  color=True,
                                                  shuffle=True,
                                                  flatten=True)


    filename = "./model_tmp/saved/" + intelligence_id + "_model.ckpt"
    model_objectkey = intelligence_id + "_model.ckpt"
    bucket = "aigomodel"

    training.execute_train(train_image, train_label, model_blueprint, savename=filename)


if __name__ == '__main__':
    config.load("dev")
    train_test()
    #first_train(["rice", "lemon"])
    #train("cb0f61d569e44e0186a8a378d43d6471_170111133024908199")

