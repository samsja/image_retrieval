from image_retrieval.script.train_classification import train


def test_full_training(data_path):

    train(epoch=1, data_path=data_path, debug=True)
