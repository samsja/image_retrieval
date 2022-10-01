from image_retrieval.data import CIFAR100


def test_cifar(data_path):

    data = CIFAR100(root_path=data_path, debug=True)

    data.setup()

    sample, _ = data.train_dataset[0]
    assert sample.shape == (3, 32, 32)

    sample, _ = data.test_dataset[0]
    assert sample.shape == (3, 32, 32)
