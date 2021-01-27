import torch

from flash.vision import ImageClassificationData, ImageClassifier


def _dummy_image_loader(filepath):
    return torch.rand(3, 224, 224)


def test_classification(tmpdir):
    data = ImageClassificationData.from_filepaths(
        train_filepaths=["a", "b"],
        train_labels=[0, 1],
        train_transform=lambda x: x,
        loader=_dummy_image_loader,
        num_workers=0,
    )
    model = ImageClassifier(2, backbone="resnet18")
    model.fit(data, fast_dev_run=True, default_root_dir=tmpdir)