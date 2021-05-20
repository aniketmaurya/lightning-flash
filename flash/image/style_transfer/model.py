# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, cast, Dict, List, Mapping, NoReturn, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _IMAGE_STLYE_TRANSFER
from flash.image.style_transfer import STYLE_TRANSFER_BACKBONES

if _IMAGE_STLYE_TRANSFER:
    import pystiche.demo
    from pystiche import enc, loss, ops
    from pystiche.image import read_image
else:

    class enc:
        Encoder = None
        MultiLayerEncoder = None

    class ops:
        EncodingComparisonOperator = None
        FeatureReconstructionOperator = None
        MultiLayerEncodingOperator = None

    class loss:

        class PerceptualLoss:
            pass


from flash.image.style_transfer.utils import raise_not_supported

__all__ = ["StyleTransfer"]


class StyleTransfer(Task):
    """Task that transfer the style from an image onto another.

    Example::

        from flash.image.style_transfer import StyleTransfer

        model = StyleTransfer(image_style)

    Args:
        style_image: Image or path to an image to derive the style from.
        model: The model by the style transfer task.
        backbone: A string or model to use to compute the style loss from.
        content_layer: Which layer from the backbone to extract the content loss from.
        content_weight: The weight associated with the content loss. A lower value will lose content over style.
        style_layers: Layers from the backbone to derive the style loss from.
        optimizer: Optimizer to use for training the model.
        optimizer_kwargs: Optimizer keywords arguments.
        scheduler: Scheduler to use for training the model.
        scheduler_kwargs: Scheduler keywords arguments.
        learning_rate: Learning rate to use for training, defaults to ``1e-3``.
        serializer: The :class:`~flash.core.data.process.Serializer` to use when serializing prediction outputs.
    """

    backbones: FlashRegistry = STYLE_TRANSFER_BACKBONES

    def __init__(
        self,
        style_image: Optional[Union[str, torch.Tensor]] = None,
        model: Optional[nn.Module] = None,
        backbone: str = "vgg16",
        content_layer: str = "relu2_2",
        content_weight: float = 1e5,
        style_layers: Union[Sequence[str], str] = ("relu1_2", "relu2_2", "relu3_3", "relu4_3"),
        style_weight: float = 1e10,
        optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-3,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
    ):

        if not _IMAGE_STLYE_TRANSFER:
            raise ModuleNotFoundError("Please, pip install -e '.[image_style_transfer]'")

        self.save_hyperparameters(ignore="style_image")

        if style_image is None:
            style_image = self.default_style_image()
        elif isinstance(style_image, str):
            style_image = read_image(style_image)

        if model is None:
            model = pystiche.demo.transformer()

        if not isinstance(style_layers, (List, Tuple)):
            style_layers = (style_layers, )

        perceptual_loss = self._get_perceptual_loss(
            backbone=backbone,
            content_layer=content_layer,
            content_weight=content_weight,
            style_layers=style_layers,
            style_weight=style_weight,
        )
        perceptual_loss.set_style_image(style_image)

        super().__init__(
            model=model,
            loss_fn=perceptual_loss,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            learning_rate=learning_rate,
            serializer=serializer,
        )

        self.perceptual_loss = perceptual_loss

    def default_style_image(self) -> torch.Tensor:
        return pystiche.demo.images()["paint"].read(size=256)

    @staticmethod
    def _modified_gram_loss(encoder: enc.Encoder, *, score_weight: float) -> ops.EncodingComparisonOperator:
        # The official PyTorch examples as well as the reference implementation of the original author contain an
        # oversight: they normalize the representation twice by the number of channels. To be compatible with them, we
        # do the same here.
        class GramOperator(ops.GramOperator):

            def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
                repr = super().enc_to_repr(enc)
                num_channels = repr.size()[1]
                return repr / num_channels

        return GramOperator(encoder, score_weight=score_weight)

    def _get_perceptual_loss(
        self,
        *,
        backbone: str,
        content_layer: str,
        content_weight: float,
        style_layers: Sequence[str],
        style_weight: float,
    ) -> loss.PerceptualLoss:
        mle, _ = cast(enc.MultiLayerEncoder, self.backbones.get(backbone)())
        content_loss = ops.FeatureReconstructionOperator(
            mle.extract_encoder(content_layer), score_weight=content_weight
        )
        style_loss = ops.MultiLayerEncodingOperator(
            mle,
            style_layers,
            lambda encoder, layer_weight: self._modified_gram_loss(encoder, score_weight=layer_weight),
            layer_weights="sum",
            score_weight=style_weight,
        )
        return loss.PerceptualLoss(content_loss, style_loss)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        input_image = batch[DefaultDataKeys.INPUT]
        self.perceptual_loss.set_content_image(input_image)
        output_image = self(input_image)
        return self.perceptual_loss(output_image).total()

    def validation_step(self, batch: Any, batch_idx: int) -> NoReturn:
        raise_not_supported("validation")

    def test_step(self, batch: Any, batch_idx: int) -> NoReturn:
        raise_not_supported("test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Any:
        input_image = batch[DefaultDataKeys.INPUT]
        return self(input_image)
