from typing import Any, Callable, List, Type, Union

import torch
from pl_bolts.models.gans import Pix2Pix
from pl_bolts.models.gans.pix2pix.pix2pix_module import Generator, PatchGAN
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.nn import functional as F

from flash.core.generation import GenerationTask
from flash.core.registry import FlashRegistry
from flash.data.data_source import DefaultDataKeys

PAIRED_IMG_TRANSLATION_BACKBONES = FlashRegistry("backbones")


def load_pix2pix():
    model = Pix2Pix(3, 3)
    gen = model.gen
    disc = model.patch_gan
    return dict(generator=gen, discriminator=disc)


PAIRED_IMG_TRANSLATION_BACKBONES(fn=load_pix2pix, name="pix2pix", namespace="gan", package="bolts")


class PairedImageTranslation(GenerationTask):
    backbones: FlashRegistry = PAIRED_IMG_TRANSLATION_BACKBONES

    def __init__(
        self,
        generator: Union[nn.Module, str] = Generator,
        discriminator: Union[nn.Module, str] = PatchGAN,
        loss_fn: Union[Callable, List[Callable]] = F.cross_entropy,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD,
        gen_kwargs=None,
        disc_kwargs=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if not gen_kwargs:
            gen_kwargs = {}

        if not disc_kwargs:
            disc_kwargs = {}

        self.model = {}
        self._load_network(generator, PairedImageTranslation.GENERATOR, **gen_kwargs)
        self._load_network(discriminator, PairedImageTranslation.DISCRIMINATOR, **disc_kwargs)

    def _load_network(self, model: Union[nn.Module, str], network_type: str, **model_kwargs):
        if isinstance(model, nn.Module):
            self.model[network_type] = model
        elif isinstance(model, str):
            # TODO: load model with model name
            self.model[network_type] = None
        else:
            raise MisconfigurationException(f"backbone should be either a string or a nn.Module. Found: {model}")

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().training_step(batch, batch_idx)
