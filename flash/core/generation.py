from typing import Any

from flash.core.model import Task


class GenerationTask(Task):
    GENERATOR = "GENERATOR"
    DISCRIMINATOR = "DISCRIMINATOR"

    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:

        super().__init__(
            *args,
            **kwargs,
        )

    def forward(self, x):
        return self.model[GenerationTask.GENERATOR](x)

    def step(self, batch: Any, batch_idx: int, optimizer_idx: int) -> Any:
        x, y = batch
        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(x, y)
            self.log('Discriminator Loss', loss)
        elif optimizer_idx == 1:
            loss = self._gen_step(x, y)
            self.log('Generator Loss', loss)

        return loss
