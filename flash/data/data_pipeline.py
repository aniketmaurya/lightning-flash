import os
from functools import wraps
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data._utils.collate import default_collate, default_convert
from torch.utils.data.dataloader import DataLoader

from flash.data.auto_dataset import AutoDataset
from flash.data.batch import _PostProcessor, _PreProcessor, default_uncollate


class Preprocess:

    def load_data(self, data: Any) -> Any:
        """Loads entire data from Dataset"""
        return data

    def load_sample(self, sample: Any) -> Any:
        """Loads single sample from dataset"""
        return sample

    def pre_collate(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis)"""
        return sample

    def post_collate(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency)

        .. note::
            This option is mutually exclusive with :meth:`device_pre_collate`, since if both are specified, uncollation has to be applied.
        """
        return batch

    def device_pre_collate(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).

        .. note::
            This option is mutually exclusive with :meth:`post_collate`, since if both are specified, uncollation has to be applied.

        .. note::
            This function won't be called within the dataloader workers, since to make that happen each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return sample

    def device_post_collate(self, batch: Any) -> Any:
        """
        Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::
            This function won't be called within the dataloader workers, since to make that happen each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return batch


class Postprocess:

    def __init__(self, save_path: Optional[str] = None):
        self._saved_samples = 0
        self._save_path = save_path

    def pre_uncollate(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch before uncollation to single samples.
        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return batch

    def post_uncollate(self, sample: Any) -> Any:
        """Transforms to apply to a single sample after splitting up the batch.
        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return sample

    def uncollate(self, batch: Any) -> Any:
        """Uncollates a batch into single samples.
        Tries to preserve the type whereever possible.
        """
        return default_uncollate(batch)

    def save_data(self, data: Any, path: str) -> None:
        """Saves all data together to a single path.
        """
        torch.save(data, path)

    def save_sample(self, sample: Any, path: str) -> None:
        """Saves each sample individually to a given path.
        """
        torch.save(sample, path)

    # TODO: Are those needed ?
    def format_sample_save_path(self, path: str) -> str:
        path = os.path.join(path, f'sample_{self._saved_samples}.ptl')
        self._saved_samples += 1
        return path

    def _save_data(self, data: Any) -> None:
        self.save_data(data, self._save_path)

    def _save_sample(self, sample: Any) -> None:
        self.save_sample(sample, self.format_sample_save_path(self._save_path))


class DataPipeline:

    PREPROCESS_FUNCS = ("load_data", "load_sample", "pre_collate", "post_collate", "device_post_collate")
    POSTPROCESS_FUNCS = ("pre_uncollate", "post_uncollate", "save_data", "save_sample")
    LOADERS_PREFIX = ('train', 'test', 'val', 'predict')

    def __init__(self, preprocess: Preprocess, postprocess: Postprocess):
        self._preprocess_pipeline = preprocess
        self._postprocess_pipeline = postprocess
        self._worker_preprocessor = None
        self._device_preprocessor = None
        self._postprocessor = None

    def load_data(self, data: Any) -> Any:
        """Loads entire data from Dataset"""
        return self.preprocess.load_data(data)

    def load_sample(self, sample: Any) -> Any:
        """Loads single sample from dataset"""
        return self.preprocess.load_sample(sample)

    def pre_collate(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis)"""
        return self.preprocess.pre_collate(sample)

    def post_collate(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency)

        .. note::
            This option is mutually exclusive with :meth:`device_pre_collate`, since if both are specified, uncollation has to be applied.
        """
        return self.preprocess.post_collate(batch)

    def device_pre_collate(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).

        .. note::
            This option is mutually exclusive with :meth:`post_collate`, since if both are specified, uncollation has to be applied.

        .. note::
            This function won't be called within the dataloader workers, since to make that happen each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return self.preprocess.device_pre_collate(sample)

    def device_post_collate(self, batch: Any) -> Any:
        """
        Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::
            This function won't be called within the dataloader workers, since to make that happen each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return self.preprocess.device_pre_collate(batch)

    def pre_uncollate(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch before uncollation to single samples.
        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return self.postprocess.pre_uncollate(batch)

    def post_uncollate(self, sample: Any) -> Any:
        """Transforms to apply to a single sample after splitting up the batch.
        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return self.postprocess.post_uncollate(sample)

    def uncollate(self, batch: Any) -> Any:
        """Uncollates a batch into single samples.
        Tries to preserve the type whereever possible.
        """
        return self.postprocess.uncollate(batch)

    def save_data(self, data: Any, path: str) -> None:
        """Saves all data together to a single path.
        """
        self.postprocess.save_data(data, path)

    def save_sample(self, sample: Any, path: str) -> None:
        """Saves each sample individually to a given path.
        """
        self.postprocess.save_sample(sample, path)

    def _is_overriden(self, method_name: str, super_obj: Any) -> bool:
        """Cropped Version of https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/model_helpers.py
        """
        process_obj = self.preprocess if isinstance(self.preprocess, super_obj) else self.postprocess

        if not hasattr(process_obj, method_name) or not hasattr(super_obj, method_name):
            return False

        return getattr(process_obj, method_name).__code__ != getattr(super_obj, method_name).__code__

    @staticmethod
    def _do_nothing_collate(samples: Sequence[Any]) -> Sequence[Any]:
        return samples

    @staticmethod
    def _do_nothing_uncollate(batch: Any) -> Any:
        return batch

    @property
    def worker_preprocessor(self) -> _PreProcessor:
        if self._worker_preprocessor is None:
            self._worker_preprocessor = self._create_collate_preprocessors()[0]
        return self._worker_preprocessor

    @worker_preprocessor.setter
    def worker_preprocessor(self, new_processor: _PreProcessor):
        self._worker_preprocessor = new_processor

    @property
    def device_preprocessor(self) -> _PreProcessor:
        if self._device_preprocessor is None:
            self._device_preprocessor = self._create_collate_preprocessors()[1]
        return self._device_preprocessor

    @device_preprocessor.setter
    def device_preprocessor(self, new_processor: _PreProcessor):

        self._device_preprocessor = new_processor

    @property
    def postprocessor(self) -> _PostProcessor:
        if self._postprocessor is None:
            self._postprocessor = self._create_uncollate_postprocessors()

        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, new_processor: _PostProcessor):
        self._postprocessor = new_processor

    def _create_collate_preprocessors(self,
                                      collate_fn: Optional[Callable] = None) -> Tuple[_PreProcessor, _PreProcessor]:
        if collate_fn is None:
            collate_fn = default_collate

        post_collate_overriden = self._is_overriden('post_collate', Preprocess)

        device_pre_collate_overriden = self._is_overriden('device_pre_collate', Preprocess)

        if post_collate_overriden and device_pre_collate_overriden:
            raise MisconfigurationException(
                f'{self.__class__.__name__}: post_collate and gpu_pre_collate are mutual exclusive.'
            )

        elif post_collate_overriden:
            worker_collate_fn = collate_fn
            device_collate_fn = self._do_nothing_collate

        elif device_pre_collate_overriden:
            worker_collate_fn = self._do_nothing_collate
            device_collate_fn = collate_fn

        else:
            worker_collate_fn = collate_fn
            device_collate_fn = self._do_nothing_collate

        worker_preprocessor = _PreProcessor(worker_collate_fn, self.pre_collate, self.post_collate)
        device_preprocessor = _PreProcessor(device_collate_fn, self.device_pre_collate, self.device_post_collate)
        return worker_preprocessor, device_preprocessor

    @staticmethod
    def _model_transfer_to_device_wrapper(func: Callable, preprocessor: _PreProcessor) -> Callable:

        @wraps(func)
        def new_func(*args, **kwargs):
            moved_to_device = func(*args, **kwargs)
            return preprocessor(moved_to_device)

        return new_func

    @staticmethod
    def _model_predict_step_wrapper(func: Callable, uncollater: _PostProcessor) -> Callable:

        @wraps(func)
        def new_func(*args, **kwargs):
            predicted = func(*args, **kwargs)
            predicted = uncollater(predicted)
            return predicted

        return new_func

    def _attach_preprocess_to_model(self, model: 'Task', loader_stage: str = 'all') -> None:
        if loader_stage == 'all':
            loader_stage = self.LOADERS_PREFIX

        elif isinstance(loader_stage, str):
            loader_stage = [loader_stage]

        for stage in loader_stage:
            loader_name = f'{stage}_loader'

            if hasattr(model, loader_name):
                dataloader = getattr(model, loader_name)

                if isinstance(dataloader, _PatchDataLoader):
                    wrap_patch_loader = True
                    dataloader = dataloader()

                else:
                    wrap_patch_loader = False

                if isinstance(dataloader, Sequence):
                    was_seq = True
                else:
                    dataloader = [dataloader]
                    was_seq = False

                for idx, loader in enumerate(dataloader):
                    if isinstance(loader, DataLoader):
                        dl_args = {k: v for k, v in vars(loader).items() if not k.startswith("_")}

                        dl_args['collate_fn'], device_collate_fnr = self.split_around_collate(
                            collate_fn=dl_args['collate_fn']
                        )

                        loader = type(loader)(**dl_args)

                    dataloader[idx] = loader

                if not was_seq:
                    dataloader = dataloader[0]

                if wrap_patch_loader:
                    dataloader = _PatchDataLoader(dataloader)

                setattr(model, loader_name, dataloader)

        model.transfer_batch_to_device = (
            self._model_transfer_to_device_wrapper(model.transfer_batch_to_device, device_collate_fnr)
        )

    def _create_uncollate_postprocessors(self, uncollate_fn: Optional[Callable] = None) -> _PostProcessor:
        save_per_sample = None
        save_fn = None

        if self.postprocess._save_path is not None:
            save_per_sample = self._is_overriden('save_sample', Postprocess)

            if save_per_sample:
                save_fn = self.postprocess._save_sample
            else:
                save_fn = self.postprocess._save_data

        return _PostProcessor(
            self.uncollate, self.pre_uncollate, self.post_uncollate, save_fn=save_fn, save_per_sample=save_per_sample
        )

    def _attach_postprocess_to_model(self, model: 'Task') -> 'Task':
        # TODO: move this to on_predict_end?
        model.predict_step = self._model_predict_wrapper(model.predict_step, self.uncollate_fn)
        return model

    def _attach_to_model(self, model: 'Task', loader_stage: str = 'all'):
        model._preprocess = self.preprocess
        model._postprocess = self.postprocess
        self._attach_preprocess_to_model(model, loader_stage)
        self._attach_postprocess_to_model(model)

    def _generate_auto_dataset(self, data: Union[Iterable, Any]) -> AutoDataset:
        return AutoDataset(
            data=data,
            load_data=self.load_data,
            load_sample=self.load_sample,
            load_data_overriden=self._is_overriden("load_data", Preprocess),
            load_sample_overriden=self._is_overriden("load_sample", Preprocess),
        )

    def to_dataloader(
        self, data: Union[Iterable, Any], auto_collate: Optional[bool] = None, **loader_kwargs
    ) -> DataLoader:
        if 'collate_fn' in loader_kwargs:
            if auto_collate is not None:
                raise MisconfigurationException('auto_collate and collate_fn are mutually exclusive')

        else:
            if auto_collate is None:
                auto_collate = True

            collate_fn = self.worker_collate_fn

            if collate_fn is not None:
                loader_kwargs['collate_fn'] = collate_fn

            else:
                if auto_collate:
                    loader_kwargs['collate_fn'] = default_collate
                else:
                    loader_kwargs['collate_fn'] = default_convert

        return DataLoader(self._generate_auto_dataset(data), **loader_kwargs)
