import gc
import os
import threading
from functools import wraps, partial
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from cv2 import (applyColorMap, COLORMAP_JET, COLORMAP_BONE, COLORMAP_COOL, COLORMAP_HOT)
from torchvision.utils import make_grid

import wandb
from mnts.mnts_logger import MNTSLogger
from pytorch_med_imaging.utils.visualization.segmentation_vis import draw_grid

__all__ = ['WNB_Plotter']


def check_init(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if wandb.run is None:
            self._logger.error("W&B run has not been initialized. Have you called 'init_run()'?")
            return
        return func(self, *args, **kwargs)
    return wrapper


class WNB_Plotter:
    """Plotter for PMI to log information to Weights & Biases (wandb)."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(WNB_Plotter, cls).__new__(cls)
        return cls._instance

    def __init__(self, project: str = None, entity: str = None, api_key: str = None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._logger = MNTSLogger[self.__class__.__name__]

        self.project = project or os.environ.get('WANDB_PROJECT', '')
        self.entity = entity or os.environ.get('WANDB_ENTITY', '')
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key

        self._last_writer_index = 0
        self._registered_module_config = {}
        self._write_n_iteration = 1000
        self._image_max_dim = (128, 128)

        self._initialized = True

    @classmethod
    def get_plotter(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            if MNTSLogger.global_logger:
                MNTSLogger.global_logger.warning("No W&B plotter instance exists.")
            return None
        return cls._instance

    def get_writer(self):
        return wandb.run

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def init_run(self, init_meta: Optional[dict] = None) -> None:
        if wandb.run is not None:
            self._logger.warning("An active W&B run already exists. Finishing it before starting a new one.")
            wandb.finish()
        init_meta = dict(init_meta or {})
        # Allow plotter_init_meta to override the instance-level project/entity
        project = init_meta.pop('project', self.project or None)
        entity  = init_meta.pop('entity',  self.entity  or None)
        self._logger.info("Initializing W&B run.")
        wandb.init(project=project, entity=entity, **init_meta)
        self._logger.info(f"W&B run initialized: {wandb.run.name} (id={wandb.run.id})")

    def continue_run(self, run_id: str, init_meta: Optional[dict] = None) -> None:
        """Resume an existing W&B run by its run ID."""
        init_meta = dict(init_meta or {})
        project = init_meta.pop('project', self.project or None)
        entity  = init_meta.pop('entity',  self.entity  or None)
        self._logger.info(f"Resuming W&B run: {run_id}")
        wandb.init(project=project, entity=entity, id=run_id, resume='must', **init_meta)

    def stop(self) -> None:
        if wandb.run is not None:
            self._logger.info("Finishing W&B run.")
            wandb.finish()

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------

    @check_init
    def log_dict(self, scalar_dict: dict, step: Optional[int] = None) -> None:
        """Log a dictionary of scalar key-value pairs."""
        flat = {}
        for k, v in scalar_dict.items():
            if isinstance(v, (list, tuple)):
                for i, vv in enumerate(v):
                    flat[f"{k}/{i}"] = vv
            else:
                flat[k] = v
        kwargs = {'step': step} if step is not None else {}
        wandb.log(flat, **kwargs)

    @check_init
    def log_scalar(self, label_or_index, value_or_scalars, label=None, step: Optional[int] = None) -> int:
        """Log a scalar or a dict of scalars.

        Supports two calling conventions:
            log_scalar(label: str, value: float)              # simple style (used by SolverBase)
            log_scalar(writer_index: int, scalars: dict, ...) # TB_plotter style
        """
        if isinstance(label_or_index, int):
            # TB_plotter-style: log_scalar(writer_index, scalars_dict, label)
            writer_index = label_or_index
            scalars = value_or_scalars
            self._last_writer_index = writer_index
            try:
                wandb.log(scalars, step=writer_index)
            except Exception:
                self._logger.error(f"Error logging scalars: {scalars}")
                return 1
            return 0
        else:
            # simple style: log_scalar(label, value)
            import re
            lbl = label_or_index
            value = value_or_scalars
            if isinstance(value, str):
                if re.fullmatch(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$', value) is not None:
                    value = float(value)
            try:
                value = float(value)
            except Exception:
                value = str(value)
            self._logger.info(f"Logging scalar: {lbl}: {value}")
            kwargs = {'step': step} if step is not None else {}
            wandb.log({lbl: value}, **kwargs)
            return 0

    @check_init
    def plot_loss(self, loss: float, writer_index: int) -> None:
        self._last_writer_index = writer_index
        wandb.log({'loss': loss}, step=writer_index)

    @check_init
    def plot_validation_loss(self, writer_index: int, *args) -> float:
        self._last_writer_index = writer_index
        payload = {'Validation_Loss': args[0]}
        if len(args) >= 2:
            payload['Accuracies'] = args[1]
        wandb.log(payload, step=writer_index)
        return args[0]

    # ------------------------------------------------------------------
    # Histogram logging
    # ------------------------------------------------------------------

    @check_init
    def plot_weight_histogram(self, net: torch.nn.Module, writer_index: int) -> None:
        self._last_writer_index = writer_index
        payload = {}
        for name, m in net.named_modules():
            if hasattr(m, 'weight') and m.weight is not None:
                key = name.replace('.', '/') + '/weights'
                payload[key] = wandb.Histogram(m.weight.detach().cpu().flatten().numpy())
        wandb.log(payload, step=writer_index)

    @check_init
    def plot_histogram(self, values: torch.Tensor, name: str, writer_index: int) -> None:
        self._last_writer_index = writer_index
        wandb.log({name: wandb.Histogram(values.cpu().flatten().numpy())}, step=writer_index)

    # ------------------------------------------------------------------
    # Image logging
    # ------------------------------------------------------------------

    @check_init
    def add_image(self, key: str, image, step: Optional[int] = None, **kwargs) -> None:
        try:
            if isinstance(image, np.ndarray):
                if image.ndim < 2:
                    raise ArithmeticError(f"Trying to add image with wrong shape: {image.shape}")
                # wandb.Image expects HWC; convert CHW if needed
                if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                    image = image.transpose(1, 2, 0)
            log_kwargs = {'step': step} if step is not None else {}
            wandb.log({key: wandb.Image(image, **kwargs)}, **log_kwargs)
        except Exception as e:
            self._logger.error(e)

    @check_init
    def plot_tensor(self,
                    tensor: torch.Tensor,
                    name: str,
                    writer_index: int,
                    cmap: Optional[str] = 'jet',
                    grid_by: Optional[str] = 'batch') -> None:
        _cmap = {
            'jet': COLORMAP_JET,
            'bone': COLORMAP_BONE,
            'cool': COLORMAP_COOL,
            'hot': COLORMAP_HOT,
        }
        assert cmap in _cmap, f"Available cmaps: [{', '.join(_cmap.keys())}], got `{cmap}`."

        _axis = {'batch': 0, 'ch': 1, 'slice': 4}
        assert grid_by in _axis, f"Available grid axes: [{', '.join(_axis.keys())}], got `{grid_by}`."

        if tensor.dim() == 4:
            _grid = make_grid(tensor, nrow=5, normalize=True)
            _img = (_grid * 255.).permute(1, 2, 0).numpy().astype('uint8')
            wandb.log({name: wandb.Image(_img)}, step=writer_index)

        elif tensor.dim() == 5:
            _a_grid = _axis[grid_by]
            _a_fixed = [_axis[k] for k in _axis if k != grid_by]
            _index = [tensor.shape[i] // 2 if i in _a_fixed else slice(None) for i in range(tensor.dim())]
            _tensor = tensor[tuple(_index)]
            _tensor = _tensor.unsqueeze(_a_fixed[0]).unsqueeze(_a_fixed[1])
            _tensor = _tensor.transpose(_a_grid, 0).squeeze().unsqueeze(1)

            _size = np.min(np.asarray([self._image_max_dim, _tensor.shape[-2:]]), axis=0)
            self._logger.debug(f"Resized from {_tensor.shape} -> {_size}")
            _tensor = torch.nn.functional.adaptive_avg_pool2d(_tensor, _size)

            _g = make_grid(_tensor, nrow=5, normalize=True).unsqueeze(0)
            _g = (_g * 254.).squeeze()[0].numpy().astype('uint8')
            _g = applyColorMap(_g, _cmap[cmap])  # returns BGR (HWC)
            _g = _g[..., ::-1]                    # BGR -> RGB
            wandb.log({name: wandb.Image(_g)}, step=writer_index)

    @check_init
    def plot_segmentation(self,
                          gt: torch.Tensor,
                          out: torch.Tensor,
                          img: Union[torch.Tensor, list],
                          writer_index: int,
                          Zrange: int = 40,
                          nrow: int = 3) -> None:
        self._last_writer_index = writer_index
        try:
            shape = gt.shape
            dim = sum([s > 1 for s in shape[2:]])

            if dim == 3:
                gtsum = torch.sum(gt, dim=[2, 3, 4]).squeeze()
                if gtsum.sum() == 0:
                    self._logger.warning("Mini-batch has no labels in this epoch.")
                b_index = torch.argmax(gtsum)
                gt = gt[b_index]
                out = out[b_index]
                img = img[b_index]

                Zrange = out.shape[-1] if out.shape[-1] < 40 else Zrange
                ar = torch.argmax(out, 0, keepdim=True)
                ss = img[0] if isinstance(img, list) else img

                ss = ss[..., :Zrange].permute(3, 0, 1, 2)
                ar = ar[..., :Zrange].permute(3, 0, 1, 2)
                gt = gt[..., :Zrange].permute(3, 0, 1, 2)

            elif dim == 2:
                gtsum = torch.sum(gt, dim=list(range(2, gt.dim()))).squeeze()
                if gtsum.sum() == 0:
                    self._logger.warning("Mini-batch has no labels in this iteration.")

                gt = WNB_Plotter._collapse_to_2d(gt)
                img = WNB_Plotter._collapse_to_2d(img)

                Zrange = out.shape[0] if out.shape[0] < 40 else Zrange
                ar = torch.argmax(out, 1)
                ss = img[0] if isinstance(img, list) else img

                ss = ss[:Zrange]
                ar = ar[:Zrange]
                gt = gt[:Zrange]
            else:
                raise IndexError(f"Unexpected label dimensions: {gt.shape}. Expected dim 2 or 3, got {dim}.")

            grid = draw_grid(ss, ar, ground_truth=gt, thickness=2)  # HWC numpy array
            wandb.log({'Image/Image': wandb.Image(grid)}, step=writer_index)

            del grid
            gc.collect()
        except Exception:
            self._logger.error("Error when plotting segmentation.")

    # ------------------------------------------------------------------
    # Module output hooks (mirrors TB_plotter interface)
    # ------------------------------------------------------------------

    def register_modules(self, module: torch.nn.Module, name: str, cmap: str = 'jet') -> None:
        """Register a module so its forward output is periodically logged."""
        if name in self._registered_module_config:
            self._logger.warning(f"Module '{name}' is already registered in this plotter.")
        self._registered_module_config[name] = {'cmap': cmap, 'module': module}
        handle = module.register_forward_hook(partial(self._collect_module_output, name=name))
        self._registered_module_config[name]['handle'] = handle

    def _collect_module_output(self, module: torch.nn.Module, input, output, name: str = None) -> None:
        if self._last_writer_index % self._write_n_iteration != 0:
            return
        if 'data' not in self._registered_module_config[name]:
            self._registered_module_config[name]['data'] = [output.detach().cpu()]
        else:
            self._registered_module_config[name]['data'].append(output.detach().cpu())

    @check_init
    def plot_collected_module_output(self, writer_index: int) -> None:
        self._logger.debug("Writing collected module outputs.")
        self._last_writer_index = writer_index

        for i, name in enumerate(self._registered_module_config):
            if 'data' not in self._registered_module_config[name]:
                self._logger.warning(f"No data collected for module: {name}")
                continue

            mod_output = self._registered_module_config[name]['data']
            mod_output = torch.cat(mod_output, dim=0) if len(mod_output) > 1 else mod_output[0]
            self._logger.debug(f"mod_output size: {mod_output.size()}")

            if not isinstance(mod_output, torch.Tensor):
                self._logger.error(f"Module data of '{name}' is not a tensor, got {type(mod_output)}.")
                continue

            self.plot_tensor(mod_output, f"{name}/Item_{i:d}", writer_index)
            del self._registered_module_config[name]['data']

    # ------------------------------------------------------------------
    # Metadata / config / files
    # ------------------------------------------------------------------

    @check_init
    def save_dict(self, scalar_dict: dict) -> None:
        """Save values to the run summary (non-time-series)."""
        for k, v in scalar_dict.items():
            self.save_value(k, v)

    @check_init
    def save_value(self, label: str, value: Any) -> None:
        self._logger.info(f"Saving value: {label}: {value}")
        wandb.run.summary[label] = value

    @check_init
    def save_file(self, label: str, file_path: str) -> None:
        self._logger.info(f"Uploading file: {file_path}")
        wandb.save(file_path)

    @check_init
    def save_params(self, params: dict) -> None:
        """Save hyperparameters to the run config."""
        if not isinstance(params, dict):
            raise TypeError(f"Parameters must be a dict, got {type(params)}.")
        wandb.config.update(params)

    @check_init
    def add_tag(self, tag: str) -> None:
        wandb.run.tags = wandb.run.tags + (tag,)

    def track_data(self, dataset_dir: Union[str, Path], version_tag):
        raise NotImplementedError

    def track_model(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collapse_to_2d(tensor: torch.Tensor) -> torch.Tensor:
        """Collapse a singleton spatial dim to produce a (B x C x H x W) tensor."""
        if tensor.dim() == 4:
            return tensor
        ones = tensor.shape[2:].count(1)
        if ones > 1 or ones == 0:
            raise IndexError(f"Label is not a set of 2D images: {tensor.shape}")
        d = tensor.shape[2:].index(1) + 2
        new_shape = list(tensor.shape)
        new_shape.pop(d)
        return tensor.reshape(new_shape)
