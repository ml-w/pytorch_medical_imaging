import torch
import torch.nn.functional as F
import torchio as tio
import SimpleITK as sitk
import numpy as np
import math
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Union

from .InferencerBase import InferencerBase
from ..solvers.SolverBase import SolverBaseCFG

__all__ = ['Img2ImgInferencer']


class Img2ImgInferencer(InferencerBase):
    r"""Inferencer for image-to-image modality transfer (e.g. MRI → CT).

    Companion to :class:`~pytorch_med_imaging.solvers.Img2ImgSolver`.  Only
    the generator network is used at inference time; the discriminator (if any)
    is not loaded or needed.

    Outputs are written as **float32 NIfTI** files, preserving the full
    continuous range of the synthesised images (no argmax or rounding).

    2.5-D processing
        If the data loader uses ``patch_size=[H, W, 1]``, each patch output has
        shape ``(B, C, H, W)``.  The inferencer re-expands the missing Z
        dimension before feeding into the :class:`torchio.data.GridAggregator`
        so the volume is correctly reassembled.

    Args:
        cfg (Img2ImgSolverCFG):
            The configuration — same CFG type as the solver.

    See Also:
        * :class:`~pytorch_med_imaging.solvers.Img2ImgSolver`
        * :class:`~pytorch_med_imaging.solvers.Img2ImgSolverCFG`
    """

    create_optimizer = InferencerBase._placeholder

    def __init__(self, cfg: SolverBaseCFG, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.required_attributes = ['output_dir']

    # ──────────────────────────────────────────────────────────────────────────
    # Core inference
    # ──────────────────────────────────────────────────────────────────────────

    def _write_out(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        r"""Run inference and write float NIfTI outputs.

        The method iterates subject by subject, aggregates 2.5-D patch
        predictions into full volumes via :class:`torchio.data.GridAggregator`,
        then writes each volume as a float32 NIfTI file.

        Args:
            output_dir (str or Path, Optional):
                Override the output directory set in the CFG.  If ``None``,
                :attr:`output_dir` from the configuration is used.
        """
        in_image_data = self.data_loader.data['input']

        if output_dir is not None:
            if getattr(self, 'output_dir', None) is not None:
                self._logger.warning(
                    f"Overriding original output_dir '{self.output_dir}' with {output_dir}")
            self.output_dir = output_dir

        if getattr(self, 'output_dir', None) is None:
            raise AttributeError(
                "Output directory is not specified. Supply it as an argument or "
                "set 'output_dir' in the CFG.")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            self.net = self.net.eval()

            if self.data_loader.sampler is None:
                raise NotImplementedError(
                    "Img2ImgInferencer requires a patch sampler for 2.5-D processing.")

            for index, subject in enumerate(
                    tqdm(self.data_loader.queue._get_subjects_iterable(),
                         desc="Subjects", position=0)):

                self._logger.info(f"Processing subject: {subject}")

                _queue, _aggregator = self.data_loader.create_aggregation_queue(
                    subject, self.data_loader.inf_samples_per_vol)

                dataloader    = DataLoader(_queue, batch_size=self.batch_size, num_workers=0)
                ndim          = subject.get_first_image()[tio.DATA].dim()

                for mb in tqdm(dataloader, desc="Patches", position=1, leave=False):
                    s   = self._unpack_minibatch(mb, self.unpack_key_inference)
                    s   = self._match_type_with_network(s)

                    if isinstance(s, list):
                        out = self.net.forward(*s)
                    else:
                        out = self.net.forward(s)

                    # Re-expand Z=1 squeezed by the generator for GridAggregator
                    if ndim == 4:
                        while out.dim() < 5:
                            out = out.unsqueeze(-1)

                    _aggregator.add_batch(out, mb[tio.LOCATION])

                out = _aggregator.get_output_tensor().float()  # (C, H, W, D)

                # Recover original orientation when available
                try:
                    original_orientation = ''.join(subject['orientation'])
                    self._logger.info(
                        f"Recovering orientation to: {original_orientation}")
                    _sub  = tio.Subject(a=tio.ScalarImage(tensor=out))
                    _sitk = sitk.DICOMOrient(_sub['a'].as_sitk(), original_orientation)
                    out   = torch.from_numpy(
                        sitk.GetArrayFromImage(_sitk).astype(np.float32))
                except Exception as e:
                    self._logger.debug(
                        f"Orientation recovery skipped: {e}")
                    # torchio (H, W, D) → sitk (D, W, H)
                    out = out.squeeze().permute(2, 1, 0)

                in_image_data.write_uid(out, index, self.output_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────────

    def display_summary(self) -> None:
        r"""Compute and log MAE and PSNR against ground-truth if available.

        If no ground-truth data is present in the data loader, this method
        logs a message and returns without error.
        """
        if self.data_loader.data.get('gt') is None:
            self._logger.info("Ground-truth data not specified — skipping summary.")
            return

        gt_data  = self.data_loader.data['gt']
        out_dir  = Path(self.output_dir)
        maes, psnrs = [], []

        for uid in gt_data.get_unique_IDs():
            pred_path = out_dir / f"{uid}.nii.gz"
            if not pred_path.exists():
                pred_path = out_dir / f"{uid}.nii"
            if not pred_path.exists():
                self._logger.warning(f"No prediction found for UID '{uid}', skipping.")
                continue

            pred_img = sitk.ReadImage(str(pred_path))
            gt_img   = gt_data.get_image_by_ID(uid)

            if gt_img is None:
                continue

            pred_arr = sitk.GetArrayFromImage(pred_img).astype(np.float32)
            gt_arr   = sitk.GetArrayFromImage(gt_img).astype(np.float32)

            if pred_arr.shape != gt_arr.shape:
                self._logger.warning(
                    f"Shape mismatch for UID '{uid}': pred {pred_arr.shape} vs gt {gt_arr.shape}. Skipping.")
                continue

            mae  = float(np.abs(pred_arr - gt_arr).mean())
            mse  = float(((pred_arr - gt_arr) ** 2).mean())
            psnr = 10.0 * math.log10(1.0 / (mse + 1e-10))
            maes.append(mae)
            psnrs.append(psnr)
            self._logger.info(f"  [{uid}]  MAE: {mae:.4f}  PSNR: {psnr:.2f} dB")

        if maes:
            self._logger.info(
                f"Summary — mean MAE: {np.mean(maes):.4f}  "
                f"mean PSNR: {np.mean(psnrs):.2f} dB  "
                f"(n={len(maes)})")
