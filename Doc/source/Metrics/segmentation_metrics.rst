Segmentation Metrics
====================


Voxel-overlap metrics
---------------------

These functions operate on TP/FP/TN/FN counts returned by :func:`perf_measure`.

.. currentmodule:: pytorch_med_imaging.perf.segmentation_perf

.. autofunction:: perf_measure

.. autofunction:: JAC

.. autofunction:: DICE

.. autofunction:: VS

.. autofunction:: VD

.. autofunction:: Sensitivity

.. autofunction:: Specificity


Surface-distance metrics
------------------------

Scipy-based re-implementations; no external ``surface-distance`` package required.

.. autofunction:: compute_surface_distances

.. autofunction:: compute_ASD

.. autofunction:: compute_HD

.. autofunction:: compute_HD95


Batch evaluation
----------------

.. autofunction:: EVAL


Script-level analysis
---------------------

.. currentmodule:: pytorch_med_imaging.scripts.analysis

Structural Similarity (SSIM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: SSIM

PSNR
^^^^
.. autofunction:: PSNR

Root Mean Square Error
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: RMSE

Run segmentation analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: segmentation_analysis
