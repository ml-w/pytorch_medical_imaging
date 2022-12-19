pmi_data_loader
===============

.. currentmodule:: pytorch_med_imaging.pmi_data_loader

All Configurations
------------------

.. autosummary::
    :nosignatures:

    PMIDataLoaderBaseCFG
    PMIImageDataLoaderCFG
    PMIImageFeaturePairLoaderCFG

PMIDataLoaderBase
-----------------


Configurations
^^^^^^^^^^^^^^
.. autodata:: PMIDataLoaderBaseCFG
    :annotation:

Class definition
^^^^^^^^^^^^^^^^
.. autoclass:: PMIDataLoaderBase
    :members:
    :private-members: _load_data_set_training, _load_data_set_inference, _read_config, _pack_data_into_subjects

PMIImageDataLoader
------------------

Configurations
^^^^^^^^^^^^^^
.. autodata:: PMIImageDataLoaderCFG
    :annotation:

Class definition
^^^^^^^^^^^^^^^^
.. autoclass:: PMIImageDataLoader
    :members:
    :show-inheritance:
    :private-members: _prepare_data, _create_queue, _read_image

PMIImageFeaturePairLoader
-------------------------

Configurations
^^^^^^^^^^^^^^
.. autodata:: PMIImageFeaturePairLoaderCFG
    :annotation:

Class definition
^^^^^^^^^^^^^^^^
.. autoclass:: PMIImageFeaturePairLoader
    :members:
    :private-members:

PMIImageFeaturePairLoaderConcat
-------------------------------

Configurations
^^^^^^^^^^^^^^
The configuration is the same as :class:`PMIImageFeaturePairLoaderCFG`

Class definition
^^^^^^^^^^^^^^^^
.. autoclass:: PMIImageFeaturePairLoaderConcat
    :members:
    :private-members:

PMIImageMCDataLoader
-------------------------------

Configurations
^^^^^^^^^^^^^^
.. autodata:: PMIImageMCDataLoaderCFG
    :annotation:

Class definition
^^^^^^^^^^^^^^^^
.. autoclass:: PMIImageMCDataLoader
    :members:
    :private-members: