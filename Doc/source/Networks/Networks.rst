Networks
========

.. currentmodule:: pytorch_med_imaging.networks


UNet
----
.. autoclass:: pytorch_med_imaging.networks.UNet.UNet
    :members:
    :show-inheritance:

.. autoclass:: pytorch_med_imaging.networks.UNet.UNetPosAware
    :members:
    :show-inheritance:


UNet_p
------
.. autoclass:: pytorch_med_imaging.networks.UNet_p.UNet_p
    :members:
    :show-inheritance:

.. autoclass:: pytorch_med_imaging.networks.UNet_p.UNet_p_residual
    :members:
    :show-inheritance:

Building blocks
^^^^^^^^^^^^^^^
.. autoclass:: pytorch_med_imaging.networks.UNet_p.Down
    :members:

.. autoclass:: pytorch_med_imaging.networks.UNet_p.Up
    :members:


VNet
----
.. autoclass:: pytorch_med_imaging.networks.VNet.VNet
    :members:
    :show-inheritance:


Layers
------

.. currentmodule:: pytorch_med_imaging.networks.layers

Standard 2-D layers
^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_med_imaging.networks.layers.StandardLayers
    :members:
    :show-inheritance:

Standard 3-D layers
^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_med_imaging.networks.layers.StandardLayers3D
    :members:
    :show-inheritance:

Attention gates
^^^^^^^^^^^^^^^
.. automodule:: pytorch_med_imaging.networks.layers.AttentionGates
    :members:
    :show-inheritance:

Dense layers
^^^^^^^^^^^^
.. automodule:: pytorch_med_imaging.networks.layers.DenseLayer
    :members:
    :show-inheritance:

Recurrent layers
^^^^^^^^^^^^^^^^
.. automodule:: pytorch_med_imaging.networks.layers.RecurrentLayers
    :members:
    :show-inheritance:

Transition layers
^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_med_imaging.networks.layers.Transitions
    :members:
    :show-inheritance:

Normalisation layers
^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_med_imaging.networks.layers.NormLayers
    :members:
    :show-inheritance:

Mobile inverted-bottleneck (MBConv)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_med_imaging.networks.layers.MBConv
    :members:
    :show-inheritance:

Transformer positional encoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pytorch_med_imaging.networks.layers.TransformerPositionalEncoding
    :members:
    :show-inheritance:
