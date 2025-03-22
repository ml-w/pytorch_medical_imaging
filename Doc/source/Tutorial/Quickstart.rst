Quickstart
==========

.. py:currentmodule::pytorch_med_imaging


Design principle
----------------

This package is designed to have three main instances to deal with training and inference:

- :class:`pmi_data_loader.PMIDataLoader` - Data Loader
- :class:`solvers.SolverBase` - Solver
- :class:`controller.Controller` - Controller

.. mermaid::

    ---
    title: Design pattern of PMI
    ---
    stateDiagram-v2
        [*] --> Controller
        state instance_creation {
            [*] --> read_CFG
            read_CFG --> create_solver: run_mode = train
            read_CFG --> create_inferencer: run_mode <> train
        }
        Controller --> instance_creation
        Controller --> Solver/Inferencer: For Training/Inference
        Solver/Inferencer --> DataLoader: For Loading Data

The entire process is governed by the Controller, which in turns is governed by a CFG file. The CFG file
is also a python script that creates CFG classes for each of the three components.

.. note::

    To override the settings in CFG during runtime, or to introduce dynamic changes in the variables, you can inherit
    and write a child class of :class:`controller.Controller`. Alternative, some variables can be changed by simply
    changing them in the controller's CFG class, e.g., :class:`controller.PMIControllerBaseCFG`.
