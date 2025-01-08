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

    stateDiagram-v2
        [*] --> Controller
        state instance_creation {
            [*] --> read_CFG
            read_CFG --> create_solver: run_mode = train
            read_CFG --> create_inferencer: run_mode <> train
        }
        Controller --> instance_creation
        Controller --> Solver/DataLoader: For Training/Inference
        Controller --> DataLoader: For Loading Data

The entire process is governed by the Controller, which in turns is governed by a CFG file. The CFG file
is also a python script that creats CFG classes for each of the three components.