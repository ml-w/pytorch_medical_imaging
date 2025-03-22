**************
Override Guide
**************

.. py:currentmodule:: pytorch_med_imaging

When is overriding needed?
==========================

PMI has a relatively rigid implementation of how a solver is going to train your network. If you would like more control
of how the I/O is going to be during training, you can consider overriding, or inserting additional methods in your
network.

Additionally, if your data is special and require additional processing before feeding it into the network, you can also
consider overriding the `solvers.Solver` class callbacks. Alternatively, you can assign the callback as the class method
if you are sure what you are doing.

Controller
----------

All scripts will starts with creating a controller based on configurations (CFG script). The controller is responsible
for bridging

- Reads CFG files
- Creates and configure the master logger
- Creates and configure the run monitoring plotters, which attaches to `tensorboardX` or `neptune`.
- Attach to `guildai` run
- Creates and initiate solvers/inferencers.
- Deals with DDP/`nn.Parallel` of the network.

Solvers
-------

Typically, the solver is responsible for a number of things including:

- Loading mini-batches of data.
- Preprocess of data, assigning preprocess transform (`torchio.transform`).
- Feeding the batches into the network.
- Calculating the loss.
- Updating the network parameters according to loss.
- Updating the hyperparameters according to pre-defined schedules.
- Perform validation step if data is specified.
- Writes the progress into a panel using `plotter`, either to tensorboardX or neptune

Inferencers
-----------

Inferencers are just child class of solvers that has a slightly different running pathway and writes the results to a
designated directory. In particular, these are the additional functionalities that solvers does not invoke:

- Writing the results to a specific output directory
- Computing the performance of the predicted results and print it

Override Guide:
^^^^^^^^^^^^^^^

    The PMI offers default solvers that will handle most img2img and img2class use case. However, overriding the
`Solver` will allow user the flexibility to gain control over hoe the I/O is fed towards and from the network.
First, you need to be clear about what type of flexibility your network needs

1. Evaluate the need of flexibility.
""""""""""""""""""""""""""""""""""""

- Does your network require special I/O for forward?
- Does the loss function require special I/O method
- Does the training/inference require callback function?
- Does your application requires a different validation logic and results representation method?

2. Decide what to override
""""""""""""""""""""""""""

Depending on the purpose, you will override different methods.






.. mermaid::



