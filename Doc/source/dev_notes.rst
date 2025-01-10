*****************
Developer's notes
*****************

*(Last Update: 8th Jan, 2025)*

.. py:currentmodule:: pytorch_med_imaging

.. contents:: Table of Contents

Current state of the project
============================

The PMI was initially designed with a straight framework of training and inference. However, as times goes by, the
variety of network and how they are trained rapidly evolves. It is clear that more flexible framework like pytorch
lightning has an advantage in this matter and PMI aims to stride towards that way.

Summary of weaknesses
---------------------

Difficulty for adding new networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data loading mechanisms entangles :class:`pmi_data_loader.PMIDataLoaderBase` and
:class:`med_img_dataset.PMIDataBase.PMIDataBase` classes. This makes it almost impossible to reuse the classes without
implementing a child class that overrides the defined classes, unless the network network's I/O fits the default design
in the package. This basically means one have to implement at least 3 components and define their CFG structure when
a new network comes up:

Required Implementation:
    1. Solver
    2. Inferencer
    3. DataLoader
    4. CFGs x 3
    5. (Potentially) Controller

Lack of well defined scope
^^^^^^^^^^^^^^^^^^^^^^^^^^

While the code is straight, the scope of the package is never properly defined. I end up adding more and more
functionalities to the project and it's frankly Frankenstein's monster now (I know it has a good heart though). That
being said, the goal of the project is always clear: **Codebase to fine-tune hyperparameters for custom networks**.

Lack of tutorial
^^^^^^^^^^^^^^^^

Obviously, when I am the only one using the code, I won't be writing tutorials. I did have plans to do that after the
code package is finalized. But, as you might have aware, it's not finalized and is still rapidly changing. The problem
is that I just have so many projects at my hand and sometimes when I work on another project, I forgets what I wrote.
That's when I though, maybe I need to write some tutorials.

.. math::

    \textit{When I write this piece of code, only me and God knows what it does.} \\
    \textit{Now, only God knows.        } \\ \\
    \text{- Random Internet Meme}

Identify what needs to be reusable
----------------------------------

The current implementation is not all bad. Decent efforts have been put into important espects of model training.

- Logging system attached to `guildai` and `Neptune`
- Hyperparameter settings on yaml level using `guildai` APIs
- Text-based comparison matrix with version control
- Attached to `torchio` for data classes

What's limiting the reusability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. :func:`solvers.SolverBase.solve_epoch` - Slightly different unpacking will require overriding the class
2. :func:`solvers.SolverBase.step` - Is now fixed.


Towards the future
------------------

I am slowly revamping the code and phase out old concepts with the hopes that it will be more usable. Here's a list of
stuff I planned:

.. table:: TODO List
    :widths: 1 5 3

    +---+--------------------------+-----------------+
    |   | Item                     | Date Done       |
    +===+==========================+=================+
    | 1 | Network implementation   |                 |
    |   | should decide how data   |                 |
    |   | is loaded, DataLoader    |                 |
    |   |                          |                 |
    |   | should reference the     |                 |
    |   | network for how data is  |                 |
    |   | loaded.                  |                 |
    +---+--------------------------+-----------------+


