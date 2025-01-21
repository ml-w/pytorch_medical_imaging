==============
Loggers in PMI
==============

By default, almost all instance of the PMI package carries a logger that is an :class:`MNTSLogger` from the sister
package dependence `mri_normalization_tools`. The only exception being the CFG classes, which by defaults does not have
any logger, and they are just data holders. The behavior of the logger can be controlled from PMI config, but its with
limited flexibility and it slightly affects the preparation logic.

Global Logger
-------------

The uniqueness of the :class:`MNTSLogger` is that there's a global logger accessible from the class directly through
a simple call `MNTSLogger.global_logger` (`None` returned if it's not initialized). This global logger serves as a
template for the creation of all other loggers. Almost all characters will be passed on including the log format,
handlers, error hooks, log levels...etc. This is convinient, but it also means you need to becareful about creating
logger-carrying instance carefully. If you creates a PMI instance with logger too early, you will irreversibly create
a global template logger that you can't easily change.

Class Logger
------------

Usually, instances of the same class will share the same :class:`MNTSLogger`. The logger is typically created by calling
the line: `self._logger = MNTSLogger[__class__.__name__]` in the class constructor such that the logger name is the
same as the class name.


Rich Handler
------------

The :class:`MNTSLogger` uses the package `rich` to manage traceback. The traceback is colorful and more helpful but it
also is less robust. Although traceback is hooked to the `exception` method of the global logger, it does not always
work.


Principles
----------

Becareful where you initialize a logger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically, you will want all PMI instance to be created only after you've prepared the logger. Otherwise, a default
setting will apply.

How to change the default logger settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have already initialized a global logger, you will have to change the settings by altering the instance
attributes of the logger through `logger._logger`, which is a `logging.Logger` instance. The changed settings will
apply to all newly created loggers. However, existing loggers will not be affected. To see a list of existing logger,
you can simply print any logger instance, or access the class attribute `MNTSLogger.loggers`.