from typing import Optional, Sequence, Union

import gc
import torch
import torch.multiprocessing as mpi
from mnts.mnts_logger import MNTSLogger
from tqdm import tqdm

import torchio
from torchio import Queue
from torchio.constants import TYPE
from torchio.data.subject import Subject
from torchio.transforms import Transform
from torchio.typing import TypeCallable

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

__all__ = ['LambdaAdaptor', 'CallbackQueue']

class LambdaAdaptor(Transform):
    r"""Applies a user-defined function as transform, store the results to designated attribute if specified

    Args:
        function: Callable that receives and returns a 4D
            :class:`torch.Tensor`.
        target_attribute: String as target attribute
        types_to_apply: List of strings corresponding to the image types to
            which this transform should be applied. If ``None``, the transform
            will be applied to all images in the subject.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. note::
        Kwargs include and exclude works here

    Example:
        >>> import torchio as tio
        >>> invert_intensity = tio.Lambda(lambda x: -x, types_to_apply=[tio.INTENSITY])
        >>> invert_mask = tio.Lambda(lambda x: 1 - x, types_to_apply=[tio.LABEL])
        >>> def double(x):
        ...     return 2 * x
        >>> double_transform = tio.Lambda(double)
    """  # noqa: E501
    def __init__(
            self,
            function: TypeCallable,
            target_attribute: Union[Sequence[str],str],
            types_to_apply: Optional[Sequence[str]] = None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.function = function
        self.types_to_apply = types_to_apply
        self.target_attribute = target_attribute
        self.args_names = 'function', 'target_attribute', 'types_to_apply'

    def apply_transform(self, subject: Subject) -> Subject:
        images = subject.get_images(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for image in images:
            image_type = image[TYPE]
            if self.types_to_apply is not None:
                if image_type not in self.types_to_apply:
                    continue

            function_arg = image.data
            result = self.function(function_arg)
            if not isinstance(result, torch.Tensor):
                message = (
                    'The returned value from the callable argument must be'
                    f' of type {torch.Tensor}, not {type(result)}'
                )
                raise ValueError(message)
            subject[self.target_attribute] = result
        return subject


class CallbackQueue(Queue):
    r"""
    An adaptor to execute some callback function after the queue sampled the patches. For this to work properly, you
    must set the data sharing strategy to 'file_system', otherwise, you might observe "OS Error: too many opened files"

    .. code-block:: python
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

    The call back functions should be written in PMI_data_loader.computations, symbols should be exported from the
    __init__.py file.

    .. warning::
        * Deadlock could occur easily using this Queue.

    Args:
        patch_sampling_callback (callable, Optional):
            A function that takes :class:`Subject` or :class:`dict` as input and return any output. If this is None, its
            behavior is same as `tio.Queue`. Default to None.
        create_new_attribute (str or list of str, Optional):
            If a list of str is supplied and the output of the upper is an iterable with identical lenght as this list,
            each output will be mapped to each str in the list.
    """
    def __init__(self,
                 *args,
                 patch_sampling_callback: Optional[TypeCallable] = None,
                 create_new_attribute: Optional[Union[Sequence[str], str]] = None,
                 **kwargs):
        super(CallbackQueue, self).__init__(*args, **kwargs)
        self.callback = patch_sampling_callback
        self.create_new_attribute = create_new_attribute
        self._logger = MNTSLogger[__class__.__name__]

        if self.create_new_attribute is None:
            msg = f"`create_new_attribute` is not specified!"
            raise ArithmeticError(msg)

    def _fill(self):
        super(CallbackQueue, self)._fill()
        if self.callback is None or self.create_new_attribute is None:
            return

        res = []
        if self.num_workers  > 1:
            #! Note:
            #   This results in OSError: Too many open files, don't know why, but can be worked around by
            #   setting ulimit -n [large number], > 100000
            #   Also, there is a fair chance that this will cause memory deadlock and hangs the whole process, thus the
            #   operations are repeated until it works.
            # Things tried:
            #   * use torch.mpi, seems helpful but not fully resolve
            #   * rerun super._fill(), lead to the dataloader thread incorrectly hangs
            #   * seems like the program runs correctly with pool.terminate() + pool.join() trap
            #   * If patch-size is too large, this still goes into deadlock even when number of sample per volume is small
            #   * Turns out the function torch.tensor(something) to turn a numpy array to a pytorch tensor is causing some problem
            #   * Bad File Descriptor error probably have nothing to do with this


            #  Create thread pool
            while len(res) == 0:
                [s.load() for s in self.patches_list] # pre-load the patches
                pool = mpi.Pool(self.num_workers)
                p = pool.map_async(self.callback,
                                   self.patches_list)
                try:
                    pool.close()
                    pool.join()

                    res.extend(p.get(300))
                    pool.terminate()
                except TimeoutError as e:
                    # reset and clear the pool worker and try again
                    pool.close()
                    pool.terminate()
                    pool.join()
                    res = []

                    # save the details of the failed patch list
                    self._logger.debug(f"{self.patches_list}")
                    self._logger.debug("{}".format('\n'.join(
                        [','.join([str(sub.get_first_image().path),
                                   str(sub[torchio.LOCATION])])
                         for sub in self.patches_list]))
                    )

                    # make a copy to avoid deadlock
                    self.patches_list = [Subject(sub) for sub in self.patches_list]

                pool.terminate()
                del pool, p
                gc.collect()
        else:
            # Do it in a single thread. Could be slow.
            for p in tqdm(self.patches_list):
                res.append(self.callback(p))
        self._map_to_new_attr(res)
        del res

    def _map_to_new_attr(self, res):
        # check if list or str was supplied to self.create_new_attribute
        if isinstance(self.create_new_attribute, str):
            # check length
            if not len(res) == len(self.patches_list):
                raise IndexError(f"Expect result to have the same length (got {len(res)}) as the patch list got "
                                 f"{len(self.patches_list)}).")
            for p, r in zip(self.patches_list, res):
                p[self.create_new_attribute] = r
        elif isinstance(self.create_new_attribute, (str, tuple)):
            for p, r in zip(self.patches_list, res):
                if not len(r) == len(self.create_new_attribute):
                    raise IndexError(f"Expect result to have the same length (got {len(r)}) as the patch list got "
                                     f"{len(self.patches_list)}).")
                for att, rr in zip(self.create_new_attribute, r):
                    p[att] = rr

    def __str__(self):
        return super(CallbackQueue, self).__str__().replace("Queue", "CallBackQueue")