from typing import Sequence, Optional, Union
from functools import wraps
import torch
from torchio.typing import TypeCallable
from torchio.data.subject import Subject
from torchio.constants import TYPE
from torchio.transforms import Transform
from torchio import Queue

import multiprocessing as mpi
from ..logger import Logger

__all__ = ['LambdaAdaptor', 'CallbackQueue']

class LambdaAdaptor(Transform):
    """Applies a user-defined function as transform, store the results to designated attribute if specified

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
    """
    An adaptor to execute some callback function after the queue sampled the patches.

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
        self._logger = Logger[self.__class__.__name__]

    def _fill(self):
        super(CallbackQueue, self)._fill()
        if self.callback is None or self.create_new_attribute is None:
            self._logger.warning("CallbackQueue was called but a callback function or an attribute string was not"
                                 "supplied! Using default behavior of tio.Queue")
            return

        if self.num_workers > 0 :
            # Create thread pool
            pool = mpi.Pool(self.num_workers)

            # for each patch, execute function
            p = pool.map_async(self.callback, self.patches_list)
            pool.close()
            pool.join()

            # Put patch back into place
            res = p.get()
        else:
            res = []
            for p in self.patches_list:
                res.append(self.callback(p))
        self._map_to_new_attr(res)

    def _map_to_new_attr(self, res):
        # check if list or str was supplied to self.create_new_attribute
        if isinstance(self.create_new_attribute, str):
            # check length
            if not len(res) == len(self.patches_list):
                raise IndexError(f"Expect result to have the same length (got {len(res)}) as the patch list got "
                                 f"{len(self.patches_list)}).")
            for p, r in zip(self.patches_list, res):
                p[self.create_new_attribute] = r
        elif isinstance(self.create_new_attribute, [str, tuple]):
            for p, r in zip(self.patches_list, res):
                if not len(r) == len(self.create_new_attribute):
                    raise IndexError(f"Expect result to have the same length (got {len(r)}) as the patch list got "
                                     f"{len(self.patches_list)}).")
                for att, rr in zip(self.create_new_attribute, r):
                    p[att] = rr