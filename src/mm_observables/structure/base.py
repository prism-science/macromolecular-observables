import numbers
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Self, Literal, TypeVar, Generic

import numpy as np
import torch
from biotite.structure import AffineTransformation, BondList

from biotite.structure.atoms import AtomArrayStack, AtomArray
from biotite.typing import M, N, NDArray1, NDArray2, NDArray3
from torch import Tensor

from mm_observables.forward_model.protocol import ForwardModel
from mm_observables.measurement.base import Measurement

"""
Design choices: We see two options in re-imagining AtomArray and AtomArrayStack for
use with torch tensors and ML. The first is to simply add methods that convert to/from numpy arrays, 
leaving those numpy arrays as the fundamental data objects. The second is to override methods where
necessary to use torch tensors as the fundamental data objects. For simplicity we choose the first
option but as tenets of our design
  1) any new method should be back compatible with both numpy arrays and torch tensors, 
  2) the class provides methods for "tensorizing" any existing AtomArrayStack or AtomArray,
  3) the class must be aware of its state--that is, whether it is ready for backprop or not.

A related goal is to eventually push as much of this as possible upstream to biotite, and to
work with biotite's developers to work towards tensor-compatible operations everywhere in biotite.

"""

def set_pytorch_ready_false(methods: list[str]):
    """Decorator to set pytorch_ready to False for selected methods in a class."""
    def decorator(cls):
        for method_name in methods:
            method = getattr(cls, method_name)
            def make_wrapper(m):
                def wrapper(*args, **kwargs):
                    # it turns out every instance method has a back-reference to the instance!
                    warnings.warn(
                        f"Calling the operation {cls}.{method} may leave the object in a state "
                        f"unsuitable for PyTorch operations. Setting pytorch_ready=False."
                    )
                    m.__self__.pytorch_ready = False
                    return m(*args, **kwargs)
                return wrapper
            setattr(cls, method_name, make_wrapper(method))
        return cls
    return decorator


PYTORCH_INCOMPATIBLE_METHODS = [
    "copy",
    "add_annotation",
    "__add__",
    "_subarray"
    "_del_element",
    "_set_element",
    "__add__",
    "_copy_annotations",
    "_get_array",
    "__getitem__",
]


@set_pytorch_ready_false(PYTORCH_INCOMPATIBLE_METHODS)
class Structure(AtomArrayStack, Generic[M, N]):
    """Our fundamental data container for structural information. The key determinant of whether
    something should be included in this class is whether the information is tied to properties of
    particular atoms. Information about experiments should not be included here. There will be
    some cases where it might not be clear whether something belongs here or not. For instance,
    some chemical shift data can be tied to particular atoms. We tend to think "no" for chemical
    shifts, but Debye-Waller "B" factors sit in a similar space and we tend to think "yes" for
    those. In general the answer will be whether something is directly measured (as is a
    chemical shift) or whether it is inferred from other measurements (as is a B factor).
    Measurements belong in the Measurement class, inferences belong in this class.

    The combination of measurements and structure is a StructureDetermination object.
    """
    # attributes we expect could have different shapes than AtomArray or AtomArrayStack
    b_factor: NDArray2[M, N, np.floating] | NDArray1[N, np.floating]
    occupancy: NDArray2[M, N, np.floating] | NDArray1[N, np.floating]

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pytorch_ready = False

    # methods for converting to/from numpy arrays
    @classmethod
    def from_numpy_atom_array_stack(cls, atom_array_stack: AtomArrayStack) -> Self:
        """
        Convert a numpy AtomArrayStack to a torch.Tensor-based Structure. Useful primarily
        for ease of instantiating these objects using existing methods that return AtomArrayStacks,
        like atomworks' parse and load_any.
        """
        ...

    def set_all_annotations_to_tensors(self, device: str | torch.device = "cuda") -> None:
        """
        Convert all annotations to torch.Tensor and set self._pytorch_ready = True
        to warn when operations might result in non-tensor values. Must be idempotent up to device.
        Must call this before using the structure in an PyTorch model. This method must set
        self._pytorch_ready = True.
        """
        ...

    def validate_annotations(self) -> None:
        """
        Ensure that all annotations are valid tensors, and that for annotations we define they
        have physically acceptable values. This is a compromise between the overhead of strict
        typing for each data type in the structure and not checking the values at all. Should
        raise an error if the annotations are invalid.

        Raises ValueError if the annotations are invalid.
        """
        ...

    def superimpose_homolog(
            self, mobile: AtomArray | AtomArrayStack | Self
    ) -> tuple[Self, AffineTransformation, np.ndarray]:
        """
        Superimpose the `mobile` structure onto this one using sequence alignment to choose
        coordinates to compare, in a **differentiable** fashion (i.e. must use PyTorch tensors).

        Returns:
        --------
        fitted: Structure the superimposed structure
        transform: AffineTransformation the transformation used to superimpose it
        fixed_anchor_indices, mobile_anchor_indices: np.ndarray the indices of the aligned residues,
            as returned by biotite.structure.superimpose_homologs
        """
        ...

    # biotite _AtomArrayBase methods that need overrides because they'll break if we use tensors
    def equal_annotations(
            self, item: AtomArray | AtomArrayStack | Self, equal_nan: bool=True
    ) -> bool:
        """
        Check if this object shares equal annotation arrays with the
        given :class: `AtomArray` or :class: `AtomArrayStack` or :class: `Structure`.
        We override this method because its parent would actually break if we used it with tensors.

        Parameters
        ----------
        item : AtomArray or AtomArrayStack or Structure
            The object to compare the annotation arrays with.
        equal_nan : bool
            Whether to count `nan` values as equal. Default: True.

        Returns
        -------
        equality : bool
            True, if the annotation arrays are equal.
        """
        if not isinstance(item, (AtomArray, AtomArrayStack, Structure)):
            return False
        if not self.equal_annotation_categories(item):
            return False
        for name in self.get_annotation_categories():
            # ... allowing `nan` values causes type-casting, which is
            #     only possible for floating-point arrays
            # TODO: this is the only change we've made to the upstream method--
            #  should we push to biotite?
            allow_nan = equal_nan & is_floating_point(self.get_annotation(name))
            # this works just fine with tensors.
            if not np.array_equal(
                    self.get_annotation(name), item.get_annotation(name), equal_nan=allow_nan
            ):
                return False
        return True

    def set_annotation(self, category: str, value: np.ndarray | Tensor) -> None:
        """
        Set an annotation array. If the annotation category does not
        exist yet, the category is created. Tensor safe.

        Parameters
        ----------
        category : str
            The annotation category to be set.
        value : ndarray
            The new value of the annotation category. The size of the
            array must be the same as the array length.

        """
        array = torch.as_tensor(value)
        if len(array) != self.array_length:
            raise IndexError(
                f"Expected array length {self.array_length}, but got {len(array)}"
            )
        # The parent version does some fancy type maintenance that we won't worry about here.
        self._annot[category] = array  # noqa

    def _subarray(self, index) -> Self:
        # Parent method can convert to a new copy of an AtomArrayStack, so we need to convert
        # to a new Structure.
        obj = super()._subarray(index)  # noqa
        return Structure.from_numpy_atom_array_stack(obj)

    def __setattr__(self, attr, value: np.ndarray | Tensor) -> None:
        """
        If the attribute is an annotation, the :attr:`value` is saved
        to the annotation in the dictionary.
        Exposes coordinates.
        :attr:`value` must have same length as :func:`array_length()`.
        """
        if attr == "coord":
            if value.ndim != 3:
                raise ValueError(
                    "A 3-dimensional ndarray is expected for coordinates of a Structure"
                )
            if value.shape[-2] != self.array_length:
                raise ValueError(
                    f"Expected array length {self.array_length}, but got {len(value)}"
                )
            if isinstance(value, Tensor):
                super().__setattr__("_coord", value.type(torch.float32))
            elif isinstance(value, np.ndarray):
                super().__setattr__("_coord", torch.as_tensor(value, dtype=torch.float32))
        else:
            super().__setattr__(attr, value)

        self.validate_annotations()

    def __setitem__(self, index: numbers.Integral, array: Self):
        ...

    # methods we want to add to AtomArrayStack, all should preserve gradients, e.g.,
    # any masking must pass gradients back through to the original tensor.
    def get_unique_atom_identifiers(self):
        """Return a list of chain/residue/atom identifiers for each unique atom."""
        ...

    def query(self, query_string: str, style: Literal["pymol", "pandas"] = "pymol") -> Self:
        """Return a new Structure object containing only the atoms matching the query string,
        in a such a way that gradients are preserved through tensors if attributes and coords
        are in tensor form

        Accepts either pandas/Atomworks-style queries or PyMOL-style queries, but you must
        specify which you are using.
        """
        ...

    def mask(
        self, query_string: str, style: Literal["pymol", "pandas"] = "pymol"
    ) -> np.ndarray[bool]:
        """Return a boolean mask of the atoms matching the query string.

        Accepts either pandas/Atomworks-style queries or PyMOL-style queries, but you must
        specify which you are using.

        """
        ...

    def get_indices(self, query_string: str, style: Literal["pymol", "pandas"] = "pymol") -> np.ndarray[int]:
        """Return a boolean mask of the atoms matching the query string.

        Accepts either pandas/Atomworks-style queries or PyMOL-style queries, but you must
        specify which you are using.

        """
        ...


def is_floating_point(x: Tensor | np.ndarray) -> bool:
    """Return True if the given array allows `nan` values."""
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.floating)
    elif isinstance(x, Tensor):
        return x.dtype.is_floating_point
    else:
        return False


@dataclass
class StructureDetermination:
    """Global container for a structure and the data and method used to determine it.
    It is also effectively the primary I/O interface for structures as a result.

    Note that we are explicitly not including old PDB file support here.
    """
    structure: Structure
    measurements: list[Measurement]
    # TODO: include this or not? Method descriptions are usually in PDB/CIF files.
    forward_model: ForwardModel

    def to_cif(self, output_path: Path | str) -> Path:
        """Return a mmCIF string representation of the structure.
        Returns the output path of the CIF file.

        Args:
            output_path: Path to write the CIF file to.
        """
        ...

    def from_cif(self, cif_path: Path | str) -> Self:
        """Return a StructureDetermination object from a CIF file.
        Since CIF files do not normally contain the measurements, this will typically
        construct only metadata for measurements.

        Args:
            cif_path: Path to the CIF file.
        """
        ...
