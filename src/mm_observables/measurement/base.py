import abc
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Protocol,
    runtime_checkable,
    Self,
)

import torch
from torch import Tensor


class MeasurementCoordinates(Tensor, metaclass=abc.ABCMeta):
    """
    Represents the coordinates of a measurement, for instance, Miller indices for Bragg diffraction.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate()

    @abc.abstractmethod
    def validate(self):
        """Ensure that the coordinates are physically acceptable,
        e.g., Miller indices are integers.
        """
        ...


class MeasurementValues(Tensor, metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validate()

    @abc.abstractmethod
    def validate(self):
        """Ensure that the values are physically acceptable,
        e.g., structure factor amplitudes are positive."""
        ...


@dataclass
@runtime_checkable
class MeasurementMetadata(Protocol):
    data_coordinates: MeasurementCoordinates
    ...

    def __eq__(self, other: object) -> bool:
        """
        Checks whether the metadata is the same as another metadata object. This method
        **must** check whether the data_coordinates are the same.
        """
        ...

    def validate(self):
        ...

    def to(self, device: str | torch.device) -> Self:
        """Move any necessary tensors to the specified device."""
        ...


# I've made this an actual base class because it needs to be a little stricter than a protocol,
# including defining __eq__
class Measurement(metaclass=abc.ABCMeta):
    """
    Base class for measurement objects.
    A measurement is a data container for any quantities that can be computed from a
    set of atomic coordinates, possibly some additional metadata, and a forward model. Our canonical
    example is a set of measured X-ray diffraction intensities (structure factor amplitudes), which
    are associated with a crystallographic unit cell and space group. The coordinates then specify
    the "where" the measurement was taken, e.g. Miller indices in reciprocal space.

    """
    def __init__(
            self,
            values: MeasurementValues,
            metadata: MeasurementMetadata,
            loss_implementation: Callable[[Self], Tensor],
    ):
        """
        Initialize a Measurement object.

        Args:
            values (MeasurementValues): Measurement data, e.g. structure factor amplitudes.
            metadata (MeasurementMetadata): Metadata associated with the measurement,
                e.g. unit cell and space group. Can include both data-derived quantities (e.g.,
                unit cell) and experimental setup (e.g., sample to detector distance) and must
                include the data "coordinates" (e.g., Miller indices).
            loss_implementation (Callable): Private logic unique to each measurement type.
        """
        data_coordinates = metadata.data_coordinates
        if not data_coordinates.shape[0] == values.shape[0]:
            raise ValueError(
                "metadata.data_coordinates and data_values must have the same number of entries"
            )
        self._data_y = values
        self._metadata = metadata
        # TODO: validate that the loss function can handle the measurement type
        self._loss_implementation = loss_implementation

        self.validate()

    @abc.abstractmethod
    def validate(self):
        """Ensure that the data values, metadata, and loss function are compatible"""
        ...

    @classmethod
    @abc.abstractmethod
    def from_file(
            cls, file: Path | str, loss_implementation: Callable[[Self], Tensor] | None
    ) -> Self:
        """Load a measurement from a file, e.g., an .mtz file.
        Subclasses must implement this method and provide a default loss function.
        """
        ...

    @property
    def data_coordinates(self) -> MeasurementCoordinates:
        """Coordinates of the measurement data, e.g. Miller indices."""
        return self._metadata.data_coordinates

    @property
    def data_values(self) -> MeasurementValues:
        """Measurement data, e.g. structure factor amplitudes."""
        return self._data_y

    @property
    def metadata(self) -> MeasurementMetadata:
        """Metadata associated with the measurement, e.g. unit cell and space group.
        Can include both data-derived quantities (like unit-cell) and experimental setup
        (e.g., sample to detector distance)
        """
        return self._metadata

    def __len__(self) -> int:
        return self._data_y.shape[0]

    def shape(self) -> tuple[int, ...]:
        return self._data_y.shape

    def __eq__(self, other: Self) -> bool:
        """
        Equality comparison for measurement objects should first check that measurement metadata
        are equal, and then can check the measurements themselves. E.g., first check that
        unit cell and space group are equal, then check that the structure factor amplitudes are
        equal.
        """
        if not isinstance(other, Measurement) or ():
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")
        return (
                (self._metadata == other.metadata) &
                (self._data_y == other.data_values).all().item()
        )

    def __ne__(self, other: Self) -> bool:
        """Included for completeness, otherwise this can return True
        if the objects are just not identical"""
        if not isinstance(other, Measurement):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")
        return not self.__eq__(other)

    def is_compatible(self, other: Self) -> bool:
        return self._metadata == other.metadata

    def loss(self, other: Self) -> Tensor:
        if not self.is_compatible(other):
            raise ValueError(
                "Cannot compute loss on incompatible Measurements; "
                "check metadata and data coordinates"
            )
        return self._loss_implementation(other)

    def to(self, device: str | torch.device):
        self._data_y = self._data_y.to(device)
        self._metadata = self._metadata.to(device)
        return self

