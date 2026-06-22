from typing import (
    Protocol,
    runtime_checkable,
)

from mm_observables.measurement.base import Measurement, MeasurementMetadata
from mm_observables.structure.base import Structure


@runtime_checkable
class ForwardModel(Protocol):
    """Protocol for forward models."""

    def __init__(self, measurement_metadata: MeasurementMetadata):
        """Initialize the forward model with the measurement metadata.
        This metadata should be all that is needed to prepare the forward model for use, e.g.,
        a unit cell, space group, and a set of Miller indices.
        """
        ...

    def is_metadata_compatible(self, measurement: Measurement) -> bool:
        """Return True if the forward model is compatible with the measurement."""
        ...

    def __call__(self, structure: Structure) -> Measurement:
        """Return the forward model output for the given structure."""
        ...
