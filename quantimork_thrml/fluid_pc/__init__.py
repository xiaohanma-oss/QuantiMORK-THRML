"""FluidPCBlock — IFN §10.6 drop-in attention replacement for QuantiMORK-THRML."""

from quantimork_thrml.fluid_pc.advection import ConservativeAdvection
from quantimork_thrml.fluid_pc.drift_mpc import MPCDrift
from quantimork_thrml.fluid_pc.fluid_pc_block import (
    FluidPCBlock,
    FluidPCConfig,
)
from quantimork_thrml.fluid_pc.graph import WaveletGraph
from quantimork_thrml.fluid_pc.leray import LerayProjector
from quantimork_thrml.fluid_pc.reaction import PCReaction
from quantimork_thrml.fluid_pc.readout import WaveletReadout
from quantimork_thrml.fluid_pc.topology import (
    FluidGraphTopology,
    build_topology,
)

__all__ = [
    "ConservativeAdvection",
    "FluidGraphTopology",
    "FluidPCBlock",
    "FluidPCConfig",
    "LerayProjector",
    "MPCDrift",
    "PCReaction",
    "WaveletGraph",
    "WaveletReadout",
    "build_topology",
]
