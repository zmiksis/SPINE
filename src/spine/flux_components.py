"""Transient flux component storage for recording individual fluxes.

This module provides a lightweight dataclass for storing individual flux
components during a single timestep. The FluxComponents object is created
at the beginning of each timestep, populated conditionally based on recording
flags, used for file I/O, and then discarded (garbage collected) before the
next timestep.

Memory usage: Only requested fluxes are stored (via .copy()), and only for
one timestep at a time. With N nodes and M recorded fluxes:
    Memory per timestep = M * N * 8 bytes (transient)

Example: Recording 5 fluxes with 1000 nodes = 40 KB per timestep (transient)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FluxComponents:
    """Transient container for individual flux components at one timestep.

    All fields default to None - no memory is allocated unless the field
    is explicitly populated. This allows conditional storage based on
    recording flags without wasting memory.

    Units: All fluxes are in umol/(um^2 * s)

    Attributes:
        Cytosolic/PM fluxes:
            pmca: PMCA pump flux (calcium extrusion)
            ncx: NCX exchanger flux (calcium extrusion)
            pm_leak: Plasma membrane leak flux (calcium influx)
            vdcc: Voltage-dependent calcium channel flux (calcium influx)
            synapse: Synaptic flux (calcium influx)
            synapse_ip3: Synaptic IP3 flux

        ER fluxes:
            ryr: Ryanodine receptor flux (ER calcium release)
            serca: SERCA pump flux (ER calcium uptake)
            er_leak: ER leak flux
            ip3r: IP3 receptor flux (ER calcium release)
            soc: Store-operated channel flux (calcium influx)

        Aggregates:
            total_pm: Total PM flux (sum of all PM fluxes) = JPM
            total_er: Total ER flux (sum of all ER fluxes) = JER
            total_ip3: Total IP3 flux = JIP3
    """

    # Cytosolic/PM fluxes
    pmca: Optional[np.ndarray] = None
    ncx: Optional[np.ndarray] = None
    pm_leak: Optional[np.ndarray] = None
    vdcc: Optional[np.ndarray] = None
    synapse: Optional[np.ndarray] = None
    synapse_ip3: Optional[np.ndarray] = None

    # ER fluxes
    ryr: Optional[np.ndarray] = None
    serca: Optional[np.ndarray] = None
    er_leak: Optional[np.ndarray] = None
    ip3r: Optional[np.ndarray] = None
    soc: Optional[np.ndarray] = None

    # Aggregates (always computed for solver)
    total_pm: Optional[np.ndarray] = None
    total_er: Optional[np.ndarray] = None
    total_ip3: Optional[np.ndarray] = None
