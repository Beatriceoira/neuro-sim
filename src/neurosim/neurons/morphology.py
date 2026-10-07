"""
Morphology class for multi-compartment neuron models.

This module implements the tree structure that defines the spatial
arrangement of compartments in a multi-compartment neuron.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from .compartment import Compartment


class Morphology:
    """
    Tree structure representing neuronal morphology.

    Contains compartments arranged in a tree (soma -> dendrites -> spines).
    Supports cable equation computation and spatial queries.
    """

    def __init__(
        self,
        specific_capacitance: float = 1.0,      # µF/cm²
        specific_axial_resistivity: float = 100.0,  # Ohm*cm
        resting_potential: float = -65.0,       # mV
    ):
        """
        Initialize the morphology.

        Args:
            specific_capacitance: Specific membrane capacitance (µF/cm²)
            specific_axial_resistivity: Cytoplasmic resistivity (Ohm*cm)
            resting_potential: Resting membrane potential (mV)
        """
        self.compartments: Dict[int, Compartment] = {}
        self.specific_capacitance = specific_capacitance
        self.specific_axial_resistivity = specific_axial_resistivity
        self.resting_potential = resting_potential
        self.next_id = 0

        # Default parameters for new compartments
        self.default_length = 10.0      # µm
        self.default_diameter = 10.0    # µm

    def add_compartment(
        self,
        name: str = "",
        length: Optional[float] = None,
        diameter: Optional[float] = None,
        parent_id: Optional[int] = None,
        **kwargs
    ) -> Compartment:
        """
        Add a new compartment to the morphology.

        Args:
            name: Human-readable name for the compartment
            length: Length of the compartment (µm). If None, uses default.
            diameter: Diameter of the compartment (µm). If None, uses default.
            parent_id: ID of the parent compartment. If None, this is the soma.
            **kwargs: Additional parameters passed to Compartment

        Returns:
            The created Compartment object
        """
        compartment_id = self.next_id
        self.next_id += 1

        comp_length = length if length is not None else self.default_length
        comp_diameter = diameter if diameter is not None else self.default_diameter

        params = {
            'specific_capacitance': self.specific_capacitance,
            'specific_axial_resistivity': self.specific_axial_resistivity,
            'resting_potential': self.resting_potential,
        }
        params.update(kwargs)

        compartment = Compartment(compartment_id, name, comp_length, comp_diameter, **params)
        self.compartments[compartment_id] = compartment

        if parent_id is not None and parent_id in self.compartments:
            parent = self.compartments[parent_id]
            parent.children.append(compartment)
            compartment.parent = parent

        return compartment

    def connect_compartments(self, child_id: int, parent_id: int):
        """
        Connect two existing compartments.

        Args:
            child_id: ID of the child compartment
            parent_id: ID of the parent compartment
        """
        if child_id not in self.compartments or parent_id not in self.compartments:
            raise ValueError("Compartment IDs not found")

        child = self.compartments[child_id]
        parent = self.compartments[parent_id]

        child.parent = parent
        parent.children.append(child)

    def get_soma(self) -> Compartment:
        """
        Get the soma compartment (root of the tree).

        Returns:
            The soma Compartment object

        Raises:
            ValueError: If no soma exists
        """
        for comp in self.compartments.values():
            if comp.is_soma:
                return comp
        raise ValueError("No soma compartment found in morphology")

    def get_all_compartments(self) -> List[Compartment]:
        """
        Get all compartments in the morphology.

        Returns:
            List of all Compartment objects
        """
        return list(self.compartments.values())

    def get_compartments_at_distance(
        self,
        source_id: int,
        max_distance: float,
        distance_func=None
    ) -> List[Tuple[Compartment, float]]:
        """
        Get compartments within a certain distance from a source.

        Args:
            source_id: ID of the source compartment
            max_distance: Maximum distance (µm)
            distance_func: Custom distance function. If None, uses path distance.

        Returns:
            List of (Compartment, distance) tuples
        """
        if distance_func is None:
            return self._path_distance_compartments(source_id, max_distance)
        else:
            source = self.compartments[source_id]
            result = []
            for comp in self.compartments.values():
                dist = distance_func(source, comp)
                if 0 <= dist <= max_distance:
                    result.append((comp, dist))
            return result

    def _path_distance_compartments(
        self,
        source_id: int,
        max_distance: float
    ) -> List[Tuple[Compartment, float]]:
        """Compute path distances from source to all other compartments."""
        source = self.compartments[source_id]
        result = []

        def dfs(comp: Compartment, distance: float):
            if comp.compartment_id != source_id:
                result.append((comp, distance))
            for child in comp.children:
                dfs(child, distance + child.length)

        dfs(source, 0.0)
        return [(c, d) for c, d in result if d <= max_distance]

    def get_path_length(self, compartment_id: int) -> float:
        """
        Get the path length from the soma to a compartment.

        Args:
            compartment_id: ID of the target compartment

        Returns:
            Path length in µm
        """
        if compartment_id not in self.compartments:
            raise ValueError(f"Compartment {compartment_id} not found")

        comp = self.compartments[compartment_id]
        # Path length from the soma center to the center of this compartment.
        # For the soma itself this is 0. For children it sums all ancestor
        # compartment lengths (not including the target compartment's own length).
        if comp.is_soma:
            return 0.0
        # Sum all ancestor lengths (distance from soma center to
        # the start of this compartment) plus this compartment's
        # own length (to reach its center).
        length = comp.length
        current = comp
        while current.parent is not None:
            current = current.parent
            length += current.length
        return length

    def describe(self) -> str:
        """
        Get a description of the morphology.

        Returns:
            String description
        """
        soma = self.get_soma()
        return (f"Morphology({len(self.compartments)} compartments, "
                f"soma='{soma.name}', "
                f"Ra={self.specific_axial_resistivity}Ohm*cm)")


# ------------------------------------------------------------------
# Convenience constructors
# ------------------------------------------------------------------

def create_simple_morphology(
    num_dendrites: int = 2,
    dendrite_length: float = 100.0,   # µm
    dendrite_diameter: float = 2.0,    # µm
    soma_length: float = 20.0,        # µm
    soma_diameter: float = 20.0,      # µm
    **kwargs
) -> Morphology:
    """
    Create a simple morphology with a soma and multiple dendrites.

    Args:
        num_dendrites: Number of dendritic branches
        dendrite_length: Length of each dendrite (µm)
        dendrite_diameter: Diameter of each dendrite (µm)
        soma_length: Length of the soma (µm)
        soma_diameter: Diameter of the soma (µm)
        **kwargs: Additional parameters passed to each compartment

    Returns:
        A Morphology object with soma + dendrites
    """
    morph = Morphology(**kwargs)

    soma = morph.add_compartment(
        name="soma",
        length=soma_length,
        diameter=soma_diameter,
    )
    soma_id = soma.compartment_id

    dendrite_ids = []
    for i in range(num_dendrites):
        dendrite = morph.add_compartment(
            name=f"dendrite_{i}",
            length=dendrite_length,
            diameter=dendrite_diameter,
            parent_id=soma_id,
        )
        dendrite_ids.append(dendrite.compartment_id)

    return morph


def create_ball_and_stick(
    soma_diameter: float = 20.0,
    dendrite_length: float = 200.0,
    dendrite_diameter: float = 2.0,
    **kwargs
) -> Morphology:
    """
    Create a ball-and-stick morphology (one spherical soma, one cylinder dendrite).

    This is a classic test geometry from cable theory.

    Args:
        soma_diameter: Diameter of the spherical soma (µm)
        dendrite_length: Length of the dendritic stalk (µm)
        dendrite_diameter: Diameter of the dendritic stalk (µm)
        **kwargs: Additional parameters

    Returns:
        A ball-and-stick Morphology object
    """
    return create_simple_morphology(
        num_dendrites=1,
        dendrite_length=dendrite_length,
        dendrite_diameter=dendrite_diameter,
        soma_length=soma_diameter,
        soma_diameter=soma_diameter,
        **kwargs,
    )


def create_branching_dendrite(
    soma_diameter: float = 20.0,
    branch_order: int = 2,
    base_length: float = 50.0,
    base_diameter: float = 3.0,
    length_decay: float = 0.7,
    diameter_decay: float = 0.7,
    **kwargs
) -> Morphology:
    """
    Create a branching dendritic tree.

    Args:
        soma_diameter: Diameter of the soma (µm)
        branch_order: Number of branching levels
        base_length: Length of the first-order branch (µm)
        base_diameter: Diameter of the first-order branch (µm)
        length_decay: Factor by which branch length decreases per order
        diameter_decay: Factor by which branch diameter decreases per order
        **kwargs: Additional parameters

    Returns:
        A branching Morphology object
    """
    morph = Morphology(**kwargs)

    soma = morph.add_compartment(
        name="soma",
        length=soma_diameter,
        diameter=soma_diameter,
    )
    root_id = soma.compartment_id

    def _add_branch(parent_id: int, order: int, length: float, diameter: float, suffix: str):
        if order < 0:
            return
        for i in range(2):  # each branch bifurcates into 2
            child = morph.add_compartment(
                name=f"dendrite{suffix}_{i}",
                length=length,
                diameter=diameter,
                parent_id=parent_id,
            )
            if order > 0:
                _add_branch(
                    child.compartment_id,
                    order - 1,
                    length * length_decay,
                    diameter * diameter_decay,
                    f"{suffix}_{i}",
                )

    _add_branch(root_id, branch_order, base_length, base_diameter, "0")

    return morph


# SWC file type names
SWC_TYPE_NAMES = {
    1: "soma",
    2: "axon",
    3: "dendrite",
    4: "basal_dendrite",
    5: "apical_dendrite",
    6: "fusiform",
    7: "spiny_stellate",
    8: "non_pyr",
    9: "pyramidal",
    10: "thalamo_cortical",
    11: "striatal",
    12: "golgi",
    13: "basket",
    14: "chandelier",
    15: "bipolar",
    16: "horizontal",
    17: "amacrine",
    18: "muller",
    19: "olfactory",
    20: "purkinje",
    21: "granule",
    22: "nucleus",
}


def _type_to_name(comp_type: int) -> str:
    """Convert SWC compartment type to name."""
    return SWC_TYPE_NAMES.get(comp_type, f"type_{comp_type}")


def load_swc(filepath: str) -> Morphology:
    """
    Load a morphology from an SWC file.

    Args:
        filepath: Path to the SWC file

    Returns:
        A Morphology object
    """
    morph = Morphology()

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) < 7:
            continue

        try:
            comp_id = int(parts[0])
            comp_type = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            radius = float(parts[5])
            parent_id = int(parts[6])

            name = _type_to_name(comp_type)
            diameter = radius * 2  # Convert radius to diameter

            if parent_id == -1:
                morph.add_compartment(name=name, diameter=diameter)
            else:
                morph.add_compartment(name=name, diameter=diameter, parent_id=parent_id)

        except (ValueError, IndexError):
            continue

    return morph
