QM/MM with Sander
=================

MACE-OFF Model Support for Sander QM/MM Simulations
----------------------------------------------------

DeePMD-GNN provides support for using MACE-OFF foundation models in sander QM/MM simulations for QM internal energy correction.

Overview
--------

The sander interface allows you to:

* Use pretrained MACE-OFF models for QM region energy corrections
* Automatically handle QM/MM boundaries using the DPRc mechanism
* Compute forces for both QM and MM atoms
* Support periodic boundary conditions

Quick Start
-----------

1. **Download MACE-OFF Model**::

    from deepmd_gnn import download_mace_off_model, convert_mace_off_to_deepmd
    
    # Download pretrained model
    model_path = download_mace_off_model("small")
    
    # Convert to DeePMD format
    deepmd_model = convert_mace_off_to_deepmd("small", "mace_qmmm.pth")

2. **Create Configuration File**::

    # sander_mace.conf
    model_file=mace_qmmm.pth
    dtype=float64

3. **Use in Python**::

    from deepmd_gnn import SanderInterface
    import numpy as np
    
    # Initialize interface
    interface = SanderInterface.from_config("sander_mace.conf")
    
    # Compute QM correction
    energy, forces = interface.compute_qm_correction(coords, types, box)

Type Map Convention
-------------------

For QM/MM calculations, use the following type naming convention:

* **QM atoms**: Standard element symbols (``H``, ``C``, ``N``, ``O``, etc.)
* **MM atoms**: Prefixed with ``m`` (``mH``, ``mC``, etc.) or ``HW``/``OW`` for water

Example type map::

    ["C", "H", "O", "N", "mC", "mH", "mO", "HW", "OW"]

In this example, ``C``, ``H``, ``O``, ``N`` are QM atoms, while ``mC``, ``mH``, ``mO``, ``HW``, ``OW`` are MM atoms.

API Reference
-------------

SanderInterface
~~~~~~~~~~~~~~~

.. autoclass:: deepmd_gnn.sander.SanderInterface
   :members:
   :undoc-members:

MACE-OFF Functions
~~~~~~~~~~~~~~~~~~

.. autofunction:: deepmd_gnn.mace_off.download_mace_off_model

.. autofunction:: deepmd_gnn.mace_off.load_mace_off_model

.. autofunction:: deepmd_gnn.mace_off.convert_mace_off_to_deepmd

Model Selection
---------------

MACE-OFF provides three model sizes:

.. list-table::
   :header-rows: 1

   * - Model
     - Parameters
     - Speed
     - Accuracy
     - Best For
   * - small
     - ~1M
     - Fast
     - Good
     - QM/MM, screening
   * - medium
     - ~5M
     - Medium
     - Better
     - Production runs
   * - large
     - ~20M
     - Slow
     - Best
     - High-accuracy calculations

For QM/MM simulations, the **small** model is recommended for a good balance of speed and accuracy.

Integration with Sander
------------------------

To integrate with sander:

1. **Prepare frozen model**: Use ``dp --pt freeze`` to create a frozen model
2. **Set environment**: Export ``DP_PLUGIN_PATH`` to load DeePMD-GNN plugin
3. **Configure sander**: Set up QM/MM regions in sander input
4. **Run simulation**: Use sander with MACE-OFF energy corrections

See the ``examples/sander_qmmm`` directory for complete examples.

Examples
--------

See the ``examples/sander_qmmm`` directory for:

* Configuration file examples
* Python wrapper scripts
* Complete QM/MM simulation setup

Notes
-----

* MACE-OFF models are pretrained on diverse molecular datasets
* QM/MM boundary is handled automatically via DPRc mechanism
* Forces on MM atoms from QM interactions are computed automatically
* Default units: energy in eV, forces in eV/Angstrom
* Coordinates in Angstroms

References
----------

* MACE: https://github.com/ACEsuit/mace
* DeePMD-GNN: https://gitlab.com/RutgersLBSR/deepmd-gnn
* DPRc mechanism: See DeePMD-kit documentation
