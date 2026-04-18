"""Compatibility entrypoint for Component 7 verification training.

The active training path for Component 7 is the boundary critic trainer in
``src.training.train_boundary_critic``. This module is kept so older commands
that reference ``train_component7_verification`` still work.
"""

from __future__ import annotations

from src.training.train_boundary_critic import main


if __name__ == "__main__":
	main()
