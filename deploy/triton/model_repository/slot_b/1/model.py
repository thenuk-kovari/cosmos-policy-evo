"""Triton slot B entrypoint for the EVO/UMI Cosmos Policy checkpoint."""

import sys
from pathlib import Path

_BACKENDS_DIR = Path("/opt/triton")
if str(_BACKENDS_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKENDS_DIR))

from backends.cosmos_policy_backend import CosmosPolicyBackend


class TritonPythonModel:
    def initialize(self, args):
        self.backend = CosmosPolicyBackend()
        self.backend.initialize(args)

    def execute(self, requests):
        return self.backend.execute(requests)

    def finalize(self):
        self.backend.finalize()
