#!/usr/bin/env bash
set -euo pipefail

container="${1:-cosmos-evo-q0-slotb-test}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

docker cp "${script_dir}/test_slot_b.py" "${container}:/tmp/test_slot_b.py" >/dev/null
docker exec "${container}" /usr/bin/python3 /tmp/test_slot_b.py
