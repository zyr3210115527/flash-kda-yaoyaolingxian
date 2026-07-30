#!/bin/bash
set -e

echo "=== FlashKDA-Ascend Full Test Suite ==="

# Install
pip install -e .

# Quick smoke test
echo "--- Smoke test ---"
python tests/test_fwd.py

# Full parametrized test
echo "--- Full parametrized test ---"
cd tests && pytest test_fwd_full.py -x -v -n 4

echo "=== All tests passed ==="
