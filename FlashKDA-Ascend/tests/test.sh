#!/bin/bash
set -e
pip install -e .
python tests/test_fwd.py
