#!/bin/sh
set -eu
opjax-pallas-environment-runner --task /tests/task.json --kernel /app/kernel.py --evidence-dir /logs/verifier/evidence
