#!/bin/sh
set -eu
mkdir -p /logs/artifacts
git -C /workspace diff --binary $(git -C /workspace rev-list --max-parents=0 HEAD) HEAD > /logs/artifacts/model.patch
