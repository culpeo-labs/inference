#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    clang \
    clangd \
    clang-format \
    clang-tidy \
    cmake \
    ninja-build \
    pkg-config \
    llvm-21 \
    libclang-rt-21-dev \
    g++-16

sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*