#!/bin/bash
# Clone and install d4descent at the pinned commit next to this repo.
# Usage: scripts/setup_d4descent.sh [target_dir]   (default: ../d4descent)
set -euo pipefail
D4D_COMMIT=a66b729
TARGET=${1:-"$(cd "$(dirname "$0")/../.." && pwd)/d4descent"}
if [ ! -d "$TARGET/.git" ]; then
  git clone https://github.com/milmillin/d4descent "$TARGET"
fi
git -C "$TARGET" checkout -q "$D4D_COMMIT"
(cd "$TARGET" && uv sync)
echo "d4descent ready at $TARGET ($(git -C "$TARGET" rev-parse --short HEAD))"
