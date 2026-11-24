#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLX_SWIFT_DIR="$ROOT_DIR/.build/checkouts/mlx-swift"
GENERATED_METAL_DIR="$MLX_SWIFT_DIR/Source/Cmlx/mlx-generated/metal"
KERNELS_DIR="$MLX_SWIFT_DIR/Source/Cmlx/mlx/mlx/backend/metal/kernels"
BUILD_DIR="$ROOT_DIR/.build/mlx-metal"

if [[ ! -d "$GENERATED_METAL_DIR" ]]; then
  echo "Missing mlx-swift checkout. Run swift build first." >&2
  exit 1
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/air"

for source in "$GENERATED_METAL_DIR"/*.metal "$GENERATED_METAL_DIR"/steel/attn/kernels/*.metal; do
  name="$(basename "$source" .metal)"
  xcrun -sdk macosx metal \
    -std=metal3.1 \
    -O3 \
    -I "$GENERATED_METAL_DIR" \
    -I "$KERNELS_DIR" \
    -c "$source" \
    -o "$BUILD_DIR/air/$name.air"
done

xcrun -sdk macosx metallib "$BUILD_DIR"/air/*.air -o "$ROOT_DIR/default.metallib"
ls -lh "$ROOT_DIR/default.metallib"
