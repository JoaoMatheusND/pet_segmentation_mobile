#!/usr/bin/env bash
# ============================================================
# PetSeg end-to-end build script
#   1. Train UNet on Oxford-IIIT Pet
#   2. Export to TFLite FP32
#   3. Copy model to Android assets
#   4. Compile Android APK
#
# Usage:
#   ./build.sh              # full pipeline
#   ./build.sh --skip-train # skip training (use existing best.pth)
#   ./build.sh --android-only
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TRAINING_DIR="$REPO_ROOT/training"
ANDROID_DIR="$REPO_ROOT/android"
ASSETS_DIR="$ANDROID_DIR/app/src/main/assets"
CONDA_ENV="deepl"

SKIP_TRAIN=false
ANDROID_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --skip-train)   SKIP_TRAIN=true ;;
        --android-only) ANDROID_ONLY=true; SKIP_TRAIN=true ;;
    esac
done

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[build]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn] ${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

init_conda() {
    if command -v conda &>/dev/null; then return 0; fi

    local candidates=(
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "$HOME/miniconda/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
        "/usr/local/anaconda3/etc/profile.d/conda.sh"
    )
    for f in "${candidates[@]}"; do
        if [ -f "$f" ]; then
            # shellcheck source=/dev/null
            . "$f"
            info "Sourced conda from $f"
            return 0
        fi
    done

    error "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
}

run_py() {
    if conda run -n "$CONDA_ENV" --no-capture-output python "$@" 2>&1; then
        return 0
    else
        error "Python command failed: python $*"
    fi
}

if ! $ANDROID_ONLY; then
    init_conda
    info "Checking conda environment '$CONDA_ENV'..."

    if ! conda env list | grep -q "^$CONDA_ENV "; then
        info "Creating conda env '$CONDA_ENV' (Python 3.10)..."
        conda create -y -n "$CONDA_ENV" python=3.10 --no-default-packages
        info "Installing PyTorch (CUDA 11.8)..."
        conda run -n "$CONDA_ENV" pip install torch torchvision \
            --index-url https://download.pytorch.org/whl/cu118
    else
        info "Conda env '$CONDA_ENV' found."
    fi
    info "Installing/updating Python requirements..."
    conda run -n "$CONDA_ENV" pip install -q -r "$TRAINING_DIR/requirements.txt"
fi

if ! $SKIP_TRAIN; then
    info "=== PHASE 1: Training ==="
    cd "$TRAINING_DIR"

    info "Starting training (20 epochs)..."
    run_py train.py

    info "Evaluating metrics..."
    run_py evaluate.py

    info "Generating visualizations..."
    run_py visualize.py
    info "Saved results.png in $TRAINING_DIR"
else
    info "=== PHASE 1: Skipped (--skip-train) ==="
    if [ ! -f "$TRAINING_DIR/best.pth" ]; then
        error "best.pth not found. Run without --skip-train first."
    fi
fi

if ! $ANDROID_ONLY; then
    info "=== PHASE 2: Export to TFLite ==="
    cd "$TRAINING_DIR"
    run_py export.py

    if [ ! -f "$TRAINING_DIR/model.tflite" ]; then
        error "model.tflite not generated."
    fi
    info "Export successful: $(du -h "$TRAINING_DIR/model.tflite" | cut -f1) model.tflite"
fi

info "=== PHASE 3: Copy model to Android assets ==="
mkdir -p "$ASSETS_DIR"
cp "$TRAINING_DIR/model.tflite" "$ASSETS_DIR/model.tflite"
info "Copied model.tflite → $ASSETS_DIR/"

info "=== PHASE 4: Build Android APK ==="
cd "$ANDROID_DIR"

if [ ! -f "gradlew" ]; then
    warn "gradlew not found. Trying system gradle..."
    if command -v gradle &>/dev/null; then
        gradle wrapper --gradle-version 8.0
    else
        warn "Neither gradlew nor gradle found."
        warn "To build the APK:"
        warn "  1. Open $ANDROID_DIR in Android Studio"
        warn "  2. Build > Make Project"
        warn "  3. APK at app/build/outputs/apk/debug/app-debug.apk"
        echo ""
        info "Python pipeline complete. Android requires Android Studio."
        exit 0
    fi
fi

chmod +x gradlew
info "Running Gradle build..."
./gradlew assembleDebug --no-daemon 2>&1

APK="$ANDROID_DIR/app/build/outputs/apk/debug/app-debug.apk"
if [ -f "$APK" ]; then
    info "✓ APK built: $APK  ($(du -h "$APK" | cut -f1))"
    echo ""
    info "Install on connected device:"
    echo "    adb install -r $APK"
else
    error "APK not found after build."
fi
