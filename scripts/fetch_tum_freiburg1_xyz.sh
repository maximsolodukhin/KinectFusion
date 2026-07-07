#!/usr/bin/env bash
set -euo pipefail

readonly DATASET_NAME="rgbd_dataset_freiburg1_xyz"
readonly ARCHIVE_NAME="${DATASET_NAME}.tgz"
readonly DEFAULT_DATA_DIR="data"
readonly DATASET_URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/${ARCHIVE_NAME}"

usage() {
  cat <<EOF
Usage: $0 [output-dir]

Downloads and extracts the TUM RGB-D fr1/xyz dataset.

Arguments:
  output-dir  Directory that will contain ${DATASET_NAME}.
              Defaults to ${DEFAULT_DATA_DIR}.

Environment:
  TUM_DATASET_URL  Override the download URL.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

readonly OUTPUT_DIR="${1:-${DEFAULT_DATA_DIR}}"
readonly TARGET_DIR="${OUTPUT_DIR}/${DATASET_NAME}"
readonly URL="${TUM_DATASET_URL:-${DATASET_URL}}"

has_expected_files() {
  [[ -f "${TARGET_DIR}/depth.txt" &&
     -f "${TARGET_DIR}/rgb.txt" &&
     -d "${TARGET_DIR}/depth" &&
     -d "${TARGET_DIR}/rgb" ]]
}

download_archive() {
  local url="$1"
  local output="$2"

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --show-error --output "${output}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget --output-document="${output}" "${url}"
  else
    echo "error: neither curl nor wget is available" >&2
    return 1
  fi
}

mkdir -p "${OUTPUT_DIR}"

if has_expected_files; then
  echo "Dataset already present: ${TARGET_DIR}"
  exit 0
fi

tmp_archive="$(mktemp "${OUTPUT_DIR}/${ARCHIVE_NAME}.XXXXXX")"
cleanup() {
  rm -f "${tmp_archive}"
}
trap cleanup EXIT

echo "Downloading ${URL}"
download_archive "${URL}" "${tmp_archive}"

echo "Extracting into ${OUTPUT_DIR}"
tar -xzf "${tmp_archive}" -C "${OUTPUT_DIR}"

if ! has_expected_files; then
  echo "error: extracted dataset is missing expected TUM RGB-D files" >&2
  exit 1
fi

echo "Dataset ready: ${TARGET_DIR}"
