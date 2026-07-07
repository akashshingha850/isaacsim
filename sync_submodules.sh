#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
	echo "This script must be run from inside a Git repository."
	exit 1
fi

submodule_path="${1:-}"

if [[ -n "$submodule_path" ]]; then
	git submodule sync --recursive -- "$submodule_path"
	git submodule update --init --recursive --jobs=8 -- "$submodule_path"
else
	git submodule sync --recursive
	git submodule update --init --recursive --jobs=8
fi
