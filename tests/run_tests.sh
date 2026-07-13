#!/usr/bin/env bash
set -euo pipefail

# Runs the pytest suite by category (subdirectory of tests/) and test type (a test_*.py file).

############################
# Paths (CWD-independent)
############################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$SCRIPT_DIR"                 # the tests/ folder
REPO_ROOT="$(dirname "$SCRIPT_DIR")"    # project root; pytest runs from here so imports resolve

############################
# Discover categories
############################
# A category is a subdirectory of tests/ that holds at least one test_*.py file. Discovered
# dynamically so new subsystem directories need no edit here.
VALID_CATEGORIES=()
for d in "$TESTS_DIR"/*/; do
    [[ -n "$(find "$d" -maxdepth 1 -name 'test_*.py' -print -quit)" ]] && VALID_CATEGORIES+=("$(basename "$d")")
done
if [[ ${#VALID_CATEGORIES[@]} -eq 0 ]]; then
    echo "No test categories found under $TESTS_DIR (expected subdirs with test_*.py files)." >&2
    exit 1
fi

############################
# Usage
############################
usage() {
    cat <<EOF
Usage: $0 [options] [-- pytest-args...]

Runs the pytest suite by category (subdirectory of tests/) and test type (test_*.py file).
With no -c/-t it runs every category in turn and prints a per-category PASS/FAIL summary.

Options:
  -c <category>   Run one category: $(IFS='/'; echo "${VALID_CATEGORIES[*]}")   (default: all categories)
  -t <test_type>  Run one test_*.py file within -c; name normalized so 'score',
                  'test_score' and 'test_score.py' all resolve                (default: all files in category)
  -k <expr>       pytest -k keyword expression                                (default: none)
  -v              Verbose (pytest -v)
  -n              Dry run (print commands, do not execute)
  -h              Show this help message

Everything after '--' is forwarded verbatim to pytest.

Examples:
  # Run everything with a per-category summary
  $0

  # Run one category
  $0 -c analysis

  # Run a single test file (these are equivalent)
  $0 -c analysis -t score
  $0 -c analysis -t test_score.py

  # Keyword filter across the whole tree
  $0 -k distribution

  # Forward flags to pytest
  $0 -c utils -- -x --maxfail=1
EOF
    exit 1
}

############################
# Defaults
############################
category=""
test_type=""
keyword=""
verbose=false
dry_run=false

############################
# Parse arguments
############################
while getopts ":c:t:k:vnh" opt; do
    case $opt in
        c) category="$OPTARG" ;;
        t) test_type="$OPTARG" ;;
        k) keyword="$OPTARG" ;;
        v) verbose=true ;;
        n) dry_run=true ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :)  echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done
shift $((OPTIND - 1))
passthrough=("$@")   # anything after `--` goes to pytest

############################
# Validate
############################
if [[ -n "$category" ]] && ! printf '%s\n' "${VALID_CATEGORIES[@]}" | grep -qx "$category"; then
    echo "Invalid category: $category" >&2
    echo "Valid categories: ${VALID_CATEGORIES[*]}" >&2
    exit 1
fi

resolved_file=""
if [[ -n "$test_type" ]]; then
    if [[ -z "$category" ]]; then
        echo "-t requires -c (a bare test-type name is ambiguous across categories)." >&2
        exit 1
    fi
    # Normalize: strip .py, ensure test_ prefix, re-add .py.
    file="${test_type%.py}"
    [[ "$file" != test_* ]] && file="test_$file"
    file="$file.py"
    if [[ ! -f "$TESTS_DIR/$category/$file" ]]; then
        echo "Test type not found: $category/$file" >&2
        available=()
        for f in "$TESTS_DIR/$category"/test_*.py; do
            [[ -e "$f" ]] && available+=("$(basename "$f")")
        done
        echo "Available in $category: ${available[*]}" >&2
        exit 1
    fi
    resolved_file="$file"
fi

############################
# Run
############################
run_pytest() {  # $1 = target path relative to repo root
    local cmd=(uv run --extra dev pytest "$1")
    $verbose && cmd+=(-v)
    [[ -n "$keyword" ]] && cmd+=(-k "$keyword")
    cmd+=("${passthrough[@]}")
    if $dry_run; then
        echo "[DRY RUN] (cd $REPO_ROOT && ${cmd[*]})"
        return 0
    fi
    (cd "$REPO_ROOT" && "${cmd[@]}")
}

# Single selection: run once and propagate pytest's own exit code.
if [[ -n "$resolved_file" ]]; then
    if run_pytest "tests/$category/$resolved_file"; then exit 0; else exit $?; fi
elif [[ -n "$category" ]]; then
    if run_pytest "tests/$category"; then exit 0; else exit $?; fi
fi

# Default: loop over every category and aggregate.
if $dry_run; then
    for cat in "${VALID_CATEGORIES[@]}"; do
        run_pytest "tests/$cat"
    done
    exit 0
fi

overall_rc=0
summary=()
for cat in "${VALID_CATEGORIES[@]}"; do
    echo "===== category: $cat ====="
    if run_pytest "tests/$cat"; then rc=0; else rc=$?; fi
    if [[ $rc -eq 0 ]]; then
        status="PASS"
    elif [[ $rc -eq 5 ]]; then
        status="WARN (no tests collected)"
    else
        status="FAIL (exit $rc)"
        overall_rc=1
    fi
    summary+=("$(printf '  %-12s %s' "$cat" "$status")")
done

echo
echo "===== summary ====="
printf '%s\n' "${summary[@]}"
exit $overall_rc
