#!/usr/bin/env bash
set -eu

# Usage:
#    run-clang-tidy.sh [normal|test]
#
# Options:
#    normal   : If specified, run clang-tidy on *.cc files except test files (warnings are treated as errors)
#    test     : If specified, run clang-tidy on *_test.cc files (warnings are ignored)
#
# Notes:
# - This script must be run from the build directory, in which compile_commands.json exits.
#   compile_commands.json is generated by cmake.
# - clang-tidy accepts relative paths from build directory for source files.
#   rel_source_dir is such relative path of the source root directory (chainerx/).

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
root_dir="$(realpath "$script_dir"/..)"
source_dir="$root_dir"/chainerx
build_dir="$(realpath $PWD)"
rel_source_dir="$(python -c 'import os; print(os.path.relpath("'$source_dir'", "'$build_dir'"))')"

if [ $# != 1 ]; then
    echo "Too many or too few arguments." >&2
    exit 1
fi

target="$1"

if [ "$target" != "normal" -a "$target" != "test" ]; then
    echo "Specify either 'normal' or 'test'." >&2
    exit 1
fi


# Check if there is compile_commands.json in the current directory.
if [ ! -f "compile_commands.json" ]; then
    echo "compile_commands.json is not found in the current directory." >&2
    exit 1
fi

# find_command: find command to search for source files
# grep_regex: Regex expression to search for error (or warning) lines

clang_tidy_checks=()
if [ "$target" == "normal" ]; then
    # *.cc files, but not *_test.cc nor files in chainerx/testing
    find_command=(find "$rel_source_dir" -path "$rel_source_dir"/testing -prune -o -name "*.cc" -not -name "*_test.cc" -print0)
    # TODO(niboshi): report warnings too by replacing with the next line.
    #grep_regex="^[^ ]+: (warning|error): .*"
    grep_regex="^[^ ]+: error: .*"
    # Avoid checks that cause warnings which are difficult to fix
    clang_tidy_checks+=(
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay  # on assert
    )

else
    # *_test.cc files in any directories and *_.cc files in chainerx/testing
    find_command=(find "$rel_source_dir" \( -path "$rel_source_dir/testing/*" -name "*.cc" -o -name "*_test.cc" \) -print0)
    grep_regex="^[^ ]+: error: .*"
    # Avoid checks that cause warnings in google-test implementation
    clang_tidy_checks+=(
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay  # on assert
        -cppcoreguidelines-pro-type-vararg  # on EXPECT_EQ, etc.
        -cppcoreguidelines-special-member-functions  # on TEST, etc.
        -cert-err58-cpp  # on TEST, etc.
        -modernize-use-equals-default  # on TEST, etc.
        -modernize-use-equals-delete  # on TEST, etc.
        -cppcoreguidelines-owning-memory  # on TEST, etc.
        -cppcoreguidelines-avoid-goto  # on EXPECT_THROW
        -readability-function-size  # Easily raised with macros
        -google-readability-function-size  # ditto.
    )
fi

clang_tidy_opts=()
if [ ${#clang_tidy_checks[@]} -gt 0 ]; then
   clang_tidy_opts+=(-checks=$(printf "%s," "${clang_tidy_checks[@]}"))
fi

# Run clang-tidy.
# xargs can split into multiple invocations of clang-tidy depending on the number of input files.
# Currently it does not cause a problem because the awk script simply counts the matching lines line-by-line.
# Keep that in mind when the script is to be modified.

set +e  # Temporarily disable error detection to capture errors in the pipes
"${find_command[@]}" | parallel --no-notice -0 clang-tidy "${clang_tidy_opts[@]+"${clang_tidy_opts[@]}"}" | awk '
    { print }
    /'"$grep_regex"'/ { n = n == 255 ? 255 : n+1 }
    END { exit n }'

PIPESTATS=("${PIPESTATUS[@]}")
set -e  # Restore error detection

# If any command in the pipes (except awk) fails, return the maximum possible number
if [ ${PIPESTATS[0]} != 0 -o ${PIPESTATS[1]} != 0 ]; then
    echo "Some command in the pipes has failed.">&2
    echo "PIPESTATUS: ${PIPESTATS[@]}" >&2
    exit 255
fi

# Report the number of clang-tidy errors (return code of awk)
exit ${PIPESTATS[2]}
