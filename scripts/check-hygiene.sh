#!/usr/bin/env bash
# scripts/check-hygiene.sh
# Two repository-wide rules, enforced from the first commit:
#   1. no Chinese characters anywhere in tracked files
#   2. no AI attribution in tracked files or in any commit message
# Rule 2 targets attribution *shapes*. The words "Claude" and "Anthropic" are
# legitimate here: the project is named nano-claude-code, NOTICE carries a
# trademark statement, and providers/anthropic.py is a real module.
#
# Everything below matches at the byte level with plain `grep -E`, on
# purpose: stock macOS ships a grep with no -P (PCRE) support, and its bundled
# bash (3.2) does not support the \u/\U Unicode escapes in $'...' quoting
# (both silently do the wrong thing rather than error, which is worse than
# either failing loudly or just not being used). A raw byte sequence, or a
# byte-range pattern, matches the same way regardless of grep vendor as long
# as byte-oriented (not collation-aware) matching is forced explicitly below,
# rather than left to whatever locale the runner happens to start in.
set -uo pipefail
export LC_ALL=C

fail=0

# CJK Unified Ideographs, U+4E00-U+9FFF, is a contiguous block of 3-byte UTF-8
# sequences. Split at the byte1=0xE4 boundary because that leading byte is
# shared with codepoints below U+4E00 (down to U+4000), so its byte2 range is
# narrowed to what U+4E00 and up actually produces; 0xE9 needs no such split
# because U+9000-U+9FFF (0xE9's whole range) sits entirely inside the block.
cjk_pattern=$'\xe4[\xb8-\xbf][\x80-\xbf]|[\xe5-\xe8][\x80-\xbf][\x80-\xbf]|\xe9[\x80-\xbf][\x80-\xbf]'
if git ls-files -z | xargs -0 grep -nE "$cjk_pattern" 2>/dev/null; then
  echo "hygiene: Chinese characters found in tracked files (see above)" >&2
  fail=1
fi

# The attribution shapes below live in their own data file rather than in this
# script, so that only that small, needle-bearing file — not this whole
# script — has to sit outside the scan it feeds. This file is otherwise
# scanned like any other tracked file: a shape added directly here, outside
# that data file, is caught like it would be anywhere else.
script_dir="$(cd "$(dirname "$0")" && pwd)"
patterns_file="$script_dir/attribution-patterns.txt"

# A missing or unreadable sidecar must fail loudly, the same way on every
# bash this project supports, rather than differ by version: on bash 3.2
# (stock macOS) the redirect below fails and the later "${patterns[@]}"
# expansion aborts under `set -u` because that version treats an empty
# array as an unset parameter; bash 4.4+ (ubuntu-latest, the platform CI
# actually enforces this on) fixed that, so the same missing file instead
# leaves `patterns` empty, the loop below runs zero times, and the script
# would print "hygiene: ok" having scanned nothing. Require the file first
# so both versions fail the same way, loudly, before that split matters.
[ -r "$patterns_file" ] || {
  echo "hygiene: cannot read $patterns_file" >&2
  exit 1
}

patterns=()
while IFS= read -r line || [ -n "$line" ]; do
  patterns+=("$line")
done <"$patterns_file"

# A present-but-corrupted sidecar (truncated, emptied, one line lost to a bad
# edit) would pass the readability check above and still reach the same
# checked-nothing-but-said-ok state, so this is a floor on the count. It is
# a floor and not an exact count *deliberately* — do not "tighten" it back
# to -eq: ci.yml's hygiene job runs only this script, with no Python and no
# pytest, so this is the one place that can catch a drop below five without
# a test suite; but pinning it to exactly five would make adding a
# legitimate sixth pattern break this script too, which is part of what
# moving the patterns out of it was for. The exact set — not just the count,
# so a swap to five *wrong* lines cannot pass either — is asserted in
# tests/test_hygiene.py, which already reads this file's content directly
# and is the place to update when the pattern set changes.
[ "${#patterns[@]}" -ge 5 ] || {
  echo "hygiene: expected at least 5 attribution patterns in $patterns_file, found ${#patterns[@]}" >&2
  exit 1
}

for pattern in "${patterns[@]}"; do
  if git ls-files -z -- . ':(exclude)scripts/attribution-patterns.txt' \
      | xargs -0 grep -nEi "$pattern" 2>/dev/null; then
    echo "hygiene: AI attribution found in tracked files: $pattern" >&2
    fail=1
  fi
  if git log --format=%B 2>/dev/null | grep -nEi "$pattern" >/dev/null; then
    echo "hygiene: AI attribution found in a commit message: $pattern" >&2
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then echo "hygiene: ok"; fi
exit "$fail"
