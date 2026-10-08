# 0016: The hygiene check, and why it looks for shapes

Status: accepted, enforced by `scripts/check-hygiene.sh` in `scripts/preflight.sh` and in the `hygiene` job of CI.

Two rules are checked on every tracked file and on every commit message in the history, from the first commit onward. No file contains a character from the CJK Unified Ideographs block, the range U+4E00 to U+9FFF; the design discussions that led to this project are kept outside the repository, and everything inside it is English. And nothing in the repository, in a file or in a commit message, credits a model with the work. This project is its author's own work, and the repository carries no credit to a tool.

## Why

The second rule is the interesting one, and it does not look for words. The names of the vendor and of its models are all over a legitimate repository: the project is called nano-claude-code, `NOTICE` carries a required statement of the trademark and of the lack of affiliation, `providers/anthropic.py` is a module, and the model ids in the configuration, the price book and the tests contain them. A check that fails on a word would fail on every one of those. So it looks for five attribution shapes, and these are described here in words because writing one out would make this note fail the check it describes:

- a commit trailer that names a model or its vendor as a co-author;
- a footer line saying that a change was made with a named assistant tool;
- a sentence in which something was written, created or authored by, or with, a named assistant or "an" AI;
- the robot-face emoji;
- the hyphenated adjective that labels content as machine-made.

The patterns themselves are one per line in `scripts/attribution-patterns.txt`, matched without regard to case. They are in their own file so that the one file that has to contain the needles is the only one the scan skips, and a shape added to the script itself would be found like anywhere else. The exact set is pinned by `tests/test_hygiene.py`, which splits each pattern across two source lines so that the test file does not match what it checks, and the script refuses to run with fewer than five (`test_attribution_patterns_file_has_the_five_expected_patterns`).

It is on from the first commit because it cannot depend on being remembered. Tools that write commit messages add their own trailer by default, and the rule is therefore an act of deleting it and not of not writing it; one missed instance has to be undone by rewriting history. The history is scanned, so CI checks out every commit and not only the latest.

The first rule is a byte pattern, not a Unicode-aware one, on purpose. Stock macOS ships a `grep` without `-P`, and its `bash` 3.2 does not understand `\u` escapes in `$'...'` strings, and both fail by quietly doing the wrong thing. The script forces `LC_ALL=C` and matches the three-byte UTF-8 sequences of the block directly, so it behaves the same on a developer's laptop and on the CI runner (`test_repository_passes_hygiene`, `test_legitimate_claude_mentions_are_not_flagged`).

## Costs

It finds five shapes and no others. An attribution in other words passes, and a check that has never failed in review has not been shown to catch the next phrasing. It is a net for the usual forms and not a proof.

It scans tracked files, and `git ls-files` does not list a file that has not been added. A new file is checked from the moment it is staged, and not before, so the check is run after `git add`.

The first rule covers one block. Chinese punctuation, the CJK extension blocks, kana and hangul are not matched.

Writing about the policy is awkward. Anything that quotes a shape fails the scan, so documentation of it, this note included, has to describe the shapes. A file that has to name them, the patterns file, is excluded from the file scan, and a commit message that needs to explain a change to it has to do so without quoting one.

## Rejected alternatives

- Forbid the words. The project's own name would fail it.
- Review alone. A rule that depends on being remembered is forgotten, and finding out afterwards means rewriting a commit history.
- A pre-commit hook alone. It is local and can be skipped, and the rule has to hold for what is pushed. The hook is a fine addition and does not replace CI.
- A Unicode-aware pattern with `grep -P`. It does not run on the macOS that the author uses.
- Strip trailers when pushing, in CI. A history that is rewritten on the way in is one whose commits differ from those on the author's machine.
