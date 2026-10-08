# 0007: Edit by content, not by line number

Status: accepted, implemented in `tools/edit.py` and `tools/fs.py`.

`Edit` takes a path and a list of edits, each with an `old_string`, a `new_string` and an optional `replace_all`. It finds the old text in the file and replaces it. The batch is applied in order, each edit to the result of the one before, and either every edit applies or the file is left as it was.

## Why

Models miscount lines and do not miscount text. A wrong line number silently edits the wrong place; a wrong string is an error the model can see and correct. So an edit is addressed by what it replaces, and four rules follow from that.

The file must have been read in this session, and unchanged since. `Read` records a stamp of what it saw (a hash, a size and a time), `Edit` and `Write` compare it with the file as it is, and a file that changed on disk is refused with an instruction to read it again. A model that edits from a stale picture of a file is the common way an edit lands in the wrong place (`test_editing_without_reading_first_is_refused`, `test_editing_a_file_that_changed_on_disk_is_refused`).

An `old_string` must match exactly once, unless `replace_all` is set. A second match is not guessed at: the error says how many there were, which is what tells the model to add context or ask for all of them (`test_a_non_unique_old_string_is_an_error_naming_the_count`).

The batch is atomic. A rename that touches four places must not be able to apply three of them, because the result is neither the old code nor the new and does not compile. A later edit that fails discards the whole batch, and the file is byte for byte what it was (`test_a_failing_later_edit_discards_the_whole_batch`, `test_a_failing_edit_in_a_batch_leaves_the_file_byte_identical`). The write itself goes to a temporary file in the same directory, is flushed, and is moved into place, so a crash cannot leave half of a source file.

What a model types and what a file holds disagree in two places, both handled and not reported. Curly quotes in the file match straight quotes from the model and the other way round (`test_curly_quotes_in_the_file_match_straight_quotes_from_the_model`), and a file's line endings are detected and kept, so an edit to a CRLF file does not make a file with both (`test_crlf_line_endings_survive_an_edit`). An edit that would change nothing is refused. A successful edit returns a unified diff.

`Read` prefixes each line with its number and a tab. The numbers let the model talk about the file and choose a next window; they are not an addressing scheme, and the tool's description tells the model never to put one in an edit.

## Costs

The model has to reproduce the text it replaces, exactly, and for a large block that is many tokens of output for a small change. Whitespace that differs by one character is an error and not a near match, so a model working from memory of a file fails often enough to cost a turn.

Matching is lenient in one place and only one: quotes. A match found by folding curly quotes is replaced with the text as the model typed it, so a file that used curly quotes in that spot has straight ones afterwards.

A line that `Read` showed redacted (0012) cannot be matched, since the file holds what the redaction replaced. An edit that has to quote such a line fails with "not found" and the line is changed by hand.

Edits in a batch apply one after another, so an earlier edit can change what a later one matches. That is deliberate (a rename followed by a use of the new name) and it is also how a batch can fail in a way that looks odd.

## Rejected alternatives

- Line-number addressing. It is compact, and a count that is off by one edits the wrong line without any error.
- A unified diff from the model. Models write hunk headers with wrong counts and context that does not match, and a patch that applies with an offset is an edit in the wrong place that nobody noticed.
- Rewriting the whole file for every change (`Write` alone). It is simple and costs the file's length in output tokens for every edit, and the model drops the lines it did not mean to touch.
- Fuzzy matching, with whitespace folded or a similarity threshold. It turns a refusal that costs a turn into a wrong edit that costs a debugging session.
- One edit per call. Four calls for a rename, with a failure on the second, leaves the code half changed.
