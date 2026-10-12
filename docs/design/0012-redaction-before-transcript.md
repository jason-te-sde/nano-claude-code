# 0012: Scrub what a tool returns before it becomes a transcript block

Status: accepted, implemented in `permissions/redact.py` and `tools/keyblocks.py`, and used by every tool that returns file content.

Credentials are kept out of the conversation in two layers. The first is by path. A path on the credentials list ([configuration.md](../configuration.md#the-credentials-list) gives it whole: `.env` in its spellings, `*.pem`, `*.key`, the SSH keys, `.aws/`, `.ssh/` and `.gnupg/`, the login files of everyday tools such as `.netrc` and `.npmrc`, keystores and `*.tfstate`, among others) is refused by the policy under the rule id `secret.path`, whatever the mode and whatever the rules say, and is compared as written and again after case folding. The same question, whether the policy would let `Read` show the file, is asked for every other way a file reaches the model. `Grep` drops the files that fail it, an `@` mention of one is not inlined and a line says so, the project map leaves it out, and an instruction file that fails it is not loaded. The second layer is by content. Text that a tool returns passes through `Redactor.scrub` before it becomes a block of the transcript, and what looks like a credential is replaced with `[redacted:<kind>]`.

## Why

The transcript is not a display. It is stored, it is loaded again by `--resume`, and all of it is sent to the model on every turn. A secret that is hidden only when it is drawn on the screen is still in the database and still in the next request. So the scrub is at the one point every tool result has to pass, before the block exists, and what is stored and what is resumed are clean along with what is shown.

The content rules are narrow on purpose. Each is recognised by what it looks like, with a length floor where a vendor documents none. The kinds are named in the marker: `aws-key`, `npm-token`, `gitlab-token`, `stripe-key`, `sendgrid-key`, `huggingface-token`, `pypi-token`, `github-token`, `anthropic-key`, `openai-key`, `slack-token`, `google-key`, `private-key`, `jwt`, `bearer-token`, `url-password` and `assigned-secret`. The last three are not a fixed shape. A bearer token is what follows `Authorization: Bearer`, and is left alone when it is a placeholder written in one case with no digit. A URL password is what sits between the colon and the `@` of `scheme://user:password@host`, and is left alone when it is a reference (`${DB_PASSWORD}`, `<password>`). An assigned secret is the one generic rule: a name that says secret (`SECRET`, `TOKEN`, `PASSWORD`, `API_KEY` and a few more, in any case, as a word of the name where it is not in capitals) followed by `=` or `:` and a value of sixteen or more characters whose Shannon entropy is at least 3.5 bits per character. An unquoted value that is code (a call, an attribute chain, a subscript, `None`, `null`, `true`) is left alone. High entropy alone is not a rule, since a base64 image, a git hash and a minified script are all high-entropy and none is a secret.

The tests that matter most are the ones that say what must not be hit: `test_ordinary_content_is_left_alone`, `test_a_low_entropy_assignment_is_left_alone` and `test_paths_that_merely_look_similar_are_not`. A scrubber that is tested only on what it catches cannot be told from one that catches everything.

A private-key block is masked whole, from its `BEGIN` line to its `END` line, and not line by line: its body is base64, and nothing in one line of it says that it is a key. That matters wherever a window of a file is shown (`Read` with an offset and a limit, `Grep`'s matches and the lines around them, the context of the diff `Edit` returns), because a window that begins or ends inside a block would show the key. So the extent of a block is worked out on the whole file first (`tools/keyblocks.py`), and a line inside one is shown as a marker. A public certificate or a public key is not a block of this kind and stays readable. Pinned by `test_no_window_of_a_file_shows_a_line_of_the_key_in_it`, `test_a_search_for_a_piece_of_the_key_does_not_show_the_line_it_is_in` and `test_the_diff_does_not_show_the_end_of_a_key_as_context_before_a_change`.

The same scrubber is applied wherever a tool's text could carry a credential in: `Read` and `Grep` output, the diff that `Edit` returns (it holds the lines around the edit as they are on disk, `test_the_diff_an_edit_returns_is_scrubbed_like_every_other_tool_output`), a mentioned file (scrubbed before it is cut to its size limit, so a key that straddles the cut cannot survive as an unrecognised half), an instruction file (the same), the text of an internal error, and the arguments of a call as they are written to the audit table. The last matters because the audit row is the one place a secret could survive: it records what the model asked for and not what the tool returned (`test_arguments_are_stored_redacted`).

`--allow-secrets` turns both layers off together, with a warning at start, and an allow rule for one path does not (the path rule comes before every allow rule). One redactor is shared by the tools and the audit, so the switch cannot turn one off and leave the other on (`test_allow_secrets_stops_the_redaction_and_the_refusal_of_credentials_files`).

## Costs

A pattern catches what it knows the shape of. A token in a provider's own format, a password in prose, a short password: none is caught, and a value in an assignment is replaced only when it is long and random enough. `expand_mentions` says as much about itself and refuses secret-looking paths for that reason, rather than relying on the content rules.

What the person types is theirs and is not scrubbed, and neither is what the model writes: the prompt and the model's own text and tool arguments go into the transcript as they are. Redaction covers what `Read`, `Grep`, `Edit`, mentions, instruction files and internal errors return, and the audit's copy of what the model asked for.

A false positive costs something. A line replaced by a marker is a line the model cannot quote, so an edit that needs it fails to match and that line is changed by hand. A bare word after a name that says secret is replaced when it is random enough, because an identifier cannot be told from a password typed without quotes (`password = a_long_function_name` looks like a password and sixteen random letters), so some code is over-redacted.

Masking a key block has two gaps, and SECURITY.md states both. `Grep`'s `files` and `count` modes and its match counts say whether a pattern matches inside a block, though they never show its text. And an `Edit` that changes the `BEGIN` line of a block, once it is approved, leaves a block that is no longer recognised, so its diff and later reads show the rows.

The path list is a convention and not a guarantee: a credential in a file called `config.yaml` is judged by its content alone.

Redaction arrived in v0.1, where the original plan had it in v0.2, because `Read`, `Grep` and mentions all needed it on the first day.

## Rejected alternatives

- Scrub when the text is drawn. The secret is then in the stored transcript and in every later request, which is what the scrub is for.
- Scrub the request on its way to the provider. The request would be clean and the database not, and `--resume` would load the secret again.
- Rely on the path rule alone. It misses a key pasted into a source file and a token printed by a command.
- A high-entropy rule that stands alone. It redacts images, hashes and bundled scripts, and the model then works on a file full of markers.
- Scrub the lines of a window as they are shown. A window that begins or ends inside a key block holds base64 that nothing in it says is a key, so the block's extent has to be found on the whole file.
- Rely on the provider not to retain what it is sent. It is not a promise the project can make for a provider.
