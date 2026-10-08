# 0012: Scrub what a tool returns before it becomes a transcript block

Status: accepted, implemented in `permissions/redact.py` and used by every tool that returns file content.

Credentials are kept out of the conversation in two layers. The first is by path. A path that is a credential by convention (`.env` and `.env.*`, `*.pem`, `*.key`, `id_rsa` and the other SSH keys, `credentials` and `credentials.*`, anything under `.aws/` or `.ssh/`, `*.p12`, `*.pfx`, `secrets.*`) is refused by the policy under the rule id `secret.path`, whatever the mode and whatever the rules say. `Grep` drops matches from such files, and an `@` mention of one is left as written and not inlined. The second is by content. Text that a tool returns passes through `Redactor.scrub` before it becomes a block of the transcript, and what looks like a credential is replaced with `[redacted:<kind>]`.

## Why

The transcript is not a display. It is stored, it is loaded again by `--resume`, and all of it is sent to the model on every turn. A secret that is hidden only when it is drawn on the screen is still in the database and still in the next request. So the scrub is at the one point every tool result has to pass, before the block exists, and what is stored and what is resumed are clean along with what is shown.

The content rules are narrow on purpose. Nine patterns for eight kinds of credential are recognised, each by what it looks like: an AWS access key id, a GitHub token (the prefixed forms and the fine-grained one), an Anthropic key, an OpenAI key, a Slack token, a Google API key, a private-key header, and a three-part JWT. And one generic rule: a name that says secret (`SECRET`, `TOKEN`, `PASSWORD`, `API_KEY` and a few more) followed by `=` or `:` and a value of sixteen or more characters whose Shannon entropy is at least 3.5 bits per character. High entropy alone is not a rule, since a base64 image, a git hash and a minified script are all high-entropy and none is a secret.

The tests that matter most are the ones that say what must not be hit: `test_ordinary_content_is_left_alone`, `test_a_low_entropy_assignment_is_left_alone` and `test_paths_that_merely_look_similar_are_not`. A scrubber that is tested only on what it catches cannot be told from one that catches everything.

The same scrubber is applied wherever a tool's text could carry a credential in: `Read` and `Grep` output, a mentioned file (scrubbed before it is cut to its size limit, so a key that straddles the cut cannot survive as an unrecognised half), the text of an internal error, and the arguments of a call as they are written to the audit table. The last matters because the audit row is the one place a secret could survive: it records what the model asked for and not what the tool returned (`test_arguments_are_stored_redacted`).

`--allow-secrets` turns both layers off together, with a warning at start, and an allow rule for one path does not (the path rule comes before every allow rule). One redactor is shared by the tools and the audit, so the switch cannot turn one off and leave the other on (`test_allow_secrets_stops_the_redaction_and_the_refusal_of_credentials_files`).

## Costs

A pattern catches what it knows the shape of. A database URL with the password inline, a token in a provider's own format, a password in prose: none is caught. `expand_mentions` says as much about itself and refuses secret-looking paths for that reason, rather than relying on the content rules.

What the person types is theirs and is not scrubbed, and neither is what the model writes: the prompt and the model's own text and tool arguments go into the transcript as they are. Redaction covers what `Read`, `Grep`, mentions and internal errors return, and the audit's copy of what the model asked for.

The diff that `Edit` returns is not scrubbed. It holds the lines around the edit as they are on disk, so a credential that sits near the line being changed in a source file reaches the transcript in clear, though `Read` showed it redacted. The path rule keeps the usual credential files out of reach of `Edit`, since a file cannot be edited before it has been read; this is the gap in the content layer that remains, and it is not covered by a test.

A false positive costs something. A line replaced by a marker is a line the model cannot quote, so an edit that needs it fails to match and that line is changed by hand.

The path list is a convention and not a guarantee: a credential in a file called `config.yaml` is judged by its content alone.

Redaction arrived in v0.1, where the original plan had it in v0.2, because `Read`, `Grep` and mentions all needed it on the first day.

## Rejected alternatives

- Scrub when the text is drawn. The secret is then in the stored transcript and in every later request, which is what the scrub is for.
- Scrub the request on its way to the provider. The request would be clean and the database not, and `--resume` would load the secret again.
- Rely on the path rule alone. It misses a key pasted into a source file and a token printed by a command.
- A high-entropy rule that stands alone. It redacts images, hashes and bundled scripts, and the model then works on a file full of markers.
- Rely on the provider not to retain what it is sent. It is not a promise the project can make for a provider.
