# Releasing

Releases go to PyPI from GitHub Actions, through
[trusted publishing](https://docs.pypi.org/trusted-publishers/): PyPI accepts an upload only
from `.github/workflows/release.yml` in this repository, running in the `pypi` environment.
No API token exists, so none can leak.

There has been no release yet. The README plans the first package on PyPI for v0.4.

## One-time setup

Both steps need the maintainer's own accounts.

1. **PyPI.** Sign in at pypi.org, open *Your account → Publishing*, and add a pending publisher:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `nano-claude-code` |
   | Owner | `jason-te-sde` |
   | Repository name | `nano-claude-code` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The project does not exist on PyPI yet; the first successful upload creates it.

2. **GitHub.** In the repository's *Settings → Environments*, create an environment named
   `pypi`. Adding yourself as a required reviewer makes every upload wait for a click, which is
   worth it: a published version can be yanked but never replaced.

## Cutting a release

1. Set the version in `pyproject.toml` and move the `Unreleased` notes in `CHANGELOG.md` under a
   heading for that version. Merge that through a pull request like any other change.
2. Tag the merge commit on `main` and push the tag:

   ```bash
   git tag -a v0.4.0 -m "v0.4.0"
   git push origin v0.4.0
   ```

3. The `release` workflow then:
   - runs the whole CI suite on the tagged commit;
   - refuses a tag that does not match the version in `pyproject.toml`;
   - builds the sdist and the wheel once, checks them with `twine check --strict`, and runs
     `ncc --version` from the wheel installed outside the checkout;
   - waits for the `pypi` environment's approval, if you set one, and uploads.

A failed run publishes nothing. Fix the cause, delete the tag locally and on GitHub, and tag
again; PyPI never saw that version.
