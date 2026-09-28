# Version and Branch Usage

This project uses branches for work in progress and Git tags/releases for stable,
installable versions. A branch is not a released version.

## Which reference should I use?

| Need | Git reference | Example |
| --- | --- | --- |
| Stable production release | Git tag or GitHub Release | `v0.2.0` |
| Test unreleased work | Feature branch | `feature/DATA-007-hf-query-compatibility` |
| Pin a source installation | Git tag in the install URL | `@v0.2.0` |

## Use a stable release

```bash
git clone https://github.com/sytse06/multimodal-rag.git
cd multimodal-rag
git checkout v0.2.0
uv sync
```

The version in `pyproject.toml` and the Git tag must match. A tagged release is the
recommended reference for production or colleague onboarding.

The project can also be installed from a pinned Git tag when an environment already
uses `uv`:

```bash
uv pip install "git+https://github.com/sytse06/multimodal-rag.git@v0.2.0"
```

## Test unreleased work

Epic work is developed on a feature branch branched from `main`:

```bash
git fetch origin
git switch --track origin/feature/DATA-007-hf-query-compatibility
uv sync
```

Feature branches are suitable for review and functional testing. They are moving
references and must not be treated as production dependencies.

## Release workflow

1. Create a feature branch from the current `main`.
2. Implement and test the change on that branch.
3. Merge the accepted work into `main`.
4. Update the project version in `pyproject.toml`.
5. Commit the version bump on `main`.
6. Create the matching annotated Git tag and push it with the release commit.
7. Publish the corresponding GitHub Release.

Example:

```bash
git switch main
git pull --ff-only origin main
# update pyproject.toml to 0.2.0, then commit
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin main
git push origin v0.2.0
```

Do not ask users to install from a moving feature branch for production use. A separate
long-lived version branch is only justified when two product variants must be maintained
in parallel; otherwise, use one feature branch followed by a merge and release tag.

## Epic 10 example

Epic 10 is developed on:

```text
feature/DATA-007-hf-query-compatibility
```

The current Ollama/static-collection path remains available on `main` while the hosted
Hugging Face query-embedding path is validated. The project version should not be bumped
until Epic 10 is accepted and merged.
