# TokenCode CLI Runtime for Python SDK

Platform-specific runtime package consumed by the published `tokencode-sdk`.

This package is staged during release so the SDK can pin an exact TokenCode CLI
version without checking platform binaries into the repo.

`tokencode-cli-bin` is intentionally wheel-only. Do not build or publish an
sdist for this package.
