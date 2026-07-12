# TokenCode Python SDK (Beta)

Build Python applications that start TokenCode threads, run turns, stream progress,
and control workspace access.

## Install

Install the SDK:

```bash
pip install tokencode-sdk
```

## Quickstart

The SDK reuses your existing TokenCode authentication when one is already
available:

```python
from tokencode_sdk import Codex

with Codex() as codex:
    thread = codex.thread_start()
    result = thread.run("Explain this repository in three bullets.")
    print(result.final_response)
```

`thread.run(...)` returns a `TurnResult` containing the final response,
collected items, and token usage.

## Authentication

Existing TokenCode authentication is reused automatically. To start ChatGPT
browser login explicitly:

```python
from tokencode_sdk import Codex

with Codex() as codex:
    login = codex.login_chatgpt()
    print(login.auth_url)
    print(login.wait().success)
```

For device-code login:

```python
with Codex() as codex:
    login = codex.login_chatgpt_device_code()
    print(login.verification_url, login.user_code)
    login.wait()
```

For API-key login:

```python
with Codex() as codex:
    codex.login_api_key("sk-...")
```

## Built-In Help

Use Python's standard `help(tokencode_sdk)`, `help(Codex)`, or
`python -m pydoc tokencode_sdk` documentation tools.

## Documentation

- [Getting started](https://github.com/miuiadmin/tokencode/blob/main/src/sdk/python/docs/getting-started.md)
- [API reference](https://github.com/miuiadmin/tokencode/blob/main/src/sdk/python/docs/api-reference.md)
- [FAQ](https://github.com/miuiadmin/tokencode/blob/main/src/sdk/python/docs/faq.md)
- [Examples](https://github.com/miuiadmin/tokencode/blob/main/src/sdk/python/examples/README.md)

The package is licensed under the
[repository Apache License 2.0](https://github.com/miuiadmin/tokencode/blob/main/src/LICENSE).
