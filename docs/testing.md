# Testing

The default test suite is deterministic and does not access NASA POWER or
require a model checkpoint:

```powershell
E:\QuzhouAITrials\.venv\Scripts\python.exe -m unittest discover -s tests
```

Tests that construct the historical environments backed by NASA POWER are
classified as network integration tests. Run them explicitly with:

```powershell
$env:RUN_NETWORK_TESTS="1"
E:\QuzhouAITrials\.venv\Scripts\python.exe -m unittest discover -s tests
```

The historical model-loading test is a model-integration test. It also needs
the explicit `QUZHOU_TEST_MODEL_PATH` environment variable and, unless the
statistics file is adjacent to the model, `QUZHOU_TEST_ENV_STATS_PATH`.
These variables must point to local files supplied by the operator; no model
or VecNormalize artifact is stored in the repository.
