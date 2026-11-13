# LLM-serving-webpage

## Static analysis and tests

Install the lint tool and run static analysis:

```sh
python -m pip install -r requirements-dev.txt
ruff check .
```

Run the unit tests with:

```sh
python -m unittest discover -v
```

The deployment workflow runs static analysis before the unit tests.