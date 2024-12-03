# Development

Development uses pre-commit for code linting and formatting. To setup development with pre-commit follow these steps after cloning the repository:

1. Create a virtual environment with python 3.10:

```
python3.10 -m venv venv
```

> [!NOTE]
> It may be necessary to install python 3.10 if it is not already on your system. Some system have python 3.10 as default and  'python' or 'python3' may be used instead.

2. Activate virtual environment:

```
source venv/bin/activate
```

3. Install the requiret python packages:

```
pip install -r requirements.txt
```

4. Install pre-commit package

```
pip install pre-commit
```

5. Install pre-commit hooks

```
pre-commit install
```

You are now ready to contribute.
