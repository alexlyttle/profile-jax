# Profile JAX

These are my experiements profiling JAX code. Currently, `main.py` does the same as in
the [Profiling Computation](https://docs.jax.dev/en/latest/profiling.html) page of the
JAX documentation.

## Usage

1. Install uv

2. Run application

    ```bash
    uv run main.py
    ```

3. Open URL in browser

### BlueBEAR

To open the Perfetto URL in the browser on your local machine, you first need to forward
the port by running:

```bash
ssh -L 9001:127.0.0.1:9001 <username>@bluebear.bham.ac.uk
```

and then from the BlueBEAR login node:

```bash
ssh -L 9001:127.0.0.1:9001 bear-pgXXXXX
```

where `bear-pgXXXXX` is the name of the compute node running the application.
