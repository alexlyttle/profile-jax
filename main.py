import os
import pathlib
import jax


def main():
    log_dir = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "jax-trace"
    with jax.profiler.trace(log_dir, create_perfetto_link=True):
        # Run the operations to be profiled
        key = jax.random.key(0)
        x = jax.random.normal(key, (5000, 5000))
        y = x @ x
        y.block_until_ready()


if __name__ == "__main__":
    main()
