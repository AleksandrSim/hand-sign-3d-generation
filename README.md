# Hand Sign 3D Generation


## Container build and start options

```
#Python 3.11 + CUDA:
make build
make run
# To start with X11 support
make run-x11

#Python 3.10 + CUDA:
make build-p310
make run-p310

#Python 3.10 + CPU
make build-p310-cpu
make run-p310-cpu
```

To stop a running container: `make stop`; if it doesn't help, try `make kill`.

## Repo structure

* `scripts` - Small, stand alone scripts and entrypoints to run train and other similar tasks.
* `src` - The main project code location.
* `configs` - The configuration files for the project.
* `notebooks` - Jupyter notebooks.
* `tests` - Test code with pytest.


