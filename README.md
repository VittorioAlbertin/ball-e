
to activate the venv from container
    `source ~/ml_env/bin/activate`

`ml_env` is bind mouted in the host `.venv` with all the python packages installed in order to be persistant between builds and not force a reinstallation each time some small change is made to `.devcontainer/`.

`.venv` is not actually recognized as a virtual enviroment by the host system, its just saved and used in the container.