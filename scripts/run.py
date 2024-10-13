import subprocess


def docker():
    # Build the Docker image
    build_command = [
        "docker",
        "build",
        "-t",
        "passive_sound_localization",
        ".",
    ]
    subprocess.run(build_command, check=True)

    # Run the Docker container
    run_command = [
        "docker",
        "run",
        "--name",
        "passive_sound_localization",
        "passive_sound_localization",
    ]
    subprocess.run(run_command, check=True)
