from setuptools import setup

setup(
    name="slim",
    version="0.4.0",
    packages=["slim", "slim.types", "slim.gui_utils", "slim.simulation"],
    url="https://github.com/resistance-modelling/slim",
    license="MIT",
    author="Enrico Trombetta, Jessica Enright, Anthony O' Hare",
    author_email="",
    description="Sea Lice Simulator",
    entry_points={
        "console_scripts": [
            "slim=slim:launch",
        ]
    },
)
