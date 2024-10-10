# Synaesthesia
## Create general Pytorch data pipelines from simple Python extendable to any sensors

Synaesthesia is a Python library that forms the foundation of a dataset stack in any Pytorch/Pythorch Lightning project.
It contains base datasets and structures that enable combination, sequencing, concatenation and other transformations through composition mechanisms.

## Usage
The easiest way of using Synaesthesia is to clone it as a submodule of your system, like in:
```bash
git submodule add git@github.com:danieledema/synaesthesia.git .submodules/synaesthesia
```
Finally, you can use `poetry` to manage the required packages, by including the submodule in the installation path, as:
```bash
poetry add .submodules/synaesthesia
```
