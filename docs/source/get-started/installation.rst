Installation
============

As it is still in heavy development, to install Synaesthesia, we currently support only the installation through a submodule.

To install Synaesthesia, you can run the following command:
.. code-block:: bash

    git submodule add git@github.com:danieledema/synaesthesia.git .submodules/synaesthesia

This will clone the repository in the `synaesthesia` folder.

You can then install the requirements by running:
.. code-block:: bash

    uv add .submodules/synaesthesia

Then, you can import the module in your Python code as follows:
.. code-block:: python

    from synaesthesia.abstract.multi_signal_dataset import MultiSignalDataset
