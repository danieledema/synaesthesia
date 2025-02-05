Installation
============

You can install Synaesthesia in three different ways:

1. Using `pip` (or like here `uv`)

.. code-block:: bash

    uv add synaesthesia

2. Installing from source

.. code-block:: bash

    git clone git@github.com:danieledema/synaesthesia.git
    cd synaesthesia
    uv build

and then you can install the package you can find in the `dist` folder:

3. Using Synaesthesia as a submodule

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
