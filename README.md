# napari-conference

[![License BSD-3](https://img.shields.io/pypi/l/napari-conference.svg?color=green)](https://github.com/kephale/napari-conference/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-conference.svg?color=green)](https://pypi.org/project/napari-conference)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-conference.svg?color=green)](https://python.org)
[![tests](https://github.com/kephale/napari-conference/workflows/tests/badge.svg)](https://github.com/kephale/napari-conference/actions)
[![codecov](https://codecov.io/gh/kephale/napari-conference/branch/main/graph/badge.svg)](https://codecov.io/gh/kephale/napari-conference)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-conference)](https://napari-hub.org/plugins/napari-conference)

A simple plugin that allows you to use napari + your webcam in video
calls

[napari_conference_example.png](napari_conference_example.png)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

### Prerequisites

You will need to:

- follow `pyvirtualcam`'s installation instructions:
https://github.com/letmaik/pyvirtualcam#installation
- install `napari` from source to get the new async slicing updates 

Note: I needed to install `pyvirtualcam` from source on my MacOS M1
with python=3.10.



You can install `napari-conference` via [pip]:

    pip install napari-conference



To install latest development version :

    pip install git+https://github.com/kephale/napari-conference.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-conference" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/kephale/napari-conference/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
