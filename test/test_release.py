"""Tests for the release utils."""

import os
import shutil
import sys

import pytest

from extra.release import changelog_as_markdown

pytestmark = pytest.mark.skipif(
    not (
        (os.environ.get("GITHUB_ACTIONS") == "true" and sys.platform != "win32")
        or bool(shutil.which("pandoc"))
    ),
    reason="pandoc isn't available",
)


@pytest.fixture
def rst_changelog():
    return """New features:

* :doc:`/plugins/substitute`: Some substitute
  multi-line change.
  :bug:`5467`
* :ref:`list-cmd` Update.

Bug fixes:

* Some fix that refers to an issue.
  :bug:`5467`
* Some fix that mentions user :user:`username`.

Empty section:

Other changes:

* Changed `bitesize` label to `good first issue`. Our `contribute`_ page is now
  automatically populated with these issues. :bug:`4855`

.. _contribute: https://github.com/beetbox/beets/contribute

2.1.0 (November 22, 2024)
-------------------------

Bug fixes:

* Fixed something."""


@pytest.fixture
def md_changelog():
    return r"""### New features

- Command **`list`**: Update.
- Plugin **`substitute`**: Some substitute multi-line change. :bug: (\#5467)

### Bug fixes

- Some fix that mentions user @username.
- Some fix that refers to an issue. :bug: (\#5467)

### Other changes


# 2.1.0 (November 22, 2024)
- Changed `bitesize` label to `good first issue`. Our [contribute](https://github.com/beetbox/beets/contribute) page is now automatically populated with these issues. :bug: (\#4855)

### Bug fixes

- Fixed something."""  # noqa: E501


def test_convert_rst_to_md(rst_changelog, md_changelog):
    actual = changelog_as_markdown(rst_changelog)

    assert actual == md_changelog
