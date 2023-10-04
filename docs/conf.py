# isort: skip_file
import sys
from datetime import datetime
from pathlib import Path
from sphinx.application import Sphinx
import pytometry

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from lamin_sphinx import *  # noqa
from lamin_sphinx import html_context, html_theme_options, extensions  # noqa

project = "Pytometry"
html_title = "Pytometry"
release = pytometry.__version__
html_context["github_repo"] = "pytometry"  # noqa
# We're actually using this for the link behind the brand of the page!
html_theme_options["logo"] = {"link": "pytometry"}  # noqa
extensions += ["sphinx.ext.intersphinx"]

ogp_site_url = "https://pytometry.netlify.app"

author = "pytometry developers"
copyright = f"{datetime.now():%Y}, {author}"

html_context = {
    "default_mode": "auto",
    "github_user": "buettnerlab",
    "github_version": "main",
}

html_logo = (
    "https://raw.githubusercontent.com/buettnerlab/pytometry/main/docs/_static/logo.svg"
)
html_favicon = (
    "https://raw.githubusercontent.com/buettnerlab/pytometry/main/docs/_static/logo.ico"
)
templates_path = ["_templates", "../lamin_sphinx/_templates"]
html_static_path = ["_static", "../lamin_sphinx/_static"]
ogp_image = (
    "https://raw.githubusercontent.com/buettnerlab/pytometry/main/docs/_static/logo.svg"
)
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable", None),
}


def setup_replacement(app: Sphinx):
    app.warningiserror = False
    app.add_css_file("custom.css")


setup = setup_replacement
