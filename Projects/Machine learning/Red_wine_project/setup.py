# import setuptools

# with open("README.md", "r", encoding="utf-8") as f:
#     long_description = f.read()


# __version__ = "0.0.0"

# REPO_NAME = "End-to-end-ML-Project"
# AUTHOR_USER_NAME = "entbappy"
# SRC_REPO = "Red_Wine_Prediction"
# AUTHOR_EMAIL = "entbappy73@gmail.com"


# setuptools.setup(
#     name=SRC_REPO,
#     version=__version__,
#     author=AUTHOR_USER_NAME,
#     author_email=AUTHOR_EMAIL,
#     description="A small python package for ml app",
#     long_description=long_description,
#     long_description_content="text/markdown",
#     url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
#     project_urls={
#         "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
#     },
#     package_dir={"": "src"},
#     packages=setuptools.find_packages(where="src")
# )





#imp.

# src becomes the package root because setup.py explicitly tells Python to look inside src/ for packages.

# The 2 lines that do everything
# package_dir={"": "src"},
# packages=setuptools.find_packages(where="src"),

# What they mean

# package_dir={"": "src"}
# → “All Python packages live inside the src folder”

# find_packages(where="src")
# → “Find all folders inside src that contain __init__.py”

# Example
# src/
# └── Red_wine_Prediction/
#     ├── __init__.py
#     ├── utils/
#     └── pipeline/


# Python sees:

# Red_wine_Prediction
# Red_wine_Prediction.utils
# Red_wine_Prediction.pipeline

# Important note

# ❌ You never import src

# ✅ You import the package inside it

# from Red_wine_Prediction.utils import common

# Why this is used

# Prevents accidental imports

# Works the same in local, Docker, CI/CD

# Industry best practice

# One-line takeaway

# src is just a container; the real Python package is the folder inside it.



import setuptools
from typing import List

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

__version__ = "0.0.0"

REPO_NAME = "End-to-end-ML-Project"
AUTHOR_USER_NAME = "shivansh_vyas"
SRC_REPO = "Projects/Projects/Machine learning/Red_wine_project/"
AUTHOR_EMAIL = "shivanshvyas1729@gmail.com"

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-e'):
                continue
            requirements.append(line)
    return requirements

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ML application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=get_requirements("requirements.txt"),
)

