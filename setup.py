"""
File Name: setup.py
Author: Panyi Dong
GitHub: https://github.com/PanyiDong/
Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)

Project: InsurAutoML
Latest Version: 0.2.8
Relative Path: /setup.py
File Created: Wednesday, 16th November 2022 7:39:46 pm
Author: Panyi Dong (panyid2@illinois.edu)

-----
Last Modified: Sunday, 15th November 2026 11:42:17 pm
Modified By: Panyi Dong (panyid2@illinois.edu)

-----
MIT License

Copyright (c) 2022 - 2026, Panyi Dong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Build script of InsurAutoML.

The package metadata (name, dependencies, extras, package data, ...) is defined in
pyproject.toml. This file handles the dynamic parts:
- the version: written to InsurAutoML/VERSION from the latest git tag (vX.Y.Z), read
  by pyproject.toml; kept as is when building from the sdist (no git checkout)
- the optional C++ extension of the InformedAutoML version, built with
  INFORMEDAUTOML=1 pip install "InsurAutoML[informed]"
- requirements.txt, requirements_nn.txt and package.json, generated from pyproject.toml
"""

from pathlib import Path
import os
import sys
import glob
import json
import logging
import importlib
import subprocess
from typing import Optional

from setuptools import setup, Extension

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10, tomli is a build requirement in pyproject.toml
    import tomli as tomllib

logging.basicConfig()
log = logging.getLogger(__file__)

torch_spec = importlib.util.find_spec("torch")

if sys.version_info < (3, 10):
    log.info("Python >= 3.10 is required for the pipeline!")

ROOT = Path(__file__).resolve().parent

# package metadata
with open(ROOT / "pyproject.toml", "rb") as fh:
    PROJECT = tomllib.load(fh)["project"]
DEPENDENCIES = PROJECT["dependencies"]
OPTIONAL_DEPENDENCIES = PROJECT["optional-dependencies"]

EXT_MODULES = []
CMDCLASS = {}


def _pybind11_includes():
    """Return a list of pybind11 include dirs if pybind11 is importable, else empty list.

    This defers importing pybind11 until build-time so that import-time
    execution of setup.py does not fail when pybind11 is not present in the
    build isolation environment.
    """
    try:
        import pybind11

        return [pybind11.get_include(), pybind11.get_include(user=True)]
    except Exception:
        return []


# Automatically get release version
# from the git tag in a git checkout, otherwise (e.g. building from the sdist,
# which has no .git) the InsurAutoML/VERSION file is used as it is
def _git_version():
    try:
        version = subprocess.run(
            ["git", "describe", "--tags"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=ROOT,
        ).stdout.decode("utf-8")
    except OSError:  # git not installed
        return ""
    # remove the first character 'v' and the commits after the tag (e.g. -2-g97af73b)
    return version.strip()[1:].split("-")[0]


VERSION_FILE = ROOT / "InsurAutoML" / "VERSION"

# assert version file
assert (ROOT / "InsurAutoML" / "version.py").is_file()
if _git_version():
    # write version to VERSION file (read by pyproject.toml)
    VERSION_FILE.write_text("%s\n" % _git_version(), encoding="utf-8")
InsurAutoML_version = VERSION_FILE.read_text(encoding="utf-8").strip()

# check R installation


def r_home_from_subprocess() -> Optional[str]:
    """Return the R home directory from calling 'R RHOME'."""
    cmd = ("R", "RHOME")
    log.debug("Looking for R home with: {}".format(" ".join(cmd)))
    try:
        tmp = subprocess.check_output(cmd, universal_newlines=True)
    except Exception as e:  # FileNotFoundError, WindowsError, etc
        log.error(f"Unable to determine R home: {e}")
        return None
    r_home = tmp.split(os.linesep)
    if r_home[0].startswith("WARNING"):
        res = r_home[1]
    else:
        res = r_home[0].strip()
    return res


# TODO: move all Windows all code into an os-specific module ?
def r_home_from_registry() -> Optional[str]:
    """Return the R home directory from the Windows Registry."""
    from packaging.version import Version

    try:
        import winreg  # type: ignore
    except ImportError:
        import _winreg as winreg  # type: ignore
    # There are two possible locations for RHOME in the registry
    # We prefer the user installation (which the user has more control
    # over). Thus, HKEY_CURRENT_USER is the first item in the list and
    # the for-loop breaks at the first hit.
    for w_hkey in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
        try:
            with winreg.OpenKeyEx(w_hkey, "Software\\R-core\\R") as hkey:
                # >v4.x.x: grab the highest version installed
                def get_version(i):
                    try:
                        return Version(winreg.EnumKey(hkey, i))
                    except Exception:
                        return None

                latest = max(
                    (
                        v
                        for v in (
                            get_version(i) for i in range(winreg.QueryInfoKey(hkey)[0])
                        )
                        if v is not None
                    )
                )

                with winreg.OpenKeyEx(hkey, f"{latest}") as subkey:
                    r_home = winreg.QueryValueEx(subkey, "InstallPath")[0]

                # check for an earlier version
                if not r_home:
                    r_home = winreg.QueryValueEx(hkey, "InstallPath")[0]
        except Exception:  # FileNotFoundError, WindowsError, OSError, etc.
            pass
        else:
            # We have a path RHOME
            if sys.version_info[0] == 2:
                # Python 2 path compatibility
                r_home = r_home.encode(sys.getfilesystemencoding())
            # Break the loop, because we have a hit.
            break
    else:
        # for-loop did not break - RHOME is unknown.
        log.error("Unable to determine R home.")
        r_home = None
    return


def get_r_home() -> Optional[str]:
    """Get R's home directory (aka R_HOME).
    If an environment variable R_HOME is found it is returned,
    and if none is found it is trying to get it from an R executable
    in the PATH. On Windows, a third last attempt is made by trying
    to obtain R_HOME from the registry. If all attempt are unfruitful,
    None is returned.
    """

    r_home = os.environ.get("R_HOME")

    if not r_home:
        r_home = r_home_from_subprocess()
    if not r_home and os.name == "nt":
        r_home = r_home_from_registry()
    log.info(f"R home found: {r_home}")
    return r_home


# # get R Home environment variable
# # if found, install rpy2
# # otherwise, do not install rpy2
# R_HOME = get_r_home()
# if not R_HOME:
#     log.info("""The R home directory could not be determined.""")

# # only install for Linux
# if R_HOME and not os.environ.get("R_HOME") and sys.platform == "linux":
#     os.environ["R_HOME"] = R_HOME
#     OPTIONAL_DEPENDENCIES["informed"].append("rpy2")
#     OPTIONAL_DEPENDENCIES["nn"].append("rpy2")

# --- External native extension: suppsplit (pybind11) ---
# only built for the informed installation: INFORMEDAUTOML=1 pip install ...
INFORMED = os.environ.get("INFORMEDAUTOML", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def build_informed_extensions():
    """C++ extensions (pybind11) used by the informed pipelines."""
    if sys.platform == "win32":
        log.warning("The informed C++ extensions are not supported on Windows.")
        return []

    extra_compile_args = []
    extra_link_args = []
    include_dirs = []
    library_dirs = []
    libraries = []
    if sys.platform == "darwin":
        extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
        extra_link_args += ["-lomp"]
        # Homebrew paths (adjust if needed)
        include_dirs += ["/usr/local/include", "/opt/homebrew/include"]
        library_dirs += ["/usr/local/lib", "/opt/homebrew/lib"]
        libraries += ["omp"]
    else:
        extra_compile_args += ["-O3", "-std=c++14", "-fopenmp"]
        extra_link_args += ["-fopenmp"]

    def _extension(name, folder, sources):
        path = os.path.join("InsurAutoML", "ext", name, folder)
        return Extension(
            "InsurAutoML.ext.{}.{}".format(name, folder),
            sources=[os.path.join(path, source) for source in sources],
            include_dirs=_pybind11_includes() + [path] + include_dirs,
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            library_dirs=library_dirs,
            libraries=libraries,
        )

    return [
        # SuppsPlit
        _extension("suppsplit", "suppsplit_cpp", ["sp.cpp", "sPlit.cpp", "bindings.cpp"]),
    ]


if INFORMED:
    EXT_MODULES += build_informed_extensions()


# torch extensions build (not used at this moment)
def build_torch_extensions():
    torch_extensions = []

    if torch_spec is not None:
        from torch.utils.cpp_extension import include_paths, BuildExtension

        # Extensions
        torch_extensions.append(
            Extension(
                name="pytorch_ext",
                sources=glob.glob("**/*.cpp", recursive=True)
                + glob.glob("**/*.cu", recursive=True),
                include_dirs=include_paths(),
                language="c++",
            )
        )

    CMDCLASS["build_ext"] = BuildExtension

    return torch_extensions


# prepare for package.json file
def get_package():
    result = {}
    result["name"] = PROJECT["name"]
    result["author"] = PROJECT["authors"][0]
    result["repository"] = {
        "type": "git",
        "url": PROJECT["urls"]["Repository"],
    }
    result["main"] = "main.py"
    result["private"] = False
    result["version"] = InsurAutoML_version
    result["description"] = PROJECT["description"]
    result["install_requires"] = DEPENDENCIES
    result["extras_require"] = OPTIONAL_DEPENDENCIES

    if os.path.exists("package.json"):
        os.remove("package.json")

    with open("package.json", "w") as f:
        json.dump(result, f, indent=4)


def _requirements(extra: str) -> list:
    # dependencies + extra, without environment markers and self-references
    items = DEPENDENCIES + OPTIONAL_DEPENDENCIES[extra]
    items = [item.split(";")[0].strip() for item in items]
    return sorted({item for item in items if not item.startswith(PROJECT["name"])})


def get_requirements():
    for file_name, extra in [
        ("requirements.txt", "informed"),
        ("requirements_nn.txt", "nn"),
    ]:
        with open(file_name, "w") as fp:
            for item in _requirements(extra):
                # write each item on a new line
                fp.write("%s\n" % item)


def setup_package():
    # metadata in pyproject.toml, only the optional extension is configured here
    setup(ext_modules=EXT_MODULES, cmdclass=CMDCLASS)


def main():
    # # check whether need to build pytorch extensions
    # EXT_MODULES += build_torch_extensions()

    # setup package
    setup_package()
    log.info(
        "{}-{} installation succeeded.".format(PROJECT["name"], InsurAutoML_version)
    )


if __name__ == "__main__":
    get_requirements()
    main()
    get_package()
