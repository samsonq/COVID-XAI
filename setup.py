from setuptools import setup, find_packages
import pip
import logging
import pkg_resources


def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception as e:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

setup(
    name="covidxai",
    version="0.1.0",
    url="https://github.com/samsonq/COVID-LRP",
    description="Using layer-wise relevance propagation and other explainability algorithms on CNN architectures to identify and explain regions of bacteria in COVID-19 chest X-rays.",
    install_requires=install_reqs,
    python_requires=">=3.4",
    keywords="lrp computer-vision explainable-ai lime deconvolution covid-19 pneumonia-detection"
)