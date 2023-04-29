from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_str: str) -> List[str]:
    '''
    This function returns the list of requirements from the requirements.txt file
    :param file_str:
    :return:
    '''
    requirements = []
    with open(file_str, 'r') as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="mlproject",
    version="0.1",
    author="Pranshu",
    author_email="sharmapranshu317@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)