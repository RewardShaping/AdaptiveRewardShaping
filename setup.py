from setuptools import find_packages, setup

VERSION = "0.0.1"
DESCRIPTION = "Use Partial Satisfiability to tarin RL agent more efficiently guaranteeing that partial satisfaction for a given specification"
LONG_DESCRIPTION = """
We give a partial credit for each completion on automaton graph, and propose a logic based reward shaping based on 
a given automaton graph structure guaranteeing that we can achieve partial satisfiability
""" 

setup(
    name="psltl",
    packages=find_packages(),
    package_data={"psltl": ["py.typed", "version.txt"]},
    version=VERSION, 
    install_requires=[       
        "stable-baselines3==1.7.0", # must be 1.7.0 version of stable baselines!, but because of qrm..crm...
        "gym==0.21.0",
        "gymnasium",
        "cython<3",
        "logaut", 
        "pythomata",
        "pickle5", # for save files, and load files
    ],
    description="Adaptive Reward Design for Reinforcement Learning in Complex Robotic Tasks",
    keywords="Adaptive Reward Design for Reinforcement Learning in Complex Robotic Tasks", 
    license="MIT",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=">=3.6", # because of baseline
    # PyPI package information.
    classifiers=[
    "Programming Language :: Python :: 3.7",
	"Programming Language :: Python :: 3.8",
    ],
)

# python setup.py sdist bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
