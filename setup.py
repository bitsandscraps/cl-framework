from setuptools import setup

setup(
    name='clfw',
    version='0.10',
    packages=['clfw'],
    url='https://github.com/bitsandscraps/continual-learning-framework',
    license='MIT',
    author='bitsandscraps',
    author_email='daniel9607@hanmail.net',
    description='A framework for training and testing continual learning algorithms.',
    install_requires=['tensorflow', 'tensorflow-datasets']
)
