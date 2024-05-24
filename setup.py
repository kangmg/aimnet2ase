from setuptools import setup, find_namespace_packages

setup(
    name='aimnet2ase',
    version='0.0.1',
    author='Kang mingi',
    author_email='kangmg@korea.ac.kr',
    description='AIMNet2 calculation with IPython ASE interface',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  
    url='https://github.com/kangmg/aimnet2ase',
    keywords=['chemistry','computational chemistry','machine learning'],
    include_package_data=True,
    packages=find_namespace_packages(), 
    install_requires=[
        'numpy',
        'ase>=3.22.1',
        'torch>=2.2.1'
    ],
    classifiers=[ 
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    python_requires='>=3.7.0',
)