#!/usr/bin/env python3
# -*- tab-width: 4 -*- ;; Emacs
# vi: set ts=4 sw=4 noet :: Vi/ViM
############################################################ IDENT(1)
#
# $Title: Python setup for bpydot $
# $Copyright: 2025 Devin Teske. All rights reserved. $
# $FrauBSD$
#
############################################################ LICENSE
#
# BSD 2-Clause
#
############################################################ DOCSTRING

"""Setup script for bpydot."""

############################################################ IMPORTS

from setuptools import setup, find_packages
from pathlib import Path

############################################################ GLOBALS

# Read version from version.py
version = {}
with open('bpydot/version.py') as f:
	exec(f.read(), version)

# Read README if it exists
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

############################################################ SETUP

setup(
	name='bpydot',
	version=version['VERSION'],
	description='Python code analysis and visualization tool',
	long_description=long_description,
	long_description_content_type='text/markdown',
	author='Devin Teske',
	author_email='dteske@FreeBSD.org',
	url='https://github.com/FrauBSD/bpydot',
	packages=find_packages(),
	entry_points={
		'console_scripts': [
			'bpydot=bpydot.__main__:main',
		],
	},
	python_requires='>=3.6',
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Developers',
		'Topic :: Software Development :: Code Generators',
		'Topic :: Software Development :: Documentation',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8',
		'Programming Language :: Python :: 3.9',
		'Programming Language :: Python :: 3.10',
		'Programming Language :: Python :: 3.11',
		'Programming Language :: Python :: 3.12',
		'Programming Language :: Python :: 3.13',
	],
	keywords='code-analysis visualization call-graph graphviz ast introspection',
)

################################################################################
# END
################################################################################
