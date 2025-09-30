# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for pip package."""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="cosmos-tokenizer",
    version="1.0.0",
    author="NVIDIA Corporation",
    description="A suite of image and video neural tokenizers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/Cosmos-Tokenizer",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    exclude=["assets", "test_data"],
    entry_points={
        'console_scripts': [
            'cosmos-image=cosmos_tokenizer.image_cli:main',
            'cosmos-video=cosmos_tokenizer.video_cli:main',
        ],
    },
)