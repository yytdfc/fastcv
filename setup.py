from setuptools import setup, find_packages

PKG_NAME = 'fastcv'

fastcv = __import__(PKG_NAME)


setup(
    name=PKG_NAME,
    version=fastcv.__version__,
    description=fastcv.__description__,
    long_description=open('./README.md').read(),
    maintainer='yytdfc',
    maintainer_email='fuchen@foxmail.com',
    keywords=[],
    url='https://github.com/yytdfc',
    packages=find_packages(),
    include_package_data=True,
    license='MIT',
    install_requires=[],
    setup_requires=[],
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
