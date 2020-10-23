from setuptools import setup, find_packages

setup(name='torch_burn',
      version='0.0.3',
      url='https://github.com/Kitsunetic/torch_burn',
      license='MIT',
      author='Kitsunetic',
      author_email='1996.jh.shim@gmail.com',
      description='torch burn',
      long_description=open('README.md').read(),
      packages=setuptools.find_packages(),
      zip_safe=False,
      setup_requires=[])
