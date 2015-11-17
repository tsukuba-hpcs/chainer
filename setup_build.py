from __future__ import print_function
import os
from os import path
import sys

import setuptools
from setuptools.command import build_ext
from setuptools.command import install


dummy_extension = setuptools.Extension('chainer', ['x.c'])


MODULES = [
    {
        'name': 'cuda',
        'file': [
            'cupy.core',
            'cupy.cuda.cublas',
            'cupy.cuda.curand',
            'cupy.cuda.device',
            'cupy.cuda.driver',
            'cupy.cuda.memory',
            'cupy.cuda.module',
            'cupy.cuda.runtime',
            'cupy.util',
        ],
        'include': [
            'cublas_v2.h',
            'cuda.h',
            'cuda_runtime.h',
            'curand.h',
        ],
        'libraries': [
            'cublas',
            'cuda',
            'cudart',
            'curand',
        ],
    },
    {
        'name': 'cudnn',
        'file': [
            'cupy.cuda.cudnn',
        ],
        'include': [
            'cudnn.h',
        ],
        'libraries': [
            'cudnn',
        ],
    }
]


def get_compiler_setting():
    include_dirs = []
    library_dirs = []
    define_macros = []
    if sys.platform == 'win32':
        include_dirs = [localpath('windows')]
        library_dirs = []
        cuda_path = os.environ.get('CUDA_PATH', None)
        if cuda_path:
            include_dirs.append(path.join(cuda_path, 'include'))
            library_dirs.append(path.join(cuda_path, 'bin'))
            library_dirs.append(path.join(cuda_path, 'lib', 'x64'))
    else:
        include_dirs = get_path('CPATH') + ['/usr/local/cuda/include']
        library_dirs = get_path('LD_LIBRARY_PATH') + [
            '/usr/local/cuda/lib64',
            '/opt/local/lib',
            '/usr/local/lib']

    return {
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'define_macros': define_macros,
        'language': 'c++',
    }


def localpath(*args):
    return path.abspath(path.join(path.dirname(__file__), *args))


def get_path(key):
    splitter = ';' if sys.platform == 'win32' else ':'
    return os.environ.get(key, "").split(splitter)


def check_include(dirs, file_path):
    return any(path.exists(path.join(dir, file_path)) for dir in dirs)

def check_readthedocs_environment():
    return os.environ.get('READTHEDOCS', None) == 'True'

def make_extensions(options):

    """Produce a list of Extension instances which passed to cythonize()."""

    no_cuda = options['no_cuda']
    settings = get_compiler_setting()

    try:
        import numpy
        numpy_include = numpy.get_include()
    except AttributeError:
        # if numpy is not installed get the headers from the .egg directory
        import numpy.core
        numpy_include = path.join(
            path.dirname(numpy.core.__file__), 'include')
    include_dirs = settings['include_dirs']
    include_dirs.append(numpy_include)

    settings['include_dirs'] = [
        x for x in include_dirs if path.exists(x)]
    settings['library_dirs'] = [
        x for x in settings['library_dirs'] if path.exists(x)]

    if options['linetrace']:
        settings['define_macros'].append(('CYTHON_TRACE', '1'))
    if no_cuda:
        settings['define_macros'].append(('CUPY_NO_CUDA', '1'))

    ret = []
    for module in MODULES:
        print('Include directories:', settings['include_dirs'])
        print('Library directories:', settings['library_dirs'])

        include = [i for i in module['include']
                   if not check_include(include_dirs, i)]
        if not no_cuda and include:
            print('Missing include files:', include)
            continue

        s = settings.copy()
        if not no_cuda:
            s['libraries'] = module['libraries']
        ret.extend([
            setuptools.Extension(
                f, [localpath(path.join(*f.split('.')) + '.pyx')], **s)
            for f in module['file']])
    return ret


_arg_options = {}


def parse_args():
    global _arg_options
    _arg_options['profile'] = '--cupy-profile' in sys.argv
    if _arg_options['profile']:
        sys.argv.remove('--cupy-profile')

    _arg_options['linetrace'] = '--cupy-linetrace' in sys.argv
    if _arg_options['linetrace']:
        sys.argv.remove('--cupy-linetrace')

    _arg_options['no_cuda'] = '--cupy-no-cuda' in sys.argv
    if _arg_options['no_cuda']:
        sys.argv.remove('--cupy-no-cuda')
    if check_readthedocs_environment():
        _arg_options['no_cuda'] = True


class chainer_build_ext(build_ext.build_ext):

    """`build_ext` command for cython files."""

    def finalize_options(self):
        from Cython.Build import cythonize

        ext_modules = self.distribution.ext_modules
        if dummy_extension in ext_modules:
            print('Executing cythonize()')
            print('Options:', _arg_options)

            directive_keys = ('linetrace', 'profile')
            directives = {key: _arg_options[key] for key in directive_keys}

            extensions = cythonize(
                make_extensions(_arg_options),
                force=True,
                compiler_directives=directives)

            # Modify ext_modules for cython
            ext_modules.remove(dummy_extension)
            ext_modules.extend(extensions)

        build_ext.build_ext.finalize_options(self)


class chainer_install(install.install):

    """`install` command for read the docs."""

    def run(self):
        install.install.run(self)

        # Hack for Read the Docs
        if check_readthedocs_environment():
            print('Executing develop command for Read the Docs')
            self.run_command('develop')
