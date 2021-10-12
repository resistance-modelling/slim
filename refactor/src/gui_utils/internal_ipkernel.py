"""
Largely inspired from Ipython's "internal_ipkernel.py" example

See https://github.com/ipython/ipykernel/blob/63e904e68d061edadc55b796526d0fa33171fa53/examples/embedding/internal_ipkernel.py
"""

from IPython.kernel.connect import connect_qtconsole
from ipykernel.kernelapp import IPKernelApp

def mpl_kernel(gui: str):
    """Launch and return an IPython kernel with matplotlib support for the desired gui

    :param gui a matplotlib backend
    """
    kernel = IPKernelApp.instance()
    kernel.initialize(['python'
                       #'--log-level=10'
                       ])
    return kernel

class InternalIPKernel(object):
    """A mixin to add kernel-loading support.
    The intended usage is to subclass a QWidget and this class at the same time.
    """

    def init_ipkernel(self, backend: str):
        self.ipkernel = mpl_kernel(backend)
        self.consoles = []

        # TODO: add Simulator in this namespace
        self.namespace = self.ipkernel.shell.user_ns

    def new_qt_console(self, _evt=None):
        return connect_qtconsole(self.ipkernel.abs_connection_file)

    def cleanup_consoles(self, _evt=None):
        for c in self.consoles:
            c.kill()