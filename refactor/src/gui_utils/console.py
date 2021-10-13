from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

class ConsoleWidget(RichJupyterWidget):
    def __init__(self, *args, **kwargs):
        super(ConsoleWidget, self).__init__(*args, **kwargs)

        # TODO make this tuneable
        self.font_size = 6
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = 'qt'
        self.kernel_client = kernel_client = self.kernel_manager.client()
        kernel_client.start_channels()

        def stop():
           kernel_client.stop_channels()
           kernel_manager.shutdown_kernel()

        self.exit_requested.connect(stop)

    def push_vars(self, variableDict):
        self.kernel_manager.kernel.shell.push(variableDict)

    def clear(self):
        self._control.clear()

    def print_text(self, text):
        self._append_plain_text(text)

    def execute_command(self, command):
        self._execute(command, False)
