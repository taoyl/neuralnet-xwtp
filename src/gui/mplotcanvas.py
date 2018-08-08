# standard libraries
import random

# thrid-party libraries
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy


class MplotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=90, labels=None):
        # font = matplotlib.font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.initial_figure(labels, font=None)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self, labels):
        pass

    def clear(self):
        self.axes.cla()


class StaticMplotCanvas(MplotCanvas):
    """Simple canvas with a sine plot."""

    def initial_figure(self, labels, font=None):
        if labels:
            if font:
                self.axes.set_title(labels['t'], fontproperties=font)
                self.axes.set_xlabel(labels['x'], fontproperties=font)
                self.axes.set_ylabel(labels['y'], fontproperties=font)
            else:
                self.axes.set_title(labels['t'])
                self.axes.set_xlabel(labels['x'])
                self.axes.set_ylabel(labels['y'])


    def update_plot(self, x, y):
        marker_style = dict(linestyle='-', color='cornflowerblue',
                            marker='x', markersize=8, fillstyle='none')
        self.axes.plot(x, y, **marker_style)
        self.draw()

    def error_plot(self, x, y):
        # self.axes.errorbar(x, y, yerr=yerr)
        self.axes.bar(x, y, 0.3, color='r')
        self.draw()
