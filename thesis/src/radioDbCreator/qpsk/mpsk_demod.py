#!/usr/bin/env python
##################################################
# Gnuradio Python Flow Graph
# Title: MPSK Demod Demo
# Generated: Sat Jun 11 16:44:17 2016
##################################################

from PyQt4 import Qt
from gnuradio import blocks
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import qtgui
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from grc_gnuradio import blks2 as grc_blks2
from optparse import OptionParser
import numpy
import sip
import sys

class mpsk_demod(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "MPSK Demod Demo")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("MPSK Demod Demo")
        try:
             self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
             pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "mpsk_demod")
        self.restoreGeometry(self.settings.value("geometry").toByteArray())


        ##################################################
        # Variables
        ##################################################
        self.samps_per_sym = samps_per_sym = 4
        self.samp_rate = samp_rate = 50000000

        ##################################################
        # Blocks
        ##################################################
        self.notebook = Qt.QTabWidget()
        self.notebook_widget_0 = Qt.QWidget()
        self.notebook_layout_0 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.notebook_widget_0)
        self.notebook_grid_layout_0 = Qt.QGridLayout()
        self.notebook_layout_0.addLayout(self.notebook_grid_layout_0)
        self.notebook.addTab(self.notebook_widget_0, "Constellation")
        self.notebook_widget_1 = Qt.QWidget()
        self.notebook_layout_1 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.notebook_widget_1)
        self.notebook_grid_layout_1 = Qt.QGridLayout()
        self.notebook_layout_1.addLayout(self.notebook_grid_layout_1)
        self.notebook.addTab(self.notebook_widget_1, "Spectrum")
        self.top_layout.addWidget(self.notebook)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
        	1024, #size
        	firdes.WIN_BLACKMAN_hARRIS, #wintype
        	0, #fc
        	samp_rate, #bw
        	"QT GUI Plot", #name
        	1 #number of inputs
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.pyqwidget(), Qt.QWidget)
        self.notebook_layout_1.addWidget(self._qtgui_freq_sink_x_0_win)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
        	1024, #size
        	"QT GUI Plot", #name
        	1 #number of inputs
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_x_axis(-2, 2)
        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.pyqwidget(), Qt.QWidget)
        self.notebook_layout_0.addWidget(self._qtgui_const_sink_x_0_win)
        self.digital_dxpsk_mod_0 = digital.dqpsk_mod(
        	samples_per_symbol=samps_per_sym,
        	excess_bw=0.35,
        	mod_code="gray",
        	verbose=False,
        	log=False)
        	
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate)
        self.blks2_tcp_sink_0 = grc_blks2.tcp_sink(
        	itemsize=gr.sizeof_gr_complex*1,
        	addr="127.0.0.1",
        	port=3305,
        	server=True,
        )
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 2**8, 1000000)), True)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_0, 0), (self.digital_dxpsk_mod_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.digital_dxpsk_mod_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blks2_tcp_sink_0, 0))


# QT sink close method reimplementation
    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "mpsk_demod")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_samps_per_sym(self):
        return self.samps_per_sym

    def set_samps_per_sym(self, samps_per_sym):
        self.samps_per_sym = samps_per_sym

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

if __name__ == '__main__':
    import ctypes
    import os
    if os.name == 'posix':
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"
    parser = OptionParser(option_class=eng_option, usage="%prog: [options]")
    (options, args) = parser.parse_args()
    qapp = Qt.QApplication(sys.argv)
    tb = mpsk_demod()
    tb.start()
    tb.show()
    def quitting():
        tb.stop()
        tb.wait()
    qapp.connect(qapp, Qt.SIGNAL("aboutToQuit()"), quitting)
    qapp.exec_()
    tb = None #to clean up Qt widgets

