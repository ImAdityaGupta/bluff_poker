import sys
import numpy as np
import itertools
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from setup import StrategyAllStorage_MC, string_to_move
from logging_weights import strat_from_weights

# Custom colormap for the left heatmap: red → black → green, centered at 0
LEFT_CMAP = LinearSegmentedColormap.from_list(
    'red_black_green',
    ['red', 'white', 'green'],
    N=256
)

# Custom colormap for the right heatmap: red → yellow → green
RIGHT_CMAP = LinearSegmentedColormap.from_list(
    'red_yellow_green',
    ['black', 'blue', 'white'],
    N=256
)


class HeatmapWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.im = None
        self.colorbar = None

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_heatmap(self,
                       data: np.ndarray,
                       cmap: str = 'viridis',
                       vmin=None,
                       vmax=None,
                       norm=None):
        """
        Efficiently update the heatmap display with new data.
        If `norm` is provided, it takes precedence over vmin/vmax.
        """
        if self.im is None:
            kw = dict(cmap=cmap)
            if norm is not None:
                kw['norm'] = norm
            else:
                kw['vmin'] = vmin
                kw['vmax'] = vmax

            self.im = self.ax.imshow(data, aspect='auto', **kw)
            self.ax.set_xticks(np.arange(data.shape[1]))
            self.ax.set_yticks(np.arange(data.shape[0]))
            self.ax.set_xlabel('Column')
            self.ax.set_ylabel('Row')
            self.ax.set_title('Heatmap')
            self.colorbar = self.figure.colorbar(self.im, ax=self.ax)
        else:
            self.im.set_data(data)
            if norm is not None:
                self.im.set_norm(norm)
            else:
                if vmin is not None and vmax is not None:
                    self.im.set_clim(vmin=vmin, vmax=vmax)
            self.colorbar.update_normal(self.im)

        self.canvas.draw_idle()


class StrategyViewer(QtWidgets.QMainWindow):
    def __init__(self, left_data_list, right_shape, strategy, parent=None):
        super().__init__(parent)
        self.setWindowTitle('RL Strategy Visualizer')
        self.strategy = strategy
        self.n = strategy.n

        # precompute all hands once
        self.hands = list(itertools.combinations_with_replacement(range(self.n), 2))

        # Store inputs
        self.left_data_list = left_data_list
        self.right_shape = right_shape
        self.left_index = 0

        # Compute global vmin/vmax for left heatmaps (exclude placeholder)
        stacked = np.stack(self.left_data_list[:-1], axis=0)
        self.left_vmin = float(np.min(stacked))
        self.left_vmax = float(np.max(stacked))

        # Create a TwoSlopeNorm so that 0 is always the center
        self.left_norm = TwoSlopeNorm(
            vmin=self.left_vmin,
            vcenter=0.0,
            vmax=self.left_vmax
        )

        # Setup layouts
        central = QtWidgets.QWidget()
        h_layout = QtWidgets.QHBoxLayout(central)

        # Left panel: heatmap + slider
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        self.left_heatmap = HeatmapWidget()
        left_layout.addWidget(self.left_heatmap)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.left_data_list) - 1)
        self.slider.setValue(0)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)

        # Debounced live updates for smoother performance
        from PyQt5.QtCore import QTimer
        self.slider_timer = QTimer(self)
        self.slider_timer.setSingleShot(True)
        self.slider_timer.timeout.connect(
            lambda: self.on_slider_change(self.slider.value())
        )
        self.slider.valueChanged.connect(lambda _: self.slider_timer.start(100))

        left_layout.addWidget(self.slider)
        h_layout.addWidget(left_widget)

        # Right panel: heatmap + text input + button
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        self.right_heatmap = HeatmapWidget()
        right_layout.addWidget(self.right_heatmap)

        self.text_input = QtWidgets.QPlainTextEdit()
        font = self.text_input.font()
        font.setPointSize(12)
        self.text_input.setFont(font)
        self.text_input.setFixedHeight(80)
        self.text_input.setPlaceholderText(
            "Enter space-separated moves (or leave blank)"
        )
        right_layout.addWidget(self.text_input)

        btn = QtWidgets.QPushButton('Update Right Heatmap')
        btn.clicked.connect(self.on_update_right)
        right_layout.addWidget(btn)

        h_layout.addWidget(right_widget)

        self.setCentralWidget(central)

        # Initial rendering
        self.update_left_heatmap()

        # show blank right heatmap and immediately relabel axes
        blank = np.zeros(self.right_shape)
        self.right_heatmap.update_heatmap(
            blank,
            cmap=RIGHT_CMAP,
            vmin=0,
            vmax=1
        )
        ax = self.right_heatmap.ax
        # set row labels from self.hands
        row_labels = [f"{i}{j}" for (i, j) in self.hands]
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_ylabel("Hand")
        # set column labels: 0H..(n-1)H,0P..,0T..,Call
        n = self.n
        col_labels = [f"{i}H" for i in range(n)] \
                   + [f"{i}P" for i in range(n)] \
                   + [f"{i}T" for i in range(n)] \
                   + ['Call']
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90)
        ax.set_xlabel("Action")
        self.right_heatmap.canvas.draw_idle()

    def on_slider_change(self, value):
        self.left_index = value
        self.update_left_heatmap()
        # also update the right panel automatically
        self.on_update_right()

    def update_left_heatmap(self):
        data = self.left_data_list[self.left_index]
        self.left_heatmap.update_heatmap(
            data,
            cmap=LEFT_CMAP,
            norm=self.left_norm
        )

    def on_update_right(self):
        text = self.text_input.toPlainText().strip()

        # parse moves or default to empty history
        if not text:
            history = []
        else:
            tokens = text.split()
            try:
                history = [string_to_move(self.n, tok) for tok in tokens]
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Input Error",
                    f"Could not parse one of the moves:\n{e}"
                )
                return

        # select current weight matrix or mixture
        if self.left_index == len(self.left_data_list) - 1:
            pv = self.strategy.prob_vector
            current_W = sum(p * W for p, W in zip(pv, self.strategy.weight_matrices))
        else:
            current_W = self.strategy.weight_matrices[self.left_index]

        # sample action probabilities for each hand
        probs_list = []
        for hand in self.hands:
            _, probs, _, _ = self.strategy.sample_action(
                history=history,
                player_cards=list(hand),
                who_first=len(history) % 2,
                which_weights=current_W
            )
            probs_list.append(probs)

        # stack into array
        arr = np.vstack(probs_list)

        # update heatmap
        self.right_heatmap.update_heatmap(
            arr,
            cmap=RIGHT_CMAP,
            vmin=0,
            vmax=1
        )
        ax = self.right_heatmap.ax

        # reset and relabel rows
        row_labels = [f"{i}{j}" for (i, j) in self.hands]
        ax.set_yticks(np.arange(arr.shape[0]))
        ax.set_yticklabels(row_labels)
        ax.set_ylabel("Hand")

        # reset and relabel columns
        n = self.n
        col_labels = [f"{i}H" for i in range(n)] \
                   + [f"{i}P" for i in range(n)] \
                   + [f"{i}T" for i in range(n)] \
                   + ['Call']
        ax.set_xticks(np.arange(arr.shape[1]))
        ax.set_xticklabels(col_labels, rotation=90)
        ax.set_xlabel("Action")

        self.right_heatmap.canvas.draw_idle()


def main(strategy: StrategyAllStorage_MC):
    n = strategy.n

    # prepare left data slices and append a placeholder for mixture
    left_list = [W[:, -6:] for W in strategy.weight_matrices]
    left_list.append(np.zeros_like(left_list[0]))

    # right heatmap has one row per unordered pair with replacement
    right_shape = ((n * (n + 1)) // 2, strategy.M)

    app = QtWidgets.QApplication(sys.argv)
    viewer = StrategyViewer(left_list, right_shape, strategy)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(strat_from_weights(0, 24, -1))
