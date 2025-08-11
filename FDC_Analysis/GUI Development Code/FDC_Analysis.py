import os
from typing import Optional
import sys
from typing import Dict, Any
import lumicks.pylake as lk
import numpy as np
import pandas as pd
import denoise
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QWidget,
    QPushButton,
    QCheckBox,
    QTextEdit,
    QLineEdit,
    QLabel,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def baseline_processing(force, force_baseline, distance, distance_baseline):
    p = np.polyfit(distance_baseline, force_baseline, deg=7)
    force_baseline_fit = np.polyval(p, distance)
    new_force = np.array(force) - force_baseline_fit
    return new_force
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
def safe_list(v):
    if isinstance(v, list):
        return v
    elif hasattr(v, 'tolist'):
        return v.tolist()
    else:
        return [v]
def safe_len(v):
    try:
        return len(v)
    except TypeError:
        return 1
# -----------------------------------------------------------------------------
# Helper Dialog – shows FDC plot + textual output
# -----------------------------------------------------------------------------

class FDCWindow(QtWidgets.QDialog):
    """Dedicated window for visualising FDC curves."""

    # def __init__(self, parent: QtWidgets.QWidget | None, data: Dict[str, Any]):
    def __init__(self, parent: Optional[QtWidgets.QWidget], data: Dict[str, Any]):
        super().__init__(parent)

        self.setWindowTitle("FDC Display")
        self.resize(1000, 850)

        self._data = data

        self._init_ui()
        self._plot()

    # ------------------------- UI assembly ----------------------------------

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Matplotlib canvas
        self._figure = Figure()

        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas)

        # Text output
        self._text = QTextEdit(readOnly=True)
        layout.addWidget(self._text)

        layout.setStretch(0, 8)
        layout.setStretch(1, 2)

        fold_num = self._data["fold_num"]
        actual_structure_size = [f'{num:.2f}' for num in self._data["actual_structure_size"]]
        free_energy = [f'{num:.2f}' for num in self._data["free_energy"]]
        # fold_site = [f'{num:.2f}' for num in self._data["fold_site"]]
        self._text.append(f'Fold Count：{fold_num}')
        self._text.append(f'Actual Fold Size (nm)：{", ".join(actual_structure_size)}')
        self._text.append(f'Free Energy (kJ/mol)：{", ".join(free_energy)}')
        for i in range(len(self._data["fold_site"])):
            start_1 = f'{self._data["fold_site"][i][2]:.2f}'
            end_1 = f'{self._data["fold_site"][i][0]:.2f}'
            start_2 = f'{self._data["fold_site"][i][3]:.2f}'
            end_2 = f'{self._data["fold_site"][i][1]:.2f}'
            self._text.append(f'Fold Site{i + 1}_Initiation-Termination (nm,pN)：({start_1}, {end_1}), ({start_2}, {end_2})')
    # ------------------------- Plot logic -----------------------------------

    def _plot(
        self,
        show_original: bool = True,
        show_downsample: bool = True,
        show_peaks: bool = True,
        show_wlc: bool = True,
    ) -> None:
        """(Re)draw the canvas based on visibility flags."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self.label_font_size=15



        if ("denoise_distance" not in self._data or "denoise_force" not in self._data
                or len(self._data["denoise_distance"]) == 0 or len(self._data["denoise_force"]) == 0):
            ax.text(0.5, 0.5, "No data loaded", ha="center", va="center")
            self._canvas.draw()
            return


        if show_original:
            ax.plot(self._data["origin_distance"], self._data["origin_force"], label="Raw FDC",color='gray')

        if show_downsample:
            ax.plot(self._data["denoise_distance"], self._data["denoise_force"], color='red', label="Denoised FDC")

        if show_peaks:
            for point_4X in self._data['fold_site']:
                ax.scatter([point_4X[2],point_4X[3]],[point_4X[0],point_4X[1]], marker='+',s=200, zorder=90, linewidth=2)
                # ax.scatter(self._data['fold_site'], self._data['fold_site'], marker="o", label="Peaks")

        if show_wlc:
            point_index=self._data['point_index']
            for i in range(len(point_index) + 1):
                PLC=self._data["PLC"][i]
                if i == len(point_index):
                    index_1 = point_index[i - 1][1] + 2
                    index_2 = len(self._data["denoise_distance"])
                if i != 0 and i != len(point_index):
                    index_1 = point_index[i - 1][1] + 2
                    index_2 = point_index[i][0] - 2
                if i == 0:
                    index_1 = 0
                    index_2 = point_index[i][1] - 2
                x_data = np.array(self._data["denoise_distance"])[index_1:index_2]
                # y_data = np.array(self._data["denoise_force"])[index_1:index_2]
                x_fit=np.linspace(min(self._data["denoise_distance"]),max(x_data)+5,index_2)
                # if i==len(point_index):
                #     x_fit=np.linspace(min(self._data["denoise_distance"]),max(x_data),index_2)
                ax.plot(x_fit,denoise.wlc_model(x_fit, P=PLC[0], L=PLC[1],C=PLC[2]),linewidth=1.5)
        ax.set_xlabel("Distance (nm)", fontsize=self.label_font_size)
        ax.set_ylabel("Force (pN)", fontsize=self.label_font_size)
        ax.set_title(f'{self._data["fold_num"]} Fold FDC', fontsize=self.label_font_size)
        ax.set_ylim(min(self._data["denoise_force"])-1,max(self._data["denoise_force"])*1.1)
        ax.tick_params(axis='x', which='major', labelsize=self.label_font_size)
        ax.tick_params(axis='y', which='major', labelsize=self.label_font_size)
        ax.legend(fontsize=self.label_font_size)
        self._canvas.draw()
        # print(f'origin_force：{force_origin}')
        # print(f'origin_distance：{distance_origin}')
        # print(f'denoise_force：{force_p}')
        # print(f'denoise_distance：{distance_p}')
        # print(f'fold_num：{len(point_list)}')
        # print(f'PLC：{P_L_C_list}')
        # print(f'actual_structure_size：{structure_size} nm')
        # print(f'structure_size_direct：{structure_size_direct} nm')
        # print(f'free_energy：{S_fold_list} kJ/mol')
        # print(f'fold_site：{point_list} kJ/mol')


    # ------------------------- Public helpers -------------------------------

    def update_visibility(
        self,
        show_original: bool,
        show_downsample: bool,
        show_peaks: bool,
        show_wlc: bool,
    ) -> None:
        self._plot(show_original, show_downsample, show_peaks, show_wlc)

    def append_text(self, message: str) -> None:  # noqa: D401
        self._text.append(message)


# -----------------------------------------------------------------------------
# Main window – mirrors the provided mock‑up
# -----------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FDC Analysis")
        self.resize(1000, 720)
        self._data_analysis: Dict[str, Any] = {}
        self._data: Dict[str, Any] = {}
        self._fdc_window: Optional[FDCWindow] = None
        # self._fdc_window: FDCWindow | None = None
        self._init_ui()

    # ------------------------- UI assembly ----------------------------------

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_vbox = QVBoxLayout(central)

        # ---- Input section -------------------------------------------------
        input_hbox = QHBoxLayout()
        main_vbox.addLayout(input_hbox)

        # H5 group
        h5_group = QGroupBox("Input Pattern: H5")
        input_hbox.addWidget(h5_group, 1)
        h5_vbox = QVBoxLayout(h5_group)

        btn_h5 = QPushButton("FDC.H5")
        btn_h5.setMinimumHeight(50)
        btn_h5.setMaximumHeight(70)


        btn_h5.clicked.connect(self._load_h5)
        h5_vbox.addWidget(btn_h5)

        btn_baseline = QPushButton("Offset Baseline")
        btn_baseline.setMinimumHeight(50)
        btn_baseline.setMaximumHeight(70)
        btn_baseline.clicked.connect(self._load_baseline)
        h5_vbox.addWidget(btn_baseline)

        btn_baseline_proc = QPushButton("Baseline Processing")
        btn_baseline_proc.setMinimumHeight(50)
        btn_baseline_proc.setMaximumHeight(70)
        btn_baseline_proc.clicked.connect(self._process_baseline)
        h5_vbox.addWidget(btn_baseline_proc)

        # CSV group
        csv_group = QGroupBox("Input Pattern: CSV")
        input_hbox.addWidget(csv_group, 1)
        csv_vbox = QVBoxLayout(csv_group)

        btn_force_csv = QPushButton("Force.CSV")
        btn_force_csv.setMinimumHeight(50)
        btn_force_csv.setMaximumHeight(70)
        btn_force_csv.clicked.connect(self._load_force_csv)
        csv_vbox.addWidget(btn_force_csv)

        btn_distance_csv = QPushButton("Distance.CSV")
        btn_distance_csv.setMinimumHeight(50)
        btn_distance_csv.setMaximumHeight(70)
        btn_distance_csv.clicked.connect(self._load_distance_csv)
        csv_vbox.addWidget(btn_distance_csv)

        btn_fd_csv = QPushButton("FDC.CSV")
        btn_fd_csv.setMinimumHeight(50)
        btn_fd_csv.setMaximumHeight(70)
        btn_fd_csv.clicked.connect(self._load_fd_csv)
        csv_vbox.addWidget(btn_fd_csv)



        self.row_input = QLineEdit()
        self.row_input.setPlaceholderText("Input line number (default is 0)")
        csv_vbox.addWidget(QLabel("Select the line number of the CSV data to be loaded:"))
        csv_vbox.addWidget(self.row_input)
        self.row_input.returnPressed.connect(self._on_row_selected)


        # ---- Analysis button ---------------------------------------------
        self._btn_fdc = QPushButton("FDC Analysis")
        self._btn_fdc.setMinimumHeight(50)
        self._btn_fdc.setMaximumHeight(70)
        self._btn_fdc.setEnabled(False)
        self._btn_fdc.setStyleSheet("""
            background-color: red; 
            color: white; 
            font-weight: bold; 
            font-size: 16px;
            border-radius: 10px;         
            padding: 8px 16px;          
            border: 2px solid #cc0000;  
            """)
        self._btn_fdc.clicked.connect(self._data_analyse)
        main_vbox.addWidget(self._btn_fdc)

        # ---- Display options ---------------------------------------------
        display_group = QGroupBox("Display of FDC analysis results")
        main_vbox.addWidget(display_group)
        display_vbox = QVBoxLayout(display_group)

        self.btn_diplay_fdc = QPushButton("FDC Display")
        self.btn_diplay_fdc.setMinimumHeight(50)
        self.btn_diplay_fdc.setMaximumHeight(70)
        self.btn_diplay_fdc.setEnabled(False)
        self.btn_diplay_fdc.clicked.connect(self.show_fdc_window)
        display_vbox.addWidget(self.btn_diplay_fdc)



        self._cb_original = QCheckBox("Raw FDC", checked=True)
        self._cb_down = QCheckBox("Denoised FDC")
        self._cb_peaks = QCheckBox("Fold Site")
        self._cb_wlc = QCheckBox("WLC Fitting")

        for cb in (self._cb_original, self._cb_down, self._cb_peaks, self._cb_wlc):

            cb.stateChanged.connect(self._refresh_fdc_visibility)
            display_vbox.addWidget(cb)

        # ---- Save button ---------------------------------------------------
        btn_save = QPushButton("Save the analysis results as a CSV file")
        btn_save.clicked.connect(self._save_csv)
        main_vbox.addWidget(btn_save, alignment=QtCore.Qt.AlignRight)

    # ------------------------- File loaders ---------------------------------
    # def plotlingshi(self) -> None:
    #     plt.plot(self._data["distance"],self._data["force_new"])
    #     plt.show()
    def _load_h5(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "chose H5 file", "", "HDF5 Files (*.h5 *.hdf5)")
        if path:
            f = lk.File(path)
            self._data["distance"] = np.array(f.distance1.data) * 1000
            self._data["force"] = np.array(f.downsampled_force2x.data)
            if self._data["force"][0] > self._data["force"][-1]:
                self._data["force"] = -self._data["force"]
            self.statusBar().showMessage(f"loaded {os.path.basename(path)}")
            self._check_ready()
            self._check_ready_display()
        else:
            self.statusBar().showMessage("No file selected, operation cancelled")

    def _load_baseline(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "chose H5 file", "", "HDF5 Files (*.h5 *.hdf5)")
        if path:
            f=lk.File(path)
            self._data["distance_baseline"] = np.array(f.distance1.data)*1000
            self._data["force_baseline"] = np.array(f.downsampled_force2x.data)
            self.statusBar().showMessage("Baseline file has been loaded")
            self._check_ready()
            self._check_ready_display()
        else:

            self.statusBar().showMessage("No file selected, operation cancelled")
    def _process_baseline(self) -> None:
        if "distance_baseline" not in self._data:
            QtWidgets.QMessageBox.warning(self, "Prompt", "Please load the baseline file first!！")
            return

        self._data["force"] = baseline_processing(self._data["force"],
                                                      self._data["force_baseline"],
                                                      self._data["distance"],
                                                      self._data["distance_baseline"])
        self.statusBar().showMessage("Baseline processing completed")
        self._check_ready()
        self._check_ready_display()

    def _load_force_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "chose Force CSV", "", "CSV Files (*.csv)")
        # return

        if path:
            csv_data=pd.read_csv(path,header=None)
            if len(csv_data) == 1:
                self._data["force"] = csv_data.iloc[0, :].values
            if len(csv_data.columns) == 1:
                self._data["force"] = csv_data.iloc[:, 0].values
            else:
                self._data["force"] = csv_data.iloc[0, :].values
            self.statusBar().showMessage("Force data has been loaded")
            self._check_ready()
            self._check_ready_display()

    def _load_distance_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, " chose Distance CSV", "", "CSV Files (*.csv)")
        # return
        if path:
            csv_data=pd.read_csv(path,header=None)
            if len(csv_data) == 1:
                self._data["distance"] = csv_data.iloc[0, :].values
            if len(csv_data.columns) == 1:
                self._data["distance"] = csv_data.iloc[:, 0].values
            else:
                self._data["distance"] = csv_data.iloc[0, :].values
            self.statusBar().showMessage("Distance data has been loaded")
            self._check_ready()
            self._check_ready_display()

    def _load_fd_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "chose FDC CSV", "", "CSV Files (*.csv)")
        if path:
            self._data["force_list_all"] = pd.read_csv(path,header=None).iloc[:, 0:2000].values
            self._data["distance_list_all"] = pd.read_csv(path,header=None).iloc[:, 2000:4000].values
            self._data["force"] = denoise.extract_and_resample(self._data["force_list_all"][0])
            self._data["distance"] = denoise.extract_and_resample(self._data["distance_list_all"][0])
            self.statusBar().showMessage(f'FDC data has been loaded, with the first row selected by default.(Total:{len(self._data["force_list_all"])})')
            self._check_ready()
            self._check_ready_display()
    # ------------------------- Utility -------------------------------------

    def _on_row_selected(self) -> None:
        text = self.row_input.text().strip()
        if ("force_list_all" not in self._data or "distance_list_all" not in self._data
                or len(self._data["distance_list_all"]) == 0 or len(self._data["force_list_all"]) == 0):
            QtWidgets.QMessageBox.warning(self, "Prompt", "Please load the CSV file first!(only for FDC.CSV button)")
            return
        if not text:
            row_number = 0
        else:
            try:
                row_number = int(text)
                if row_number < 0:
                    raise ValueError("Line numbers cannot be negative.")
                if row_number >= len(self._data["force_list_all"]):
                    raise ValueError("The line number exceeds the total number of data entries.")

            except ValueError as e:

                QtWidgets.QMessageBox.warning(self, "Input error", f"Invalid line number: {str(e)}")
                return


        self._process_row_selection(row_number)
        self._data_analyse()


    def _process_row_selection(self, row_number: int) -> None:

        self._data["force"]=denoise.extract_and_resample(self._data["force_list_all"][row_number])
        self._data["distance"]=denoise.extract_and_resample(self._data["distance_list_all"][row_number])
        status = f"Row {row_number} has been selected."
        self.statusBar().showMessage(status)
        self._check_ready()

    def _check_ready(self) -> None:
        if all(k in self._data for k in ("force","distance")):
            self._btn_fdc.setEnabled(True)
            # self._btn_fdc.setStyleSheet("background-color: green; color: white; "
            #                             "font-weight: bold; font-size: 16px;"
            #                             "border-radius: 10px; padding: 8px 16px;border: 2px solid green;")
            self._btn_fdc.setStyleSheet("""
                QPushButton {
                    background-color: green;
                    color: white;
                    font-weight: bold;
                    font-size: 16px;
                    border-radius: 10px;
                    padding: 8px 16px;
                    border: 2px solid green;
                }
                QPushButton:hover {
                    background-color: #228B22;  
                    border: 2px solid #006400;  
                }
                QPushButton:pressed {
                    background-color: #006400;  
                }
            """)
    def _check_ready_display(self) -> None:
        if all(k in self._data_analysis for k in ("denoise_force","denoise_distance")):
            self.btn_diplay_fdc.setEnabled(True)
            # self.btn_diplay_fdc.setStyleSheet("background-color: green; color: white; "
            #                                   "font-weight: bold; font-size: 16px;")
            self.btn_diplay_fdc.setStyleSheet("""
                QPushButton {
                    background-color: green;
                    color: white;
                    font-weight: bold;
                    font-size: 16px;
                    border-radius: 10px;
                    padding: 8px 16px;
                    border: 2px solid green;
                }
                QPushButton:hover {
                    background-color: #228B22;  
                    border: 2px solid #006400;  
                }
                QPushButton:pressed {
                    background-color: #006400; 
                }
            """)


    # ------------------------- Analysis ------------------------------------


    def _data_analyse(self) -> None:
        if len(self._data['force']) != len(self._data['distance']):
            QtWidgets.QMessageBox.warning(self, "The data lengths are inconsistent", "Please keep the lengths of force and distance consistent.")
            return
        try:
            self._data_analysis = denoise.model_out(
                force=self._data['force'], distance=self._data['distance']
            )
        except Exception as e:
            import traceback, os, datetime
            msg = traceback.format_exc()
            QtWidgets.QMessageBox.critical(self, "FDC analysis failed", msg)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"error_{ts}.log", "w", encoding="utf-8") as f:
                f.write(msg)
            return
        self.statusBar().showMessage("FDC analysis completed")
        self.show_fdc_window()

    def show_fdc_window(self) -> None:

        if ("distance" not in self._data or "force" not in self._data
                or len(self._data["distance"]) == 0 or len(self._data["force"]) == 0):
            QtWidgets.QMessageBox.warning(self, "Prompt", "Please load the data first!！")

            return

        # display_data = {
        #     "distance": self._data["distance"],
        #     "force": self._data["force_new"],
        # }
        if self._fdc_window and self._fdc_window.isVisible():
            self._fdc_window.close()

        self._fdc_window = FDCWindow(self, self._data_analysis)
        self._fdc_window.show()
        self._refresh_fdc_visibility()
        self._check_ready_display()

    def _refresh_fdc_visibility(self) -> None:
        if self._fdc_window:
            self._fdc_window.update_visibility(
                self._cb_original.isChecked(),
                self._cb_down.isChecked(),
                self._cb_peaks.isChecked(),
                self._cb_wlc.isChecked(),
            )

    # ------------------------- Save CSV ------------------------------------

    def _save_csv(self) -> None:
        if not all(k in self._data for k in ("force", "distance")):
            QtWidgets.QMessageBox.warning(self, "Prompt", "Please complete the analysis and load the data first.！")

            return
        path, _ = QFileDialog.getSaveFileName(self, "save CSV", "fdc_result.csv", "CSV Files (*.csv)")
        if path:
            out = {
                    "origin_force": safe_list(self._data_analysis["origin_force"]),
                    "origin_distance": safe_list(self._data_analysis["origin_distance"]),
                    "denoise_force": safe_list(self._data_analysis["denoise_force"]),
                    "denoise_distance": safe_list(self._data_analysis["denoise_distance"]),
                    "fold_num": safe_list(self._data_analysis["fold_num"]),
                    "PLC": safe_list(flatten(self._data_analysis["PLC"])),
                    "actual_structure_size": safe_list(self._data_analysis["actual_structure_size"]),
                    "structure_size_direct": safe_list(self._data_analysis["structure_size_direct"]),
                    "free_energy": safe_list(self._data_analysis["free_energy"]),
                    "fold_site": safe_list(flatten(self._data_analysis["fold_site"])),
                }
            max_len = max(safe_len(v) for v in out.values())

            for k, v in out.items():
                if safe_len(v) < max_len:
                    out[k] = v + [np.nan] * (max_len - safe_len(v))
            # print('ok')

            out=pd.DataFrame(out)
            out.to_csv(path, index=False)
            self.statusBar().showMessage(f"The analysis results have been saved to {path}")

        # print(f'origin_force：{force_origin}')
        # print(f'origin_distance：{distance_origin}')
        # print(f'denoise_force：{force_p}')
        # print(f'denoise_distance：{distance_p}')
        # print(f'fold_num：{len(point_list)}')
        # print(f'PLC：{P_L_C_list}')
        # print(f'actual_structure_size：{structure_size} nm')
        # print(f'structure_size_direct：{structure_size_direct} nm')
        # print(f'free_energy：{S_fold_list} kJ/mol')
        # print(f'fold_site：{point_list} kJ/mol')
# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None: 
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
