"""Magicgui launcher window for ``python -m vascumap``.

Collects folders, organoid masking mode, device width, model paths, and
advanced toggles. Two action buttons close the window and store the chosen
run mode so the caller can dispatch to the appropriate batch runner.
"""

from pathlib import Path
from typing import Optional

from magicgui import magicgui
from magicgui.widgets import Container, Label, PushButton, RadioButtons, TextEdit

DEFAULT_PIX2PIX_CKPT = Path(r"C:\Users\taylorhearn\git_repos\vascumap\luca_models\epoch=117-val_g_psnr=20.47-val_g_ssim=0.62.ckpt")
DEFAULT_UNET_CKPT = Path(r"C:\Users\taylorhearn\git_repos\vascumap\luca_models\best_full.pth")
DEFAULT_DEVICE_WIDTH_UM = 35.0

ORGANOID_CHOICES = ["No organoid masking", "Dark organoid", "Light organoid", "Infer from name"]
ORGANOID_TO_FORCE_MASK = {"No organoid masking": False, "Dark organoid": "dark", "Light organoid": "light", "Infer from name": None}


class VascuMapLauncherGUI:
    """Standalone magicgui launcher window for the VascuMap pipeline.

    After the user clicks either action button the window closes and
    ``config`` is populated with all validated settings.
    """

    def __init__(self):
        """Build all widget panels and show the launcher window."""
        self.closed = False
        self.config: Optional[dict] = None

        self.folder_panel = magicgui(self.folder_stub, source_dir={"label": "Source Folder", "mode": "d"}, output_dir={"label": "Output Folder", "mode": "d"}, skip_dir={"label": "Skip Folder (optional)", "mode": "d"}, call_button=False)

        self.organoid_widget = RadioButtons(label="Organoid Masking", choices=ORGANOID_CHOICES, value="No organoid masking")

        self.options_panel = magicgui(self.options_stub, device_width_um={"label": "Device Width (µm)", "value": DEFAULT_DEVICE_WIDTH_UM, "min": 0.0, "max": 10000.0, "step": 0.5}, brightfield_channel={"label": "Brightfield Channel", "value": 0, "min": 0, "max": 20}, require_merged={"label": "Require 'merged' in name", "value": True}, save_all_interim={"label": "Save All Intermediates", "value": False}, call_button=False)

        self.model_panel = magicgui(self.model_stub, pix2pix_ckpt={"label": "Pix2Pix Checkpoint", "mode": "r", "value": DEFAULT_PIX2PIX_CKPT}, unet_ckpt={"label": "U-Net Checkpoint", "mode": "r", "value": DEFAULT_UNET_CKPT}, call_button=False)

        self.btn_gui = PushButton(text="Use GUI (Curation)")
        self.btn_gui.clicked.connect(self.on_gui_clicked)
        self.btn_auto = PushButton(text="Run Automatically")
        self.btn_auto.clicked.connect(self.on_auto_clicked)

        self.log_widget = TextEdit(value="Configure folders, masking, device width, and model paths.\nThen click Use GUI (Curation) or Run Automatically.\nThe window will close and progress will appear in the terminal.\n")
        self.log_widget.min_height = 120
        try:
            self.log_widget.native.setReadOnly(True)
        except Exception:
            pass

        self.widget = Container(widgets=[Label(value="── Folders ──"), self.folder_panel, Label(value="── Organoid Masking ──"), self.organoid_widget, Label(value="── Options ──"), self.options_panel, Label(value="── Models ──"), self.model_panel, Label(value="── Run ──"), self.btn_gui, self.btn_auto, self.log_widget], label="VascuMap Launcher")
        self.widget.native.setWindowTitle("VascuMap Launcher")
        self.widget.native.setMinimumWidth(560)

        launcher = self
        original_close = self.widget.native.closeEvent

        def on_native_close(event):
            """Mark the launcher as closed when the window X button is clicked."""
            launcher.closed = True
            if original_close:
                original_close(event)

        self.widget.native.closeEvent = on_native_close
        self.folder_panel.source_dir.changed.connect(self.on_source_changed)
        self.widget.show()

    @staticmethod
    def folder_stub(source_dir: Path = Path(), output_dir: Path = Path(), skip_dir: Path = Path()):
        """Stub function backing the folder picker panel."""
        return None

    @staticmethod
    def options_stub(device_width_um: float = DEFAULT_DEVICE_WIDTH_UM, brightfield_channel: int = 0, require_merged: bool = True, save_all_interim: bool = False):
        """Stub function backing the options panel."""
        return None

    @staticmethod
    def model_stub(pix2pix_ckpt: Path = DEFAULT_PIX2PIX_CKPT, unet_ckpt: Path = DEFAULT_UNET_CKPT):
        """Stub function backing the model checkpoint picker panel."""
        return None

    def log(self, message: str):
        """Append a line to the status log widget."""
        current = self.log_widget.value.rstrip()
        self.log_widget.value = (current + "\n" + message) if current else message

    @staticmethod
    def dir_or_none(value) -> Optional[str]:
        """Return the string path if it points to an existing directory, else None."""
        path = Path(str(value))
        if str(path) in (".", "") or not path.is_dir():
            return None
        return str(path)

    @staticmethod
    def file_or_none(value) -> Optional[str]:
        """Return the string path if it points to an existing file, else None."""
        path = Path(str(value))
        if str(path) in (".", "") or not path.is_file():
            return None
        return str(path)

    def on_source_changed(self, value):
        """Log the number of compatible image files when the source folder changes."""
        source = self.dir_or_none(value)
        if source is None:
            return
        n = sum(1 for p in Path(source).iterdir() if p.is_file() and p.suffix.lower() in (".lif", ".tif", ".tiff"))
        self.log(f"[INFO] Found {n} .lif/.tif/.tiff file(s) in {source}")

    def collect_and_validate(self) -> Optional[dict]:
        """Read all widget values, validate them, and return a config dict or None.

        Appends error messages to the log widget and returns None if validation
        fails so the window stays open for the user to correct the issue.
        """
        errors: list[str] = []

        source_dir = self.dir_or_none(self.folder_panel.source_dir.value)
        if source_dir is None:
            errors.append("Source Folder is required and must exist.")

        output_value = Path(str(self.folder_panel.output_dir.value))
        if str(output_value) in (".", ""):
            errors.append("Output Folder is required.")
            output_dir = None
        else:
            try:
                output_value.mkdir(parents=True, exist_ok=True)
                output_dir = str(output_value)
            except Exception as exc:
                errors.append(f"Could not create Output Folder: {exc}")
                output_dir = None

        skip_dir = self.dir_or_none(self.folder_panel.skip_dir.value)
        device_width_um = float(self.options_panel.device_width_um.value)
        if device_width_um <= 0:
            errors.append("Device Width (µm) must be > 0.")

        force_mask = ORGANOID_TO_FORCE_MASK[self.organoid_widget.value]
        pix2pix_ckpt = self.file_or_none(self.model_panel.pix2pix_ckpt.value)
        if pix2pix_ckpt is None:
            errors.append("Pix2Pix Checkpoint must point to an existing file.")
        unet_ckpt = self.file_or_none(self.model_panel.unet_ckpt.value)
        if unet_ckpt is None:
            errors.append("U-Net Checkpoint must point to an existing file.")

        if errors:
            self.log("--- Validation failed ---")
            for error in errors:
                self.log(f"  [ERROR] {error}")
            return None

        return {"source_dir": source_dir, "output_dir": output_dir, "skip_dir": skip_dir, "force_mask": force_mask, "device_width_um": device_width_um, "brightfield_channel": int(self.options_panel.brightfield_channel.value), "require_merged": bool(self.options_panel.require_merged.value), "save_all_interim": bool(self.options_panel.save_all_interim.value), "pix2pix_ckpt": pix2pix_ckpt, "unet_ckpt": unet_ckpt}

    def on_gui_clicked(self):
        """Validate settings, store mode='gui', and close the launcher window."""
        cfg = self.collect_and_validate()
        if cfg is None:
            return
        cfg["mode"] = "gui"
        self.config = cfg
        self.closed = True
        self.widget.close()

    def on_auto_clicked(self):
        """Validate settings, store mode='auto', and close the launcher window."""
        cfg = self.collect_and_validate()
        if cfg is None:
            return
        cfg["mode"] = "auto"
        self.config = cfg
        self.closed = True
        self.widget.close()


def launch() -> VascuMapLauncherGUI:
    """Create and show the launcher GUI. The caller must run the Qt event loop."""
    return VascuMapLauncherGUI()
