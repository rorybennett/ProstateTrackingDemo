# Clarius Cast Dual Display Demo

A small Python/PySide6 demo application for connecting to a Clarius scanner using the Clarius Cast libraries.

The app shows two ultrasound image panels:

- **Original ultrasound image** – the image received from the scanner
- **Processed image** – currently a copy of the original image, but this is where custom processing can be added

## Project structure

```text
.
├── sidecaster.py
└── libraries/
    ├── pyclariuscast.pyd # Windows
    ├── cast.dll          # Windows
    ├── libcast.so        # Linux
    └── pyclariuscast.so  # Linux
```

The required Clarius Cast library files must be placed in the `libraries` subdirectory next to `sidecaster.py`.

## Requirements

- Python 3
- PySide6
- Clarius Cast library files in `libraries/`
- A Clarius scanner available on the network

Install the Python dependency with:

```bash
pip install PySide6
```

## Running the app

From the project directory, run:

```bash
python sidecaster.py
```

The default connection settings are:

- IP address: `192.168.1.1`
- Port: `5828`

Change these in the app window if needed, then click **Connect**.

## Controls

- **Connect / Disconnect** – connect to or disconnect from the scanner
- **Run / Freeze** – start or freeze the live image stream
- **Depth** – increase or decrease imaging depth
- **Gain** – increase or decrease image gain
- **Capture Image** – trigger image capture on the scanner
- **Capture Movie** – trigger cine/movie capture on the scanner
- **Save Local** – save the current original and processed images locally
- **B Mode** – switch to B-mode imaging
- **Color Mode** – switch to colour flow imaging

Saved local images are written to:

```text
~/Pictures/clarius_original_image.png
~/Pictures/clarius_processed_image.png
```

## Adding image processing

Custom image processing can be added in `processImageForDisplay()` inside `sidecaster.py`.

At the moment it simply returns a copy of the original image:

```python
def processImageForDisplay(self, original_img):
    return original_img.copy()
```

Modify this function to update what is shown in the processed image panel.

## Notes

- On Windows, the app loads `cast.dll` from the `libraries` directory.
- On Linux, the app loads `libcast.so` and `pyclariuscast.so` from the `libraries` directory.
- If the app connects but no image is visible, check firewall and network settings.
