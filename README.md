# Clarius Cast Sidecaster

Minimal PySide6 app for streaming Clarius ultrasound and showing two views:

- **Original ultrasound image**
- **Processed image** with YOLO detection or UNet segmentation

`sidecaster.py` imports the UNet model from `UNet.py`.

## Expected layout

`sidecaster.py` treats its parent directory as the asset directory:

```text
<asset-dir>/
├── libraries/
│   ├── cast.dll              # Windows
│   ├── libcast.so            # Linux
│   └── pyclariuscast.so      # Linux
├── models/
│   ├── yolo_x_phantom_best.pt
│   ├── unet_phantom_latest.pth
│   └── training_parameters.txt   # optional
└── <app-dir>/
    ├── sidecaster.py
    ├── UNet.py
    └── src/                  # optional guidance icons
```

## Requirements

- Python 3
- PySide6
- NumPy
- Ultralytics
- PyTorch
- Clarius Cast library files
- Clarius scanner on the network

```bash
pip install PySide6 numpy ultralytics torch
```

## Run

```bash
python sidecaster.py
```

Default connection:

```text
IP:   192.168.1.1
Port: 5828
```

Click **Connect**, then **Run**.

## Main controls

- **YOLO / UNet** – choose detection or segmentation
- **ROI** – limit processing to the centre of the image
- **UNet Threshold** – adjust the segmentation threshold
- **RL / AP / SI** – show measurements on the processed image
- **Record Boundaries** – plot UNet boundary points in 3D when IMU data is available
- **Save Local** – save current original and processed images

Saved images:

```text
~/Pictures/clarius_original_image.png
~/Pictures/clarius_processed_image.png
```

## Notes

- YOLO and UNet are mutually exclusive in the processed view.
- UNet uses `models/unet_phantom_latest.pth` and the architecture in `UNet.py`.
- If the app connects but no image appears, check firewall and network settings.
