# Real-Time Drowsiness Detection System

A real-time computer vision system that monitors eye activity through a webcam and detects prolonged eye closure as a potential indicator of drowsiness.

The system uses **MediaPipe Face Mesh** to track facial landmarks, calculates the **Eye Aspect Ratio (EAR)** from selected eye landmarks, and triggers a visual and audible alert when the eyes remain closed for a sustained number of frames.

---

## Overview

Driver and user drowsiness can be associated with prolonged eye closure and reduced visual attention. This project explores a lightweight computer vision approach for detecting this behavior in real time using facial landmark analysis.

The application:

* Captures live video through a webcam
* Detects facial landmarks using MediaPipe Face Mesh
* Tracks landmark points around both eyes
* Calculates the **Eye Aspect Ratio (EAR)**
* Identifies prolonged eye closure using a configurable threshold
* Displays real-time EAR and detection status
* Activates a looping alarm when drowsiness is detected

> **Note:** This project is a computer vision prototype and is not intended to replace certified driver-monitoring or safety systems.

---

## How It Works

The detection pipeline follows these steps:

```text
Webcam Feed
     │
     ▼
Frame Capture
     │
     ▼
MediaPipe Face Mesh
     │
     ▼
Eye Landmark Extraction
     │
     ▼
Eye Aspect Ratio (EAR)
     │
     ▼
Eye Closure Detection
     │
     ▼
Consecutive Frame Check
     │
     ├── Normal → Continue Monitoring
     │
     └── Prolonged Closure → Drowsiness Alert + Alarm
```

### Eye Aspect Ratio (EAR)

The system estimates whether the eyes are open or closed using the **Eye Aspect Ratio**.

For each eye:

$$
EAR = \frac{A + B}{2C}
$$

where:

* **A** and **B** represent vertical distances between selected eye landmarks
* **C** represents the horizontal distance across the eye

When the EAR falls below the configured threshold, the system considers the eyes to be closed.

The current implementation uses:

```text
EAR threshold: 0.15
Consecutive closed frames: 20
```

These values can be adjusted in `app.py` depending on lighting, camera position, and individual eye geometry.

---

## Features

* **Real-time webcam monitoring**
* **Facial landmark tracking** using MediaPipe Face Mesh
* **Eye Aspect Ratio-based detection**
* **Continuous eye-closure monitoring**
* **Visual drowsiness warning**
* **Audible looping alarm**
* **Live EAR display**
* Lightweight local processing with no external API dependency

---

## Tech Stack

| Technology              | Purpose                                             |
| ----------------------- | --------------------------------------------------- |
| **Python**              | Core programming language                           |
| **OpenCV**              | Webcam capture, image processing, and visualization |
| **MediaPipe Face Mesh** | Facial and eye landmark detection                   |
| **NumPy**               | Distance calculations and numerical operations      |
| **Windows `winsound`**  | Audible alarm playback                              |

---

## Project Structure

```text
DrowsinessDetector/
│
├── app.py
│   └── Main application and real-time detection pipeline
│
├── modules/
│   └── face_detector.py
│       └── Face detection/landmark helper module
│
├── alarm.wav
│   └── Audio alert used when drowsiness is detected
│
├── requirements.txt
│   └── Pinned project dependencies
│
├── .gitignore
│   └── Files and folders excluded from version control
│
└── README.md
    └── Project documentation
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd DrowsinessDetector
```

### 2. Create a Virtual Environment

Using a virtual environment is recommended to keep project dependencies isolated.

**Windows PowerShell:**

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\Activate.ps1
```

If activation is successful, your terminal will show:

```text
(venv)
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

The project uses pinned dependency versions to maintain compatibility with the current MediaPipe-based implementation.

---

## Usage

Start the application with:

```powershell
python app.py
```

The webcam window will open and begin monitoring the detected face.

The application displays:

* Current **EAR value**
* **EYES CLOSED!** when the EAR falls below the configured threshold
* **DROWSINESS ALERT!** when prolonged eye closure is detected

When the drowsiness condition is reached, the application plays the alarm continuously until the eyes are detected as open again.

### Controls

| Key | Action                           |
| --- | -------------------------------- |
| `q` | Close the webcam window and exit |

---

## Configuration

The primary detection parameters can be adjusted directly in `app.py`:

```python
EAR_THRESHOLD = 0.15
DROWSY_FRAMES = 20
```

### `EAR_THRESHOLD`

Determines the EAR value below which the eyes are considered closed.

### `DROWSY_FRAMES`

Determines how many consecutive frames must indicate closed eyes before the drowsiness alert is triggered.

These parameters are intentionally kept configurable because detection behavior can vary with camera placement, lighting conditions, face position, and individual facial characteristics.

---

## Requirements

* **Python 3.x**
* A working **webcam**
* **Windows** for the built-in `winsound` alarm functionality

The current dependency versions are pinned in `requirements.txt` for reproducibility.

---

## Limitations

This implementation is intentionally lightweight and has several limitations:

* Detection depends on adequate webcam visibility and lighting.
* Facial landmark tracking may be affected by significant head movement or occlusion.
* A fixed EAR threshold may not work equally well for every individual.
* The system focuses primarily on prolonged eye closure and does not model other indicators of drowsiness.
* The current alarm implementation uses Windows-specific `winsound` functionality.

---

## Future Improvements

Potential extensions include:

* Personalized EAR calibration
* Blink-rate and blink-duration analysis
* Yawning detection
* Head-pose estimation
* Improved temporal smoothing
* Multi-condition drowsiness scoring
* Cross-platform audio alerts
* Performance and robustness evaluation across different lighting and camera conditions
* A more structured user interface for configuration and monitoring
