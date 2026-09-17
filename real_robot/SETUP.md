# Real xArm7 environment setup

Run commands from the repository root. These instructions describe the Ubuntu
24.04 / Python 3.12 / RTX 5090 installation used by this branch. They install
software and model assets; they do not command robot motion. On the existing
workstation, the environments and model assets are already installed.

## 1. Host prerequisites

```bash
sudo apt update
sudo apt install python3.12-venv python3-tk git wget unzip build-essential libglib2.0-0
```

The click-to-select scene helper uses Matplotlib **TkAgg**, so `python3-tk` and a
local desktop session are needed even though OpenCV is the headless build.

Install Docker Engine and the NVIDIA Container Toolkit following the
[Docker Ubuntu guide](https://docs.docker.com/engine/install/ubuntu/) and
[NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The configured FoundationPose container needs a compatible NVIDIA driver;
[its upstream requirements](https://github.com/nvidia-isaac/foundation-pose-inference-library#requirements)
specify driver 580 or newer. Verify GPU access:

```bash
nvidia-smi
docker pull nvcr.io/nvidia/pytorch:26.05-py3
docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.05-py3 nvidia-smi
```

Connect the D435i over USB 3. If device permissions prevent capture, follow the
[RealSense Linux setup](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md)
to install its udev rules. Close other applications using the camera before capture.
The pipeline and scene helper reset the camera once at startup.

## 2. Host execution environment: `.venv-real`

This environment runs scene preparation, RGB-D capture, planning, plotting,
identity association, and the official xArm SDK. It does not need PyTorch.

```bash
python3.12 -m venv .venv-real
.venv-real/bin/python -m pip install --upgrade pip
.venv-real/bin/python -m pip install -r requirements-real.txt
.venv-real/bin/python -m pip check
```

`requirements-real.txt` includes the shared NumPy/SciPy/NetworkX/Matplotlib/PyYAML
requirements, Pillow, and the installed camera/SDK/OpenCV versions:

- `xarm-python-sdk==1.18.4`
- `pyrealsense2==2.58.4.10922`
- `opencv-contrib-python-headless==5.0.0.93`

For the exact recorded host package versions, use this **instead of** the generic
requirements installation above:

```bash
.venv-real/bin/python -m pip install -r requirements-real.lock.txt
```

The lock file records the current working environment's runtime dependencies;
installing it on a clean machine has not been retested here. PyBullet and Trimesh
are not required by the current Cartesian execution pipeline. The separate
legacy joint-planning prototype is not part of this setup.

Either use the explicit `.venv-real/bin/python` commands throughout, or activate
it once per terminal to use the shorter `python` commands:

```bash
source .venv-real/bin/activate
python -c 'import sys; print(sys.executable)'
```

## 3. SAM2 environment: `real_robot/.venv-pose`

SAM2 runs in its own host environment. The installed source revision and CUDA
wheel versions are pinned below; do not install them into Isaac's environment.
The clone commands are for a fresh checkout; skip cloning directories that
already contain the required source revision.

```bash
mkdir -p real_robot/external real_robot/models
python3.12 -m venv real_robot/.venv-pose
real_robot/.venv-pose/bin/python -m pip install --upgrade pip setuptools wheel
real_robot/.venv-pose/bin/python -m pip install \
  torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/facebookresearch/sam2.git real_robot/external/sam2
git -C real_robot/external/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM2_BUILD_CUDA=0 real_robot/.venv-pose/bin/python -m pip install \
  --no-build-isolation -e real_robot/external/sam2
real_robot/.venv-pose/bin/python -m pip check

wget -O real_robot/models/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

The editable install brings in SAM2's Python dependencies, including Hydra,
OmegaConf, iopath, Pillow, NumPy, and tqdm. `SAM2_BUILD_CUDA=0` explicitly skips
its optional hole-filling extension, matching the pipeline's existing operation
without that extension. A warning about `sam2._C` is separate from a failed
instance-mask/identity check. To enable the optional extension, install a CUDA
toolkit matching the PyTorch CUDA version and follow [SAM2's installation notes](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).

```bash
real_robot/.venv-pose/bin/python -c \
  'import torch, sam2; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

## 4. FoundationPose runtime and model assets

FoundationPose executes **inside Docker**, using the Python bindings from the
source tree through `PYTHONPATH`. Its CUDA/TensorRT shared library is built in the
same container image. Installing a host Python package alone does not supply this
library, and the SAM2 CUDA 12.8 environment does not replace the container.

```bash
git clone https://github.com/nvidia-isaac/foundation-pose-inference-library.git \
  real_robot/external/foundation-pose-inference-library
git -C real_robot/external/foundation-pose-inference-library checkout \
  9eabc9b21ad5fe66a895a8f3ad2dc95861b6d6e9

bash real_robot/external/foundation-pose-inference-library/scripts/download_weights.sh \
  "$PWD/real_robot/models/foundationpose"

docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$PWD:/project" \
  -w /project/real_robot/external/foundation-pose-inference-library \
  nvcr.io/nvidia/pytorch:26.05-py3 \
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    '-DCMAKE_CUDA_ARCHITECTURES=75;80;86;90;100;120' \
    -DTENSORRT_ROOT=/usr -DFOUNDATION_POSE_WITH_NVDIFFRAST=OFF

docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -v "$PWD:/project" \
  -w /project/real_robot/external/foundation-pose-inference-library \
  nvcr.io/nvidia/pytorch:26.05-py3 cmake --build build -j4
```

These flags match the installed CMake cache, including the RTX 5090 `120` target.
The resulting library must be at
`real_robot/external/foundation-pose-inference-library/build/libfoundation_pose_nvidia.so`.
Weights must be named `real_robot/models/foundationpose/refiner_net.onnx` and
`score_net.onnx`. See the [upstream build guide](https://github.com/nvidia-isaac/foundation-pose-inference-library#getting-started)
for container-specific build troubleshooting.

Generate the local cuboid meshes from this branch's model dimensions:

```bash
.venv-real/bin/python real_robot/make_box_mesh.py --kind protex --out real_robot/models/protex_90x60x28mm.obj
.venv-real/bin/python real_robot/make_box_mesh.py --kind beach --out real_robot/models/beach_110x65x25mm.obj
.venv-real/bin/python real_robot/make_box_mesh.py --kind toothpaste --out real_robot/models/toothpaste_210x50x40mm.obj
```

Model assets, external source trees, virtual environments, and experiment captures
are intentionally ignored by Git. Cloning this branch does not download them.

## 5. Validate and prepare a scene

Import checks and the offline suite do not connect to the arm or capture images:

```bash
.venv-real/bin/python -c 'import numpy, scipy, networkx, yaml, matplotlib, tkinter, cv2, PIL, pyrealsense2; from xarm.wrapper import XArmAPI; print("Host imports OK")'
.venv-real/bin/python -m unittest discover -s real_robot -p 'test_*.py' -v
.venv-real/bin/python real_robot/run_pipeline.py --config real_robot/experiment_design/configs/S07.json --check-config
```

`--check-config` checks motion settings, not model installation, reachability, or
physical calibration. The real-robot configs include this workstation's camera
transform, manual XY correction, robot IP, and agreed geometry/contact settings.

Next follow the [S07/S12/S13 preparation, preview, and execution commands](README.md#prepare-configuration-and-execute-s07-s12-or-s13).
Prepare a new configuration after rearranging the scene. Only `--execute` connects
the pipeline to the robot; a preview captures and processes images without motion.

## Installed environment record

Recorded on 2026-09-17: Python 3.12, host xArm SDK 1.18.4, RealSense Python SDK
2.58.4.10922, SAM2 torch 2.11.0+cu128 / torchvision 0.26.0+cu128, and FoundationPose
container `nvcr.io/nvidia/pytorch:26.05-py3`. Source revisions above were read from
the installed checkouts; both external checkouts had no local modifications.
The full installation/download/build sequence is documented, not rerun during
this documentation update. No existing environment was upgraded.
