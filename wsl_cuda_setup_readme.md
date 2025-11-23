# CUDA on WSL2 — Complete Setup Guide (Working Instructions)

This README provides the exact steps needed to get **CUDA working inside WSL2 (Ubuntu)** using an NVIDIA GPU on Windows 11. These steps are based on a real working configuration and avoid all the pitfalls that commonly break CUDA in WSL.

---

## 🚀 Overview
WSL2 supports GPU acceleration through a special virtualization layer (`dxgkrnl`). To actually use CUDA inside WSL, you must:
- Use Windows 11
- Install the regular Windows NVIDIA driver
- Update WSL
- Install the Ubuntu CUDA Toolkit (NOT the Linux NVIDIA driver)

This guide walks through the full working setup.

---

# 1️⃣ Windows Requirements

### ✔ Windows 11 (23H2 or 24H2 recommended)
You must be on a modern Windows 11 build:
- **23H2 or later**
- Build number **≥ 22621**

Check your version:
```
winver
```

If you are on 24H2, everything works out of the box.

---

# 2️⃣ Install NVIDIA GPU Driver (on Windows)
You only need the **normal Windows NVIDIA driver**.
Do **not** install Linux drivers.

Download from NVIDIA's website or use GeForce Experience.

Verify:
```
nvidia-smi
```

This must show your GPU.

---

# 3️⃣ Update WSL and Kernel
Open PowerShell as Admin:

```
wsl --update
wsl --status
```

You must see:
- **WSL2 kernel**
- GPU acceleration supported
- Kernel ≥ **6.1**

---

# 4️⃣ Install CUDA Toolkit INSIDE WSL (correct method)
Do **not** install `nvidia-driver-*` or NVIDIA's Linux CUDA repo.
WSL uses Ubuntu's own CUDA toolkit.

Run:
```
sudo apt update
sudo apt install -y nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc
```

This installs:
- NVCC compiler
- CUDA runtime
- cuBLAS, cuFFT, Thrust
- CUDA examples

Check NVCC:
```
nvcc --version
```

---

# 5️⃣ Verify WSL GPU Virtualization
WSL exposes the GPU via `/dev/dxg`.

Check:
```
ls -l /dev/dxg
```

WSL GPU libraries:
```
ls /usr/lib/wsl/lib
```

Run the WSL version of nvidia-smi:
```
/usr/lib/wsl/lib/nvidia-smi
```

If this works → GPU passthrough is healthy.

---

# 6️⃣ Test CUDA Program (VectorAdd)
Ubuntu places CUDA examples here:
```
/usr/share/doc/nvidia-cuda-toolkit/examples
```

Copy them to your home folder:
```
cp -r /usr/share/doc/nvidia-cuda-toolkit/examples ~/cuda-examples
cd ~/cuda-examples/vectorAdd
nvcc vectorAdd.cu -o vectorAdd
./vectorAdd
```

Expected output:
```
Vector addition succeeded!
```

---

# 7️⃣ (Optional) Run deviceQuery
Ubuntu ships a portable deviceQuery:

```
nvcc -o deviceQuery /usr/share/doc/nvidia-cuda-toolkit/examples/deviceQuery/deviceQuery.cpp
./deviceQuery
```

You should see:
```
Detected 1 CUDA Capable devices
Result = PASS
```

---

# 🟢 CUDA is Now Fully Working Inside WSL
Everything is ready for CUDA kernels, cuBLAS, cuFFT, PyTorch, TensorFlow, or custom GPU compute.

---

# ❌ Common Mistakes That Break WSL CUDA
Avoid these:

- Do **NOT** install Linux NVIDIA drivers:
  ```
  sudo apt install nvidia-driver-XXX   # WRONG
  ```

- Do **NOT** install CUDA from NVIDIA’s Linux repo
- Do **NOT** run WSL1

Only the Windows driver + Ubuntu CUDA toolkit is needed.

---

# 📌 Ultra-Short Copy/Paste Setup
```
# Windows
nvidia-smi
wsl --update

# WSL
sudo apt update
sudo apt install -y nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc
/usr/lib/wsl/lib/nvidia-smi
nvcc --version

# Test CUDA
cp -r /usr/share/doc/nvidia-cuda-toolkit/examples ~/cuda-examples
cd ~/cuda-examples/vectorAdd
nvcc vectorAdd.cu -o vectorAdd
./vectorAdd
```

---

# 📘 License
Feel free to copy/paste into GitHub, documentation, forums, or your project README.

---

If you want, I can also add:
- A troubleshooting section
- A version for Arch/WSL
- A separate guide for PyTorch or TensorFlow GPU setup

