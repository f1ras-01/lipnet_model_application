"""
gpu_utils.py
------------
Detects and selects the NVIDIA GPU explicitly, hiding all other GPU devices
(Intel iGPU, AMD iGPU) from TensorFlow.

On laptops with both an Intel integrated GPU and an NVIDIA dedicated GPU,
TensorFlow picks GPU:0 (Intel) by default. Intel iGPUs don't support full
CUDA compute — they crash mid-training with "Unexpected Event status: 1".

This module:
  1. Lists all physical GPU devices TF can see.
  2. Finds the NVIDIA one by checking the device name string.
  3. Sets ONLY that device as visible to TF.
  4. Enables memory growth so the GPU VRAM isn't fully reserved at startup.

Import this at the very top of any entry-point script, BEFORE any other
TF or project imports:

    from gpu_utils import setup_gpu
    setup_gpu()
"""

import tensorflow as tf


def setup_gpu() -> None:
    """
    Select the NVIDIA GPU and hide all other GPU devices from TensorFlow.

    If no NVIDIA GPU is found, falls back to the first available GPU,
    or CPU if no GPUs exist at all.
    """
    all_gpus = tf.config.list_physical_devices("GPU")

    if not all_gpus:
        print("No GPU found — running on CPU.")
        return

    # Find NVIDIA GPU by name (TF device names contain "GPU" + driver info)
    # On Windows the name looks like: /physical_device:GPU:0
    # We query the device details to get the actual hardware name
    nvidia_gpu = None
    for gpu in all_gpus:
        details = tf.config.experimental.get_device_details(gpu)
        device_name = details.get("device_name", "").upper()
        if "NVIDIA" in device_name or "GEFORCE" in device_name or "RTX" in device_name or "GTX" in device_name or "QUADRO" in device_name or "TESLA" in device_name:
            nvidia_gpu = gpu
            break

    if nvidia_gpu is None:
        # Fallback: use last GPU (on most laptops NVIDIA is listed after Intel)
        nvidia_gpu = all_gpus[-1]
        print(f"⚠  No NVIDIA GPU identified by name — using last device: {nvidia_gpu}")
        print(f"   All detected GPUs: {[g.name for g in all_gpus]}")
    
    # Hide all other devices — TF will only see the selected NVIDIA GPU
    try:
        tf.config.set_visible_devices(nvidia_gpu, "GPU")
        tf.config.experimental.set_memory_growth(nvidia_gpu, True)
        details = tf.config.experimental.get_device_details(nvidia_gpu)
        hw_name = details.get("device_name", nvidia_gpu.name)
        print(f"GPU: {hw_name}  ({nvidia_gpu.name})")
        if len(all_gpus) > 1:
            hidden = [g.name for g in all_gpus if g != nvidia_gpu]
            print(f"   Hidden (iGPU/other): {hidden}")
    except RuntimeError as e:
        # set_visible_devices must be called before any GPU ops — if TF has
        # already initialized the GPU context this will fail
        print(f"⚠  Could not restrict GPU visibility: {e}")
        print(f"   Using default GPU selection.")
        try:
            tf.config.experimental.set_memory_growth(all_gpus[0], True)
        except RuntimeError:
            pass
