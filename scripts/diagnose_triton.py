"""One-off Modal probe: is Triton usable for torch.compile?"""

from __future__ import annotations

import modal

from modal_utils import MODAL_ENVIRONMENT, build_image


app = modal.App("diagnose-triton")
image = build_image()


@app.function(image=image, gpu="H100", timeout=600)
def diagnose_triton():
    import os
    import shutil
    import subprocess
    import sys
    import traceback

    out: dict = {
        "python": sys.version,
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = torch.cuda.is_available()
        out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        out["torch_err"] = repr(e)
        out["torch_tb"] = traceback.format_exc()
        return out

    try:
        import triton

        out["triton"] = getattr(triton, "__version__", "?")
        out["triton_file"] = getattr(triton, "__file__", None)
    except Exception as e:
        out["triton_import_err"] = repr(e)
        out["triton_tb"] = traceback.format_exc()
        return out

    try:
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, n):
            pid = tl.program_id(0)
            if pid < n:
                y_ptr[pid] = x_ptr[pid] + 1

        out["triton_jit_ok"] = True
    except Exception as e:
        out["triton_jit_err"] = repr(e)
        out["triton_jit_tb"] = traceback.format_exc()

    try:
        from torch._inductor.utils import has_triton

        out["has_triton"] = bool(has_triton())
    except Exception as e:
        out["has_triton_err"] = repr(e)
        out["has_triton_tb"] = traceback.format_exc()

    for bin_name in ("ptxas", "nvcc", "nvidia-smi"):
        out[f"which_{bin_name}"] = shutil.which(bin_name)

    try:
        out["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    except Exception as e:
        out["nvidia_smi_err"] = repr(e)

    # Minimal inductor compile probe.
    try:
        import torch

        @torch.compile(mode="reduce-overhead")
        def f(x):
            return x * 2 + 1

        x = torch.randn(1024, device="cuda")
        y = f(x)
        out["compile_ok"] = True
        out["compile_out_mean"] = float(y.mean().item())
    except Exception as e:
        out["compile_err"] = repr(e)
        out["compile_tb"] = traceback.format_exc()[-2000:]

    return out


def main():
    with modal.enable_output():
        with app.run(environment_name=MODAL_ENVIRONMENT):
            print(diagnose_triton.remote())


if __name__ == "__main__":
    main()
