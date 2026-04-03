import subprocess

def detect_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout.splitlines()
        gpu_name = None
        for line in output:
            if "Tesla" in line or "RTX" in line or "A100" in line or "V100" in line:
                gpu_name = line.split("|")[1].strip()
                gpu_name = gpu_name.split(" ", 1)[1].strip().split("  ")[0]
                break

        return {
            "available": True,
            "gpu_name": gpu_name
        }
    except Exception:
        return {
            "available": False,
            "name": None
        }
