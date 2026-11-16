"""Collect information about the machine running benchmarks."""

import platform
import os
import psutil


class HardwareInspector:
    """Inspect CPU, memory and available GPU information."""

    def get_cpu_info(self):
        return {
            "processor": platform.processor(),
            "architecture": platform.machine(),
            "logical_cores": os.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
        }

    def get_memory_info(self):
        memory = psutil.virtual_memory()

        return {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "used_percent": memory.percent,
        }

    def get_gpu_info(self):
        try:
            import torch

            if not torch.cuda.is_available():
                return []

            devices = []

            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)

                devices.append({
                    "index": index,
                    "name": properties.name,
                    "memory_gb": round(
                        properties.total_memory / (1024 ** 3), 2
                    ),
                })

            return devices

        except ImportError:
            return []

    def get_system_report(self):
        return {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "cpu": self.get_cpu_info(),
            "memory": self.get_memory_info(),
            "gpu": self.get_gpu_info(),
        }