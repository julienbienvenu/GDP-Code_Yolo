import psutil

# Get current CPU frequency in Hz
def get_cpu_frequency():
    cpu_freq = psutil.cpu_freq()
    if cpu_freq is not None:
        return cpu_freq.current
    else:
        return None

# Get number of instructions per cycle (IPC)
def get_cpu_ipc():
    cpu_info = psutil.cpu_info()
    if 'model name' in cpu_info[0]:
        model_name = cpu_info[0]['model name']
        if ' @ ' in model_name:
            ipc_str = model_name.split(' @ ')[-1]
            if 'IPC=' in ipc_str:
                ipc = ipc_str.split('IPC=')[-1]
                try:
                    ipc = float(ipc)
                    return ipc
                except ValueError:
                    pass
    return None

# Calculate feasible number of operations per second
def calculate_operations_per_second():
    cpu_freq = get_cpu_frequency()
    ipc = get_cpu_ipc()
    if cpu_freq is not None and ipc is not None:
        return cpu_freq * ipc
    else:
        return None

# Example usage
operations_per_second = calculate_operations_per_second()
if operations_per_second is not None:
    print(f"Feasible number of operations per second: {operations_per_second / 1e9} GOp/s") # Convert Hz to GHz
else:
    print("Failed to retrieve CPU frequency or IPC.")
