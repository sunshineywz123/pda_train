import subprocess
import re

import logging
import time
import psutil
import numpy as np

# 获取内存使用情况

# 将结果以字典形式返回，方便查看和使用


# 配置logging模块
logging.basicConfig(filename='/mnt/data/home/linhaotong/temperature_info.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

sleep_duration = 5

while True:

    # 执行sensors命令并获取输出
    result = subprocess.run(['sensors'], capture_output=True, text=True)
    output = result.stdout
    memory_usage = psutil.virtual_memory()

    # 初始化字典以存储温度信息
    temperature_info = {
        'ISA adapter': 0,
        'CPU Core Max': 0,
        'acpitz-acpi-0': 0,
        'nvme-pci-0200': 0,
        'gpu1': 0,
        'gpu2': 0,
        'total_memory': 0,
        'used_memory': 0,
        'free_memory': 0,
        'memory_usage_percent': 0
    }
    
    # 正则表达式匹配不同组件的温度
    patterns = {
        'ISA adapter': r'Package id 0:\s+\+(\d+\.\d+)°C',
        'CPU Core': r'Core \d+:\s+\+(\d+\.\d+)°C',
        'acpitz-acpi-0': r'temp1:\s+\+(\d+\.\d+)°C  \(crit',
        'nvme-pci-0200': r'Composite:\s+\+(\d+\.\d+)°C',
        'gpu1': r'Sensor 1:\s+\+(\d+\.\d+)°C',
        'gpu2': r'Sensor 2:\s+\+(\d+\.\d+)°C',
    }

    # 解析并提取温度信息
    for key, pattern in patterns.items():
        matches = re.findall(pattern, output)
        if matches:
            if key == 'CPU Core':
                temperature_info['CPU Core Max'] = max(map(float, matches))
            else:
                temperature_info[key] = float(matches[0])

    # 输出结果
    temperature_info['total_memory'] = np.round(memory_usage.total / 1024 / 1024 / 1024, 1)
    temperature_info['used_memory'] = np.round(memory_usage.used / 1024 / 1024 / 1024, 1)
    temperature_info['free_memory'] = np.round(memory_usage.free / 1024 / 1024 / 1024, 1)
    temperature_info['memory_usage_percent'] = np.round(memory_usage.percent, 1)
    print(temperature_info)
    logging.info(f'Temperature Information: {temperature_info}')
    # 等待一段时间
    time.sleep(sleep_duration)

# print(output)
# print(temperature_info)
# import ipdb;ipdb.set_trace()%