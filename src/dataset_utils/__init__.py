import os
import importlib.util
import inspect
from .interface import DatasetBase

# a mapping from dataset name to Dataset class
dataset_mapping = {}

for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename not in ["interface.py", "__init__.py"]:
        module_name = filename[:-3]
        filepath = os.path.join(os.path.dirname(__file__), filename)

        # 根据文件路径 (filepath) 生成一个模块规范对象 (spec)，告诉 Python 如何加载此模块。
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        # 根据 spec 创建一个空的模块对象
        module = importlib.util.module_from_spec(spec)
        # 执行模块文件中的代码（相当于 import 语句），将模块中定义的类、函数、变量等加载到 module 对象中
        spec.loader.exec_module(module)

        # 获取模块中的 Dataset 类
        DatasetClass = getattr(module, "Dataset", None)
        if DatasetClass is None:
            continue

        if inspect.isclass(DatasetClass) and issubclass(DatasetClass, DatasetBase):
            for dataset in DatasetClass.support_datasets:
                dataset_mapping[dataset] = DatasetClass
        else:
            raise ValueError(
                f"Module {__name__}.{module_name} does not contain a class that named Dataset and inherits from DatasetBase."
            )

# 控制 from module import * 的行为
__all__ = ["dataset_mapping", "DatasetBase"]
