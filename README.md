### 运行指南

安装依赖环境

```bash
pip install -r requirements.txt
```

运行程序

```
python main.py
```



### 代码结构

- `ImageRegistrant.py`：用于图像配准的类，封装了图像配准计算流程的各个函数与成员变量
- `algorithm_utils.py`：以上图像配准类中所需的但又不必封装的工具函数，如双线性插值、图像归一化、各种损失函数等，可以在此处扩展损失函数而无需修改图像配准类的代码
- `window.py`：由QT Designer生成的前端窗口布局
- `button_func.py`：自定义前端各按钮实现的功能
- `main.py`：窗口主函数，运行的入口
