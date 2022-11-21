# Image Registration System

This is an image registration system supporting linear and nonlinear transformations.

For linear transformation, an SGD is implemented from scratch to do the optimization. The loss function can be chosen from mean squared error, correlation coefficient, cross correlation, mutual information and K-L divergence. 

For non-linear transformation, the optical-flow method is used.



## Environment and Running

Environment:

```bash
pip install -r requirements.txt
```

Run the system:

```
python main.py
```



## Code Structure

- `ImageRegistrant.py`：The class for conducting image registration. Implemented SGD and the pipeline for registration.
- `algorithm_utils.py`：Utilities like bi-linear interpolation, normalization and various loss functions. The registration algorithm can be extended by adding code here.
- `window.py`：Front-end layout generated by QT Designer.
- `button_func.py`：The functionality for the buttons in the front-end.
- `main.py`：The main-loop for the window.
