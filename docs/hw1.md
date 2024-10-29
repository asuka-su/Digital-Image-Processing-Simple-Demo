# HW1 Report

本次作业主要实现了：
- 图像RGB色域与HSL色域的相互转化
- 基于HSL色域的对色相、饱和度、亮度的独立编辑
- RGB三通道的可视化和可见性编辑
- 对界面UI的改进

以下分别进行简单总结。

## RGB2HSL

首先对输入RGB图像归一化，将像素值统一至 $[0,1]$ 区间上。然后套用公式
```math
L=\frac{R+G+B}{3}
```
```math
S=1-\frac{3\min(R,G,B)}{R+G+B}
```
```math
\theta=\arccos(\frac{2R-G-B}{2\sqrt{R^2+G^2+B^2-RG-RB-BG}})\quad H=\left\{ \begin{aligned} \theta\quad\quad&B\le G\\2\pi-\theta\quad&B>G\end{aligned}\right.
```
计算。  

需要注意一些特殊情况：
- 计算 $S$ 时，若 $R=G=B=0$ （白色），定义其饱和度为0。
- 计算 $H$ 时，若 $R=G=B$ （灰色），定义其色相为0。
- 特别地，计算 $H$ 时，当 $G=B$ 时， $\arccos$ 方程可能失败，猜测是括号内的值因浮点数精度限制略超 $\pm1$ ，输出值变为 $nan$ 。为此手动化简公式。  

相关代码在[utils.py](../utils.py)中实现。

## HSL2RGB

与RGB2HSL相比，反向的转化只需要套用表格公式即可，没有什么特殊情况需要处理，将最终的输出值取整到np.uint8类型即可。  

相关代码在[utils.py](../utils.py)中实现。

## Independent Editing Based on HSL Space

有了上面两个函数后，即可基于HSL色域空间对输入图像的色相、饱和度、亮度进行调整。允许用户输入的offset均为 $[-1,1]$ 之间的值。  

对于色相，将 $offset\times 2\pi$ 作为偏差值加在原图色相域上，对超出范围的取周期模使其落入 $[0,2\pi]$ 范围。  
对于饱和度和亮度，将原图的饱和度和亮度等比例扩张（缩小）为 $1+offset$ 倍，使用np.clip函数框定输出在 $[0,1]$ 范围。

相关代码在[functions.py](../functions.py)中实现。

## Channel Visibility [extension]

在输出编辑后的图像的同时，在下方分别以灰度图形式展示RGB三通道取值。用户可以点击通道下方 $\fbox{visible}$ 选项框控制该通道是否出现在输出图像中。控制该通道图像不可见不会影响该通道灰度图显示。

相关代码在[functions.py](../functions.py)中实现。

## UI Improvements

- 去除了“运行”按钮，改为根据用户输入的变更实时计算输出图像。主要用到了gradio提供的change/input/release函数。
- 在用户上传图像前，色相/饱和度/亮度控制条和通道可见性控制框不可见。当输入图像被清除时，控制条和控制框同时复位。
- 如何让右侧的RGB通道及对应的 $\fbox{visible}$ 选项框显示在同一行列？原先它们会成一竖列六个框排开，发现问题在gradio的Column的默认最小宽度设置上。Column默认最小宽度为320px，当无法达到时，它将自动换行。将其改为100px即可。

相关代码在[UI.py](../UI.py)中实现。
