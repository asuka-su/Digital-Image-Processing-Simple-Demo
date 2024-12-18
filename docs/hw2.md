# HW2 Report

本次作业主要实现了：
- 图像使用Nearest、Bilinear、Bicubic放缩x0.5/x1/x2/x4倍
- 调用cv2库的Nearest、Bilinear、Bicubic、Lanczos4放缩x0.5/x1/x2/x4倍
- 图像以某点（可选参数）为中心，逆时针旋转某角度（可选参数）
- 图像选定边及放缩比例进行斜切

以下分别进行简单总结

## Nearest/Bilinear/Bicubic

Nearest插值最简单，生成新的横纵坐标列再除以放缩因子后使用np.around取整即可。  
这里还用到了numpy的“高级索引”功能，即`input_image[w_ind[:,np.newaxis], h_ind]`所展示的，可以直接得到由`w_ind[:,np.newaxis]`和`h_ind`（会自动被broadcast为二维形式）组成的索引得到的图像。

Bilinear插值稍微复杂一点，同样生成横纵坐标、除以放缩因子后，需要进行两次线性插值。  
写的时候出了一个小bug，就是以offset生成周围四点的加权时，为了生成合适的形状，`w_offset`需要以`[:,np.newaxis]`广播，而`h_offset`需要以`[np.newaxis,:]`广播。

Bicubic插值最复杂，需要引入更复杂的加权函数，这里直接硬编码了16个点的加权函数。  
这里用到了np.vectorize函数，可以将作用在单个数上的函数变为作用在矩阵上的函数，非常方便。

除此之外，在该界面添加了 $\fbox{Use opencv-python}$ 的选项，可以使用opencv库对图像进行放缩。

相关代码在[utils.py](../utils.py)中实现。

## Rotation [extension]

允许用户给定旋转中心和旋转角度。  

当旋转中心位于图片之外时，会给出警告，但仍会给出运行结果。  

允许用户指定补充色（当图片旋转出屏幕范围时的填充颜色），默认设置为黑色。为了解决旋转后边界像素超出范围后出现单行/单列像素的补充色偏差（主要是在90/180/270/360度时），略微放松了bound的限制。  

经过逆变换得到的新像素位置在原图片的像素位置后，使用双线性插值得到其值。

相关代码在[utils.py](../utils.py)中实现。

## Shearing [extension]

允许用户选择边及比例进行斜切操作。

这里斜切比例范围限制在 $[-1,1]$ ，比例太大的话看上去也没有什么意义了。  

同样允许用户指定补充色，使用双线性插值得到新像素值。

相关代码在[utils.py](../utils.py)中实现。

## UI Design

仅简要介绍一些（比较）特别的UI：
- 当用户的输入图像被清空时，对应的输出图像也会被清空。
- 在ROTATION & SHEARING界面，选择Rotation后，会出现旋转中心、旋转角度的设定；选择Shearing后会出现边、比例的设定。在切换之间保留记忆。
- 在ROTATION & SHEARING界面，右下角可选择默认补充色设置。这里提供了两个快速颜色设置黑和白，此时RGB三通道不可调；也可以选择 $\fbox{Other}$ 键主动设置三通道取值。