**请注意！！你需要确保你创建Python代码的位置路径不包含中文！否则会出现一些意料之外的问题**

# PyWin32实现基础的游戏脚本 --小Huugh

首先，确保你安装好python(建议版本3.8开头的)，然后将这段指令直接粘贴进cmd里，**要是出报错不要管，这可能是我的问题，不过不影响后续的操作！**

```shell
pip install opencv-python==4.4.0.40  -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install pywin32 -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install win32gui -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install win32con -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install win32ui -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install win32api -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install pypiwin32 -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install pillow==9.2.0 -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install numpy==1.19.3 -i https://pypi.mirrors.ustc.edu.cn/simple/
```

然后就可以快乐地新建你的py文件，然后粘贴下面的代码啦！

```python
import win32gui, win32con, win32api, win32ui
import cv2
from PIL import Image
from numpy import average, dot, linalg
import numpy as np
import time
```

## 获取窗口句柄(hwnd)

通过指定句柄的方式定位游戏窗口，你有两种方法可以定位游戏窗口

### 方案一

通过按键抓抓工具之类的程序获取句柄

### 方案二

通过win32gui进行搜索，如下给出示例代码（用于搜索微端窗口）

```python
'''
	这段代码执行后获取到的hwnd就是窗口句柄了
'''
try:
    hwnd = win32gui.FindWindow("ApolloRuntimeContentWindow",None)
    process_id = win32process.GetWindowThreadProcessId(hwnd)
    hwnd_child=[]
    win32gui.EnumChildWindows(hwnd,lambda hWnd,param:param.append(hWnd),hwnd_child)
    hwnd=(hwnd_child[0])
    res = -1
except:
    print("未检测到美食，请使用微端打开美食")
    while True:
        pass
```

## 发送鼠标操作

### 鼠标左击

```python
'''
  传入需要点击的窗口内的坐标和窗口句柄hwnd即可，调用该函数用以鼠标左键单击
'''
def leftclick(cx, cy, hwnd):
    try:
        win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
        lp = win32api.MAKELONG(cx, cy)
        win32api.SendMessage(hwnd, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON, lp)
        win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lp)
        time.sleep(0.005)
        win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, lp)
    finally:
        lock.release()
```

### 鼠标右击

```python
'''
  传入需要点击的窗口内的坐标和窗口句柄hwnd即可，调用该函数用以鼠标右键单击
'''
def leftclick(cx, cy, hwnd):
    try:
        win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
        lp = win32api.MAKELONG(cx, cy)
        win32api.SendMessage(hwnd, win32con.WM_MOUSEMOVE, win32con.MK_RBUTTON, lp)
        win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_RBUTTON, lp)
        time.sleep(0.005)
        win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, win32con.MK_RBUTTON, lp)
    finally:
        lock.release()
```

## 发送键盘操作

在你的程序中添加如下函数，然后调用即可

### 发送一串字符串

```python
def send_str(text):
    astrToint = [ord(c) for c in text]
    for item in astrToint:
        win32api.PostMessage(hwnd, win32con.WM_CHAR, item, 0)
```

###发送一个按键

```python
def send_key(id,hwnd):
    win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, id, 0)
    win32api.SendMessage(hwnd, win32con.WM_KEYUP, id, 0)
```

例子:send_str("你好",hwnd)就是将你好作为字符串发送给窗口，这个过程类似于用户的键盘输入

## 涉及到游戏中的图像处理

### 截图

在你的代码中添加这段代码，然后调用capture(你获取到的句柄)即可获取到游戏的图像（返回值），这个图像是以opencv格式表达的，因此可以对它进行一些变换操作。

```python
def capture(hWnd):
    win32api.PostMessage(hwnd, win32con.WM_SETFOCUS, 0, 0)
    # 获取后台窗口的句柄，注意后台窗口不能最小化
    # 获取句柄窗口的大小信息
    left, top, right, bot = win32gui.GetWindowRect(hWnd)
    width = right - left
    height = bot - top
    hWndDC = win32gui.GetWindowDC(hWnd)
    mfcDC = win32ui.CreateDCFromHandle(hWndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)
    saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
    signedIntsArray = saveBitMap.GetBitmapBits(True)
    im_opencv = np.frombuffer(signedIntsArray, dtype='uint8')
    im_opencv.shape = (height, width, 4)
    cv2.cvtColor(im_opencv, cv2.COLOR_BGRA2RGB)
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hWnd, hWndDC)
    return im_opencv
```

如果你要进行接下来的相似度计算与找图点击，请务必先加入这段代码，然后继续往下看

```python
def selectRectangle(hwnd):
    img=capture(hwnd)
    roi = cv2.selectROI(img);
    cv2.destroyAllWindows()
    return roi
```

###模板图像获取

首先在你的代码中复制粘贴下面这段代码

```python
# 对图片进行统一化处理
def get_thum(image, size=(128, 256), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image
# 计算图片之间的余弦距离
def cosSim(image1, image2):
    image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_turple in image.getdata():
            vector.append(average(pixel_turple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res
# 计算相似度
def calculate(img, img1):
    # 计算图img的直方图
    H1 = cv2.calcHist([img], [1], None, [256], [0, 256])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
    # 计算图img2的直方图
    H2 = cv2.calcHist([img1], [1], None, [256], [0, 256])
    H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
    # 利用compareHist（）进行比较相似度
    similarity = cv2.compareHist(H1, H2, 0)
    return (similarity)
def make_regalur_image(img, size=(256, 256)):
    """我们有必要把所有的图片都统一到特别的规格，在这里我选择是的256x256的分辨率。"""
    return img.resize(size).convert('RGB')
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
def calc_similar(li, ri):
    li = make_regalur_image(Image.fromarray(cv2.cvtColor(li, cv2.COLOR_BGR2RGB)))
    ri = make_regalur_image(Image.fromarray(cv2.cvtColor(ri, cv2.COLOR_BGR2RGB)))
    return sum(hist_similar(l.histogram(), r.histogram()) for l, r in zip(split_image(li), split_image(ri))) / 16.0
def split_image(img, part_size=(64, 64)):
    w, h = img.size
    pw, ph = part_size
    assert w % pw == h % ph == 0
    return [img.crop((i, j, i + pw, j + ph)).copy() for i in range(0, w, pw) \
            for j in range(0, h, ph)]
def compareImg(temp, hwnd):
    img=capture(hwnd)
     #= cv2.imread("img_Winapi.bmp");
    # 系数匹配法，越接近1表示匹配度越高
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = temp.shape[:2]  # 获取需要检测的模板大小
    res5 = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF)
    # cv2.normalize(res5, res5, 0, 1, cv2.NORM_MINMAX)  # 归一化TM_CCOEFF算法处理结果，便于显示
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res5)  # 获取归一化相关性匹配中的匹配度最高的位置
    r = img.copy()
    cv2.rectangle(r, maxLoc, (maxLoc[0] + w, maxLoc[1] + h), 255, 5)  # 匹配度最高的位置画白色矩形框
    return r, (calc_similar(temp, img[maxLoc[1]:maxLoc[1] + h, maxLoc[0]:maxLoc[0] + w])), maxLoc
class SetInfo:
    def __init__(self, slogan, saveName, hwnd):
        self.slogan = slogan;
        self.saveName = saveName;
        self.hwnd = hwnd;

    def setting(self):
        print(self.slogan);
        img=capture(self.hwnd)
        #img = cv2.imread("img_Winapi.bmp");
        roi = cv2.selectROI(img);
        print(roi);
        select = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        cv2.imwrite(path + self.saveName, select);
        a, ra, _ = compareImg(select, self.hwnd)
        cv2.imshow('yb', a)
        print("相似度", ra)
        click(int((roi[0] + roi[0] + roi[2]) / 2), int((roi[1] + roi[1] + roi[3]) / 2), self.hwnd)
        cv2.waitKey(0);
        cv2.destroyAllWindows()
```

下面将给你一个例子，可以看到"圈出铲子"是给用户的提示语，"chanzi.png"是模板图像的保存位置,hwnd是窗口句柄，通过调用这个函数可以获取模板图片并进行存储**请注意！！图像名称不能有中文或空格**

```python
SetInfo("圈出铲子", "chanzi.png", hwnd).setting();
```

### 找图点击

前面你已经完成了模板图片的保存，接下来你需要读取这张模板图片

```python
铲子 = cv2.imread('chanzi.png');
```

然后加入这段图像点击的代码，你可以按照你的想法对这段代码进行调整

```python
def clickImg(temp, hwnd):
    try:
        img=capture(hwnd)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = temp.shape[:2]  # 获取需要检测的模板大小
        res5 = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res5)  # 获取归一化相关性匹配中的匹配度最高的位置
        cv2.rectangle(img, maxLoc, (maxLoc[0] + w, maxLoc[1] + h), 255, 5)  # 匹配度最高的位置画白色矩形框
        siml = calc_similar(temp, img[maxLoc[1]:maxLoc[1] + h, maxLoc[0]:maxLoc[0] + w])
        print("匹配度", (int(siml * 10000)) / 100.0, "%")
        if (siml > 0.60):
            click(maxLoc[0] + int(w / 2), maxLoc[1] + int(h / 2), hwnd)
        else:
            print("匹配度过低，可能未出现该匹配项，一秒后将重新匹配")
            time.sleep(1)
            clickImg(temp, hwnd)
    except:
        clickImg(temp, hwnd)
```

 接下来是调用了

```python
clickImg(铲子,hwnd)
```

---

**纠错联系邮箱:2396392765@qq.com**"# game-auto" 
