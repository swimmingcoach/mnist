from win32 import win32gui
from win32 import win32con
from win32 import win32api

# 从顶层窗口向下搜索主窗口，无法搜索子窗口
# FindWindow(lpClassName=None, lpWindowName=None)  窗口类名 窗口标题名
handle = win32gui.FindWindow("Notepad", None)


# 获取窗口位置
left, top, right, bottom = win32gui.GetWindowRect(handle)
#获取某个句柄的类名和标题
title = win32gui.GetWindowText(handle)
clsname = win32gui.GetClassName(handle)

# 打印句柄
# 十进制
print(handle)
# 十六进制
print("%x" %(handle) )


# 搜索子窗口
# 枚举子窗口
hwndChildList = []
win32gui.EnumChildWindows(handle, lambda hwnd, param: param.append(hwnd),  hwndChildList)

# FindWindowEx(hwndParent=0, hwndChildAfter=0, lpszClass=None, lpszWindow=None) 父窗口句柄 若不为0，则按照z-index的顺序从hwndChildAfter向后开始搜索子窗体，否则从第一个子窗体开始搜索。 子窗口类名 子窗口标题
subHandle = win32gui.FindWindowEx(handle, 0, "EDIT", None)

# 获得窗口的菜单句柄
menuHandle = win32gui.GetMenu(subHandle)
# 获得子菜单或下拉菜单句柄
# 参数：菜单句柄 子菜单索引号
subMenuHandle = win32gui.GetSubMenu(menuHandle, 0)
# 获得菜单项中的的标志符，注意，分隔符是被编入索引的
# 参数：子菜单句柄 项目索引号
menuItemHandle = win32gui.GetMenuItemID(subMenuHandle, 0)
# 发送消息，加入消息队列，无返回
# 参数：句柄 消息类型 WParam IParam
win32gui.postMessage(subHandle, win32con.WM_COMMAND, menuItemHandle, 0)


# wParam的定义是32位整型，high word就是他的31至16位，low word是它的15至0位。
# 当参数超过两个，wParam和lParam不够用时，可以将wParam就给拆成两个int16来使用。
# 这种时候在python里记得用把HIWORD的常数向左移16位，再加LOWORD，即wParam = HIWORD<<16+LOWORD。

# 下选框内容更改
# 参数：下选框句柄； 消息内容； 参数下选框的哪一个item，以0起始的待选选项的索引；如果该值为-1，将从组合框列表中删除当前选项，并使当前选项为空； 参数
# CB_Handle为下选框句柄，PCB_handle下选框父窗口句柄
if win32api.SendMessage(CB_handle, win32con.CB_SETCURSEL, 1, 0) == 1:
# 下选框的父窗口命令
# 参数：父窗口句柄； 命令； 参数：WParam：高位表示类型，低位表示内容；参数IParam，下选框句柄
# CBN_SELENDOK当用户选择了有效的列表项时发送，提示父窗体处理用户的选择。 LOWORD为组合框的ID. HIWORD为CBN_SELENDOK的值。
            win32api.SendMessage(PCB_handle, win32con.WM_COMMAND, 0x90000, CB_handle)
# CBN_SELCHANGE当用户更改了列表项的选择时发送，不论用户是通过鼠标选择或是通过方向键选择都会发送此通知。LOWORD为组合框的ID. HIWORD为CBN_SELCHANGE的值。
            win32api.SendMessage(PCB_handle, win32con.WM_COMMAND, 0x10000, CB_handle)


# 设置文本框内容，等窗口处理完毕后返回true。中文需编码成gbk
# 参数：句柄；消息类型；参数WParam，无需使用； 参数IParam，要设置的内容，字符串
win32api.SendMessage(handle, win32con.WM_SETTEXT, 0, os.path.abspath(fgFilePath).encode('gbk'))


# 控件点击确定,处理消息后返回0
# 参数:窗口句柄; 消息类型; 参数WParam HIWORD为0（未使用），LOWORD为控件的ID; 参数IParam  0（未使用）,确定控件的句柄
win32api.SendMessage(Mhandle, win32con.WM_COMMAND, 1, confirmBTN_handle)


# 获取窗口文本不含截尾空字符的长度
# 参数：窗口句柄； 消息类型； 参数WParam； 参数IParam
bufSize = win32api.SendMessage(subHandle, win32con.WM_GETTEXTLENGTH, 0, 0) +1
# 利用api生成Buffer
strBuf = win32gui.PyMakeBuffer(bufSize)
print(strBuf)
# 发送消息获取文本内容
# 参数：窗口句柄； 消息类型；文本大小； 存储位置
length = win32gui.SendMessage(subHandle, win32con.WM_GETTEXT, bufSize, strBuf)
# 反向内容，转为字符串
# text = str(strBuf[:-1])

address, length = win32gui.PyGetBufferAddressAndLen(strBuf)
text = win32gui.PyGetString(address, length)
# print('text: ', text)

# 鼠标单击事件
#鼠标定位到(30,50)
win32api.SetCursorPos([30,150])
#执行左单键击，若需要双击则延时几毫秒再点击一次即可
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP | win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
#右键单击
win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP | win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)

def click1(x,y):                #第一种
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

def click2(x,y):               #第二种
    ctypes.windll.user32.SetCursorPos(x,y)
    ctypes.windll.user32.mouse_event(2,0,0,0,0)
    ctypes.windll.user32.mouse_event(4,0,0,0,0)

def click_it(pos):          #第三种
    handle= win32gui.WindowFromPoint(pos)
    client_pos =win32gui.ScreenToClient(handle,pos)
    tmp=win32api.MAKELONG(client_pos[0],client_pos[1])
    win32gui.SendMessage(handle, win32con.WM_ACTIVATE,win32con.WA_ACTIVE,0)
    win32gui.SendMessage(handle, win32con.WM_LBUTTONDOWN,win32con.MK_LBUTTON,tmp)
    win32gui.SendMessage(handle, win32con.WM_LBUTTONUP,win32con.MK_LBUTTON,tmp)

# 发送回车
win32api.keybd_event(13,0,0,0)
win32api.keybd_event(13,0,win32con.KEYEVENTF_KEYUP,0)


# 关闭窗口
win32gui.PostMessage(win32lib.findWindow(classname, titlename), win32con.WM_CLOSE, 0, 0)


# 检查窗口是否最小化，如果是最大化
if(win32gui.IsIconic(hwnd)):
#     win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    win32gui.ShowWindow(hwnd, 8)
    sleep(0.5)

# SW_HIDE：隐藏窗口并激活其他窗口。nCmdShow=0。
# SW_MAXIMIZE：最大化指定的窗口。nCmdShow=3。
# SW_MINIMIZE：最小化指定的窗口并且激活在Z序中的下一个顶层窗口。nCmdShow=6。
# SW_RESTORE：激活并显示窗口。如果窗口最小化或最大化，则系统将窗口恢复到原来的尺寸和位置。在恢复最小化窗口时，应用程序应该指定这个标志。nCmdShow=9。
# SW_SHOW：在窗口原来的位置以原来的尺寸激活和显示窗口。nCmdShow=5。
# SW_SHOWDEFAULT：依据在STARTUPINFO结构中指定的SW_FLAG标志设定显示状态，STARTUPINFO 结构是由启动应用程序的程序传递给CreateProcess函数的。nCmdShow=10。
# SW_SHOWMAXIMIZED：激活窗口并将其最大化。nCmdShow=3。
# SW_SHOWMINIMIZED：激活窗口并将其最小化。nCmdShow=2。
# SW_SHOWMINNOACTIVE：窗口最小化，激活窗口仍然维持激活状态。nCmdShow=7。
# SW_SHOWNA：以窗口原来的状态显示窗口。激活窗口仍然维持激活状态。nCmdShow=8。
# SW_SHOWNOACTIVATE：以窗口最近一次的大小和状态显示窗口。激活窗口仍然维持激活状态。nCmdShow=4。
# SW_SHOWNORMAL：激活并显示一个窗口。如果窗口被最小化或最大化，系统将其恢复到原来的尺寸和大小。应用程序在第一次显示窗口的时候应该指定此标志。nCmdShow=1。