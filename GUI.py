# -*- coding: utf-8 -*-
import wx
import numpy as np
import matplotlib

import pandas as pd
import csv
import tushare as ts
import lstm
import time
from run import plot_results, save_result

# matplotlib采用WXAgg为后台,将matplotlib嵌入wxPython中
matplotlib.use("WXAgg")

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.ticker import MultipleLocator, FuncFormatter

import pylab
from matplotlib import pyplot


######################################################################################
class MPL_Panel_base(wx.Panel):
    ''''' #MPL_Panel_base面板,可以继承或者创建实例'''

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=-1)

        self.Figure = matplotlib.figure.Figure(figsize=(4, 3), facecolor='white')
        self.axes = self.Figure.add_subplot(111)
        #self.axes = self.Figure.add_axes([0.1, 0.1, 0.8, 0.8])
        self.FigureCanvas = FigureCanvas(self, -1, self.Figure)

        # 继承鼠标移动显示鼠标处坐标的事件
        self.FigureCanvas.mpl_connect('motion_notify_event', self.MPLOnMouseMove)

        self.NavigationToolbar = NavigationToolbar(self.FigureCanvas)
        
        #显示坐标点
        self.StaticText = wx.StaticText(self, -1, label='Show Stock Index')

        self.SubBoxSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SubBoxSizer.Add(self.NavigationToolbar, proportion=0, border=2, flag=wx.ALL | wx.EXPAND)
        self.SubBoxSizer.Add(self.StaticText, proportion=4, border=4, flag=wx.ALL | wx.EXPAND)


        self.TopBoxSizer = wx.BoxSizer(wx.VERTICAL)
        self.TopBoxSizer.Add(self.SubBoxSizer, proportion=4, border=2, flag=wx.ALL | wx.EXPAND)
        self.TopBoxSizer.Add(self.FigureCanvas, proportion=-10, border=2, flag=wx.ALL | wx.EXPAND)

        self.SetSizer(self.TopBoxSizer)

        ###方便调用
        self.pylab = pylab
        self.pl = pylab
        #self.pyplot = pyplot
        self.numpy = np
        self.np = np
        self.plt = pyplot

        # 显示坐标值
    def MPLOnMouseMove(self, event):
        ex = event.xdata  # 这个数据类型是numpy.float64
        ey = event.ydata  # 这个数据类型是numpy.float64
        if ex and ey:
            # 可以将numpy.float64类型转化为float类型,否则格式字符串可能会出错
            self.StaticText.SetLabel('%10.5f,%10.5f' % (float(ex), float(ey)))
            #self.StaticText1.SetLabel('%10.5f' % (float(ex)))
            #self.StaticText2.SetLabel('%10.5f' % (float(ey)))
    def UpdatePlot(self):
        '''''#修改图形的任何属性后都必须使用self.UpdatePlot()更新GUI界面 '''
        self.FigureCanvas.draw()

    def plot(self, *args, **kwargs):
        '''''#最常用的绘图命令plot '''
        self.axes.plot(*args, **kwargs)
        self.UpdatePlot()

    def semilogx(self, *args, **kwargs):
        ''''' #对数坐标绘图命令 '''
        self.axes.semilogx(*args, **kwargs)
        self.UpdatePlot()

    def semilogy(self, *args, **kwargs):
        ''''' #对数坐标绘图命令 '''
        self.axes.semilogy(*args, **kwargs)
        self.UpdatePlot()

    def loglog(self, *args, **kwargs):
        ''''' #对数坐标绘图命令 '''
        self.axes.loglog(*args, **kwargs)
        self.UpdatePlot()

    def grid(self, flag=True):
        ''''' ##显示网格  '''
        if flag:
            self.axes.grid()
        else:
            self.axes.grid(False)

    def title_MPL(self, TitleString="基于长短期循环神经网络的股票预测软件V1.0"):
        ''''' # 给图像添加一个标题   '''
        self.axes.set_title(TitleString)

    def xlabel(self, XabelString="X"):
        ''''' # Add xlabel to the plotting    '''
        self.axes.set_xlabel(XabelString)

    def ylabel(self, YabelString="Y"):
        ''''' # Add ylabel to the plotting '''
        self.axes.set_ylabel(YabelString)

    def xticker(self, major_ticker=1.0, minor_ticker=0.1):
        ''''' # 设置X轴的刻度大小 '''
        self.axes.xaxis.set_major_locator(MultipleLocator(major_ticker))
        self.axes.xaxis.set_minor_locator(MultipleLocator(minor_ticker))

    def yticker(self, major_ticker=1.0, minor_ticker=0.1):
        ''''' # 设置Y轴的刻度大小 '''
        self.axes.yaxis.set_major_locator(MultipleLocator(major_ticker))
        self.axes.yaxis.set_minor_locator(MultipleLocator(minor_ticker))

    def legend(self, *args, **kwargs):
        ''''' #图例legend for the plotting  '''
        self.axes.legend(*args, **kwargs)

    def xlim(self, x_min, x_max):
        ''' # 设置x轴的显示范围  '''
        self.axes.set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        ''' # 设置y轴的显示范围   '''
        self.axes.set_ylim(y_min, y_max)

    def savefig(self, *args, **kwargs):
        ''' #保存图形到文件 '''
        self.Figure.savefig(*args, **kwargs)

    def cla(self):
        ''' # 再次画图前,必须调用该命令清空原来的图形  '''
        self.axes.clear()
        self.Figure.set_canvas(self.FigureCanvas)
        self.UpdatePlot()

    def ShowHelpString(self, HelpString="Show Help String"):
        ''''' #可以用它来显示一些帮助信息,如鼠标位置等 '''
        self.StaticText.SetLabel(HelpString)
        ################################################################

class MPL_Panel(MPL_Panel_base):
    ''''' #MPL_Panel重要面板,可以继承或者创建实例 '''

    def __init__(self, parent):
        MPL_Panel_base.__init__(self, parent=parent)

        # 测试一下
        self.FirstPlot()
        # 仅仅用于测试和初始化,意义不大

    def FirstPlot(self):
        # self.rc('lines',lw=5,c='r')
        self.cla()
        x = np.arange(-5, 5, 0.25)
        y = np.sin(x)
        self.yticker(0.5, 0.1)
        self.xticker(1.0, 0.2)
        self.xlabel('X')
        self.ylabel('Y')
        self.title_MPL("图像")
        self.grid()
        self.plot(x, y, '--^g')

        ###############################################################################

# MPL_Frame添加了MPL_Panel的1个实例
class MPL_Frame(wx.Frame):
    """MPL_Frame可以继承,并可修改,或者直接使用"""

    def __init__(self, title="lstm股票预测软件", size=(1000, 800)):
        wx.Frame.__init__(self, parent=None, title=title, size=size)

        self.MPL = MPL_Panel_base(self)
        # 静态文本
        label_name = '股票预测软件'
        font = wx.Font(24, wx.ROMAN, wx.NORMAL, wx.LIGHT)
        self.header = wx.StaticText(parent=self.MPL, label=label_name,
                                    size=(450, 20), style=wx.ALIGN_CENTER, pos=(350, 1))
        self.header.SetFont(font)

 
        # 创建FlexGridSizer
        self.FlexGridSizer = wx.FlexGridSizer(rows=9, cols=10, vgap=5, hgap=5)
        self.FlexGridSizer.SetFlexibleDirection(wx.BOTH)

        #self.RightPanel = wx.Panel(self, -1)
        self.downPanel = wx.Panel(self,-1)

        #加载按键
        self.load_bt = wx.Button(parent=self.downPanel, label="加载股票数据", pos=(10, 10), size=(100, 40))
        self.Bind(wx.EVT_BUTTON, self.load, self.load_bt)
        #训练按键
        self.data_bt = wx.Button(parent=self.downPanel, label="训练预测模型", pos=(10, 10), size=(100, 40))
        self.Bind(wx.EVT_BUTTON, self.train_data, self.data_bt)

        # 预测按键
        self.Button1 = wx.Button(self.downPanel, -1, "预测", size=(100, 40), pos=(10, 10))
        self.Button1.Bind(wx.EVT_BUTTON, self.Button1Event)

        #保存按键
        self.save_bt = wx.Button(parent=self.downPanel, label="保存结果", pos=(10, 10), size=(100, 40))
        self.Bind(wx.EVT_BUTTON, self.save, self.save_bt)

        # 测试按钮2
        self.Button2 = wx.Button(self.downPanel, -1, "关于本软件", size=(100, 40), pos=(10, 10))
        self.Button2.Bind(wx.EVT_BUTTON, self.Button2Event)
        
        #股票代码
        self.input_text = wx.TextCtrl(self.downPanel,-1,"股票代码",size=(100, 40))
        
        self.FlexGridSizer.Add(self.input_text, proportion=0, border=5, flag=wx.ALL | wx.EXPAND)
        #self.FlexGridSizer.Add(self.submit_bt, proportion=0, border=5, flag=wx.ALL | wx.EXPAND)
        self.FlexGridSizer.Add(self.load_bt, proportion=0, border=5, flag=wx.ALL | wx.EXPAND)
        self.FlexGridSizer.Add(self.data_bt, proportion=0, border=5, flag=wx.ALL | wx.EXPAND)
        self.FlexGridSizer.Add(self.Button1, proportion=0, border=5, flag=wx.ALL | wx.EXPAND)
        self.FlexGridSizer.Add(self.save_bt, proportion=0, border=5, flag=wx.ALL | wx.EXPAND)
        self.FlexGridSizer.Add(self.Button2, proportion=0, border=5, flag=wx.ALL | wx.EXPAND)
        

        self.downPanel.SetSizer(self.FlexGridSizer)

        #self.BoxSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.BoxSizer = wx.BoxSizer(wx.VERTICAL)
        self.BoxSizer.Add(self.MPL, proportion=10, border=2, flag=wx.ALL | wx.EXPAND)
        self.BoxSizer.Add(self.downPanel, proportion=0, border=2, flag=wx.ALL | wx.EXPAND)

        self.SetSizer(self.BoxSizer)

        # 状态栏
        self.makeMenuBar()

        # MPL_Frame界面居中显示
        self.Centre(wx.BOTH)      
    
    
    def load(self, event):
        gp_num=self.input_text.GetValue()
        a=ts.get_hist_data(gp_num,ktype = 'W')
        ar=np.array(a)
        br=ar[:,1]
        
        #stock_name=self.input_text.GetValue()
        seq_len = 50
        print('> 加载数据中... ')
        filename=br
        
        try:
            self.X_train, self.y_train, self.X_test, self.y_test = \
                lstm.load_data(filename, seq_len, True)
        except:
            wx.MessageBox("数据加载失败，请检查操作!!!")
        else:
            print('> Data Loaded. Compiling...')
            wx.MessageBox("数据加载完成")

    def train_data(self, event):
        global_start_time = time.time()
        epochs = 1

        try:
            self.model = lstm.build_model([1, 50, 100, 1])
            self.model.fit(
                self.X_train,
                self.y_train,
                batch_size=512,
                nb_epoch=epochs,
                validation_split=0.05)
            self.predicted = lstm.predict_point_by_point(self.model, self.X_test)
        except:
            wx.MessageBox("模型训练失败，请检查操作!!!")
        else:
            wx.MessageBox("模型训练完成, 训练用时： %d s" % (time.time() - global_start_time))

    def save(self, event):
        try:
            save_result(y_test=self.y_test, predicted_values=self.predicted)
        except:
            wx.MessageBox("结果保存失败，请检查操作!!!")
        else:
            wx.MessageBox("结果保存成功！")

    # 定义菜单栏
    def makeMenuBar(self):

        menuBar = wx.MenuBar()
        self.CreateStatusBar(number=3)
        menu = wx.Menu()
        self.about_me(menu)
        self.support(menu)
        self.help_view(menu)
        menuBar.Append(menu, "&帮助")
        self.SetMenuBar(menuBar)

    def help_view(self, menu):
        menu.AppendSeparator()
        versionItem = menu.Append(-1, "Version", "V 1.0")
        menu.AppendSeparator()
        exitItem = menu.Append(wx.ID_EXIT)

        self.Bind(wx.EVT_MENU, self.OnExit, exitItem)
        self.Bind(wx.EVT_MENU, self.version, versionItem)

    def version(self, event):
        wx.MessageBox("Version: 1.0")

    def OnExit(self, event):
        """Close the frame, terminating the application."""
        self.Close(True)

    def about_me(self, menu):
        item = wx.MenuItem(menu, id=122, text="关于我们", kind=wx.ITEM_NORMAL)
        self.Bind(wx.EVT_MENU, self.about1_click, item)
        menu.Append(item)

    def about1_click(self, event):
        wx.MessageBox("版本号 V1.0 \n 2018.5.11'")

    def support(self, menu):
        menu.AppendSeparator()
        item = wx.MenuItem(menu, id=122, text="技术支持", kind=wx.ITEM_NORMAL)
        self.Bind(wx.EVT_MENU, self.support_click, item)
        menu.Append(item)

    def support_click(self, event):
        wx.MessageBox("1、加载数据：首先带你是")


    # 按钮事件,用于测试

    def Button1Event(self, event):
        self.MPL.cla()  # 必须清理图形,才能显示下一幅图
        #x = np.arange(-10, 10, 0.25)
        x,y=self.predicted, self.y_test
        #y = np.cos(x)
        self.MPL.axes.plot(x)
        self.MPL.axes.plot(y)
        self.MPL.xlabel('Time')
        self.MPL.ylabel('Range')
        #self.MPL.ylim(0,500)
        #self.MPL.axes.plot(x, label='True Data')
        #self.MPL.plot(y, label='Prediction')
        self.MPL.legend(['True', 'Prediction'], loc='upper right')
        #self.MPL.xticker([0, 100, 200, 300, 400],
                   #['2017-01', '2017-03', '2017-05', '2017-07', '2017-09'])
        self.MPL.grid()
        self.MPL.ShowHelpString()
        self.MPL.UpdatePlot()  # 必须刷新才能显示


    def Button2Event(self, event):
        self.AboutDialog()


        # 打开文件,用于测试

    def DoOpenFile(self):
            wildcard = r"Data files (*.dat)|*.dat|Text files (*.txt)|*.txt|ALL Files (*.*)|*.*"
            open_dlg = wx.FileDialog(self, message='Choose a file', wildcard=wildcard, style='')
            if open_dlg.ShowModal() == wx.ID_OK:
                path = open_dlg.GetPath()
                try:
                    file = open(path, 'r')
                    text = file.read()
                    file.close()
                except:
                    dlg = wx.MessageDialog(self, 'Error opening file\n')
                    dlg.ShowModal()

            open_dlg.Destroy()
    #加载背景
    def OnEraseBack(self, event):
        dc = event.GetDC()
        if not dc:
            dc = wx.ClientDC(self)
            rect = self.GetUpdateRegion().GetBox()
            dc.SetClippingRect(rect)
        dc.Clear()
        bmp = wx.Bitmap(r"C:\Prectice_project\lstm_end\picture\background.bmp")
        dc.DrawBitmap(bmp, 0, 0)


        # 自动创建状态栏

    def StatusBar(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetFieldsCount(5)
        self.statusbar.SetStatusWidths([-1, -1, -1,-1,-1])


        # About对话框

    def AboutDialog(self):
        dlg = wx.MessageDialog(self,
                               '\t基于长短期记忆循环神经网络的股票预测软件'
                               '\t\n本软件使用深度学习的方法构建预测模型，实现对个股开盘价格走势的准确预测 '
                               '\n版本号 V1.0 \n 2018.5.11',
                               '关于此软件', wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

        ###############################################################################

# 主程序测试
if __name__ == '__main__':
    app = wx.App()
    # frame = MPL2_Frame()
    frame = MPL_Frame()
    frame.Center()
    frame.Show()
    app.MainLoop()