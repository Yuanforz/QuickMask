import sys
from PyQt5.QtWidgets import QApplication
from app_state import AppState
from main_window import MainWindow

if __name__ == '__main__':
    # 1. 创建应用实例
    app = QApplication(sys.argv)

    # 2. 创建核心状态管理器
    #    程序将自动在当前目录下寻找 'images' 和 'masks' 文件夹
    app_state = AppState(image_dir="images", mask_dir="masks")

    # 3. 创建主窗口
    main_window = MainWindow(app_state)
    main_window.show()

    # 4. 加载文件并启动应用
    #    在主窗口显示后再加载，可以避免启动时界面卡顿
    app_state.load_files()

    # 5. 运行事件循环
    sys.exit(app.exec_())