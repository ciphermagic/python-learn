import numpy as np
from PIL import ImageGrab
import cv2
import time

wait_time = 5
window_size = (320, 104, 704, 448)  # 384,344  192,172 96,86

if __name__ == '__main__':
    for i in list(range(wait_time))[::-1]:
        print(i + 1)
        time.sleep(1)

    last_time = time.time()
    while True:

        print_screen = np.array(ImageGrab.grab(bbox=window_size))

        screen_gray = cv2.cvtColor(print_screen, cv2.COLOR_BGR2GRAY)  # 灰度图像收集
        screen_reshape = cv2.resize(screen_gray, (96, 86))

        cv2.imshow('window3', print_screen)

        # 测试时间用
        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()
