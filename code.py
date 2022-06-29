import cv2
import mediapipe as mp
import math
import socket


client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PORT = 1111
server_address = ("127.0.0.1", PORT)


def vector_angle(x1, y1, x2, y2):  # 求解二维向量的角度
    try:
        angle_ = math.degrees(math.acos((x1 * x2 + y1 * y2) /  # 通过计算余弦求解角度
                                        (((x1 ** 2 + y1 ** 2) ** 0.5) * ((x2 ** 2 + y2 ** 2) ** 0.5))))
    except (ValueError, ArithmeticError):
        angle_ = 181  # 如果识别出错，将角度设置为一个大于180度的角度
        print("EXCEPTION!")

    return angle_


def hand_angle(hand):  # 获取对应手相关向量的二维角度,根据角度确定手势
    angle_list = []
    # thumb 大拇指角度
    thumb_angle = vector_angle(
        int(hand[0][0]) - int(hand[2][0]), int(hand[0][1]) - int(hand[2][1]),
        int(hand[3][0]) - int(hand[4][0]), int(hand[3][1]) - int(hand[4][1]))
    angle_list.append(thumb_angle)
    # index 食指角度
    index_angle = vector_angle(
        int(hand[0][0]) - int(hand[6][0]), int(hand[0][1]) - int(hand[6][1]),
        int(hand[7][0]) - int(hand[8][0]), int(hand[7][1]) - int(hand[8][1]))
    angle_list.append(index_angle)
    # middle 中指角度
    middle_angle = vector_angle(
        int(hand[0][0]) - int(hand[10][0]), int(hand[0][1]) - int(hand[10][1]),
        int(hand[11][0]) - int(hand[12][0]), int(hand[11][1]) - int(hand[12][1]))
    angle_list.append(middle_angle)
    # ring 无名指角度
    ring_angle = vector_angle(
        int(hand[0][0]) - int(hand[14][0]), int(hand[0][1]) - int(hand[14][1]),
        int(hand[15][0]) - int(hand[16][0]), int(hand[15][1]) - int(hand[16][1]))
    angle_list.append(ring_angle)
    # pink 小拇指角度
    pink_angle = vector_angle(
        int(hand[0][0]) - int(hand[18][0]), int(hand[0][1]) - int(hand[18][1]),
        int(hand[19][0]) - int(hand[20][0]), int(hand[19][1]) - int(hand[20][1]))
    angle_list.append(pink_angle)

    return angle_list


def hand_gesture(angle_list):  # 二维约束的方法定义手势
    thr_angle = 65  # 普通手指弯曲大于65度视为握紧
    thr_angle_thumb = 53  # 大拇指弯曲大于53度视为握紧
    thr_angle_s = 49  # 所有手指弯曲小于49度视为伸展
    ges = None
    if 181 not in angle_list:  # 当检测无异常时，判断手势类型
        if angle_list[0] > thr_angle_thumb and \
                angle_list[1] > thr_angle and \
                angle_list[2] > thr_angle and \
                angle_list[3] > thr_angle and \
                angle_list[4] > thr_angle:
            ges = "Rock"  # 五指握紧视为石头
        elif angle_list[0] < thr_angle_s and \
                angle_list[1] < thr_angle_s and \
                angle_list[2] < thr_angle_s and \
                angle_list[3] < thr_angle_s and \
                angle_list[4] < thr_angle_s:
            ges = "Paper"  # 五指伸展视为布
        elif angle_list[0] > thr_angle_thumb and \
                angle_list[1] < thr_angle_s and \
                angle_list[2] < thr_angle_s and \
                angle_list[3] > thr_angle and \
                angle_list[4] > thr_angle:
            ges = "Scissor"  # 只伸展食指和中指视为剪刀

    return ges


def hand_num_gesture(angle_list):  # 二维约束的方法定义手势
    thr_angle = 65  # 普通手指弯曲大于65度视为握紧
    thr_angle_thumb = 53  # 大拇指弯曲大于53度视为握紧
    thr_angle_s = 49  # 所有手指弯曲小于49度视为伸展
    ges = None
    if 181 not in angle_list:  # 当检测无异常时，判断手势类型
        if angle_list[0] > thr_angle_thumb and \
                angle_list[1] < thr_angle_s and \
                angle_list[2] > thr_angle and \
                angle_list[3] > thr_angle and \
                angle_list[4] > thr_angle:
            ges = "One"  # 只伸展食指视为一
        elif angle_list[0] > thr_angle_thumb and \
                angle_list[1] < thr_angle_s and \
                angle_list[2] < thr_angle_s and \
                angle_list[3] > thr_angle and \
                angle_list[4] > thr_angle:
            ges = "Two"  # 只伸展食指和中指视为二
        elif angle_list[0] > thr_angle_thumb and \
                angle_list[1] < thr_angle_s and \
                angle_list[2] < thr_angle_s and \
                angle_list[3] < thr_angle_s and \
                angle_list[4] > thr_angle:
            ges = "Three"  # 只伸展食指和中指和无名指视为三
        elif angle_list[0] > thr_angle_thumb and \
                angle_list[1] < thr_angle_s and \
                angle_list[2] < thr_angle_s and \
                angle_list[3] < thr_angle_s and \
                angle_list[4] < thr_angle_s:
            ges = "Four"  # 只握紧大拇指视为四
        elif angle_list[0] < thr_angle_s and \
                angle_list[1] < thr_angle_s and \
                angle_list[2] < thr_angle_s and \
                angle_list[3] < thr_angle_s and \
                angle_list[4] < thr_angle_s:
            ges = "Five"  # 所有手指全部握紧视为五

    return ges


def determine(ls):  # 通过手势结果判断对局情况
    draw = "Draw"
    left = "Left Win"
    right = "Right Win"
    winner = None
    if ls[0][1] == ls[1][1] and ls[0][1] is not None:
        winner = draw  # 若双方结果均不出现异常的情况下相等，则为平局
    elif ls[0][1] == "Scissor" and ls[1][1] == "Paper":
        winner = left
    elif ls[0][1] == "Scissor" and ls[1][1] == "Rock":
        winner = right
    elif ls[0][1] == "Rock" and ls[1][1] == "Paper":
        winner = right
    elif ls[0][1] == "Rock" and ls[1][1] == "Scissor":
        winner = left
    elif ls[0][1] == "Paper" and ls[1][1] == "Rock":
        winner = left
    elif ls[0][1] == "Paper" and ls[1][1] == "Scissor":
        winner = right
    # 若有任意一只手识别失败，则视为没有对局结果，返回空值
    return winner


def str_to_num(s):  # 将字符串转化成对应的数字
    num = 0
    if s == "One":
        num = 1
    elif s == "Two":
        num = 2
    elif s == "Three":
        num = 3
    elif s == "Four":
        num = 4
    elif s == "Five":
        num = 5

    return num


def win_num():
    mp_drawing = mp.solutions.drawing_utils  # 用于绘制
    mp_hands = mp.solutions.hands            # 用于识别手势
    hands = mp_hands.Hands(
        static_image_mode=False,             # 动态识别手势
        max_num_hands=1,                     # 可以识别一只手
        model_complexity=1,                  # 增强识别性能
        min_detection_confidence=0.5,        # 检测置信度为0.5
        min_tracking_confidence=0.5)         # 追踪置信度为0.5
    cap = cv2.VideoCapture(0)
    num = 0
    while True:
        # 配置图像框架
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间
        frame = cv2.flip(frame, 1)                      # 图像翻转
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换颜色空间
        # 当检测到手势时，进行手势识别和处理
        if results.multi_hand_landmarks:
            # 绘制手型
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            # 记录手的21个点的坐标位置
            hand_local = []
            for i in range(21):
                x = results.multi_hand_landmarks[0].landmark[i].x * frame.shape[1]
                y = results.multi_hand_landmarks[0].landmark[i].y * frame.shape[0]
                hand_local.append((x, y))
            # 根据坐标判断出每个手指的角度和手的姿势
            angle = hand_angle(hand_local)
            gesture = hand_num_gesture(angle)
            client_socket.sendto(str(gesture).encode(), server_address)
            # 绘制手势
            x = int(hand_local[0][0]) - 50
            y = min(int(hand_local[i][1]) for i in range(21)) - 50
            cv2.putText(frame, gesture, (x, y), 0, 1.3, (0, 0, 255), 3)
            num = str_to_num(gesture)
        else:
            client_socket.sendto("None".encode(), server_address)
        # 展示图象
        cv2.imshow('MediaPipe Hands', frame)
        # 当按下了Enter键时，退出循环
        if cv2.waitKey(1) & 0xFF == 13:
            break

    return num


def detect():
    mp_drawing = mp.solutions.drawing_utils     # 用于绘制
    mp_hands = mp.solutions.hands               # 用于识别手势
    hands = mp_hands.Hands(
            static_image_mode=False,            # 动态识别手势
            max_num_hands=2,                    # 可以识别两只手
            model_complexity=1,                 # 增强识别性能
            min_detection_confidence=0.5,       # 检测置信度为0.5
            min_tracking_confidence=0.5)        # 追踪置信度为0.5
    cap = cv2.VideoCapture(0)
    round_now = 0                               # 记录回合数
    last_condition = [None, None]               # 记录上次识别结果
    score = [0, 0]                              # 记录比分
    win_round = win_num()                       # 记录回合制
    flag = 0                                    # 记录胜利情况，0代表未决胜负，1代表左手方胜利，2代表右手方胜利
    while True:
        # 配置图像框架
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间
        frame = cv2.flip(frame, 1)                      # 图像翻转
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换颜色空间
        # 若胜负已定，则不再继续识别，只显示比赛结果
        if flag == 1:
            cv2.putText(frame, "Left is Winner!", (205, 75), 0, 1.3, (0, 0, 255), 3)
        elif flag == 2:
            cv2.putText(frame, "Right is Winner!", (205, 75), 0, 1.3, (0, 0, 255), 3)
        # 绘制比赛相关信息
        num = 0
        cv2.putText(frame, "Round" + str(round_now), (250, 150), 0, 1.3, (0, 0, 255), 3)
        cv2.putText(frame, str(score[0]) + " : " + str(score[1]), (260, 200), 0, 1.3, (0, 0, 255), 3)
        # 当检测到手势时，进行手势识别和处理
        if results.multi_hand_landmarks:
            demo = []  # 记录左右手信息
            for hand_landmarks in results.multi_hand_landmarks:
                num += 1
                # 绘制手型
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # 记录手的21个点的坐标位置
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x * frame.shape[1]
                    y = hand_landmarks.landmark[i].y * frame.shape[0]
                    hand_local.append((x, y))
                # 根据坐标判断出每个手指的角度和手的姿势
                angle = hand_angle(hand_local)
                gesture = hand_gesture(angle)
                client_socket.sendto(str(gesture).encode(), server_address)
                # 绘制手势
                x = int(hand_local[0][0]) - 50
                y = min(int(hand_local[i][1]) for i in range(21)) - 50
                cv2.putText(frame, gesture, (x, y), 0, 1.3, (0, 0, 255), 3)
                # 更新左右手信息
                demo.append([int(hand_local[0][0]), gesture])
                if num == 2 and flag == 0:
                    # 当有一方胜利数达到阈值时，比赛结束
                    if score[0] == win_round:
                        flag = 1
                        continue
                    elif score[1] == win_round:
                        flag = 2
                        continue
                    # 按左右手顺序填进列表
                    if demo[0][0] > demo[1][0]:
                        temp = demo[0]
                        demo[0] = demo[1]
                        demo[1] = temp
                    # 判断当前对局结果并展示
                    result = determine(demo)
                    cv2.putText(frame, result, (240, 75), 0, 1.3, (0, 0, 255), 3)
                    # 当对局结果有效时，更新比赛信息
                    if result is not None and last_condition[0] is None and last_condition[1] is None:
                        round_now += 1
                        last_condition[0] = demo[0][1]
                        last_condition[1] = demo[1][1]
                        if result == "Left Win":
                            score[0] += 1
                        elif result == "Right Win":
                            score[1] += 1
        else:
            client_socket.sendto("None".encode(), server_address)
        # 当无法检测到两只手时，视为无效对局
        if num != 2:
            last_condition = [None, None]
        # 展示图像
        cv2.imshow('MediaPipe Hands', frame)
        # 当按下了Esc键时，退出循环
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()


if __name__ == '__main__':
    detect()
