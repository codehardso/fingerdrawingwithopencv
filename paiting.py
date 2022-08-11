# A toy project with OpenCV, PyMunk and Mediapipe
import pymunk
import cv2
import numpy as np
import mediapipe as mp
from utils.cv_utils import rescale_frame, get_emojis, overlay
from utils.physics_utils import add_emojis_to_space, add_fingers_to_space, create_static_line


mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
# define the space for handling physics
space = pymunk.Space()
space.gravity = 0, -500

# define balls as dynamic bodies for physics engine
balls_radius = 12
balls = [(640 + np.random.uniform(-640, 640), 0 + 5 * i + 0.5 * i ** 2) for i in range(500)]#0-1200
balls_body = [pymunk.Body(100.0, 1666, body_type=pymunk.Body.DYNAMIC) for b in balls]
for i, ball in enumerate(balls_body):
    balls_body[i].position = balls[i]
    shape = pymunk.Circle(balls_body[i], balls_radius)
    space.add(balls_body[i], shape)

###

emoji_list = get_emojis()
emoji_radius = 30
number_of_emojis = 68

emojis = [(640 + np.random.uniform(-640, 640), 0 + 5 * i + 0.5 * i ** 2) for i in
                  range(number_of_emojis)]
emojis_body = [pymunk.Body(100.0, 1666, body_type=pymunk.Body.DYNAMIC) for _ in emojis]
add_emojis_to_space(space, emojis_body, emojis, emoji_radius)


##
# define fingers as kinematic bodies for physics engine
fingers_radius = 20
fingers = [pymunk.Body(10, 1666, body_type=pymunk.Body.KINEMATIC) for i in range(21)]
for i, finger in enumerate(fingers):
    finger_shape = pymunk.Circle(fingers[i], fingers_radius)
    space.add(fingers[i], finger_shape)

# a few color for drawing balls
colors = [(219, 152, 52), (34, 126, 230), (182, 89, 155),
          (113, 204, 46), (94, 73, 52), (15, 196, 241),
          (60, 76, 231)]

cap = cv2.VideoCapture(0)

xp, yp = 0, 0

brushThickness = 18
eraserThickness = 30

def create_static_line(space, x1, y1, x2, y2):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, (x1, y1), (x2, y2), 10)
    space.add(body, shape)
    return shape

create_static_line(space, 0, 0, 1280, 0)#横线
create_static_line(space, 0, 0, 0, 720)#左边竖线
create_static_line(space, 1280, 0, 1280, 720)#右边竖线

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

def addball(event, x, y, flags, param): # on mouse click add a ball
    if event == cv2.EVENT_LBUTTONDOWN:
        ball_body = pymunk.Body(100.0,1666, body_type=pymunk.Body.DYNAMIC)
        ball_body.position = (x,475-y)
        shape = pymunk.Circle(ball_body, balls_radius)
        space.add(ball_body, shape)
        balls_body.append(ball_body)
# reading the video from webcam

with mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        success, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                print("hand",hand_landmarks)
                for i, finger in enumerate(fingers):
                    # converting the coordinates
                    x = int(hand_landmarks.landmark[i].x * image.shape[1])
                    y = image.shape[0] - int(hand_landmarks.landmark[i].y * image.shape[0])
                    # update the velocity of balls
                    fingers[i].velocity = 14.0 * (x - fingers[i].position[0]), 14.0 * (y - fingers[i].position[1])

                    ###gesture
                    #index
                    x_index, y_index = int(hand_landmarks.landmark[8].x * image.shape[1]), int(hand_landmarks.landmark[8].y * image.shape[0])

                    x_indexdown, y_indexdown = int(hand_landmarks.landmark[7].x * image.shape[1]), int(
                        hand_landmarks.landmark[7].y * image.shape[0])
                    #middle
                    x_middle, y_middle = int(hand_landmarks.landmark[12].x * image.shape[1]), int(
                        hand_landmarks.landmark[12].y * image.shape[0])

                    x_middledown, y_middledown = int(hand_landmarks.landmark[11].x * image.shape[1]), int(
                        hand_landmarks.landmark[11].y * image.shape[0])

                    #ring
                    x_ring, y_ring = int(hand_landmarks.landmark[16].x * image.shape[1]), int(
                        hand_landmarks.landmark[16].y * image.shape[0])

                    x_ringdown, y_ringedown = int(hand_landmarks.landmark[13].x * image.shape[1]), int(
                        hand_landmarks.landmark[13].y * image.shape[0])
                    #pinky
                    x_pinky, y_pinky = int(hand_landmarks.landmark[20].x * image.shape[1]), int(
                        hand_landmarks.landmark[20].y * image.shape[0])

                    x_pinkydown, y_pinkydown = int(hand_landmarks.landmark[17].x * image.shape[1]), int(
                        hand_landmarks.landmark[17].y * image.shape[0])


                    if y_index < y_indexdown and y_middle < y_middledown and y_ring < y_ringedown and y_pinky < y_pinkydown:
                        xp, yp = 0, 0
                        print("Selection Mode")

                    if y_index < y_indexdown and y_middle > y_middledown and y_ring > y_ringedown and y_pinky > y_pinkydown:#食指
                        cv2.circle(image, (x_index, y_index), 10, (230, 230 ,250), cv2.FILLED)
                        ##
                        for i, emoji in enumerate(emojis_body):
                            xb = int(emoji.position[0])
                            yb = int(image.shape[0] - emoji.position[1])
                            image = overlay(image, emoji_list[i % len(emoji_list)], xb, yb, 46, 46)
                        ##

                        if xp == 0 and yp == 0:
                            xp, yp = x_index, y_index

                        cv2.line(image, (xp, yp), (x_index, y_index), (255, 238, 210), brushThickness)
                        cv2.line(imgCanvas, (xp, yp), (x_index, y_index), (255, 238, 210), brushThickness)
                        xp, yp = x_index, y_index
                        print("index up")

                    if y_index < y_indexdown and y_middle < y_middledown and y_ring > y_ringedown and y_pinky > y_pinkydown:  # 橡皮
                        xp, yp = 0, 0
                        if xp == 0 and yp == 0:
                            xp, yp = x_index, y_index
                        cv2.line(image, (xp, yp), (x_index, y_index), (230, 230 ,250), eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (x_index, y_index), (0, 0, 0), eraserThickness)


        # getting the position of balls from physics engine and drawing
        for i, ball in enumerate(balls_body):
            xb = int(ball.position[0])
            yb = int(image.shape[0] - ball.position[1])
            cv2.circle(image, (xb, yb), balls_radius, colors[i % len(colors)], -1)

        # take a simulation step
        space.step(0.02)

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        image = cv2.bitwise_and(image, imgInv)
        image = cv2.bitwise_or(image, imgCanvas)

        cv2.imshow("Image", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

