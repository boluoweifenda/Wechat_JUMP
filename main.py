import os
import cv2
import numpy as np

Height = 800  # we rescale screen to 800x450 (if ratio is 16:9)
Width = None
Scale = None
HW = None  # the resolution of the screen
MEAN = 77  # the average grayscale value of the chess
R_chess = 12  # the relative radius of the chess head
Y_bias = 68
K = 3.75  # scale coefficient for press time
B = 15  # bias coefficient for press time
Wait = 1500  # step period

def adb(command):
  p = os.popen(command)
  return p.read()


print(adb('adb version'))
print(adb('adb devices'))



def format(x):
  x = int(round(x))
  if x < 0:
    x = 0
  return x


def findOrigin(frame):
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  # search small circle
  circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=10, param2=20, minRadius=10, maxRadius=15)[0, :]

  origin_fig = np.zeros_like(frame)
  if circles is not None:
    for circle in circles:
      # draw the outer circle
      cv2.circle(origin_fig, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
      # draw the center of the circle
      cv2.circle(origin_fig, (circle[0], circle[1]), 2, (0, 0, 255), 2)

  cv2.imshow('origin_fig', origin_fig)

  numCross = len(circles)

  minSub = 100
  minIndex = 0
  for i in range(numCross):
    c = (circles[i][0], circles[i][1])
    r = R_chess

    def getSquarePatch(c,r,gray):
      x0 = format(c[0] - r)
      y0 = format(c[1] - r)
      x1 = format(c[0] + r)
      y1 = format(c[1] + r)
      patch = gray[y0:y1, x0:x1]
      return patch

    patch = getSquarePatch(c,r,gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (patch.shape[1], patch.shape[0]))

    sum = np.sum(np.multiply(patch, kernel))
    mean = 1.0 * sum / np.sum(kernel)

    sub = abs(mean - MEAN)
    if sub < minSub:
      minSub = sub
      minIndex = i

  return circles[minIndex]

def findLine(frame):

  lines = []
  for channel in range(3):
    frame_channel = frame[..., channel]
    canny = cv2.Canny(frame_channel, 1, 50)
    lines.append(cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=5, minLineLength=20, maxLineGap=5))

  lines = np.concatenate(lines, axis=0)
  line_fig = np.zeros_like(frame)

  minY = [100000, 100000]
  minIndex = [0, 0]
  if lines is not None:
    for i in range(len(lines)):
      line = lines[i][0]
      if line[3] > 200:
        rad = (line[3] - line[1])/(line[2] - line[0] + 1e-3)
        if rad > 0.5 and rad < 0.6:
          cv2.line(line_fig, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
          if line[1] < minY[0]:
            minY[0] = line[1]
            minIndex[0] = i

        if rad > -0.6 and rad < -0.5:
          cv2.line(line_fig, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
          if line[3] < minY[1]:
            minY[1] = line[3]
            minIndex[1] = i

  line1 = lines[minIndex[0]][0]
  line2 = lines[minIndex[1]][0]


  k1 = 1.0*(line1[3] - line1[1]) / (line1[2] - line1[0])
  b1 = 1.0*(line1[1] - k1 * line1[0])

  k2 = 1.0*(line2[3] - line2[1]) / (line2[2] - line2[0])
  b2 = 1.0*(line2[1] - k2 * line2[0])

  k12 = k1 - k2 + 1e-3 if abs(k1 - k2) < 1e-3 else k1 - k2
  x = np.round((b2 - b1) / k12)
  y = np.round(k1 * x + b1)

  cv2.imshow('line', line_fig)

  return np.float32([x,y])

def findDistortCircle(frame_distort):

  circles = []

  for channel in range(3):
    frame_channel = frame_distort[..., channel]
    circle = cv2.HoughCircles(frame_channel, cv2.HOUGH_GRADIENT, 1, 50, param1=3, param2=30, minRadius=5, maxRadius=50)
    if circle is not None:
      circles.append(circle[0, ...])

  if len(circles) !=0:
    circles = np.concatenate(circles, axis=0)


  minY = 10000
  minIndex = -1
  if circles is not None:
    for i in range(len(circles)):
      c_y = circles[i][1]
      if c_y < minY and c_y > 100:
        minY = c_y
        minIndex = i

      # draw the outer circle
      cv2.circle(frame_distort, (circles[i][0], circles[i][1]), circles[i][2], (0, 255, 0), 2)
      # draw the center of the circle
      cv2.circle(frame_distort, (circles[i][0], circles[i][1]), 2, (0, 0, 255), 2)

  cv2.imshow('frame_distort', frame_distort)

  if minIndex == -1:
    return None
  return circles[minIndex]

def get_button_position(HW):
  x_center = 0.5
  y_center = 0.825
  width = 0.005
  height = 0.005

  rand = np.clip(np.random.randn(4), -1, 1)

  x1 = format(HW[1]*(rand[0]*width + x_center))
  x2 = format(HW[1]*(rand[1]*width + x_center))
  y1 = format(HW[0]*(rand[2]*height + y_center))
  y2 = format(HW[0]*(rand[3]*height + y_center))
  return x1, y1, x2, y2


step = 0
while True:

  cv2.waitKey(Wait + int(500 * np.random.rand()))

  adb('adb shell /system/bin/screencap -p /sdcard/screenshot.png')
  adb('adb pull /sdcard/screenshot.png  screenshot.png')
  frame = cv2.imread('screenshot.png')

  if Width is None:
    HW = frame.shape[0:-1]  # get screen resolution
    Scale = 1.0 * frame.shape[0]/Height
    Width = format(frame.shape[1]/Scale)
    print(frame.shape[0],'x',frame.shape[1],'->',Height,'x',Width,'Scale =',Scale)

  frame = cv2.resize(frame,
                   dsize=(format(frame.shape[1]*Height/frame.shape[0]), Height),
                   interpolation=cv2.INTER_AREA)

  frame_distort = cv2.resize(frame,
                   dsize=(int(frame.shape[1]*Height/(frame.shape[0]*1.73)), Height),
                   interpolation=cv2.INTER_AREA)


  circle = findOrigin(frame)
  p2x,p2y = findLine(frame)
  circle_distort = findDistortCircle(frame_distort)

  # draw original point
  cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
  p1x = circle[0]
  p1y = circle[1] + np.float32(Y_bias)
  cv2.circle(frame, (p1x, p1y), 2, (255, 0, 0), 2)

  # draw original line
  cv2.line(frame, (p1x, 0), (p1x, Height), (0, 0, 255), 2)

  # check whether to use findLine or findDistortCircle
  if circle_distort is not None:
    circle_distort[0] = round(circle_distort[0] * 1.73)
    if circle_distort[1] < p1y - 10 and circle_distort[1] < p2y:
      p2x = circle_distort[0]
      p2y = circle_distort[1]
      cv2.circle(frame, (circle_distort[0], circle_distort[1]), circle_distort[2], (0, 0, 255), 2)

  # draw destination
  cv2.line(frame, (p2x, 0), (p2x, Height), (255, 0, 0), 2)

  cv2.imshow('frame', frame)
  cv2.waitKey(100)

  swipe_x1,swipe_y1,swipe_x2,swipe_y2 = get_button_position(HW)
  duration = np.clip(B + format(K*abs(p2x - p1x)), 200, 1000)


  # if step%10 == 0:
  #   duration = duration + format(50 * np.clip(np.random.randn(),-1,+1))

  cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
    x1=swipe_x1,
    y1=swipe_y1,
    x2=swipe_x2,
    y2=swipe_y2,
    duration=duration
  )

  print(duration)

  adb(cmd)

  step = step + 1
