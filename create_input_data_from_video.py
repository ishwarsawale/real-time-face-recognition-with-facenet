import cv2
import os
import shutil
import sys
# print(cv2.__version__)

print("Add Name for User:")
user_input = raw_input("Enter name: ")
path = user_input
if not os.path.exists(path):
    os.makedirs(path)
print("Please start turning slowly your head  like for apple FaceID. Press 'q' to save and add this new user to the dataset");
raw_input("Press Enter to continue...")

vidcap = cv2.VideoCapture(0)
success,image = vidcap.read()
count = 0
success = True
while success and count < 500:
  frame = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
  cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
  cv2.imshow('Frame',frame)
  cv2.imwrite("frames%d.jpg" % count, image)     # save frame as JPEG file
  print ('Saving frames%d.jpg to disk' %count)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  count += 1
vidcap.release()
cv2.destroyAllWindows()
sourcepath='./'
sourcefiles = os.listdir(sourcepath)
destinationpath = path
for file in sourcefiles:
    if file.endswith('.jpg'):
        shutil.move(os.path.join(sourcepath,file), os.path.join(destinationpath,file))
