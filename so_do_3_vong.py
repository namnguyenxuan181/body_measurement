import cv2
import math

protoFile = "mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "mpi/pose_iter_160000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
# file = input('nhap ten file hoac duong dan day du: ')
frame = cv2.imread('aman2.jpg')
blurred = cv2.GaussianBlur(frame, (5, 5), 0) 
edge = cv2.Canny(blurred,100,200)
# Specify the input image dimensions
inWidth = 368
inHeight = 368
 
# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
 
# Set the prepared object as the input blob of the network
net.setInput(inpBlob)

output = net.forward()

H = output.shape[2]
W = output.shape[3]
# Empty list to store the detected keypoints
frameHeight, frameWidth, chanel = frame.shape
points = []
for i in range(15):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
 
    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
     
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H
    threshold=0
    if prob > threshold : 
        # cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        # cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
 
        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else :
        points.append(None)
body={}
body['Head'] = points[0]
body['Neck'] = points[1]
body['Right Shoulder'] = points[2]
body['Right Elbow'] = points[3]
body['Right Wrist'] = points[4]
body['Left Shoulder'] = points[5]
body['Left Elbow'] = points[6]
body['Left Wrist'] = points[7]
body['Right Hip'] = points[8]
body['Left Hip'] = points[11]
body['Chest'] = points[14]

def measurement(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def distance(a,b,x):
    return a/b*x
high = body['Left Ankle'][1] - body['Head'][1]
shoulder = measurement(body['Right Shoulder'],body['Left Shoulder'])
hip = measurement(body['Right Hip'],body['Left Hip'])
body['Right Hip'] =[body['Right Hip'][0] -hip/2,body['Right Hip'][1]]
body['Left Hip'] =[body['Left Hip'][0] +hip/2,body['Left Hip'][1]]
points[8] = body['Right Hip'] 
points[11] = body['Left Hip'] 
nguc = body['Neck'][1] + ( body['Chest'][1]-body['Neck'][1])/2
nguc_phai = [body['Right Shoulder'][0]+(body['Right Hip'][0]-body['Right Shoulder'][0])/2, nguc]
nguc_trai = [body['Left Hip'][0]+(body['Left Shoulder'][0]-body['Left Hip'][0])/2, nguc]
eo = body['Chest'][1] + ( body['Right Hip'][1]-body['Chest'][1])/2
eo_trai = [body['Chest'][0],eo]
eo_phai = [body['Chest'][0],eo]
print(eo)
while(edge[eo_phai[0],eo_phai[1] !=255]):
    eo_phai[0]-=1
while(edge[eo_trai[0],eo_trai[1] !=255]):
    eo_trai[0]+=1

points.append(nguc_phai)
points.append(nguc_trai)
points.append(eo_phai)
points.append(eo_trai)
eo = measurement(eo_phai,eo_trai)
hip = measurement(body['Right Hip'],body['Left Hip'])
nguc = measurement(nguc_phai,nguc_trai)
ba_vong_ngang = [shoulder,nguc,eo,hip]
# points[14]=[0,0]
# i=0
l1=range(19)
l = [0,1,3,5,6,7,9,10,15]
for i in l1:
    cv2.circle(frame, (int(points[i][0]), int(points[i][1])), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(frame, "{}".format(i), (int(points[i][0]), int(points[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    # i+=1
# high_cm = int(input('nhap chieu cao: '))
high_cm = 160
fact_high = high_cm - 5

width = round(frameWidth/frameHeight*480)
height = 480
dim = (width, height)
resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Output-Keypoints",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
