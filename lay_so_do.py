import cv2
import math
import numpy as np
def do_chieu_day(file,high_cm,demo):
    high_cm -= 5
    protoFile = "mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "mpi/pose_iter_160000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # file = input('nhap ten file hoac duong dan day du: ')
    frame = cv2.imread(file)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0) 
    edge = cv2.Canny(blurred,50,100)
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
    body['Right Knee'] = points[9]
    body['Right Ankle'] = points[10]
    body['Left Hip'] = points[11]
    body['Left Knee'] = points[12]
    body['Left Ankle'] = points[13]
    body['Chest'] = points[14]

    def measurement(a,b):
        return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    def distance(a,b,x):
        return a/b*x
    high = body['Left Ankle'][1] - body['Head'][1]
    nguc = body['Neck'][1] + ( body['Chest'][1]-body['Neck'][1])/2
    nguc_phai = [int(body['Right Shoulder'][0]+(body['Right Hip'][0]-body['Right Shoulder'][0])/2), int(nguc)]
    nguc_trai = [int(body['Left Hip'][0]+(body['Left Shoulder'][0]-body['Left Hip'][0])/2), int(nguc)]
    eo = body['Chest'][1] + (min([ body['Right Hip'][1],body['Left Hip'][1]])-body['Chest'][1])/2
    eo_phai = [int(body['Right Hip'][0]+(body['Chest'][0]-body['Right Hip'][0])/2),int(eo)]
    eo_trai = [int(body['Chest'][0]+(body['Left Hip'][0]-body['Chest'][0])/2),int(eo)]

    while(np.sum(edge[eo_phai[1]-10:eo_phai[1]+10,eo_phai[0]]) ==0):
        eo_phai[0]-=1

    while(np.sum(edge[eo_trai[1]-10:eo_trai[1]+10,eo_trai[0]])==0):
        eo_trai[0]+=1
    # print(np.sum(edge[nguc_phai[1],nguc_phai[0]]))
    while(np.sum(edge[nguc_phai[1],nguc_phai[0]]) ==0):
        nguc_phai[0]-=1

    while(np.sum(edge[nguc_trai[1],nguc_trai[0]])==0):
        nguc_trai[0]+=1
    mong_phai = [body['Right Hip'][0],body['Right Hip'][1]]
    mong_trai = [body['Left Hip'][0],body['Left Hip'][1]]
    while(np.sum(edge[mong_phai[1],mong_phai[0]]) ==0):
        mong_phai[0]-=1

    while(np.sum(edge[mong_trai[1],mong_trai[0]])==0):
        mong_trai[0]+=1
    mong_phai[1] = mong_trai[1]
    # ,points[0],points[13]
    points = [nguc_phai,nguc_trai,eo_phai,eo_trai,mong_phai,mong_trai]
    eo = measurement(eo_phai,eo_trai)
    hip = measurement(body['Right Hip'],body['Left Hip'])
    nguc = measurement(nguc_phai,nguc_trai)
    # ba_vong_ngang = [shoulder,nguc,eo,hip]
    # points[14]=[0,0]
    # i=0
    for i in range(len(points)):
        cv2.circle(frame, (int(points[i][0]), int(points[i][1])), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        # cv2.putText(frame, "{}".format(i), (int(points[i][0]), int(points[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
        # i+=1
    # high_cm = int(input('nhap chieu cao: '))
    # high_cm = 160
    # fact_high = high_cm - 5
    # print(eo_phai,eo_trai)
    width = round(frameWidth/frameHeight*480)
    height = 480
    dim = (width, height)
    nguc =measurement(nguc_phai,nguc_trai) 
    eo = measurement(eo_phai,eo_trai)
    mong = measurement(mong_phai,mong_trai)
    day_nguc = distance(nguc,high,high_cm)
    day_eo = distance(eo,high,high_cm)
    day_mong = distance(mong,high,high_cm)


    if demo == True:
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Output-Keypoints",resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return day_nguc,day_eo,day_mong



def do_chieu_ngang(file,high_cm,demo):
    high_cm -= 5
    protoFile = "mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "mpi/pose_iter_160000.caffemodel"

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # file = input('nhap ten file hoac duong dan day du: ')
    frame = cv2.imread(file)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0) 
    edge = cv2.Canny(blurred,10,20)
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
    body['Right Knee'] = points[9]
    body['Right Ankle'] = points[10]
    body['Left Hip'] = points[11]
    body['Left Knee'] = points[12]
    body['Left Ankle'] = points[13]
    body['Chest'] = points[14]

    def measurement(a,b):
        return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    def distance(a,b,x):
        return a/b*x


    high = body['Left Ankle'][1] - body['Head'][1]
    hip = measurement(body['Right Hip'],body['Left Hip'])
    body['Right Hip'] =[body['Right Hip'][0] -hip/2,body['Right Hip'][1]]
    body['Left Hip'] =[body['Left Hip'][0] +hip/2,body['Left Hip'][1]]
    mong_phai = [int(body['Right Hip'][0]),int(body['Right Hip'][1])]
    mong_trai = [int(body['Left Hip'][0]),int(body['Left Hip'][1])]

    while(edge[mong_phai[1] , mong_phai[0]] ==0):
        mong_phai[0]-=1

    while(np.sum(edge[mong_trai[1],mong_trai[0]])==0):
        mong_trai[0]+=1
    mong_phai[1] = mong_trai[1]
    vai_phai = list(body['Right Shoulder'])
    vai_trai = list(body['Left Shoulder'])

    nguc = body['Neck'][1] + ( body['Chest'][1]-body['Neck'][1])/2
    nguc_phai = [int(vai_phai[0]+(mong_phai[0]-vai_phai[0])/2), int(nguc)]
    nguc_trai = [int(mong_trai[0]+(vai_trai[0]-mong_trai[0])/2), int(nguc)]

    eo = body['Chest'][1] + (min([ body['Right Hip'][1],body['Left Hip'][1]])-body['Chest'][1])/2
    eo_phai = [int(mong_phai[0]+(body['Chest'][0]-mong_phai[0])/2),int(eo)]
    eo_trai = [int(body['Chest'][0]+(mong_trai[0]-body['Chest'][0])/2),int(eo)]

    while(np.sum(edge[eo_phai[1],eo_phai[0]]) ==0):
        eo_phai[0]-=1

    while(np.sum(edge[eo_trai[1]-3:eo_trai[1]+3,eo_trai[0]])==0):
        eo_trai[0]+=1

    while(np.sum(edge[vai_phai[1],vai_phai[0]]) ==0):
        vai_phai[0]-=1

    while(np.sum(edge[vai_trai[1]-3:vai_trai[1]+3,vai_trai[0]])==0):
        vai_trai[0]+=1

    # mong_phai = [int(vai_phai[0]),int(body['Right Hip'][1])]
    # mong_trai = [int(vai_trai[0]),int(body['Left Hip'][1])]
    # while(edge[mong_phai[1] , mong_phai[0]] ==0):
    #     mong_phai[0]+=1

    # while(np.sum(edge[mong_trai[1],mong_trai[0]])==0):
    #     mong_trai[0]-=1
    # mong_phai[1] = mong_trai[1]

    kc_eo = max([ int(body['Chest'][0]) - eo_phai[0], eo_trai[0] - int(body['Chest'][0]) ])
    eo_phai = [body['Chest'][0] - kc_eo,int(eo)]
    eo_trai = [body['Chest'][0]+kc_eo,int(eo)]


    # while(np.sum(edge[nguc_phai[1],nguc_phai[0]]) ==0):
    #     nguc_phai[0]-=1

    # while(np.sum(edge[nguc_trai[1],nguc_trai[0]])==0):
    #     nguc_trai[0]+=1

    points = [nguc_phai,nguc_trai,eo_phai,eo_trai,mong_phai,mong_trai]

    shoulder = measurement(vai_phai,vai_trai)
    nguc =measurement(nguc_phai,nguc_trai)
    # print(nguc,high)
    eo = measurement(eo_phai,eo_trai)
    mong = measurement(mong_phai,mong_trai)
    ngang_vai = distance(shoulder,high,high_cm)
    ngang_nguc = distance(nguc,high,high_cm)
    ngang_eo = distance(eo,high,high_cm)
    ngang_mong = distance(mong,high,high_cm)
    # points[14]=[0,0]
    # i=0
    for i in range(len(points)):
        cv2.circle(frame, (int(points[i][0]), int(points[i][1])), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        # cv2.putText(frame, "{}".format(i), (int(points[i][0]), int(points[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
        # i+=1
    # high_cm = int(input('nhap chieu cao: '))
    # high_cm = 160
    # fact_high = high_cm - 5
    if demo == True:
        width = round(frameWidth/frameHeight*480)
        height = 480
        dim = (width, height)
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Output-Keypoints1",resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return ngang_nguc,ngang_eo,ngang_mong



ngang_nguc,ngang_eo,ngang_mong = do_chieu_ngang('aman2.jpg',160,True)
day_nguc,day_eo,day_mong=do_chieu_day('aman3.jpg',160,True)
print("vong 1: ngang: {}, day: {}".format(ngang_nguc,day_nguc))
print("vong 2: ngang: {}, day: {}".format(ngang_eo,day_eo))
print("vong 3: ngang: {}, day: {}".format(ngang_mong,day_eo))