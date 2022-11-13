import cv2
import apriltag
import numpy
import imutils

#We are going to measure in feet 

"""
families in April Tags

 - Tag36h11
 - TagStandard41h12
 - TagStandard52h13
 - TagCircle21h7
 - TagCircle49h12
 - TagCustom48h12

 To Do:
 get angle
 get distance
"""

vid = cv2.VideoCapture(0)

def convert(pt):
    return (int(pt[0]), int(pt[1]))

#We need to know the width of the april tag beforehand
KNOWN_WIDTH = 0.2
#We also need the initial distance to the april tag
KNOWN_DISTANCE = 1.0
#Length per pixel

#Computes the distance to the camera
def distanceToCam(knownWidth, focalLength, perWidth):
    return knownWidth * focalLength / perWidth;

def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)

    return cv2.minAreaRect(c)

def main():
    oneFoot = cv2.imread("1ft.jpg")
    marker = find_marker(oneFoot)
    focalLen = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

    while(True): 
        ret, image = vid.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)
        
        print("Number of results: ", len(results))
        for r in results:
            #get bounding box of apriltag
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = convert(ptA)
            ptB = convert(ptB)
            ptC = convert(ptC)
            ptD = convert(ptD)

            cv2.line(image, ptA, ptB, (0, 0, 255), 2)
            cv2.line(image, ptB, ptC, (0, 0, 255), 2)
            cv2.line(image, ptC, ptD, (0, 0, 255), 2)
            cv2.line(image, ptD, ptA, (0, 0, 255), 2)
        
            (cX, cY) = (int(r.center[0]), int(r.center[1]));
            cv2.circle(image, (cX, cY), 5, (0, 0, 255,), -1)
          
            side = numpy.sqrt((ptA[0] - ptB[0]) * (ptA[0] - ptB[0]) + (ptA[1] - ptB[1]) * (ptA[1] - ptB[1]))
            print(side) 

            dist = round(distanceToCam(KNOWN_WIDTH, focalLen, side) * 100.0) / 100.0

            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #distance 
            cv2.putText(image, "Distance: " + str(dist) + "ft", (ptA[0], ptA[1] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 
            cv2.putText(image, "Angle: " + str(dist) + "ft", (ptA[0], ptA[1] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 
            print("[INFO] Tag family: {}".format(tagFamily)) 
            

        cv2.imshow("Image", image)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    vid.release()

if __name__ == "__main__":
    main()
