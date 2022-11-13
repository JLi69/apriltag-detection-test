import cv2
import apriltag

"""
families in April Tags

 - Tag36h11
 - TagStandard41h12
 - TagStandard52h13
 - TagCircle21h7
 - TagCircle49h12
 - TagCustom48h12
"""

def convert(pt):
    return (int(pt[0]), int(pt[1]))

def main():
    image = cv2.imread("april.png")
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

        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
    
        (cX, cY) = (int(r.center[0]), int(r.center[1]));
        cv2.circle(image, (cX, cY), 5, (0, 0, 255,), -1)
        
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] Tag family: {}".format(tagFamily))

    cv2.imshow("Image", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
