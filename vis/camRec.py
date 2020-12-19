import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('testMovie.avi', fourcc, 20.0, (640,480))

while (True):
    ret, frame = cap.read()
    colr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retval2, threshold2 = cv2.threshold(colr, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Otsu threshold', threshold2)
    # cv2.imshow('original', colr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
