import cv2

img = cv2.imread('./result/res_at_iteration_0.png')

cv2.imshow('sample image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
