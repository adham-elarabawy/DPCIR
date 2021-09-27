import cv2

gray = cv2.imread('/mikQNAP/aelarabawy/DIV2K_valid_HR/801.png', 0)

cv2.imwrite('grayed.png', gray)

