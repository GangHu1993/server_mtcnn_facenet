from face_recon import Face_recon
import cv2

img=cv2.imread('3115.JPG',cv2.IMREAD_COLOR)

face_recon = Face_recon()
res = face_recon.face_rec(img)
res1 = face_recon.face_rec(img)
print (res)
