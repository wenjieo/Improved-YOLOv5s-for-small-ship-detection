import os.path
import skimage
import cv2

def select_has_object(txt_from, img_from, txt_save,img_save):
    num = 0
    for txt in os.listdir(txt_from):
        txt_name = txt[0:-4]
        txt = txt_from + '\\' + txt
        img_name = txt_name
        img = img_from + '\\' + img_name + '.png'
        img = cv2.imread(img)
        if os.path.getsize(txt) != 0:
            num = num + 1
            fp = open(txt, 'r')
            for line in fp:
                fq = open(txt_save + '\\' + txt_name + '.txt', 'a+')
                fq.write(line)
            fp.close()
            fq.close()
            cv2.imwrite(img_save + '\\' + img_name + '.png', img)
    print("包含目标的图片数量：", num)

txt_from = r'E:\otherdata\LEVIR-Ship\our-division\test\labels'
img_from = r'E:\otherdata\LEVIR-Ship\our-division\test\images'
txt_save = r'C:\Users\dell\Desktop\2'
img_save = r'C:\Users\dell\Desktop\1'

select_has_object(txt_from, img_from, txt_save,img_save)


