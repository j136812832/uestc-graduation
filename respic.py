from PIL import Image
import os


# im=Image.open('test.jpg')
# out=im.resize((20,20))#
# out.save("out.png","PNG") #
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        count = 1
        # 当前文件夹所有文件
        print(files)
        for i in files:
            # 判断是否以.jpg结尾
            if i.endswith('.png'):
                # 如果是就改变图片像素为28 28
                i = file_dir + "/" + i
                im = Image.open(i)

                out = im.resize((224, 224))
                out.save('./image/test/label/' + str(count) + '.png', 'PNG')
                count += 1
                print(i)


file_name('/home/zhangyijie/Desktop/vaismall/test/label')  # 当前文件夹

