from PIL import Image
from PIL import ImageEnhance



class image_enhance():
    def __init__(self,image):
        self.image = image

    # 量度增强
    def brightness(self, brightness):
        enh_bri = ImageEnhance.Brightness(self.image)
        image_brightened = enh_bri.enhance(brightness)
        return image_brightened

    # 颜色强度
    def Color(self, color):
        enh_col = ImageEnhance.Color(self.image)
        image_color = enh_col.enhance(color)
        return image_color
        # 对比度强度
    def Constrast(self,constrast):
        enh_con = ImageEnhance.Contrast(self.image)
        image_contrasted = enh_con.enhance(constrast)
        return image_contrasted

    # 瑞度增强
    def Sharpness(self,sharpness):
        enh_sha = ImageEnhance.Sharpness(self.image)
        image_sharped = enh_sha.enhance(sharpness)
        return image_sharped
    # 图像旋转
    def Rotate(self,rotate):
        img = self.image.rotate(rotate)
        return img
    def Mirror_left_right(self):
        img = self.image.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    #镜像
    def Mirror_top_bottom(self):
        img = self.image.transpose(Image.FLIP_TOP_BOTTOM)
        return img

filepath = "./2.jpg"
image = Image.open(filepath)
Enhance = image_enhance(image)
s = Enhance.Mirror_top_bottom()
s.show()
s = Enhance.Sharpness(0.5)
s.show()

