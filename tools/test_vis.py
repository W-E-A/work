from PIL import Image, ImageDraw, ImageFont  
import os
import os.path as osp

root_path = '/mnt/auto-labeling/wyc/deepaccident/data/vis/correlation_heatmap0514/type1_subtype1_accident_Town01_scenario00004_1'
images = os.listdir(root_path)
images.sort()

with Image.open(osp.join(root_path, images[0])) as img:  
    total_width = img.width * 4 
    total_height = img.height * 2 
# 创建一个新的空白图片来放置所有图片  
new_image = Image.new('RGB', (total_width+15, total_height+5), color=(255, 255, 255))  
  
  
# 初始化x坐标（用于放置图片）  
x_offset = 0  
# 绘制图片和注释  
draw = ImageDraw.Draw(new_image)  
for idx, img_name in enumerate(images):  
    with Image.open(osp.join(root_path, img_name)) as img:  
        new_image.paste(img, (x_offset, (idx%2)*(img.height+15)))
        caption = img_name.split('.')[0]
        left, top, right, bottom = draw.textbbox((0, 0), caption)
        text_width, text_height = right - left, bottom - top
        # 计算注释的位置（这里假设注释在图片下方居中）  
        text_x = x_offset + (img.width - text_width) // 2
        text_y = (idx%2+1) * img.height  
        # 添加注释  
        draw.text((text_x, text_y), caption, fill='black')  
        # 更新x坐标以便放置下一张图片  
        x_offset += (idx%2) * (img.width+5)  
  
# 保存合成后的图片  
new_image.save(osp.join(root_path,'combined_with_captions.jpg'))