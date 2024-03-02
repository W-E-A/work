import imageio
from glob import glob

# 把场景序列文件放到一个文件夹，然后输出gif图像

imgpath = glob('./data/step_vis_data/temp/*')
name = imgpath[0].split('/')[-1].split('.')[0].split('_')[:-1]
name = '_'.join(name)
output_gif_path = f'./data/step_vis_data/{name}.gif'

png_files = [f for f in imgpath if f.endswith('.png')]

png_files = sorted(png_files, key=lambda x : int(x.split('_')[-1].split('.')[0]))

images = []
for png_file in png_files:
    images.append(imageio.imread(png_file))

imageio.mimsave(output_gif_path, images, 'GIF', fps=2, loop=0) # type: ignore


# import imageio
# from glob import glob

# # 把场景序列文件放到一个文件夹，然后输出gif图像

# imgpath = glob('./data/step_vis_data/temp/*')
# name = imgpath[0].split('/')[-1].split('_')[:-1]
# name[-1] = name[-1].rstrip('.png')
# name = '_'.join(name)
# output_gif_path = f'./data/step_vis_data/{name}.gif'

# png_files = [f for f in imgpath if f.endswith('.png')]

# # png_files = sorted(png_files, key=lambda x : int(x.split('_')[-1].split('.')[0])) # 自定义排序
# png_files = sorted(png_files, key=lambda x : int(x.split('_')[-6]))

# images = []
# for png_file in png_files:
#     images.append(imageio.imread(png_file))

# imageio.mimsave(output_gif_path, images, 'GIF', fps=2, loop=0) # type: ignore