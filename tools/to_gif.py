import imageio
from glob import glob
import os


def to_gif(root_path, fps):
# 把场景序列文件放到一个文件夹，然后输出gif图像

    imgpath = glob(os.path.join(root_path, '*'))
    if len(imgpath) <= 0:
        print(f"empty folder: {root_path}, return")
        return
    output_gif_path = os.path.join(root_path, 'vis.gif')

    png_files = [f for f in imgpath if f.endswith('.png')]

    png_files = sorted(png_files, key=lambda x : int(x.split('_')[-1].split('.')[0]))

    images = []
    for png_file in png_files:
        images.append(imageio.imread(png_file))

    imageio.mimsave(output_gif_path, images, 'GIF', fps=fps, loop=0) # type: ignore
    print("to_gif done")


def prefix_to_gif(root_path, prefix, fps):
    imgpath = glob(os.path.join(root_path, '*'))
    if len(imgpath) <= 0:
        print(f"empty folder: {root_path}, return")
        return
    output_gif_path = os.path.join(root_path, f'{prefix}.gif')

    prefix_png_files = [f for f in imgpath if (f.endswith('.png') and os.path.basename(f).split('_')[0] == f'{prefix}')]
    prefix_png_files = sorted(prefix_png_files, key=lambda x : int(x.split('_')[-1].split('.')[0]))
    images = []
    for png_file in prefix_png_files:
        images.append(imageio.imread(png_file))
    
    print(prefix_png_files)
    imageio.mimsave(output_gif_path, images, 'GIF', fps=fps, loop=0) # type: ignore
    print(f"{prefix}_to_gif done")
    

if __name__ == '__main__':
    # ROOT_PATH = './data/motion/2282'
    # fps = 2
    # to_gif(ROOT_PATH, fps)
    prefix_to_gif('./data/motion/2282', 'instance', 2)
    prefix_to_gif('./data/motion/2282', 'center', 2)


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