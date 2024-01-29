import argparse
import os
import os.path as osp
import json
import tqdm
import pdb

from mmdet3d.utils import register_all_modules
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# DATA_ROOT = '/ai/volume/dataset/V2X-Seq-SPD'
# LABEL_EXT = '.json'

def parse_args():
    parser = argparse.ArgumentParser(description='Debug Program')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    data_loader = Runner.build_dataloader(cfg.train_dataloader)
    item = next(iter(data_loader))
    pdb.set_trace()
    

if __name__ == '__main__':
    # register_all_modules()
    # label_root = osp.join(DATA_ROOT, 'cooperative/label')
    # label_list = os.listdir(label_root)
    # print(len(label_list))
    # cls_names = set()
    # for idx, label_name in tqdm.tqdm(enumerate(label_list)):
    #     with open(osp.join(label_root, label_name)) as f:
    #         label_dicts = json.load(f)
    #     for label_dict in label_dicts:
    #         cls_name = label_dict['type']
    #         cls_names.add(cls_name)
    # print(cls_names)
    # main()
    import numpy as np
    try:
        import open3d as o3d
        from open3d import geometry
        from open3d.visualization import Visualizer
    except ImportError:
        o3d = geometry = Visualizer = None
    if o3d is None or geometry is None:
        raise ImportError(
            'Please run "pip install open3d" to install open3d first.')
    import os
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="QStandardPaths: XDG_RUNTIME_DIR not set, defaulting to '/tmp/runtime-root'")
    import matplotlib.pyplot as plt
    path = '/ai/volume/DeepAccident/data/DeepAccident_Data/type1_subtype1_accident/ego_vehicle/lidar01/Town01_type001_subtype0001_scenario00001/Town01_type001_subtype0001_scenario00001_001.npz'
    points = np.load(path)['data']
    save_path = './pointss.png'
    
    o3d_vis = o3d.visualization.Visualizer()
    o3d_vis.create_window(width=1024, height=1024)
    view_control = o3d_vis.get_view_control()
    o3d_vis.get_view_control().set_up([0, 0, 1])
    o3d_vis.get_render_option().point_size = 0.1
    o3d_vis.get_render_option().background_color = [0, 0, 0]
    # return o3d_vis
    pcd = geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    points_colors = np.tile(np.array((1,1,1)), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(points_colors)

    # 定义裁剪范围
    min_bound = np.array([-51.2, -51.2, -5.0])  # 裁剪范围的最小值
    max_bound = np.array([51.2, 51.2, 3.0])    # 裁剪范围的最大值
    # 创建裁剪框
    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    # 裁剪点云
    cropped_point_cloud = pcd.crop(crop_box)

    mesh_frame = geometry.TriangleMesh.create_coordinate_frame(**dict(size=3.5, origin=[0, 0, 0]))
    o3d_vis.add_geometry(mesh_frame)
    o3d_vis.add_geometry(cropped_point_cloud)
    
    # o3d_vis.capture_screen_image(save_path, do_render=True)
    image_data = o3d_vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image_data)

    fig, ax = plt.subplots(1,1)
    ax.imshow(image)
    ax.set_axis_off()
    fig.savefig('./tefasdf.png', dpi=1000, bbox_inches='tight', pad_inches=0)