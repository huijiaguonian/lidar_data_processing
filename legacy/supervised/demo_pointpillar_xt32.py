import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import torch
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate


class PandarXTDemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, ext='.bin'):
        super().__init__(dataset_cfg, class_names, training, root_path)
        self.root_path = root_path
        self.ext = ext

    def __len__(self):
        return 1

    def __getitem__(self, index):
        points = np.fromfile(self.root_path, dtype=np.float32).reshape(-1, 4)

        # XT32 parser coordinates: +Y is forward at azimuth 0 deg.
        # OpenPCDet/KITTI coordinates: +X forward, +Y left, +Z up.
        # Labels used for training must receive this exact same transform.
        x_hesai = points[:, 0].copy()
        y_hesai = points[:, 1].copy()
        points[:, 0] = y_hesai
        points[:, 1] = -x_hesai

        # Parser output intensity is already normalized to [0, 1].
        # Apply an optional vertical calibration exactly once.
        points[:, 2] += args.z_offset

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        # Batching is handled once by DatasetTemplate.collate_batch in main.
        return data_dict


def main():
    logger = common_utils.create_logger()
    logger.info('---------------- Quick Demo for PandarXT32 ----------------')

    # ==== 配置文件 ====
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger.info(cfg)

    dataset = PandarXTDemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=args.data_path,
        ext=args.ext
    )

    if args.validate_only:
        sample = dataset[0]
        points = sample['points']
        voxel_count = len(sample.get('voxel_coords', []))
        logger.info(f"Prepared points: {len(points)}, voxels: {voxel_count}")
        voxel_point_count = int(sample['voxel_num_points'].sum())
        logger.info(
            f"Points entering voxels: {voxel_point_count} ({voxel_point_count / len(points):.1%})")
        logger.info(f"Prepared X: {points[:, 0].min():.2f} ~ {points[:, 0].max():.2f}")
        logger.info(f"Prepared Y: {points[:, 1].min():.2f} ~ {points[:, 1].max():.2f}")
        logger.info(f"Prepared Z: {points[:, 2].min():.2f} ~ {points[:, 2].max():.2f}")
        logger.info(
            f"Prepared intensity: {points[:, 3].min():.4f} ~ {points[:, 3].max():.4f}"
        )
        return
    # The real dataset supplies grid, voxel, and feature metadata to the model.
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset
    )

    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()


    with torch.no_grad():
        sample = dataset[0]
        data_dict = dataset.collate_batch([sample])
        load_data_to_gpu(data_dict)

        # === Debug 输出点云范围 ===
        pts = data_dict['points'][:, 1:5]
        logger.info(f"Point range X: {pts[:,0].min():.2f} ~ {pts[:,0].max():.2f}")
        logger.info(f"Point range Y: {pts[:,1].min():.2f} ~ {pts[:,1].max():.2f}")
        logger.info(f"Point range Z: {pts[:,2].min():.2f} ~ {pts[:,2].max():.2f}")

        pred_dicts, _ = model.forward(data_dict)

        logger.info(f"✅ Inference done. Detected objects: {len(pred_dicts[0]['pred_boxes'])}")

        if len(pred_dicts[0]['pred_boxes']) == 0:
            logger.warning("⚠️ No boxes detected — check scale or z_offset settings.")

        # === 可视化保存 ===
        pred_boxes = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dicts[0]['pred_labels'].detach().cpu().numpy()
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            points=pts.detach().cpu().numpy(),
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
        )
        logger.info(f"Predictions saved: {output_path}")
        if args.visualize and 'pred_boxes' in pred_dicts[0]:
            from pcdet.visualization import visualize_utils as V
            import matplotlib.pyplot as plt
            V.draw_scenes(
                points=pts.cpu().numpy(),
                gt_boxes=None,
                ref_boxes=pred_dicts[0]['pred_boxes'].detach().cpu().numpy(),
                ref_scores=pred_dicts[0]['pred_scores'].detach().cpu().numpy(),
                ref_labels=pred_dicts[0]['pred_labels'].detach().cpu().numpy(),
            )
            out_path = "output/demo_pandar128_result.png"
            plt.savefig(out_path, dpi=300)
            logger.info(f"✅ Visualization saved: {out_path}")

    logger.info('---------------- Done ----------------')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='OpenPCDet PandarXT32 Demo')
    parser.add_argument('--cfg_file', type=str, required=True, help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='point cloud path')
    parser.add_argument('--ext', type=str, default='.bin', help='file extension')
    parser.add_argument('--z_offset', type=float, default=0.0,
                        help='optional vertical calibration in metres; applied once')
    parser.add_argument('--output_path', type=str,
                        default='output/demo_xt32_predictions.npz',
                        help='compressed points and predictions output')
    parser.add_argument('--visualize', action='store_true', help='open the optional Mayavi view')
    parser.add_argument('--validate_only', action='store_true',
                        help='prepare and inspect the input without CUDA inference')
    args = parser.parse_args()

    main()
