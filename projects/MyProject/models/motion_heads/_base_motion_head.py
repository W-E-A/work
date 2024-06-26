import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import scipy
from mmdet3d.registry import MODELS
from ..dense_heads import BaseTaskHead
from ..loss_utils import MotionSegmentationLoss, SpatialRegressionLoss, ProbabilisticLoss, GaussianFocalLoss, SpatialProbabilisticLoss
from ..utils import BevFeatureSlicer
from ...utils import FeatureWarper, predict_instance_segmentation_and_trajectories
from ..modules.motion_modules import ResFuturePrediction, SpatialDistributionModule, DistributionModule


@MODELS.register_module()
class BaseMotionHead(BaseTaskHead):
    def __init__(
        self,
        # settings: temporal
        receptive_field=3,
        n_future=0,
        future_discount=0.95,
        # settings: model
        task_dict={},
        in_channels=256,
        inter_channels=None,
        n_gru_blocks=3,
        grid_conf = None,
        new_grid_conf = None,
        probabilistic_enable=True,
        prob_on_foreground=False,
        future_dim=6,
        prob_latent_dim=32,
        distribution_log_sigmas=None,
        using_spatial_prob=False,
        using_prob_each_future=False,
        detach_state=True,
        # settings: loss
        class_weights=None,
        use_topk=True,
        topk_ratio=0.25,
        ignore_index=255,
        posterior_with_label=False,
        sample_ignore_mode='all_valid',
        using_focal_loss=False,
        focal_cfg=dict(type='mmdet.GaussianFocalLoss', reduction='none'),
        loss_weights=None,
        # settings: others
        train_cfg=None,
        test_cfg=None,
        init_cfg=dict(type='Kaiming', layer='Conv2d'),
        **kwargs,
    ):
        super(BaseMotionHead, self).__init__(
            task_dict, in_channels, inter_channels, init_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_bias = -2.19

        self.in_channels = in_channels
        self.receptive_field = receptive_field
        self.n_future = n_future
        self.probabilistic_enable = probabilistic_enable
        self.posterior_with_label = posterior_with_label
        self.prob_on_foreground = prob_on_foreground
        self.using_spatial_prob = using_spatial_prob
        self.using_prob_each_future = using_prob_each_future

        self.sample_ignore_mode = sample_ignore_mode
        assert self.sample_ignore_mode in ['all_valid', 'past_valid', 'none']

        self.prob_latent_dim = prob_latent_dim if self.probabilistic_enable else 0
        self.future_dim = future_dim
        self.loss_weights = loss_weights
        self.ignore_index = ignore_index
        self.using_focal_loss = using_focal_loss

        self.warper = FeatureWarper(pc_range=new_grid_conf[0])
        self.cropper =  BevFeatureSlicer(grid_conf, new_grid_conf)
        if self.n_future > 0:
            distri_min_log_sigma, distri_max_log_sigma = distribution_log_sigmas
            # 当前的 feature + 未来的 label
            future_distribution_in_channels = self.in_channels + self.n_future * future_dim
            # future_distribution_in_channels = self.in_channels + 4 * future_dim

            if self.probabilistic_enable:
                prob_latent_dim = self.prob_latent_dim * \
                    self.n_future if self.using_prob_each_future else self.prob_latent_dim

                if self.using_spatial_prob:
                    self.present_distribution = SpatialDistributionModule(
                        self.in_channels,
                        prob_latent_dim,
                        min_log_sigma=distri_min_log_sigma,
                        max_log_sigma=distri_max_log_sigma,
                    )

                    self.future_distribution = SpatialDistributionModule(
                        future_distribution_in_channels,
                        prob_latent_dim,
                        min_log_sigma=distri_min_log_sigma,
                        max_log_sigma=distri_max_log_sigma,
                    )
                else:
                    self.present_distribution = DistributionModule(
                        self.in_channels,
                        prob_latent_dim,
                        min_log_sigma=distri_min_log_sigma,
                        max_log_sigma=distri_max_log_sigma,
                    )

                    self.future_distribution = DistributionModule(
                        future_distribution_in_channels,
                        prob_latent_dim,
                        min_log_sigma=distri_min_log_sigma,
                        max_log_sigma=distri_max_log_sigma,
                    )

            # adapt for different modules of FuturePrediction

            self.future_prediction = ResFuturePrediction(
                in_channels=self.in_channels,
                latent_dim=self.prob_latent_dim,
                n_future=self.n_future,
                prob_each_future=self.using_prob_each_future,
                detach_state=detach_state,
                n_gru_blocks=n_gru_blocks,
            )

        # loss functions
        # 1. loss for foreground segmentation
        self.seg_criterion = MotionSegmentationLoss(
            class_weights=torch.tensor(class_weights),
            use_top_k=use_topk,
            top_k_ratio=topk_ratio,
            future_discount=future_discount,
        )

        # 2. loss for instance center heatmap
        self.reg_instance_center_criterion = SpatialRegressionLoss(
            norm=2,
            future_discount=future_discount,
        )

        self.cls_instance_center_criterion = GaussianFocalLoss(
            focal_cfg=focal_cfg,
            ignore_index=ignore_index,
            future_discount=future_discount,
        )

        # 3. loss for instance offset
        self.reg_instance_offset_criterion = SpatialRegressionLoss(
            norm=1,
            future_discount=future_discount,
            ignore_index=ignore_index,
        )

        # 4. loss for instance flow
        self.reg_instance_flow_criterion = SpatialRegressionLoss(
            norm=1,
            future_discount=future_discount,
            ignore_index=ignore_index,
        )

        self.probabilistic_loss = ProbabilisticLoss(
            foreground=self.prob_on_foreground)

    def forward(self, bevfeats, future_distribution_inputs=None, noise=None):
        '''
        the forward process of motion head:
        1. get present & future distributions
        2. iteratively get future states with ConvGRU
        3. decode present & future states with the decoder heads
        '''
        bevfeats = bevfeats[0]

        if self.training:
            assert future_distribution_inputs is not None

        res = {}
        if self.n_future > 0:
            present_state = bevfeats.unsqueeze(dim=1).contiguous()

            # sampling probabilistic distribution
            sample, output_distribution = self.distribution_forward(
                present_state, future_distribution_inputs, noise
            )

            b, _, _, h, w = present_state.shape
            hidden_state = present_state[:, 0]

            future_states = self.future_prediction(sample, hidden_state)
            future_states = torch.cat([present_state, future_states], dim=1)
            # flatten dimensions of (batch, sequence)
            batch, seq = future_states.shape[:2]
            flatten_states = future_states.flatten(0, 1)

            res.update(output_distribution)
            for task_key, task_head in self.task_heads.items():
                res[task_key] = task_head(
                    flatten_states).view(batch, seq, -1, h, w)
        else:
            b, _, h, w = bevfeats.shape
            for task_key, task_head in self.task_heads.items():
                res[task_key] = task_head(bevfeats).view(b, 1, -1, h, w)

        return res

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        if 'instance_center' in self.task_heads:
            self.task_heads['instance_center'][-1].bias.data.fill_(
                self.init_bias)

    def distribution_forward(
        self,
        present_features,
        future_distribution_inputs=None,
        noise=None,
    ):
        """
        Parameters
        ----------
            present_features: 5-D output from dynamics module with shape (b, 1, c, h, w)
            future_distribution_inputs: 5-D tensor containing labels shape (b, s, cfg.PROB_FUTURE_DIM, h, w)
            noise: a sample from a (0, 1) gaussian with shape (b, s, latent_dim). If None, will sample in function

        Returns
        -------
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
            present_distribution_mu: shape (b, s, latent_dim)
            present_distribution_log_sigma: shape (b, s, latent_dim)
            future_distribution_mu: shape (b, s, latent_dim)
            future_distribution_log_sigma: shape (b, s, latent_dim)
        """
        b, s, _, h, w = present_features.size()  # b, 1, 384, 200, 200
        assert s == 1
        # import pdb;pdb.set_trace()
        present_mu, present_log_sigma = self.present_distribution(
            present_features) # b 1 384 200 200

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            # Concatenate future labels to z_t
            future_features = future_distribution_inputs[:, 1:].contiguous().view(
                b, 1, -1, h, w) #(1,5,6,200,200)->(1,1,30,200,200) # B len 1+1+2+2 200, 200 -> B len-1 6 200 200 -> B 1 (len-1)*6 200 200
            future_features = torch.cat(
                [present_features, future_features], dim=2) # B 1 384+(len-1)*6 200 200

            future_mu, future_log_sigma = self.future_distribution(
                future_features)#future_mu (1,32,50,50) future_log_sigma (1,32,50,50)

        # import pdb

        if noise is None:
            if self.training:
                noise = torch.randn_like(present_mu)
            else:
                noise = torch.zeros_like(present_mu)
                # noise = 0.5 * torch.ones_like(present_mu)

        # 在测试模式下使用 label 生成 sample时，未来预测的结果很差
        if self.training or self.posterior_with_label:
            mu = future_mu # B 1 dim
            sigma = torch.exp(future_log_sigma) # B 1 dim
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)

        # pdb.set_trace()
        # sample = mu + sigma * noise
        # sample = mu
        # sample = mu + sigma * noise

        # http://www.sefidian.com/2021/12/04/steps-to-sample-from-a-multivariate-gaussian-normal-distribution-with-python-code/

        if not self.training:
            def pdf_multivariate_gauss(x, mu, cov):
                '''
                Caculate the multivariate normal density (pdf)

                Keyword arguments:
                    x = numpy array of a "d x 1" sample vector
                    mu = numpy array of a "d x 1" mean vector
                    cov = "numpy array of a d x d" covariance matrix
                '''
                assert (mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
                assert (x.shape[0] > x.shape[1]), 'x must be a row vector'
                assert (cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
                assert (mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
                assert (mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
                # part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
                # part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
                cov_det = np.prod(cov)

                part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
                part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))

                return float(part1 * np.exp(part2))

            n = 5 # number of samples (+1 for mean sample)
            np.random.seed(7)

            samples = [mu + 1.0 * sigma * torch.randn_like(mu) for _ in range(n + 1)]

            samples[-1] = mu

            for i, sample in enumerate(samples):
                # upsampling sample to the spatial size of features
                if self.using_spatial_prob:
                    samples[i] = F.interpolate(
                        sample, size=present_features.shape[-2:],
                        mode='bilinear')  # (1, 32, 50, 50) -> torch.Size([1, 32, 200, 200])
                else:
                    samples[i] = sample.view(b, -1, 1, 1).expand(b, -1, h, w)  # (1, 32, 1, 1) -> (1, 32, 200, 200)
        else:
            samples = mu + sigma * noise # 训练的时候从未来采样，同时训练对齐当前生成器生成和未来生成器的生成结果分布靠近（KL损失）
            # upsampling sample to the spatial size of features
            if self.using_spatial_prob:
                samples = F.interpolate(
                    samples, size=present_features.shape[-2:], mode='bilinear') # B 1 dim 200 200
            else:
                samples = samples.view(b, -1, 1, 1).expand(b, -1, h, w)

        output_distribution = {
            'present_mu': present_mu, # 当前采样 # B 1 dim
            'present_log_sigma': present_log_sigma, # 当前采样 # B 1 dim
            'future_mu': future_mu, # 测试时没有 # B 1 dim
            'future_log_sigma': future_log_sigma, # 测试时没有 # B 1 dim
        }

        return samples, output_distribution # B 1 dim 200 200 and dict

    def prepare_future_labels(self, batch, label_sampling_mode='nearest', center_sampling_mode='nearest'):
        labels = {}
        future_distribution_inputs = []
        segmentation_labels = torch.stack(batch['motion_segmentation']) # B, len, H, W
        instance_center_labels = torch.stack(batch['instance_centerness']) # B, len, H, W
        instance_offset_labels = torch.stack(batch['instance_offset']) # B, len, 2, H, W
        instance_flow_labels = torch.stack(batch['instance_flow']) # B, len, 2, H, W
        gt_instance = torch.stack(batch['motion_instance']) # B, len, H, W
        future_egomotion = torch.stack(batch['future_egomotion']) # B, len, 4, 4
        bev_transform = batch.get('aug_transform', None)

        if bev_transform is not None:
            bev_transform = bev_transform.float()

        segmentation_labels = self.warper.cumulative_warp_features_reverse(
            segmentation_labels.float().unsqueeze(2), # B, len, 1, H, W
            future_egomotion,
            mode=label_sampling_mode, bev_transform=bev_transform,
        ).long().contiguous() # to long(0 or 1)
        labels['segmentation'] = segmentation_labels  # B, len, 1, H, W
        future_distribution_inputs.append(segmentation_labels)

        # Warp instance labels to present's reference frame
        gt_instance = self.warper.cumulative_warp_features_reverse(
            gt_instance.float().unsqueeze(2), # B, len, 1, H, W
            future_egomotion,
            mode=label_sampling_mode, bev_transform=bev_transform,
        ).long().contiguous()[:, :, 0] # to long(0, 1~254, 255) # B, len, H, W
        labels['instance'] = gt_instance

        instance_center_labels = self.warper.cumulative_warp_features_reverse(
            instance_center_labels,
            future_egomotion,
            mode=center_sampling_mode, bev_transform=bev_transform,
        ).contiguous()
        labels['centerness'] = instance_center_labels # B, len, 1, H, W

        instance_offset_labels = self.warper.cumulative_warp_features_reverse(
            instance_offset_labels,
            future_egomotion,
            mode=label_sampling_mode, bev_transform=bev_transform,
        ).contiguous()
        labels['offset'] = instance_offset_labels # B, len, 2, H, W

        future_distribution_inputs.append(instance_center_labels)
        future_distribution_inputs.append(instance_offset_labels)

        instance_flow_labels = self.warper.cumulative_warp_features_reverse(
            instance_flow_labels,
            future_egomotion,
            mode=label_sampling_mode, bev_transform=bev_transform,
        ).contiguous()
        labels['flow'] = instance_flow_labels # B, len, 2, H, W
        future_distribution_inputs.append(instance_flow_labels)

        if len(future_distribution_inputs) > 0:
            future_distribution_inputs = torch.cat(
                future_distribution_inputs, dim=2) #  # B, len, 1+1+2+2, H, W

        return labels, future_distribution_inputs # dict tensor B len 1+1+2+2 200, 200

    def loss_by_feat(self, predictions: dict, training_labels: dict = None):
        # ['instance', 'segmentation', 'flow', 'centerness', 'offset'] training_labels
        # ['segmentation', 'instance_flow', 'instance_center', 'instance_offset'] predictions 2 2 1 2
        loss_dict = {}

        '''
        prediction dict:
            'segmentation': 2,
            'instance_center': 1,
            'instance_offset': 2,
            'instance_flow': 2,
        '''
        assert training_labels is not None
        for key, val in training_labels.items():
            training_labels[key] = val.float() # seg 01 to float others remain unchanged(float)
        # import pdb;pdb.set_trace()
        frame_valid_mask = torch.tensor([True] * (self.receptive_field + self.n_future)).unsqueeze(0) # all valid 1, len
        frame_valid_mask = frame_valid_mask.repeat(training_labels['segmentation'].shape[0], 1).bool() # B, len
        past_valid_mask = frame_valid_mask[:, :self.receptive_field]
        future_frame_mask = frame_valid_mask[:, (self.receptive_field - 1):]

        if self.sample_ignore_mode == 'all_valid':
            # only supervise when all 7 frames are valid
            batch_valid_mask = frame_valid_mask.all(dim=1)
            future_frame_mask[~batch_valid_mask] = False
            prob_valid_mask = batch_valid_mask

        elif self.sample_ignore_mode == 'past_valid':
            # only supervise when past 3 frames are valid
            past_valid = torch.all(past_valid_mask, dim=1)
            future_frame_mask[~past_valid] = False
            prob_valid_mask = past_valid

        elif self.sample_ignore_mode == 'none':
            prob_valid_mask = frame_valid_mask.any(dim=1)

        # segmentation
        loss_dict['loss_motion_seg'] = self.seg_criterion(
            predictions['segmentation'], training_labels['segmentation'].long(), # again to long
            frame_mask=future_frame_mask,
        ) # B, len, 2, 200, 200  B, len, 1, 200, 200, onehot

        # instance centerness, but why not focal loss
        if self.using_focal_loss:
            loss_dict['loss_motion_centerness'] = self.cls_instance_center_criterion(
                predictions['instance_center'], training_labels['centerness'],
                frame_mask=future_frame_mask,
            ) # B, len, 1, 200, 200  B, len, 1, 200, 200
        else:
            loss_dict['loss_motion_centerness'] = self.reg_instance_center_criterion(
                predictions['instance_center'], training_labels['centerness'],
                frame_mask=future_frame_mask,
            ) # B, len, 1, 200, 200  B, len, 1, 200, 200

        # instance offset
        loss_dict['loss_motion_offset'] = self.reg_instance_offset_criterion(
            predictions['instance_offset'], training_labels['offset'],
            frame_mask=future_frame_mask,
        ) # B, len, 2, 200, 200  B, len, 2, 200, 200

        if self.n_future > 0:
            # instance flow
            loss_dict['loss_motion_flow'] = self.reg_instance_flow_criterion(
                predictions['instance_flow'], training_labels['flow'],
                frame_mask=future_frame_mask,
            ) # B, len, 2, 200, 200  B, len, 2, 200, 200

            if self.probabilistic_enable:
                loss_dict['loss_motion_prob'] = self.probabilistic_loss(
                    predictions, # present future / mu sigma
                    foreground_mask=training_labels['segmentation'],
                    batch_valid_mask=prob_valid_mask,
                ) # ??? FIXME

        for key in loss_dict:
            loss_dict[key] *= self.loss_weights.get(key, 1.0)

        return loss_dict

    def predict_by_feat(self, predictions: dict):
        # ['segmentation', 'instance_flow', 'instance_center', 'instance_offset'] 2 2 1 2
        # output future seg and ins-seg, not traj
        seg_prediction = torch.argmax(
            predictions['segmentation'], dim=2, keepdims=True) # B, len, 1, H, W

        if self.using_focal_loss:
            predictions['instance_center'] = torch.sigmoid(
                predictions['instance_center']) # B, len, 1, H, W

        pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            predictions, compute_matched_centers=False,
        )

        return seg_prediction, pred_consistent_instance_seg
