# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch import nn

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, boxes_iou_bev
from mmdet.core import build_bbox_coder, multi_apply, reduce_mean
from mmdet.core.bbox import bbox_overlaps

from mmdet3d.core.bbox.structures.box_3d_mode import (Box3DMode, CameraInstance3DBoxes,
                              DepthInstance3DBoxes, LiDARInstance3DBoxes)
from mmdet3d.models.dense_heads import SeparateHead



@HEADS.register_module()
class CenterHead_cost_assign_1task_3detr(BaseModule):
    """CenterHead for CenterPoint.

    Args:
        in_channels (list[int] | int, optional): Channels of the input
            feature map. Default: [128].
        tasks (list[dict], optional): Task information including class number
            and class names. Default: None.
        train_cfg (dict, optional): Train-time configs. Default: None.
        test_cfg (dict, optional): Test-time configs. Default: None.
        bbox_coder (dict, optional): Bbox coder configs. Default: None.
        common_heads (dict, optional): Conv information for common heads.
            Default: dict().
        loss_cls (dict, optional): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict, optional): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict, optional): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int, optional): Output channels for share_conv
            layer. Default: 64.
        num_heatmap_convs (int, optional): Number of conv layers for heatmap
            conv layer. Default: 2.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(CenterHead_cost_assign_1task_3detr, self).__init__(init_cfg=init_cfg)

        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.fp16_enabled = False

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(builder.build_head(separate_head))

    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        # print(len(feats), feats[0].shape)
        return multi_apply(self.forward_single, feats)

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor, optional): Mask of the feature map with the
                shape of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dicts):
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which
                        boxes are valid.
        """

        batch_size = preds_dicts[0][0]['heatmap'].shape[0]
        preds_dicts_batch = []
        for i in range(batch_size):
            dict1 = {}
            for key in preds_dicts[0][0]:
                value = []
                for j in range(len(preds_dicts)):
                    value.append(preds_dicts[j][0][key][i])
                dict1[key] = value
            preds_dicts_batch.append(dict1)
            i += 1

        loss_cls_list, loss_bbox_list, pos_inds_list = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_dicts_batch)

        device = preds_dicts[0][0]['heatmap'].device
        num_total_pos = sum([max(inds, 1) for inds in pos_inds_list])
        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        print(num_total_samples)
        # Transpose heatmaps  从按图片区分变为按task分

        return sum(loss_cls_list) / num_total_samples, sum(loss_bbox_list) / num_total_samples

    def cls_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): The prediction with shape (num_query, 1, *) or
                (num_query, *).
            gt_labels (Tensor): The learning label of prediction with
                shape (num_gt, *).

        Returns:
            Tensor: Cross entropy cost matrix in shape (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction='none')
        cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + \
            torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost = cls_cost / n

        return cls_cost


    def focal_cls_cost(self, cls_pred, gt_labels):

        weight=1.
        alpha=0.25
        gamma=2
        eps=1e-12
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
        pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * weight


    def bbox_cost(self, bbox_pred, gt_bboxes):
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost

    def iou_bbox_cost(self, bboxes1,
                             bboxes2,
                             mode='giou',
                             is_aligned=False,
                             coordinate='lidar'):
    
        assert bboxes1.size(-1) == bboxes2.size(-1) >= 7
        # 只计算bev的iou
        bboxes1[:,3] = bboxes1[:,3] / self.train_cfg['out_size_factor'] / self.train_cfg['voxel_size'][0]
        bboxes1[:,4] = bboxes1[:,4] / self.train_cfg['out_size_factor'] / self.train_cfg['voxel_size'][1]
        bboxes2[:,3] = bboxes2[:,3] / self.train_cfg['out_size_factor'] / self.train_cfg['voxel_size'][0]
        bboxes2[:,4] = bboxes2[:,4] / self.train_cfg['out_size_factor'] / self.train_cfg['voxel_size'][1]

        bboxes1 = LiDARInstance3DBoxes(bboxes1, box_dim=bboxes1.shape[-1], origin=(0.5, 0.5, 0.5))
        bboxes2 = LiDARInstance3DBoxes(bboxes2, box_dim=bboxes2.shape[-1], origin=(0.5, 0.5, 0.5))

        # Change the bboxes to bev
        # box conversion and iou calculation in torch version on CUDA
        # is 10x faster than that in numpy version

        # bboxes1_bev = bboxes1.bev
        # bboxes2_bev = bboxes2.bev
        # ret = boxes_iou_bev(bboxes1_bev, bboxes2_bev)   # 带旋转角度的iou

        bboxes1_bev = bboxes1.nearest_bev
        bboxes2_bev = bboxes2.nearest_bev
        ret = bbox_overlaps(
            bboxes1_bev, bboxes2_bev, mode=mode, is_aligned=is_aligned)
        
        
        # ret = bboxes1.overlaps(bboxes1, bboxes2, mode=mode)
        return -ret


    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dicts_a_batch):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
        """
        # ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        # print([ i.shape for i in preds_dicts_a_batch['heatmap']])
        
        selectable_k = 9  # 注意这里的k
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),         # 中心点
            dim=1).to(device)
        
        # gt_bboxes_3d_decode = gt_bboxes_3d.tensor.to(device)         # 底部中心点

        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        
        cls_costs, bbox_costs = [], [] 
        for idx, task_head in enumerate(self.task_heads):
            cls_costs.append(preds_dicts_a_batch['heatmap'][idx])
            bbox_costs = [ preds_dicts_a_batch['reg'][idx], preds_dicts_a_batch['height'][idx], preds_dicts_a_batch['dim'][idx],\
                              preds_dicts_a_batch['rot'][idx], preds_dicts_a_batch['vel'][idx] ]
        cls_pred = torch.cat(cls_costs, dim=0).permute(1,2,0).reshape(-1,10)   # 10为类别数
        cls_costs = cls_pred.clone().detach()  
        reg_pred = torch.cat(bbox_costs, dim=0).permute(1,2,0)    # 10为
        bbox_costs = reg_pred.clone().detach() 

    
        # 编码
        H = feature_map_size[0]
        W = feature_map_size[1]
        num_bboxes = H * W
        grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=bbox_costs.dtype, device=device),
                torch.linspace(
                    0, W - 1, W, dtype=bbox_costs.dtype, device=device))
        bbox_costs[:, :, 0] += grid_x
        bbox_costs[:, :, 1] += grid_y
        
        # bbox_costs_decode = bbox_costs.new_zeros((bbox_costs.shape[0], bbox_costs.shape[1], 7))
        # bbox_costs_decode[:, :, 0:2] = bbox_costs[:, :, 0:2]
        bbox_costs = bbox_costs.reshape(-1,10)

        #解码到3d lidar格式(xyz仍是降采样后的结果) rot = torch.atan2(rot_sine, rot_cosine)  torch.exp(preds_dict[0]['dim'])
        bbox_costs[:, 3:6] = torch.exp(bbox_costs[:, 3:6])
        bbox_costs[:, 6] = torch.atan2(bbox_costs[:, 6], bbox_costs[:, 7])
        bbox_costs = bbox_costs[:, :7]

        all_points_x, all_points_y = grid_x.reshape(-1), grid_y.reshape(-1)


        # cls_costs = self.cls_cost(cls_costs, F.one_hot(gt_labels_3d, 10)) # detr式的cost
        
        # 注意task_masks改变了gt的顺序
        # cls_costs = -torch.log(torch.sigmoid(cls_costs[:, task_classes[0]]))
        cls_costs = self.focal_cls_cost(cls_costs, task_classes[0])
    
        for idx, task_head in enumerate(self.task_heads):
            # ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            # mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)
            anno_boxes = gt_bboxes_3d.new_zeros((num_objs, 10),
                                              dtype=torch.float32)

            anno_boxes_3d = gt_bboxes_3d.new_zeros((num_objs, 7),
                                              dtype=torch.float32)                    

            for k in range(num_objs):
                cls_id = task_classes[idx][k]

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]     # 这里是否写反了？
                # if idx == 0:
                #     print(width,length)
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                   
                    # TODO: support other outdoor dataset
                    vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]

                    anno_boxes_3d[k] = torch.cat([
                        center,                              
                        z.unsqueeze(0), box_dim,
                        rot.unsqueeze(0)
                    ])

                    if self.norm_bbox:
                        box_dim = box_dim.log()        # 这里变化
                    anno_boxes[k] = torch.cat([
                        center,                              
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        vx.unsqueeze(0),
                        vy.unsqueeze(0)
                    ])

                    
        # print(anno_boxes[0:5,0:10])

        # bbox_costs = torch.cdist(bbox_costs, anno_boxes, p=1)
        bbox_costs = self.iou_bbox_cost(bbox_costs, anno_boxes_3d)
        overlaps = - (cls_costs + bbox_costs)


        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long,
                                             device=device)
        _, topk_idxs = overlaps.topk(selectable_k, dim=0)
        candidate_idxs = topk_idxs

    
        # 限制pred的点在gt框内
        # candidate_points = all_points[topk_idxs.t().reshape(-1)]
        # is_in_gts_matrix = LiDARInstance3DBoxes(anno_boxes_3d, box_dim=anno_boxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)).points_in_boxes_all(candidate_points)
        # is_in_gts_matrix_spilt = torch.split(is_in_gts_matrix, selectable_k, dim=0)
        # is_in_gts = []
        # for i, mat in enumerate(is_in_gts_matrix_spilt):
        #     is_in_gts.append(mat[:, i])
        # print(torch.nonzero(torch.stack(is_in_gts, dim=-1)>0))
        # is_in_gts = torch.stack(is_in_gts, dim=-1) == 1


        for gt_idx in range(num_objs):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        
        # 限制anchor点在BEV框内
        bboxes_cx, bboxes_cy = all_points_x, all_points_y

        ep_bboxes_cx = bboxes_cx.reshape(1, -1).expand(
            num_objs, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.reshape(1, -1).expand(
            num_objs, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.reshape(-1)


        # 将l w根据voxel_size和stride变换
        anno_boxes_3d[:,3] = anno_boxes_3d[:,3] / self.train_cfg['out_size_factor'] / self.train_cfg['voxel_size'][0]
        anno_boxes_3d[:,4] = anno_boxes_3d[:,4] / self.train_cfg['out_size_factor'] / self.train_cfg['voxel_size'][1]
        gt_bboxes = LiDARInstance3DBoxes(anno_boxes_3d, box_dim=anno_boxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)).enlarge_bev()
        # print('中心点坐标', bboxes_cx[candidate_idxs[0]], bboxes_cy[candidate_idxs[0]])
        # print(gt_bboxes[0])
        # print(cls_costs[candidate_idxs[0]][0], bbox_costs[candidate_idxs[0]][0])


        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].reshape(-1, num_objs) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].reshape(-1, num_objs) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].reshape(-1, num_objs)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].reshape(-1, num_objs)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.001
        # print(is_in_gts.shape)
        # print(torch.nonzero(is_in_gts, as_tuple=False).squeeze().shape[0]/(is_in_gts.shape[0]*is_in_gts.shape[1]))
        # print('================')


        INF = 9999
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_in_gts.view(-1)]
        # index = candidate_idxs.view(-1)
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_objs, -1).t()
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        # print(argmax_overlaps.shape, max_overlaps[:20])
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            # print('----------------------')
            num_pos = pos_inds.numel()
            assigned_labels[pos_inds] = task_classes[0][             # task_classes[0]  gt_labels_3d
                assigned_gt_inds[pos_inds] - 1]

        else:
            num_pos = 0
        labels = gt_labels_3d.new_full((num_bboxes, ),
                                  10,
                                  dtype=torch.long)
        labels[pos_inds] = assigned_labels[pos_inds]
        loss_cls = self.loss_cls(cls_pred, labels, avg_factor=1.)     # 这里avg_factor要修改
        
        code_weights = self.train_cfg.get('code_weights', None)
        reg_pred = reg_pred.reshape(-1,10)

        bbox_weights = torch.zeros_like(reg_pred)
        bbox_targets = torch.zeros_like(reg_pred)
        bbox_targets[pos_inds, :] = anno_boxes[assigned_gt_inds[pos_inds] - 1, :]
        bbox_weights[pos_inds, :] = bbox_weights.new_tensor(code_weights)
        bbox_targets[pos_inds, 0] = bbox_targets[pos_inds, 0] - pos_inds % W
        bbox_targets[pos_inds, 1] = bbox_targets[pos_inds, 1] - torch.floor(pos_inds / W)
        # print(bbox_targets[pos_inds[0:6]])
        # print(reg_pred[pos_inds[0:6]])
        # print('================================')

        loss_bbox = self.loss_bbox(reg_pred, bbox_targets, bbox_weights, avg_factor = 1.)
        return loss_cls, loss_bbox, num_pos
        

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """

        
        cls_loss, bbox_loss = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, preds_dicts)

        loss_dict = dict()

        for task_id, preds_dict in enumerate(preds_dicts):
            
            loss_dict[f'task{task_id}.loss_heatmap'] = cls_loss
            loss_dict[f'task{task_id}.loss_bbox'] = bbox_loss
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            # Apply NMS in birdeye view

            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_gpu(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.test_cfg['nms_thr'],
                    pre_max_size=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
