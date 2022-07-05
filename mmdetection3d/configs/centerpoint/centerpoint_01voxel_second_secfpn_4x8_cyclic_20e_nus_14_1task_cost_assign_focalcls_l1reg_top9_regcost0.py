_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/models/centerpoint_01voxel_second_secfpn_nus.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]
voxel_size = [0.1, 0.1, 0.2]
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),

    pts_bbox_head=dict(
        type='CenterHead_cost_assign_1task_focalcls_l1reg',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=10, class_names=class_names),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9,
            pc_range=point_cloud_range[:2]),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        # loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            selectable_k=9,
            clss_weight = 1,
            regg_weight = 0,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            point_cloud_range=point_cloud_range,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range[:2],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))


dataset_type = 'NuScenesDataset'
data_root = '/ghome/changjh/detr3d/mmdetection3d/data/nuscenes/'
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            load_interval=4)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

evaluation = dict(interval=2, pipeline=eval_pipeline)
