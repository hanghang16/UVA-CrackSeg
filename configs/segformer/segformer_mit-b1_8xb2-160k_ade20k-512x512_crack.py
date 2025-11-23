_base_ = ['./segformer_mit-b0_8xb2-160k_ade20k-512x512_crack.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa
norm_cfg = dict(type='BN', requires_grad=True) # 只使用GPU时，BN取代SyncBN
#model settings
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[2, 2, 2, 2]),

    decode_head=dict(in_channels=[64, 128, 320, 512]))



#python tools/train.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512_crack.py --work-dir work_dirs/segformer_b1_SimAM_test_11.15

#python tools/train.py configs/segformer/segformer_mit-b1_8xb2-160k_ade20k-512x512_crack.py --work-dir work_dirs/segformer_b1_DiceLoss_test_11.10

#python tools/test.py work_dirs/segformer_b1_SimAM_shujuzengqiang_test_11.11/segformer_mit-b1_8xb2-160k_ade20k-512x512_crack.py work_dirs/segformer_b1_SimAM_shujuzengqiang_test_11.11/best_mIoU_iter_13500.pth --out work_dirs/segformer_b1_SimAM_shujuzengqiang_test_11.11/exp1

#python tools/test.py work_dirs/segformer_b1_SimAM_SE_test_11.13/segformer_mit-b1_8xb2-160k_ade20k-512x512_crack.py work_dirs/segformer_b1_SimAM_SE_test_11.13/best_mIoU_iter_8800.pth --out work_dirs/segformer_b1_SimAM_SE_test_11.13/test_masks

#python tools/train.py configs/segformer/segformer_mit-b1_8xb1-160k_cityscapes-800x800.py --resume