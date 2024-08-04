import torch
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

img_meta = dict(img_shape=(3, 4, 4),
                 pad_shape=(3, 4, 4),
                 inventado=(3, 4, 4))

data_sample = SegDataSample()

# defining gt_segmentations for encapsulate the ground truth data
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))

# add and process property in SegDataSample
data_sample.gt_sem_seg = gt_segmentations


assert 'gt_sem_seg' in data_sample
#assert 'sem_seg' in data_sample.gt_sem_seg
assert 'img_shape' in data_sample.gt_sem_seg.metainfo_keys()

print(data_sample.gt_sem_seg.metainfo_keys())
print(data_sample.gt_sem_seg.data.shape)


# delete and change property in SegDataSample
data_sample = SegDataSample()
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
data_sample.gt_sem_seg = gt_segmentations
data_sample.gt_sem_seg.set_metainfo(dict(img_shape=(4,4,9), pad_shape=(4,4,9)))

print(data_sample.gt_sem_seg.metainfo_keys())
print(data_sample.gt_sem_seg.img_shape)

# Tensor-like operations
data_sample = SegDataSample()
gt_segmentations = PixelData(metainfo=img_meta)
gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))

#cuda_gt_segmentations = gt_segmentations.cuda()
#cuda_gt_segmentations = gt_segmentations.to('cuda:0')
#cpu_gt_segmentations = cuda_gt_segmentations.cpu()
#cpu_gt_segmentations = cuda_gt_segmentations.to('cpu')