import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter


__all__ = ['FRCNN','RCNN_global']

# @ readapted from 'Concept Bottleneck Model With Additional Unsupervised Concepts' by Sawada and Nakamura, IEEE (2022)
class RCNN_global(nn.Module):

    # & Ok
    def __init__(self, cfg=None, random_select=False):
        super(RCNN_global, self).__init__()
        
        # ^ load RCNN for object detection (bdd100k_24.pth)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)#,trainable_backbone_layers=0)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,12)
        
        ckpt_path = 'bdd_models/bdd100k_24.pth'
        current_folder = os.getcwd()
        if current_folder.endswith('notebooks'):
            ckpt_path = '../bdd_models/bdd100k_24.pth'
        self.checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(self.checkpoint['model'])#['state_dict']) #self.checkpoint['state_dict']) #['model'])

        # ^ layers for global feature extraction
        self.avgpool_glob = nn.AdaptiveAvgPool2d(output_size=7)
        self.conv_glob1 = nn.Conv2d(256, 128, 3, padding=1)
        self.relu_glob1 = nn.ReLU(inplace=True)
        self.conv_glob2 = nn.Conv2d(128, 64, 3, padding=1)
        self.relu_glob2 = nn.ReLU(inplace=True)
        self.lin_glob=nn.Linear(in_features=3136, out_features=2048,bias=True)
        self.relu_glob=nn.ReLU()
        
        total_params  = sum(param.numel() for param in self.conv_glob1.parameters())
        total_params += sum(param.numel() for param in self.conv_glob2.parameters())
        total_params += sum(param.numel() for param in self.lin_glob.parameters())
        print(f"Total parameters in global feature extractor: {total_params}")
        
    # & Ok
    def forward(self, images, my_roi=None):
        """
        images: Tensor [B, 3, H, W]
        Returns:
            scene_feats: Tensor [B, 2048]
            rois:        Tensor [N, 5](batch_idx, x1, y1, x2, y2)
            scores:      Tensor [N]
            labels:      Tensor [N]
            bbox_feats:  Tensor [N, 1024]
        """
        B, _, H, W = images.shape
        self.model.eval()
        for name, param in self.model.named_parameters():   # fix RCNN
            param.requires_grad = False

        # ^ scene feature extraction
        outputs = []
        hook = self.model.backbone.register_forward_hook(
            lambda self, input, output: outputs.append(output))
        L = len(images)
        imax = torch.max(images)
        imin = torch.min(images)
        input = torch.rand((L, 3, 749, 1333), device=images.device) *(imax-imin) + imin
        input[:, :, :720, :1280] = images
        res1 = self.model(input) 
        hook.remove()

        x = outputs[0]['0']
        x = self.relu_glob1(self.conv_glob1(x))
        x = self.relu_glob2(self.conv_glob2(x))
        x = self.avgpool_glob(x)
        x = x.flatten(start_dim=1)
        scene_feats_raw = self.lin_glob(x)
        scene_feats = self.relu_glob(scene_feats_raw)
        
        # ^ all ROI feature extraction
        with torch.no_grad():   dets = self.model(images)
        all_rois, all_scores, all_labels, batch_idx = [], [], [], []
        for b, d in enumerate(dets):
            n = d['boxes'].shape[0]
            if n == 0: continue
            all_rois.append(d['boxes'])
            all_scores.append(d['scores'])
            all_labels.append(d['labels'])
            batch_idx.append(torch.full((n,1), b, device=images.device))

        rois   = torch.cat(all_rois, 0)             # [N,4]
        b_idx  = torch.cat(batch_idx, 0)            # [N,1]
        rois   = torch.cat([b_idx, rois], dim=1)    # [N,5]
        scores = torch.cat(all_scores, 0)           # [N]
        labels = torch.cat(all_labels, 0)           # [N]

        # re‐run backbone(FPN) for ROI pooling using feature map fpn_feats with shape [N,256,7,7]
        fpn_feats = self.model.backbone(images)
        pooled = self.model.roi_heads.box_roi_pool(fpn_feats, all_rois, [(H,W)] * B) 
        # manually run the heads *without* the final ReLU
        flat = pooled.flatten(start_dim=1)     # [N, 256*7*7]
        x6 = F.relu(self.model.roi_heads.box_head.fc6(flat))
        bbox_feats = self.model.roi_heads.box_head.fc7(x6)  
        
        # ^ my_roi feature extraction
        if my_roi is not None:
            pooled_m = self.model.roi_heads.box_roi_pool(fpn_feats, [my_roi] ,[(H,W)] * B) 
            flat_m   = pooled_m.flatten(start_dim=1)
            x6_m     = F.relu(self.model.roi_heads.box_head.fc6(flat_m))
            my_bbox_feats    = self.model.roi_heads.box_head.fc7(x6_m)
            
        return scene_feats, scene_feats_raw, rois, scores, labels, bbox_feats, my_bbox_feats if my_roi is not None else None