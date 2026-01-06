import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from util.misc import inverse_sigmoid

class RoIRefinementModule(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", num_encoder_layers=1, num_decoder_layers=1, roi_size=7, num_classes=91):
        super().__init__()
        self.d_model = d_model
        self.roi_size = roi_size
        
        # Standard Transformer Encoder/Decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Positional Embedding for the RoI grid
        self.row_embed = nn.Parameter(torch.rand(roi_size, d_model // 2))
        self.col_embed = nn.Parameter(torch.rand(roi_size, d_model // 2))
        
        # Prediction heads
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = nn.Linear(d_model, 4)
        
        # Initialize
        nn.init.constant_(self.bbox_embed.bias.data, 0)
        nn.init.uniform_(self.row_embed.data)
        nn.init.uniform_(self.col_embed.data)

    def get_pos_embed(self, batch_size):
        # Generate 2D pos embed for roi_size x roi_size
        pos = torch.cat([
            self.col_embed.unsqueeze(0).repeat(self.roi_size, 1, 1),
            self.row_embed.unsqueeze(1).repeat(1, self.roi_size, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1) # (H*W, 1, C)
        return pos.repeat(1, batch_size, 1) # (H*W, B, C)

    def forward(self, feature_map, boxes, query_embeds):
        """
        feature_map: [B, C, H, W] - The feature map to crop from (e.g. level 0)
        boxes: [B, K, 4] - (cx, cy, w, h) normalized [0, 1]
        query_embeds: [B, K, C] - The query embeddings to refine
        """
        B, C, H, W = feature_map.shape
        K = boxes.shape[1]
        
        # 1. Prepare Boxes for RoI Align
        # Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
        # And scale to feature map size
        boxes_x1y1 = boxes[..., :2] - boxes[..., 2:] / 2
        boxes_x2y2 = boxes[..., :2] + boxes[..., 2:] / 2
        boxes_x1y1 = boxes_x1y1 * torch.tensor([W, H], device=boxes.device)
        boxes_x2y2 = boxes_x2y2 * torch.tensor([W, H], device=boxes.device)
        
        rois = torch.cat([boxes_x1y1, boxes_x2y2], dim=-1) # [B, K, 4]
        
        # Flatten for roi_align: List of [K, 4] or single tensor with batch index
        # torchvision roi_align supports list of tensors
        rois_list = [rois[i] for i in range(B)]
        
        # 2. RoI Align
        # Output: [B*K, C, roi_size, roi_size]
        roi_features = roi_align(feature_map, rois_list, output_size=(self.roi_size, self.roi_size), spatial_scale=1.0, sampling_ratio=-1)
        
        # 3. Prepare for Transformer
        # Flatten spatial dims: [B*K, C, H*W] -> [H*W, B*K, C]
        src = roi_features.flatten(2).permute(2, 0, 1)
        
        # Positional Embeddings
        pos = self.get_pos_embed(B*K).to(src.device)
        
        # 4. Encoder
        memory = self.encoder(src + pos)
        
        # 5. Decoder
        # tgt: query_embeds [B, K, C] -> [1, B*K, C] (treating each query as a sequence of length 1)
        # Actually, we process each query independently with its own RoI.
        # So tgt is [1, B*K, C]
        tgt = query_embeds.flatten(0, 1).unsqueeze(0)
        
        # We don't need a mask because each query only attends to its own RoI memory
        hs = self.decoder(tgt, memory) # [1, B*K, C]
        
        hs = hs.squeeze(0) # [B*K, C]
        
        # 6. Prediction
        class_logits = self.class_embed(hs) # [B*K, num_classes]
        bbox_deltas = self.bbox_embed(hs)   # [B*K, 4]
        
        # Reshape back to [B, K, ...]
        class_logits = class_logits.view(B, K, -1)
        bbox_deltas = bbox_deltas.view(B, K, -1)
        
        return class_logits, bbox_deltas
