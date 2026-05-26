import torch
import torch.nn as nn

class MultimodalFusionModel(nn.Module):
    def __init__(self, speech_model, text_model, num_classes=7, fine_tune=True):
        super(MultimodalFusionModel, self).__init__()
        
        # 1. Store the pre-trained backend architectures
        self.speech_backbone = speech_model
        self.text_backbone = text_model.bert
        
        # Selective Backbone Unfreezing Core
        if fine_tune:
            for param in self.speech_backbone.parameters():
                param.requires_grad = True
            for param in self.text_backbone.parameters():
                param.requires_grad = False
            # Unfreeze only the final BERT encoder layer and the pooler
            for param in self.text_backbone.encoder.layer[-1].parameters():
                param.requires_grad = True
            for param in self.text_backbone.pooler.parameters():
                param.requires_grad = True
        else:
            for param in self.speech_backbone.parameters():
                param.requires_grad = False
            for param in self.text_backbone.parameters():
                param.requires_grad = False
            
        # 2. Linear Projections to map features to a common dimensional space (e.g., 256)
        self.speech_proj = nn.Linear(256, 256)
        self.text_proj = nn.Linear(768, 256)
        
        # 3. 🔥 THE GATING NET: Computes the attention scalar 'g'
        self.gate = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # Restricts output to range [0, 1]
        )
        
        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, speech_features, input_ids, attention_mask):
        # Extract foundational context maps from backbones
        _, _, speech_context, _ = self.speech_backbone(speech_features, return_features=True)
        bert_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_context = bert_outputs.pooler_output
        
        # Map to matching dimensionality vectors
        s_proj = torch.relu(self.speech_proj(speech_context))
        t_proj = torch.relu(self.text_proj(text_context))
        
        # Compute multi-modal gating balance scalar
        combined_proj = torch.cat((s_proj, t_proj), dim=1)
        gate_weight = self.gate(combined_proj)
        
        # 🔥 Dynamic Attentive Gated Cross-Modal Linear Fusion
        fused_vector = gate_weight * s_proj + (1.0 - gate_weight) * t_proj
        
        # Final classification inference mapping
        logits = self.classifier(fused_vector)
        return logits
