import torch  # Deep learning framework
import torch.nn as nn
import timm  # For Image models like ResNet
from transformers import AutoModel  # Pretrained text encoders


class MediLLMModel(nn.Module):
    def __init__(
        self,
        text_model_name=("emilyalsentzer/Bio_ClinicalBERT"),
        # Bio_ClinicalBERT is a pretrained model on clinical notes,
        # output to 3 classes i.e triage levels
        num_classes=3,
        dropout=0.3,
        hidden_dim=256,
        mode="multimodal",
    ):
        super(MediLLMModel, self).__init__()  # Use constructor of nn.Module
        assert mode in [
            "text",
            "image",
            "multimodal",
        ], "Mode must be one of: 'text', 'image', or 'multimodal'"
        self.mode = mode.lower()

        # Text encoder: Bio_ClinicalBERT
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name
        )  # Automodel returns base model without a classification head,
        # just embeddings
        self.text_hidden_size = self.text_encoder.config.hidden_size
        # Dimensionality of hidden states i.e embedding vector size returned by
        # the text_encoder for each token, 768 for Bert models

        # Image encoder: ResNet-50 via TIMM
        """
        Bottle neck block used in ResNet-50
        Input x
          ↓
        Conv1x1 → BN → ReLU   → (reduce size)
        Conv3x3 → BN → ReLU   → (main conv)
        Conv1x1 → BN          → (restore size)
          ↓
        + Add skip (possibly Conv-adjusted x)
          ↓
        ReLU
        """
        # Image encoder
        self.image_encoder = timm.create_model(
            "resnet50", pretrained=True, num_classes=0
        )  # Model for images ResNet-50 which means 50 layers.
        # Initial Conv + pooling layer (size reduction),
        # 16 residual blocks * 3 layers per block = 48 layers,
        # Final Fully-connected layer.
        # Each block has 3 conv layers + skip connection,
        # i.e. input is fed into the next block with the output (F(x) + x)
        self.image_hidden_size = (
            self.image_encoder.num_features
        )  # Size of ResNet output ---> 2048 for ResNet-50

        # --- Determine fusion dimension based on the mode ---
        if self.mode == "text":
            fusion_dim = self.text_hidden_size
        elif self.mode == "image":
            fusion_dim = self.image_hidden_size
        else:
            fusion_dim = self.text_hidden_size + self.image_hidden_size

        # --- Classification head ---
        # pass the combined features into the classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),  # Dense layer
            nn.ReLU(),  # Non-linear activation function
            nn.Dropout(dropout),  # randomly zeroes 30 percent of neuron outputs
            # to prevent over-fitting
            nn.Linear(hidden_dim, num_classes),  # Final Classification output
        )

    def forward(self, input_ids=None, attention_mask=None, image=None):
        # input_ids shape: [batch, seq_length]
        # attention_mask: mask to ignore padding, same shape as input_ids
        # image: [batch, 3, 224, 224]
        # Text features
        if self.mode == "text":
            text_outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # feed tokenized text into the BERT Model which returns a
            # dictionary with last_hidden_state: [batch_size, seq_len,
            # hidden_size], pooler_output: [batch_size, hidden_size]
            # (CLS embeddings), hidden_states: List of tensors,
            # attentions(weights): List of Tensors
            features = text_outputs.last_hidden_state[
                :, 0, :
            ]  # CLS token, return CLS tokens from all batches, position 0,
            # a batch of 3 sentences has 3 CLS tokens

        # Image features
        elif self.mode == "image":
            features = self.image_encoder(
                image
            )  # pass the image through ResNet, returns a [batch, 2048] tensor

        else:  # multimodal
            text_outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            text_feat = text_outputs.last_hidden_state[:, 0, :]  # CLS token
            image_feat = self.image_encoder(image)
            features = torch.cat(
                (text_feat, image_feat), dim=1
            )  # Concatenates text and image features along feature dimension
            # [CLS vector from BERT] + [ResNet image vector]
            # -> [batch_size, 2816]

        # Return logits for each class, later apply softmax during evaluation
        return self.classifier(features)
