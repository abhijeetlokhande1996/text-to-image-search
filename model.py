from sklearn.model_selection import train_test_split
import timm
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm import tqdm
import pandas as pd
from typing import List, Any
from prepare_dataset import TextToImageSearchDataset, get_model_checkpoint
import torch.nn.functional as F
import itertools
# get_device_map


def get_device_map() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


DEVICE = get_device_map()
# print("Device :: {}".format(DEVICE))

MODEL_CHECKPOINT = get_model_checkpoint()


class ImageEncoder(nn.Module):
    def __init__(self, trainable: bool = True):
        super().__init__()
        self.image_encoder = timm.create_model(
            model_name="resnet50", pretrained=True, num_classes=0, global_pool="avg")
        for param in self.image_encoder.parameters():
            param.requires_grad = trainable

    def forward(self, X):
        return self.image_encoder(X)


class TextEncoder(nn.Module):

    def __init__(self, trainable: bool = True, device: str = DEVICE) -> None:
        super().__init__()
        self.model_checkpoint = MODEL_CHECKPOINT
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_checkpoint)
        self.text_encoder = DistilBertModel.from_pretrained(
            self.model_checkpoint)
        for param in self.text_encoder.parameters():
            param.requires_grad = trainable

    def forward(self, X, return_last_hidden_state=True):
        # tokenised_input = self.tokenizer(X, return_tensors="pt", truncation=True, is_split_into_words=False, max_length=30, padding=True)
        output = self.text_encoder(**X)
        if not return_last_hidden_state:
            return output
        else:
            return output.last_hidden_state[:, 0, :]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, X):
        projected = self.projection(X)
        X = self.gelu(projected)
        X = self.fc(X)
        X = self.dropout(X)
        X = X + projected
        X = self.layer_norm(X)
        return X


class CLIPModel(nn.Module):
    def __init__(self, projection_dim: int = 512, temperature: float = 1.0) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(trainable=True)
        self.text_encoder = TextEncoder(trainable=True)
        self.projection_dim = projection_dim

        self.image_projection = ProjectionHead(
            input_dim=2048, projection_dim=self.projection_dim)
        self.text_projection = ProjectionHead(
            input_dim=768, projection_dim=self.projection_dim)

        self.temperature = temperature

    @staticmethod
    def cross_entropy(preds, targets, reduction="none"):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def forward(self, X):
        image_embeddings = self.image_encoder(X["image"])
        text_embeddings = self.text_encoder(
            {"input_ids": X["input_ids"], "attention_mask": X["attention_mask"]})

        image_embeddings = self.image_projection(image_embeddings)
        text_embeddings = self.text_projection(text_embeddings)

        # calculating the Loss using image and text embeddings
        logits = (text_embeddings @ image_embeddings.T) / self.temperature

        image_similarity = image_embeddings @ image_embeddings.T
        text_similarity = text_embeddings @ text_embeddings.T

        targets = F.softmax((image_similarity + text_similarity) /
                            (2.0 * self.temperature), dim=-1)

        text_loss = self.cross_entropy(
            preds=logits, targets=targets, reduction="none")

        image_loss = self.cross_entropy(
            preds=logits.T, targets=targets.T, reduction="none")

        loss = (text_loss + image_loss) / 2.0
        return loss.mean()


def split_dataset(image_filenames, captions, train_size=0.8, val_size=0.1, test_size=0.1):
    assert train_size + val_size + \
        test_size == 1.0, "Train, validation, and test sizes must sum to 1.0"

    # First, split into train and temp (validation + test)
    train_imgs, temp_imgs, train_caps, temp_caps = train_test_split(
        image_filenames, captions, train_size=train_size, random_state=42)

    # Then split the temp into validation and test
    val_imgs, test_imgs, val_caps, test_caps = train_test_split(
        temp_imgs, temp_caps, test_size=test_size / (test_size + val_size), random_state=42)

    return (train_imgs, train_caps), (val_imgs, val_caps), (test_imgs, test_caps)


def get_text_to_image_search_dataset():
    df_limited = pd.read_csv("./archive/dataset_limited.csv")
    image_filenames: List[str] = df_limited["image"].tolist()
    captions: List[str] = df_limited["caption"].tolist()

    # return TextToImageSearchDataset(image_filenames, captions)
    return image_filenames, captions


def get_datasets():
    image_filenames, captions = get_text_to_image_search_dataset()

    (train_imgs, train_caps), (val_imgs, val_caps), (test_imgs,
                                                     test_caps) = split_dataset(image_filenames, captions)

    train_dataset = TextToImageSearchDataset(
        train_imgs, train_caps)
    val_dataset = TextToImageSearchDataset(
        val_imgs, val_caps)
    test_dataset = TextToImageSearchDataset(
        test_imgs, test_caps)

    return train_dataset, val_dataset, test_dataset


def compute_validation_loss(model, validation_dataloader) -> float:
    running_loss = 0.0
    count = 0
    for batch in tqdm(validation_dataloader, total=len(validation_dataloader)):
        caption = batch.pop("caption")
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        valid_loss = model(batch)
        running_loss += valid_loss
        count += 1
        break

    return running_loss / count


if __name__ == "__main__":

    # dataset = get_text_to_image_search_dataset()
    train_dataset, val_dataset, test_dataset = get_datasets()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

    model = CLIPModel()
    model.to(DEVICE)
    params = [
        {"params": model.image_encoder.parameters(), "lr": 1e-4},
        {"params": model.text_encoder.parameters(), "lr": 1e-5},
        {"params": itertools.chain(model.image_projection.parameters(
        ), model.text_projection.parameters()), "lr": 1e-3, "weight_decay": 1e-3},

    ]
    optimizer = torch.optim.AdamW(params=params, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=1, factor=0.8
    )
    train_tqdm_object = tqdm(train_dataloader, total=len(train_dataloader))
    for epoch in range(1):
        model.train()
        running_loss = 0.0
        for batch in train_tqdm_object:
            caption = batch.pop("caption")
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            break

        model.eval()
        with torch.no_grad():
            valid_loss_mean = compute_validation_loss(
                model, validation_dataloader)
        lr_scheduler.step(valid_loss_mean)

        epoch_loss = running_loss / len(train_dataloader)

        print(f"Epoch: {epoch}, Loss: {
              epoch_loss}, Valid Loss: {valid_loss_mean}")
