import torch
import pandas as pd
import torch.utils
import torch.utils.data
import albumentations as A
from typing import List, Any
from transformers import DistilBertTokenizer
import cv2

MODEL_CHECKPOINT = "distilbert/distilbert-base-uncased"


class TextToImageSearchDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames: List[str], captions: List[str]) -> None:
        super().__init__()
        self.image_filenames = image_filenames
        self.captions = captions
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CHECKPOINT)
        self.encoded_captions = self.tokenizer(
            self.captions, return_tensors="pt", truncation=True, is_split_into_words=False, max_length=30, padding=True)
        self.image_transforms = A.Compose([
            A.Resize(224, 224, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ])

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]).clone().detach()
                for key, value in self.encoded_captions.items()}
        image_filepath = "./archive/Images/" + self.image_filenames[index]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transforms(image=image)["image"]
        # pytorch expects channels, height, width
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["caption"] = self.captions[index]
        return item

    def __len__(self):
        return len(self.image_filenames)


def prepare_dataset_for_clip():
    caption_filepath: str = "./archive/captions.txt"
    df = pd.read_csv(caption_filepath)
    df["image"] = df["image"].str.strip()
    df["caption"] = df["caption"].str.strip()

    df_limited = df.groupby(by="image").head(2).reset_index(drop=True)

    output_filepath: str = "./archive/dataset_limited.csv"
    df_limited.to_csv(output_filepath, index=False)
    print("Output filepath :: {}".format(output_filepath))
    return df_limited


if __name__ == "__main__":
    # df_limited: pd.DataFrame = prepare_dataset_for_clip()

    df_limited = pd.read_csv("./archive/dataset_limited.csv")
    image_filenames: List[str] = df_limited["image"].tolist()
    captions: List[str] = df_limited["caption"].tolist()

    dataset = TextToImageSearchDataset(image_filenames, captions)
