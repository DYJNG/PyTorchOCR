import json
from pathlib import Path
import random
import argparse
from tqdm import tqdm



def write_to_file(img_dir, samples, out_path):
    fp = open(str(out_path), "w", encoding="UTF-8")
    for sample in tqdm(samples):
        sample = sample.strip("\n")
        info = json.loads(sample)
        filename = info["filename"]
        img_path = Path(str(img_dir)).joinpath(filename)
        info["img_path"] = str(img_path)
        info.pop("filename")
        fp.write(json.dumps(info, ensure_ascii=False) + "\n")
    fp.close()


def gen_pubtab_label(
    img_dir, 
    label_path, 
    out_path, 
    split_train_val=False, 
    ratio=0.9, 
    extra_out_path=None
):  
    with open(str(label_path), "r", encoding="UTF-8") as f:
        samples = f.readlines()
    if not split_train_val:
        write_to_file(img_dir, samples, out_path)
    else:
        train_samples, val_samples = [], []
        num_samples= len(samples)
        random.shuffle(samples)
        for i in range(num_samples):
            if i < num_samples * ratio:
                train_samples.append(samples[i])
            else:
                val_samples.append(samples[i])
        write_to_file(img_dir, train_samples, out_path)
        write_to_file(img_dir, val_samples, extra_out_path)
        print("num of train samples: ", len(train_samples))
        print("num of valid samples: ", len(val_samples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default=".",
        help="The directory of images")
    parser.add_argument(
        "--label_path",
        type=str,
        default="table_samples.txt",
        help="The directory of annotations")
    parser.add_argument(
        "--out_path",
        type=str,
        default="out_label.txt",
        help="Output file path")
    parser.add_argument(
        "--split_train_val",
        type=bool,
        default=False,
        help="Whether split train val")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.9,
        help="Ratio of train samples")
    parser.add_argument(
        "--extra_out_path",
        type=str,
        default="val_label.txt",
        help="Output val file path")


    args = parser.parse_args()

    print("Generate pubtab label")
    gen_pubtab_label(
        args.img_dir, 
        args.label_path, 
        args.out_path, 
        args.split_train_val, 
        args.ratio, 
        args.extra_out_path)