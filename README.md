# Rice leaf disease classifier

MobileNetV3-Small model that classifies a rice-leaf photo as one of:

| Class | Meaning |
|---|---|
| Healthy | Rice leaf, no disease |
| Bacterialblight | Bacterial leaf blight (*Xanthomonas oryzae*) |
| Blast | Rice blast (*Magnaporthe oryzae*) |
| Brownspot | Brown spot (*Bipolaris oryzae*) |
| Tungro | Rice tungro virus |

**Best validation accuracy: 93.5%** (5,937 train / 1,484 val images).

Dataset: [sharmin3/Rice-Leaf-Disease](https://huggingface.co/datasets/sharmin3/Rice-Leaf-Disease) (7,421 images).

## Predict

```bash
pip install torch torchvision pillow
python predict.py path/to/leaf.jpg --ckpt rice_leaf_disease.pt
```

Place `rice_leaf_disease.pt` next to `predict.py` (download from this repo if present, or train it).

## Train

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sharmin3/Rice-Leaf-Disease', repo_type='dataset', filename='Rice Leaf Disease-20241115T062818Z-001.zip', local_dir='./data')"
unzip -o "./data/Rice Leaf Disease-20241115T062818Z-001.zip" -d ./data/raw

python train_rice_disease.py \
  --data-dir "./data/raw/Rice Leaf Disease" \
  --out-dir . \
  --epochs 4 --img-size 128 --batch-size 64 --freeze-backbone
```

Weights file `rice_leaf_disease.pt` is ~6 MB. If GitHub file-size limits block it in this repo, keep a local copy from the Grok project folder `rice_disease_model/`.
