# Rice leaf disease classifier

MobileNetV3-Small model that classifies a rice-leaf photo as Healthy, Bacterial blight, Blast, Brown spot, or Tungro.

**Best validation accuracy: 93.5%.**

## Web page (Okay / Not okay)

```bash
pip install torch torchvision pillow
# put rice_leaf_disease.pt in this folder (or next to site/app.py)
python site/app.py
```

Open http://127.0.0.1:7860 — upload a leaf photo. The page shows **Okay** if the top class is Healthy, otherwise **Not okay** plus the disease name.

## Command line

```bash
python predict.py path/to/leaf.jpg --ckpt rice_leaf_disease.pt
```

## Train

See `train_rice_disease.py`. Dataset: [sharmin3/Rice-Leaf-Disease](https://huggingface.co/datasets/sharmin3/Rice-Leaf-Disease).
