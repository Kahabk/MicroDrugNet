"""
MicroDrugNet — Hugging Face Spaces / Gradio Demo
Run: python demo/app.py
Deploy: push to HF Spaces with requirements.txt
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import gradio as gr
from rdkit import Chem

from microdrug.model import MicroDrugNet
from data.preprocess import smiles_to_graph
from torch_geometric.data import Batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL  = None

EXAMPLE_DRUGS = {
    "Aspirin"    : "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen"  : "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Caffeine"   : "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Metformin"  : "CN(C)C(=N)NC(=N)N",
    "Metronidazole": "Cc1ncc([N+](=O)[O-])n1CCO",
}

def load_model(ckpt_path=None):
    global MODEL
    MODEL = MicroDrugNet(n_taxa=1000).to(DEVICE)
    if ckpt_path and Path(ckpt_path).exists():
        MODEL.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint — using random weights (demo mode)")
    MODEL.eval()

def predict(smiles: str, microbiome_csv: str, sample_id: str = None):
    if MODEL is None:
        return "Model not loaded.", None, None

    # Validate SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string.", None, None

    graph = smiles_to_graph(smiles)
    if graph is None:
        return "Could not build molecular graph.", None, None

    # Parse microbiome CSV (or use random for demo)
    if microbiome_csv and Path(microbiome_csv).exists():
        import pandas as pd
        df = pd.read_csv(microbiome_csv, index_col=0)
        if sample_id and sample_id in df.index:
            arr = df.loc[sample_id].values[:1000].astype(np.float32)
        else:
            arr = df.values[0, :1000].astype(np.float32)
        if len(arr) < 1000:
            arr = np.pad(arr, (0, 1000 - len(arr)))
    else:
        arr = np.random.exponential(1.0, 1000).astype(np.float32)

    micro  = torch.tensor(arr).unsqueeze(0).to(DEVICE)
    batch  = Batch.from_data_list([graph]).to(DEVICE)

    with torch.no_grad():
        preds = MODEL(batch, micro)

    bio   = preds["bioavailability"].item()
    probs = torch.softmax(preds["response_logits"], dim=-1).cpu().squeeze().tolist()
    label = ["Low 🔴", "Medium 🟡", "High 🟢"][int(np.argmax(probs))]

    summary = (
        f"### Prediction Results\n"
        f"**Bioavailability Score:** `{bio:.3f}` ({'High' if bio>0.6 else 'Low'} systemic exposure)\n\n"
        f"**Patient Response:** {label}\n\n"
        f"**Class Probabilities:** Low={probs[0]:.2f} | Med={probs[1]:.2f} | High={probs[2]:.2f}"
    )
    return summary, bio, probs

def build_interface():
    with gr.Blocks(title="MicroDrugNet", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧬 MicroDrugNet\n**Predicting drug metabolism via gut microbiome (pharmacomicrobiomics)**")
        gr.Markdown("Enter a drug SMILES string and optionally upload a microbiome OTU table (CSV) to predict bioavailability and patient response.")

        with gr.Row():
            with gr.Column():
                smiles_dd  = gr.Dropdown(choices=list(EXAMPLE_DRUGS.keys()), label="Example Drugs", value="Aspirin")
                smiles_box = gr.Textbox(label="Drug SMILES", value=EXAMPLE_DRUGS["Aspirin"])
                microbiome_file = gr.File(label="Microbiome OTU Table (CSV, optional)")
                sample_id_box   = gr.Textbox(label="Sample ID (optional)")
                btn = gr.Button("🔬 Predict", variant="primary")
            with gr.Column():
                output_text = gr.Markdown()
                bio_bar     = gr.Number(label="Bioavailability Score")

        smiles_dd.change(fn=lambda x: EXAMPLE_DRUGS.get(x, ""), inputs=smiles_dd, outputs=smiles_box)
        btn.click(fn=lambda s, f, i: predict(s, f.name if f else None, i),
                  inputs=[smiles_box, microbiome_file, sample_id_box],
                  outputs=[output_text, bio_bar, gr.JSON(visible=False)])

        gr.Markdown("---\n*MicroDrugNet is a research prototype. Not for clinical use.*")
    return demo

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   default=None)
    ap.add_argument("--share",  action="store_true")
    ap.add_argument("--port",   type=int, default=7860)
    args = ap.parse_args()
    load_model(args.ckpt)
    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)
