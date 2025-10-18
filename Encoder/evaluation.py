import numpy as np
import os 
import torch
import random 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from  model import SpeakerEncoder
# for evaluation and visualization of speaker embeddings after training the encoder model.
def LoadEncoderModel(encoder_model_path):
    if not os.path.exists(encoder_model_path):
        raise FileNotFoundError(f"Encoder model file {encoder_model_path} does not exist.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model =SpeakerEncoder(device=device, loss_device=device)
    checkpoint = torch.load(encoder_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
    


def visualize( encoder_model_path, file_path, n_speaker=20,sample_per_speaker=5, method="pca"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    model= LoadEncoderModel(encoder_model_path)
    
    speaker_dirs=sorted([d for d in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, d))])
    selected_speakers =random.sample(speaker_dirs,n_speaker)
    embeddings=[]
    print(selected_speakers)
    labels=[]
    for speaker in selected_speakers:
        speaker_path= os.path.join(file_path, speaker)
        npy_files=sorted([d for d in os.listdir(speaker_path) if d.endswith('.npy')])
        selected_files= random.sample(npy_files, sample_per_speaker)
        for f in selected_files:
            utterance_path = os.path.join(speaker_path, f)
            data = np.load(utterance_path)
            data = torch.from_numpy(data).unsqueeze(0)
            embed = model.forward(data).detach().cpu().numpy()[0]
            embed = embed / np.linalg.norm(embed) 
            embeddings.append(embed)
            labels.append(speaker)
    embeddings=np.array(embeddings)
    print("Embedding created sucessfully.......")
    if method == "tsne":
        reduced = TSNE(n_components=2, perplexity=10, init="pca", learning_rate='auto').fit_transform(embeddings)
    elif method == "pca":
        reduced = PCA(n_components=2).fit_transform(embeddings)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")
    
    plt.figure(figsize=(10,8))
    for speaker in set(labels):
        idx=[i for i, label in enumerate(labels) if  label==speaker]
        plt.scatter(reduced[idx,0],reduced[idx,1],label=speaker,s=40)

    plt.title(f"Speaker Clusters using {method.upper()} (20 speakers, 5 samples each)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="small")
    plt.tight_layout()
    plt.grid(True)
    plt.show()    

if "__main__" == __name__:
    visualize(r"DIR\encoder.pt",r"DIR\encoder")   