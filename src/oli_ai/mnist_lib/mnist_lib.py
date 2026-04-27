from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import norm
import networkx as nx
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model


def plot_img(image):
    image = image.reshape((28, 28))
    plt.imshow(image, cmap='gray')
    plt.show()


def plot_imgs_labels(imgs, labels, rows=2, cols=5):
    figure = plt.figure(figsize=(10, 3))
    # plotting images from the training set
    for i in range(0, rows*cols):
        plt.subplot(rows, cols, i+1)
        plt.subplots_adjust(hspace=1, wspace=1)
        plt.title(f"Label: {labels[i]}")
        img = imgs[i].reshape((28, 28))
        plt.imshow(img, cmap='gray')       
        # Rimuovere le indicazioni numeriche dagli assi
        plt.xticks([])
        plt.yticks([])

def plot_imgs(imgs, labels, rows=2, cols=5):
    figure = plt.figure(figsize=(10, 3))
    # plotting images from the training set
    for i in range(0, rows*cols):
        plt.subplot(rows, cols, i+1)
        plt.subplots_adjust(hspace=1, wspace=1)
        #plt.title(f"Label: {labels[i]}")
        img = imgs[i].reshape((28, 28))
        plt.imshow(img, cmap='gray')
        
        # Rimuovere le indicazioni numeriche dagli assi
        plt.xticks([])
        plt.yticks([])



def predict(model,image):
  image = image.reshape(1, 28, 28)  # Aggiungi dimensione batch
  prediction = model.predict(image)
  #print(f"Raw prediction: {prediction}")  # [0.01, 0.02, 0.95, 0.01, ...]
  # Classe predetta
  predicted_class = np.argmax(prediction)
  return predicted_class


# Funzione per visualizzare i pesi
def visualize_weights(weights, title):
   normalized_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
   
   fig, axes = plt.subplots(1, 10, figsize=(15, 2))  # Altezza dimezzata
   
   for i in range(10):
       axes[i].imshow(normalized_weights[:, i].reshape(28, 28), cmap='gray')
       axes[i].axis('off')
       axes[i].set_title(f'{i}', fontsize=6)  # Font più piccolo
   
   fig.suptitle(title, fontsize=10, y=0.95)  # Titolo più piccolo
   plt.subplots_adjust(top=0.65, bottom=0.02, hspace=0, wspace=0.05)  # Margini ultra-mini
   plt.show()


import networkx as nx
import matplotlib.pyplot as plt

def show_nn_graph(layers, max_neurons=10):
    """
    Visualizza una rete neurale con architettura arbitraria
    
    Args:
        layers: lista con il numero di neuroni per ogni layer
                Es: [2,1] = 2 input, 1 output
                    [2,2,1] = 2 input, 2 hidden, 1 output  
                    [2,2,4,5,4] = 2 input, 2 hidden, 4 hidden, 5 hidden, 4 output
        max_neurons: numero massimo di neuroni da mostrare per layer (default=10)
                    Se un layer ha più neuroni, mostra i primi, poi "...", poi l'ultimo
    """
    if len(layers) < 2:
        raise ValueError("Servono almeno 2 layer (input e output)")
    
    G = nx.DiGraph()
    pos = {}
    
    # SPACING DINAMICO in base al numero massimo di neuroni visualizzati
    max_visual_neurons = min(max(layers), max_neurons)
    vertical_spacing = max(0.2, min(0.5, 3.0 / max_visual_neurons))  # Adatta spacing
    horizontal_spacing = 1.5
    
    # Crea tutti i nodi e le posizioni
    all_nodes = []
    all_actual_nodes = []  # Nodi reali (senza "...")
    node_labels = {}  # Mappa per etichette pulite
    
    for layer_idx, layer_size in enumerate(layers):
        layer_nodes = []
        layer_actual_nodes = []
        
        # Determina il tipo di layer
        if layer_idx == 0:
            layer_type = 'input'
            prefix = 'I'
        elif layer_idx == len(layers) - 1:
            layer_type = 'output'
            prefix = 'O'
        else:
            layer_type = 'hidden'
            prefix = f'H{layer_idx}'
        
        # Logica SICURA - prima calcolo esatto dei nodi poi creazione
        visual_nodes = []  # Lista di (nome_display, tipo, posizione_y)
        actual_nodes_only = []  # Solo neuroni reali per connessioni
        
        if layer_size <= max_neurons:
            # Caso semplice: tutti i neuroni (partendo da 1, dall'alto al basso)
            for i in range(layer_size):
                display_name = f'{i+1}'  # SOLO NUMERO per display
                y_pos = ((layer_size-1)/2 - i) * vertical_spacing
                visual_nodes.append((display_name, layer_type, y_pos))
                actual_nodes_only.append(display_name)
        else:
            # Caso complesso: primi + ... + ultimo (dall'alto al basso)
            total_slots = max_neurons
            y_positions_list = [((total_slots-1)/2 - i) * vertical_spacing for i in range(total_slots)]
            
            slot = 0
            # Primi neuroni (max_neurons - 2, partendo da 1)
            for i in range(max_neurons - 2):
                display_name = f'{i+1}'  # SOLO NUMERO: 1,2,3...
                visual_nodes.append((display_name, layer_type, y_positions_list[slot]))
                actual_nodes_only.append(display_name)
                slot += 1
            
            # "..." (PENULTIMO slot)
            visual_nodes.append(('...', 'dots', y_positions_list[slot]))
            slot += 1
            
            # Ultimo neurone (ULTIMO slot)
            last_display = f'{layer_size}'
            visual_nodes.append((last_display, layer_type, y_positions_list[slot]))
            actual_nodes_only.append(last_display)
        
        # Crea tutti i nodi con nomi unici per NetworkX
        x = layer_idx * horizontal_spacing
        
        for display_name, node_type, y_pos in visual_nodes:
            # Nome unico per NetworkX
            unique_name = f'{prefix}_{display_name}'
            G.add_node(unique_name, layer=node_type)
            pos[unique_name] = (x, y_pos)
            layer_nodes.append(unique_name)
            
            # Mappa per etichette pulite
            node_labels[unique_name] = display_name
        
        # Solo neuroni reali per connessioni
        layer_actual_nodes = [f'{prefix}_{name}' for name in actual_nodes_only]
        
        all_nodes.append(layer_nodes)
        all_actual_nodes.append(layer_actual_nodes)
    
    # Crea le connessioni tra layer adiacenti
    for layer_idx in range(len(layers) - 1):
        current_layer = all_actual_nodes[layer_idx]  # Solo nodi reali
        next_layer = all_actual_nodes[layer_idx + 1]  # Solo nodi reali
        
        # Connetti ogni neurone del layer corrente con ogni neurone del layer successivo
        for current_node in current_layer:
            for next_node in next_layer:
                G.add_edge(current_node, next_node)
    
    # Calcola dimensioni figura in base al numero di layer E neuroni visualizzati
    fig_width = max(6, len(layers) * 1.8)
    fig_height = max(3, max_visual_neurons * 0.8 + 1)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Colori per i diversi tipi di layer
    colors = {'input': '#3498db', 'hidden': '#2ecc71', 'output': '#e74c3c', 'dots': '#95a5a6'}
    node_colors = [colors[G.nodes[node]['layer']] for node in G.nodes()]
    
    # Dimensioni dei nodi (più piccoli per i "...")
    node_sizes = []
    for node in G.nodes():
        if G.nodes[node]['layer'] == 'dots':
            node_sizes.append(300)  # Più piccoli per "..."
        else:
            node_sizes.append(600)  # Dimensione normale
    
    # Disegna la rete con etichette pulite
    # Crea lista ordinata di nodi per garantire ordine nel drawing
    ordered_nodes = []
    for layer_nodes in all_nodes:
        ordered_nodes.extend(layer_nodes)
    
    nx.draw(G, pos,
            nodelist=ordered_nodes,  # FORZA l'ordine dei nodi
            labels=node_labels,      # Etichette solo numeriche
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=10,
            font_weight='bold',
            font_color='white',
            arrows=True,
            arrowsize=15,
            edge_color='#95a5a6',
            width=1.5,
            alpha=0.8)
    
    # Titolo con architettura
    arch_str = '→'.join(map(str, layers))
    plt.title(f'Neural Network Architecture: {arch_str}', fontsize=14, pad=20)
    
    plt.axis('off')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Invece di tight_layout()
    plt.show()


def show_nn_graph_old(layers, max_neurons=10):
    """
    Visualizza una rete neurale con architettura arbitraria
    
    Args:
        layers: lista con il numero di neuroni per ogni layer
                Es: [2,1] = 2 input, 1 output
                    [2,2,1] = 2 input, 2 hidden, 1 output  
                    [2,2,4,5,4] = 2 input, 2 hidden, 4 hidden, 5 hidden, 4 output
        max_neurons: numero massimo di neuroni da mostrare per layer (default=10)
                    Se un layer ha più neuroni, mostra i primi, poi "...", poi l'ultimo
    """
    if len(layers) < 2:
        raise ValueError("Servono almeno 2 layer (input e output)")
    
    G = nx.DiGraph()
    pos = {}
    
    # SPACING DINAMICO in base al numero massimo di neuroni visualizzati
    max_visual_neurons = min(max(layers), max_neurons)
    vertical_spacing = max(0.2, min(0.5, 3.0 / max_visual_neurons))  # Adatta spacing
    horizontal_spacing = 1.5
    
    # Crea tutti i nodi e le posizioni
    all_nodes = []
    all_actual_nodes = []  # Nodi reali (senza "...")
    node_labels = {}  # Mappa per etichette pulite
    
    for layer_idx, layer_size in enumerate(layers):
        layer_nodes = []
        layer_actual_nodes = []
        
        # Determina il tipo di layer
        if layer_idx == 0:
            layer_type = 'input'
            prefix = 'I'
        elif layer_idx == len(layers) - 1:
            layer_type = 'output'
            prefix = 'O'
        else:
            layer_type = 'hidden'
            prefix = f'H{layer_idx}'
        
        # Logica SICURA - prima calcolo esatto dei nodi poi creazione
        visual_nodes = []  # Lista di (nome_display, tipo, posizione_y)
        actual_nodes_only = []  # Solo neuroni reali per connessioni
        
        if layer_size <= max_neurons:
            # Caso semplice: tutti i neuroni (partendo da 1, dall'alto al basso)
            for i in range(layer_size):
                display_name = f'{i+1}'  # SOLO NUMERO per display
                y_pos = ((layer_size-1)/2 - i) * vertical_spacing
                visual_nodes.append((display_name, layer_type, y_pos))
                actual_nodes_only.append(display_name)
        else:
            # Caso complesso: primi + ... + ultimo (dall'alto al basso)
            total_slots = max_neurons
            y_positions_list = [((total_slots-1)/2 - i) * vertical_spacing for i in range(total_slots)]
            
            slot = 0
            # Primi neuroni (max_neurons - 2, partendo da 1)
            for i in range(max_neurons - 2):
                display_name = f'{i+1}'  # SOLO NUMERO: 1,2,3...
                visual_nodes.append((display_name, layer_type, y_positions_list[slot]))
                actual_nodes_only.append(display_name)
                slot += 1
            
            # "..." (PENULTIMO slot)
            visual_nodes.append(('...', 'dots', y_positions_list[slot]))
            slot += 1
            
            # Ultimo neurone (ULTIMO slot)
            last_display = f'{layer_size}'
            visual_nodes.append((last_display, layer_type, y_positions_list[slot]))
            actual_nodes_only.append(last_display)
        
        # Crea tutti i nodi con nomi unici per NetworkX
        x = layer_idx * horizontal_spacing
        
        for display_name, node_type, y_pos in visual_nodes:
            # Nome unico per NetworkX
            unique_name = f'{prefix}_{display_name}'
            G.add_node(unique_name, layer=node_type)
            pos[unique_name] = (x, y_pos)
            layer_nodes.append(unique_name)
            
            # Mappa per etichette pulite
            node_labels[unique_name] = display_name
        
        # Solo neuroni reali per connessioni
        layer_actual_nodes = [f'{prefix}_{name}' for name in actual_nodes_only]
        
        all_nodes.append(layer_nodes)
        all_actual_nodes.append(layer_actual_nodes)
    
    # Crea le connessioni tra layer adiacenti
    for layer_idx in range(len(layers) - 1):
        current_layer = all_actual_nodes[layer_idx]  # Solo nodi reali
        next_layer = all_actual_nodes[layer_idx + 1]  # Solo nodi reali
        
        # Connetti ogni neurone del layer corrente con ogni neurone del layer successivo
        for current_node in current_layer:
            for next_node in next_layer:
                G.add_edge(current_node, next_node)
    
    # Calcola dimensioni figura in base al numero di layer E neuroni visualizzati
    fig_width = max(6, len(layers) * 1.8)
    fig_height = max(3, max_visual_neurons * 0.8 + 1)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Colori per i diversi tipi di layer
    colors = {'input': '#3498db', 'hidden': '#2ecc71', 'output': '#e74c3c', 'dots': '#95a5a6'}
    node_colors = [colors[G.nodes[node]['layer']] for node in G.nodes()]
    
    # Dimensioni dei nodi (più piccoli per i "...")
    node_sizes = []
    for node in G.nodes():
        if G.nodes[node]['layer'] == 'dots':
            node_sizes.append(300)  # Più piccoli per "..."
        else:
            node_sizes.append(600)  # Dimensione normale
    
    # Disegna la rete con etichette pulite
    # Crea lista ordinata di nodi per garantire ordine nel drawing
    ordered_nodes = []
    for layer_nodes in all_nodes:
        ordered_nodes.extend(layer_nodes)
    
    nx.draw(G, pos,
            nodelist=ordered_nodes,  # FORZA l'ordine dei nodi
            labels=node_labels,      # Etichette solo numeriche
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=10,
            font_weight='bold',
            font_color='white',
            arrows=True,
            arrowsize=15,
            edge_color='#95a5a6',
            width=1.5,
            alpha=0.8)
    
    # Titolo con architettura
    arch_str = '→'.join(map(str, layers))
    plt.title(f'Neural Network Architecture: {arch_str}', fontsize=14, pad=20)
    
    # Aggiungi etichette per i layer
    layer_labels = ['Input']
    if len(layers) > 2:
        for i in range(1, len(layers) - 1):
            layer_labels.append(f'Hidden {i}')
    layer_labels.append('Output')
    
    # Posiziona le etichette dei layer
    for i, label in enumerate(layer_labels):
        plt.text(i * horizontal_spacing, 
                max([pos[node][1] for node in all_nodes[i]]) + 0.5,
                label, 
                ha='center', 
                va='bottom',
                fontsize=12,
                fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()





def show_xor_3d(model, h1,h2,inputs,outputs):

  # === 3. Estrai attivazioni da entrambi i hidden layer ===
  hidden1_model = Model(inputs=model.input, outputs=h1)
  hidden2_model = Model(inputs=model.input, outputs=h2)
  act1 = hidden1_model.predict(inputs)
  act2 = hidden2_model.predict(inputs)

  # === 4. Unisci le attivazioni dei neuroni ===
  all_activations = np.concatenate([act1, act2], axis=1)

  # === 5. Trova il neurone con massima separazione tra classi ===
  best_index = 0
  max_sep = -np.inf
  for i in range(all_activations.shape[1]):
      z = all_activations[:, i]
      z0 = z[outputs[:, 0] == 0]
      z1 = z[outputs[:, 0] == 1]
      sep = np.abs(np.mean(z0) - np.mean(z1))
      if sep > max_sep:
          max_sep = sep
          best_index = i

  # === 6. Prepara i dati per il grafico 3D ===
  best_neuron_values = all_activations[:, best_index]
  data_3d = np.hstack([inputs, best_neuron_values.reshape(-1, 1)])

  # === 7. Identifica il layer del neurone scelto ===
  if best_index < act1.shape[1]:
      layer_name = f"Hidden Layer 1 - Neuron {best_index}"
  else:
      layer_name = f"Hidden Layer 2 - Neuron {best_index - act1.shape[1]}"

  # === 8. Plot 3D con Plotly ===
  fig = go.Figure()

  fig.add_trace(go.Scatter3d(
      x=data_3d[:, 0],
      y=data_3d[:, 1],
      z=data_3d[:, 2],
      mode='markers+text',
      marker=dict(
          size=14,
          color=outputs.ravel(),
          colorscale='Viridis',
          opacity=1.0,
          symbol='circle',
          line=dict(color='black', width=1)
      ),
      text=[f"({x[0]},{x[1]})" for x in inputs],
      textposition="top center",
      textfont=dict(size=12, color='black'),
      hovertext=[
          f"Input: ({x[0]}, {x[1]})<br>Output XOR: {y[0]}<br>Attivazione Neurone: {z:.4f}"
          for x, y, z in zip(inputs, outputs, data_3d[:, 2])
      ],
      hoverinfo="text",
      name='XOR Points'
  ))

  fig.update_layout(
      title=dict(
          text=f'Visualizzazione 3D XOR – {layer_name}',
          font=dict(size=20)
      ),
      scene=dict(
          xaxis=dict(
              title='Input 1 (X)',
              range=[-0.1, 1.2],
              tickvals=[0, 1],
              showbackground=True,
              backgroundcolor='rgb(255, 230, 230)',
              gridcolor='red',
          ),
          yaxis=dict(
              title='Input 2 (Y)',
              range=[-0.1, 1.2],
              tickvals=[0, 1],
              showbackground=True,
              backgroundcolor='rgb(230, 255, 230)',
              gridcolor='green',
          ),
          zaxis=dict(
              title='Attivazione Neurone (Z)',
              range=[-0.1, 1.1],
              showbackground=True,
              backgroundcolor='rgb(230, 230, 255)',
              gridcolor='blue',
          ),
          camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
      ),
      height=700,
      width=900,
      margin=dict(l=0, r=0, b=0, t=40, pad=100),
  )

  fig.show()


def show_xor_error(errors_history):
  # Visualizzazione dell'andamento dell'errore durante il training
  fig_error = go.Figure()
  fig_error.add_trace(go.Scatter(
    y=errors_history,
    mode='lines',
    name='MSE',
    line=dict(width=2, color='royalblue')
  ))

  fig_error.update_layout(
    title='Andamento dell\'errore durante il training',
    xaxis_title='Epoca',
    yaxis_title='MSE (Mean Squared Error)',
    yaxis_type='log',
    height=400,
    width=800,
    margin=dict(t=100, b=100)  # Aumentato il margine superiore e inferiore
  )
  fig_error.show()


# =============================================================================
# Estensioni didattiche (aggiunte in v0.13.0)
# -----------------------------------------------------------------------------
# API in italiano per il notebook oli_ai_mnist_cosine_similarity_didattica.
# 100% retrocompatibili: nessuna funzione esistente e' stata toccata.
# =============================================================================
# -----------------------------------------------------------------------------
# Costruttore di vettori con sintassi matematica: v[1, 2, 3]
# -----------------------------------------------------------------------------
class _VettoreFactory:
    """Permette la sintassi v[1, 2, 3] per costruire un vettore.
    Dietro le quinte è un numpy array, quindi le operazioni vettoriali
    (+, -, *, prodotto per scalare) si comportano come in matematica.

    Esempi:
        v[1, 2, 3] + v[4, 5, 6]   ->  array([5., 7., 9.])
        3 * v[1, 2, 3]            ->  array([3., 6., 9.])
        v[1, 2, 3] - v[1, 0, 0]   ->  array([0., 2., 3.])
    """
    def __getitem__(self, indici):
        if not isinstance(indici, tuple):
            indici = (indici,)
        return np.array(indici, dtype=float)

    def __repr__(self):
        return "v[...]  costruttore di vettori (es. v[1, 2, 3])"


v = _VettoreFactory()


# -----------------------------------------------------------------------------
# Caricamento dati
# -----------------------------------------------------------------------------
def carica_mnist():
    """Carica MNIST e restituisce (immagini_train, etichette_train,
    immagini_test, etichette_test). Le immagini restano matrici 28x28
    di interi 0-255 (numpy array, ma indicizzabili come liste)."""
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


# -----------------------------------------------------------------------------
# Visualizzazione (alias didattici in italiano sopra le funzioni esistenti)
# -----------------------------------------------------------------------------
def mostra_cifra(immagine):
    """Mostra una singola cifra. Accetta sia matrice 28x28 sia vettore 784."""
    plot_img(immagine)


def mostra_cifre(immagini, etichette, righe=2, colonne=5):
    """Mostra una griglia di cifre con le rispettive etichette."""
    plot_imgs_labels(immagini, list(etichette), righe, colonne)


# -----------------------------------------------------------------------------
# Filtro / media (sostituiscono numpy boolean indexing e np.average)
# -----------------------------------------------------------------------------
def filtra_per_cifra(immagini, etichette, cifra):
    """Restituisce solo le immagini la cui etichetta è 'cifra'."""
    immagini = np.asarray(immagini)
    etichette = np.asarray(etichette)
    return immagini[etichette == cifra]


def media_immagini(immagini):
    """Calcola la 'cifra media' = media pixel-per-pixel di tutte le immagini.
    Restituisce una matrice 28x28."""
    return np.average(np.asarray(immagini), axis=0)


# -----------------------------------------------------------------------------
# Operazioni vettoriali esposte ai ragazzi
# -----------------------------------------------------------------------------
def appiattisci(immagine):
    """Trasforma una matrice 28x28 in un vettore di 784 numeri.
    Se è già un vettore, lo restituisce invariato."""
    a = np.asarray(immagine, dtype=float)
    return a.reshape(-1)


def prodotto_scalare(v1, v2):
    """Prodotto scalare tra due vettori (anche immagini 28x28: vengono
    appiattite automaticamente). Definizione dalle slide ripasso_vettori."""
    a = appiattisci(v1)
    b = appiattisci(v2)
    return float(np.dot(a, b))


def norma(v):
    """Lunghezza (norma euclidea) del vettore."""
    return float(np.linalg.norm(appiattisci(v)))


def similarita(v1, v2):
    """Similarità coseno tra due vettori. Funziona anche con immagini 28x28."""
    return prodotto_scalare(v1, v2) / (norma(v1) * norma(v2))


# alias con accento per coerenza con le slide (Python accetta l'identificatore)
similarità = similarita


# -----------------------------------------------------------------------------
# Predizione (sostituisce reshape + argmax_sim + comprehension)
# -----------------------------------------------------------------------------
def predici(immagine, modello):
    """Restituisce la cifra (0-9) il cui vettore medio è più simile
    all'immagine data. 'modello' è la lista dei 10 vettori medi."""
    similarita_per_cifra = [similarita(immagine, vettore_medio)
                            for vettore_medio in modello]
    return int(np.argmax(similarita_per_cifra))


def predici_tutte(immagini, modello):
    """Applica predici() a tutte le immagini. Restituisce un array di
    predizioni (una per immagine)."""
    immagini_v = np.asarray(immagini, dtype=float).reshape(len(immagini), -1)
    modello_v = np.asarray(modello, dtype=float).reshape(len(modello), -1)
    # versione vettorizzata interna (veloce); l'API esterna resta semplice
    norme_img = np.linalg.norm(immagini_v, axis=1, keepdims=True)
    norme_mod = np.linalg.norm(modello_v, axis=1, keepdims=True)
    img_norm = immagini_v / norme_img
    mod_norm = modello_v / norme_mod
    sim = img_norm @ mod_norm.T
    return np.argmax(sim, axis=1)


# -----------------------------------------------------------------------------
# Valutazione
# -----------------------------------------------------------------------------
def accuratezza(predizioni, etichette_vere):
    """Frazione di predizioni corrette (fra 0 e 1)."""
    predizioni = np.asarray(predizioni)
    etichette_vere = np.asarray(etichette_vere)
    return float((predizioni == etichette_vere).mean())


def mostra_matrice_confusione(etichette_vere, predizioni, normalizzata=False):
    """Visualizza la matrice di confusione 10x10."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    M = confusion_matrix(etichette_vere, predizioni)
    fmt = 'd'
    if normalizzata:
        M = M.astype(float) / M.sum(axis=1, keepdims=True)
        fmt = '.2f'
    plt.figure(figsize=(8, 6))
    sns.heatmap(M, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Cifra predetta')
    plt.ylabel('Cifra reale')
    plt.title('Matrice di confusione' + (' (normalizzata)' if normalizzata else ''))
    plt.show()


def cifre_piu_difficili(etichette_vere, predizioni):
    """Stampa le cifre ordinate dalla più difficile alla più facile da
    classificare, con la rispettiva accuratezza."""
    from sklearn.metrics import confusion_matrix
    M = confusion_matrix(etichette_vere, predizioni)
    accuratezza_per_cifra = np.diag(M) / M.sum(axis=1)
    ordine = np.argsort(accuratezza_per_cifra)
    print("Cifre dalla più difficile alla più facile:")
    for cifra in ordine:
        print(f"  cifra {cifra}: {accuratezza_per_cifra[cifra]*100:.1f}%")


def mostra_errori_frequenti(immagini_test, etichette_vere, predizioni,
                            quanti_tipi=3, esempi_per_tipo=5):
    """Mostra esempi visivi delle confusioni più frequenti."""
    from sklearn.metrics import confusion_matrix
    M = confusion_matrix(etichette_vere, predizioni)
    np.fill_diagonal(M, 0)
    # trova le N celle (reale, predetta) con più errori
    coppie = []
    for i in range(10):
        for j in range(10):
            if M[i, j] > 0:
                coppie.append((i, j, M[i, j]))
    coppie.sort(key=lambda t: t[2], reverse=True)
    etichette_vere = np.asarray(etichette_vere)
    predizioni = np.asarray(predizioni)
    for reale, predetta, conteggio in coppie[:quanti_tipi]:
        print(f"\nEsempi di cifra {reale} classificata come {predetta} "
              f"({conteggio} casi):")
        indici = np.where((etichette_vere == reale) & (predizioni == predetta))[0]
        indici = indici[:esempi_per_tipo]
        if len(indici) == 0:
            continue
        immagini_errate = [immagini_test[i] for i in indici]
        etichette_errate = [f"vera:{reale} pred:{predetta}"] * len(immagini_errate)
        plot_imgs_labels(immagini_errate, etichette_errate,
                         rows=1, cols=len(immagini_errate))
