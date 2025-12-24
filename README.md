# Stream Screen 📺

Application de streaming d'écran optimisée pour Windows vers Raspberry Pi (ou tout autre récepteur compatible). Conçu pour être rapide, simple et performant.

![Stream Screen Icon](stream4.ico)

## 🚀 Fonctionnalités
- **Faible Latence** : Utilise DXCam ou MSS pour une capture ultra-rapide.
- **Support Multi-écrans** : Choisissez n'importe quel moniteur connecté.
- **Écran Virtuel** : Pas de second écran ? Créez-en un avec [Virtual Display Driver](https://github.com/VirtualDrivers/Virtual-Display-Driver/releases).
- **Lancement à distance** : Lancez le script de réception sur votre Raspberry Pi directement depuis l'interface via SSH.
- **Sécurisé** : Vos identifiants SSH sont chiffrés localement.

## 🛠️ Installation

### Sur le PC (Windows)
1. Téléchargez le dossier complet.
2. Lancez **`start.bat`**.
   - Ce script installera automatiquement les dépendances Python nécessaires (`requirements.txt`) et lancera l'interface.

### Sur le Raspberry Pi (Récepteur)
1. Assurez-vous d'avoir Python 3 installé.
2. Copiez le fichier **`stream_receiver.py`** sur votre Raspberry Pi.
   - *Emplacement recommandé :* Sur le Bureau (`/home/pi/Desktop/stream_receiver.py`).
   - *Note :* L'application PC est configurée par défaut pour chercher le fichier sur le Desktop, mais vous pouvez modifier ce chemin dans l'onglet "Raspberry Pi".

## 🎮 Utilisation

### Méthode 1 : Tout depuis le PC (Recommandé)
1. Ouvrez **Stream Screen** sur votre PC.
2. Allez dans l'onglet **Raspberry Pi**.
3. Entrez l'IP du Pi, votre identifiant (ex: `pi`), votre mot de passe et le chemin vers le script `stream_receiver.py`.
4. Cochez les cases "Mémoriser" pour ne plus avoir à les ressaisir.
5. Cliquez sur **"Lancer Receiver sur Pi (SSH)"**.
   - Cela va connecter le Pi, lancer le script de réception, et démarrer automatiquement le stream sur votre PC.

### Méthode 2 : Lancement Manuel
1. **Sur le Raspberry Pi**, ouvrez un terminal et lancez :
   ```bash
   python3 stream_receiver.py
   ```
   (Ajoutez l'IP de votre PC si nécessaire, ex: `python3 stream_receiver.py 192.168.1.15`, sinon il écoutera en mode broadcast).
2. **Sur le PC**, cliquez sur le gros bouton vert **"LANCER LE FLUX"**.

## 💡 Idées d'utilisation
L'application diffuse le flux vidéo de votre écran via votre **réseau local**, ce qui permet des usages simples et pratiques :

- **Câble HDMI Virtuel** : Connectez votre PC à la TV du salon (pilotée par un Raspberry Pi) sans tirer un câble de 10 mètres à travers la pièce.
- **Monitoring Déporté** : Gardez un œil sur une tâche en cours sur votre PC (rendu 3D, compilation, téléchargement) depuis une autre pièce équipée d'un petit écran connecté au réseau.

## 🔧 Dépannage
- **Écran noir ?** Vérifiez que le firewall Windows n'autorise pas Python.
- **Lenteur ?** Passez en mode "DXCam" et baissez la résolution en 720p.
- **Pas d'écran étendu ?** Si vous n'avez qu'un seul écran physique et souhaitez streamer un bureau séparé, installez le [Virtual Display Driver](https://github.com/VirtualDrivers/Virtual-Display-Driver/releases).

## 📂 Structure des fichiers
- `StreamScreen.pyw` : L'application principale (GUI).
- `stream_receiver.py` : Script à exécuter sur le client (Raspberry Pi).
- `stream_config.json` : Fichier de configuration (généré automatiquement, **ne pas partager**).
- `start.bat` : Lanceur automatique.
