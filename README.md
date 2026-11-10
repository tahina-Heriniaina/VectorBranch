# VectorBench

**Plateforme de comparaison et d'évaluation des bases de données vectorielles**

## Présentation du projet

VectorBench est une application permettant de comparer les performances de plusieurs moteurs de bases de données vectorielles.

L'objectif est d'évaluer leurs performances selon plusieurs critères, notamment :
- La précision des résultats de recherche.
- Le temps de réponse (latence).
- La consommation des ressources.
- Les performances de l'indexation.

### Moteurs vectoriels étudiés

- Qdrant
- Milvus
- Weaviate
- Chroma
- FAISS
- pgvector
- Pinecone

## Technologies utilisées

| Partie | Technologies |
|---|---|
| Backend | Python, FastAPI |
| Frontend | React, TypeScript, Vite |
| Bases vectorielles | Qdrant, Milvus, Weaviate, Chroma, FAISS, pgvector, Pinecone |
| Conteneurisation | Docker |
| Gestion de versions | Git, GitHub |

## Prérequis

Avant de commencer, installer :

- [Python](https://www.python.org/downloads/)
- [Node.js et npm](https://nodejs.org/)
- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

## 1. Cloner le projet

Ouvrir un terminal et exécuter :

```bash
git clone https://github.com/tahina-Heriniaina/VectorBranch.git
```

Accéder au dossier du projet :

```bash
cd VectorBranch
```

## 2. Installation du backend

Accéder au dossier backend :

```bash
cd backend
```

Créer un environnement virtuel Python :

```bash
python -m venv venv
```

Activer l'environnement virtuel sous Windows PowerShell :

```powershell
.\venv\Scripts\Activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

### Configuration des variables d'environnement

Créer ou compléter le fichier `backend/.env` avec les paramètres nécessaires à la configuration des moteurs vectoriels.

Ne pas publier les clés API, mots de passe ou autres informations confidentielles sur GitHub.

### Lancement du backend

Depuis le dossier `backend`, exécuter :

```bash
uvicorn app.main:app --reload
```

Le backend sera accessible à l'adresse :

http://127.0.0.1:8000

Documentation interactive de l'API :

http://127.0.0.1:8000/docs

## 3. Installation du frontend

Ouvrir un deuxième terminal à la racine du projet :

```bash
cd frontend
```

Installer les dépendances :

```bash
npm install
```

### Configuration du frontend

Configurer le fichier `.env` du frontend si nécessaire, notamment l'adresse de l'API backend.

### Lancement du frontend

```bash
npm run dev
```

L'interface sera accessible à l'adresse :

http://localhost:5173

## 4. Lancement des services Docker

Pour démarrer les services configurés dans le fichier Docker Compose à la racine du projet :

```bash
docker compose up -d
```

Pour vérifier les conteneurs :

```bash
docker ps
```

Pour arrêter les services :

```bash
docker compose stop
```

**Attention :** ne pas utiliser `docker compose down -v` si des données locales doivent être conservées.

## 5. Structure du projet

```text
VectorBench/
│
├── backend/
│   ├── app/
│   │   ├── embeddings/
│   │   ├── engines/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── package-lock.json
│
├── docker/
│   └── milvus/
│
├── docker-compose.yml
└── README.md
```

## 6. Utilisation de l'application

Après avoir démarré le backend, le frontend et les services nécessaires :

1. Accéder à l'interface web.
2. Sélectionner un moteur vectoriel.
3. Importer un jeu de données.
4. Effectuer une recherche vectorielle.
5. Lancer un benchmark.
6. Consulter les résultats et comparer les performances des moteurs.
