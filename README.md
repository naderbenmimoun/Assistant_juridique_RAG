# Assistant juridique RAG

Petit projet universitaire de chatbot RAG local pour interroger en français le corpus `AgentPublic/dole`. Il sert une interface vanilla HTML/CSS/JavaScript avec FastAPI, recherche les embeddings BGE-M3 déjà présents dans le dataset avec Qdrant, et génère des réponses locales avec Ollama/Qwen3.

## Démarrage

```bash
cp .env.example .env
docker compose up --build
```

Ouvrez ensuite <http://localhost:8000>. Au premier démarrage, Docker télécharge `qwen3:8b` et `bge-m3`, puis importe les 4 352 embeddings existants dans Qdrant. Les téléchargements sont importants et peuvent prendre plusieurs minutes. Les volumes `ollama_data` et `qdrant_data` rendent les démarrages suivants beaucoup plus rapides.

Pour une machine avec moins de mémoire, remplacez `OLLAMA_CHAT_MODEL=qwen3:8b` par `OLLAMA_CHAT_MODEL=qwen3:4b` dans `.env` avant de lancer Compose.

## Utilisation

Posez une question en français, par exemple :

```text
Quel est le montant de l'amende encourue en cas de fraude aux prestations sociales ?
```

Les conversations restent uniquement dans la mémoire du navigateur et disparaissent au rafraîchissement. Ne saisissez pas de données personnelles sensibles. Les réponses sont informatives, ne sont pas des conseils juridiques personnalisés et doivent être vérifiées dans les sources officielles ou auprès d'un professionnel qualifié.

## API

```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Quel est le délai pour le recours contentieux ?","history":[]}'
```

La réponse contient l'answer, la requête autonome éventuellement reformulée, les cartes de sources et les métriques de récupération. Les scores et métadonnées viennent directement de Qdrant.

## Architecture et choix issus du notebook

- FastAPI sert l'UI et les routes `GET /api/health` et `POST /api/chat`.
- Au démarrage, l'import idempotent conserve les 4 352 embeddings `embeddings_bge-m3` existants : il ne ré-encode pas le corpus.
- Qdrant utilise la distance cosinus sur des vecteurs BGE-M3 de dimension 1 024.
- Chaque requête est encodée par `langchain_ollama.OllamaEmbeddings` (`bge-m3`). La recherche est limitée à 5 résultats, puis applique exactement `min_score=0.40` et un arrêt sur écart de score `>= 0.04`.
- `qwen3:8b`, via `langchain_ollama.ChatOllama`, reformule prudemment les suivis dépendants puis répond uniquement avec le contexte retenu.
- Les sources et scores affichés proviennent du backend/Qdrant, jamais du texte généré.

L'application s'abstient sans appeler le modèle de génération lorsqu'aucune source ne passe les filtres : `Le contexte fourni ne permet pas de répondre à cette question.`

## Réparer volontairement un index incomplet

Une collection existante mais incomplète n'est jamais supprimée automatiquement. Après avoir vérifié que c'est souhaité :

```bash
docker compose run --rm api python -m app.ingestion --recreate
```

## Vérifications locales

```bash
python -m compileall app
pytest
docker compose config
```

Un test Docker complet nécessite que Docker, la mémoire disponible et le téléchargement des modèles soient opérationnels. Après démarrage, vérifiez `http://localhost:8000/api/health`; il doit signaler `4352` points. Qdrant et Ollama restent internes au réseau Compose : seul le port FastAPI 8000 est publié.

## Dépannage

- Si le premier démarrage semble long, attendez la fin de `ollama-pull` : les deux modèles doivent être téléchargés avant que l'API démarre.
- Si Qdrant signale un index incomplet, utilisez explicitement la commande de réparation ci-dessus.
- Si Qwen3:8b manque de mémoire, choisissez `qwen3:4b` dans `.env`, puis relancez la pile.
