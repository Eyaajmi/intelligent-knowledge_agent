
TEST_CASES = [
    {
        "question": "Quelles sont les sanctions disciplinaires ?",
        "expected_page": 16
    },
    {
        "question": "Quelle est la durée maximale de la suspension de travail ?",
        "expected_page": 16
    },
    {
        "question": "Que se passe-t-il lorsque les sanctions disciplinaires sont épuisées dans l'année ?",
        "expected_page": 16
    },
    {
        "question": "Que se passe-t-il en cas de refus du contrôle médical ?",
        "expected_page": 11
    },

    # Questions reformulées
    {
        "question": "Quelles mesures peuvent être prises contre un employé qui commet une faute légère ?",
        "expected_page": 16
    },
    {
        "question": "Combien de jours au maximum un salarié peut-il être suspendu ?",
        "expected_page": 16
    },
    {
        "question": "Quelle sanction peut suivre plusieurs blâmes ?",
        "expected_page": 16
    },
    {
        "question": "Qu'arrive-t-il si un employé refuse le contrôle médical ?",
        "expected_page": 11
    },
  {
        "question": "Quels sont les jours fériés prévus par le règlement ?",
        "expected_page": 10
    },
    {
        "question": "Quel est le nombre de jours de congé accordé pour le mariage d'un employé ?",
        "expected_page": 11
    },
]

# Questions SANS réponse dans les documents : servent à calibrer le seuil d'abstention
# (MIN_SEMANTIC_SCORE). Ajoute-en de ton domaine : questions voisines mais absentes du corpus.
UNANSWERABLE_CASES = [
    {"question": "Quelle est la couleur du logo de Maroclear ?"},
    {"question": "Quelle est la capitale de l'Australie ?"},
    {"question": "Comment préparer une pizza margherita ?"},
    {"question": "Qui a gagné la coupe du monde de football en 2018 ?"},
]
