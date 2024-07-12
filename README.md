# Movie Recommendation System

## Project Overview

This project implements a movie recommendation system using BERT (Bidirectional Encoder Representations from Transformers) for encoding movie overviews and calculating cosine similarity for recommendations. It utilizes the `tmdb_5000_credits.csv` and `tmdb_5000_movies.csv` datasets to generate movie recommendations based on a given movie title.

## Features

- Movie Overview Encoding:** Uses BERT to encode movie overviews into dense vectors.
- Recommendation Engine:** Computes cosine similarity between movie embeddings to suggest similar movies.
- Python Libraries: Utilizes `transformers`, `torch`, `numpy`, `pandas`, and `scikit-learn` for implementation.

## Getting Started

### Prerequisites

- Python 3.7+
- Pip (Python package manager)

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

### 3. **Run the Script**

Explain how to execute the main script and provide a brief description of what the script does.

```markdown```
### Run the Script

Execute the `recommendation_system.py` script to run the recommendation engine and get movie recommendations based on a given title.

```bash
python src/recommendation_system.py

#Example Usage

from src.recommendation_system import get_recommendations
print(get_recommendations('The Dark Knight Rises'))
print(get_recommendations('The Matrix'))

```

Output: 

Recommended movies for "The Matrix":
0    The Matrix Revolutions
1    The Matrix Reloaded
2    Inception
3    The Dark Knight
4    Interstellar
5    V for Vendetta
6    Watchmen
7    Man of Steel
8    The Prestige
9    Batman Begins


We welcome contributions to improve this project! If you have suggestions, bug fixes, or feature requests, please follow these steps:

1. **Fork the repository** on GitHub.
2. **Create a new branch** for your changes.
3. **Make your changes** and commit them with descriptive messages.
4. **Push your changes** to your forked repository.
5. **Create a pull request** from your forked repository to the main repository.

### Reporting Issues

If you encounter any bugs or have suggestions for improvements, please open a new issue in the [Issues](https://github.com/yourusername/movie-recommendation-system/issues) section.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **BERT:** The BERT model is provided by [Hugging Face Transformers](https://github.com/huggingface/transformers).
- **TMDB Dataset:** The movie metadata used in this project comes from [TMDB](https://www.kaggle.com/tmdb/tmdb-movie-metadata).




