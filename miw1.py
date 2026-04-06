import numpy as np
import matplotlib.pyplot as plt

MOVES = ["P", "K", "N"]
MOVE_TO_IDX = {m: i for i, m in enumerate(MOVES)}
IDX_TO_MOVE = {i: m for i, m in enumerate(MOVES)}

BEATS = {
    "P": "K",
    "K": "N",
    "N": "P"
}


def counter_move(predicted_move):
    for move, beats in BEATS.items():
        if beats == predicted_move:
            return move
    raise ValueError("Nieznany ruch.")


def result(player_move, opponent_move):
    if player_move == opponent_move:
        return 0
    elif BEATS[player_move] == opponent_move:
        return 1
    else:
        return -1


class MarkovOpponent:
    def __init__(self, transition_matrix, initial_distribution=None, seed=None):
        self.transition_matrix = np.array(transition_matrix, dtype=float)
        self.rng = np.random.default_rng(seed)

        if initial_distribution is None:
            self.initial_distribution = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        else:
            self.initial_distribution = np.array(initial_distribution, dtype=float)

        self.last_move_idx = None

    def play(self):
        if self.last_move_idx is None:
            move_idx = self.rng.choice(3, p=self.initial_distribution)
        else:
            probs = self.transition_matrix[self.last_move_idx]
            move_idx = self.rng.choice(3, p=probs)

        self.last_move_idx = move_idx
        return IDX_TO_MOVE[move_idx]


class BayesianLearner:
    def __init__(self, prior_strength=1.0):
        self.alpha = np.ones((3, 3), dtype=float) * prior_strength
        self.last_observed_move_idx = None

    def update(self, observed_move):
        current_idx = MOVE_TO_IDX[observed_move]

        if self.last_observed_move_idx is not None:
            self.alpha[self.last_observed_move_idx, current_idx] += 1

        self.last_observed_move_idx = current_idx

    def estimated_transition_matrix(self):
        row_sums = self.alpha.sum(axis=1, keepdims=True)
        return self.alpha / row_sums

    def predict_next_move_distribution(self):
        if self.last_observed_move_idx is None:
            return np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)

        T_hat = self.estimated_transition_matrix()
        return T_hat[self.last_observed_move_idx]

    def choose_move(self):
        pred_dist = self.predict_next_move_distribution()
        predicted_idx = np.argmax(pred_dist)
        predicted_move = IDX_TO_MOVE[predicted_idx]
        return counter_move(predicted_move)


def print_matrix(matrix, title):
    print(f"\n{title}")
    print("        P       K       N")
    for i, row_name in enumerate(MOVES):
        row = matrix[i]
        print(f"{row_name}   {row[0]:.3f}   {row[1]:.3f}   {row[2]:.3f}")


def simulate_game(n_rounds=1000, seed=42):
    true_transition_matrix = np.array([
        [0.15, 0.70, 0.15],
        [0.20, 0.20, 0.60],
        [0.65, 0.20, 0.15],
    ])

    opponent = MarkovOpponent(true_transition_matrix, seed=seed)
    learner = BayesianLearner(prior_strength=1.0)

    round_scores = []
    cumulative_scores = []

    opponent_moves = []
    learner_moves = []

    total_score = 0

    for round_idx in range(n_rounds):
        learner_move = learner.choose_move()

        opponent_move = opponent.play()

        score = result(learner_move, opponent_move)
        total_score += score

        round_scores.append(score)
        cumulative_scores.append(total_score)

        learner_moves.append(learner_move)
        opponent_moves.append(opponent_move)

        learner.update(opponent_move)

    learned_transition_matrix = learner.estimated_transition_matrix()

    return {
        "true_transition_matrix": true_transition_matrix,
        "learned_transition_matrix": learned_transition_matrix,
        "round_scores": round_scores,
        "cumulative_scores": cumulative_scores,
        "opponent_moves": opponent_moves,
        "learner_moves": learner_moves,
        "final_score": total_score
    }


def plot_cumulative_score(cumulative_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_scores, label="Skumulowany wynik gracza uczącego się")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Runda")
    plt.ylabel("Skumulowany wynik")
    plt.title("Papier–Kamień–Nożyce: skumulowany wynik w czasie")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = simulate_game(n_rounds=2000, seed=123)

    print_matrix(results["true_transition_matrix"], "Prawdziwa macierz przejść przeciwnika:")
    print_matrix(results["learned_transition_matrix"], "Wyuczona macierz przejść (bayesowsko):")

    print(f"\nKońcowy wynik gracza uczącego się po {len(results['round_scores'])} rundach: {results['final_score']}")

    plot_cumulative_score(results["cumulative_scores"])