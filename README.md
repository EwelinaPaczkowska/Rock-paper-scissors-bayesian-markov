# Rock–Paper–Scissors with Bayesian Learning and Markov Model

## Description
This project implements a simulation of the Rock–Paper–Scissors game where one player follows a probabilistic strategy based on a Markov chain, and the other player learns and predicts this strategy using Bayesian updating.

The learning agent observes transitions between moves, estimates the opponent's transition probabilities, and selects counter-moves to maximize its score over time.

## Features
- Markov-based opponent with a predefined transition matrix
- Bayesian learning of transition probabilities (frequency-based update)
- Prediction of opponent's next move
- Strategy optimization using counter-moves
- Simulation of 1000+ rounds
- Tracking and visualization of cumulative score
- Comparison between true and learned transition matrices

## Technologies
- Python
- NumPy
- Matplotlib

## How it works
1. The opponent selects moves based on a transition matrix (Markov chain).
2. The learner observes move transitions.
3. Transition probabilities are updated using a Bayesian approach (counting + normalization).
4. The learner predicts the opponent’s next move.
5. A counter-move is chosen to maximize the chance of winning.
6. The process repeats over multiple rounds.

## Results
- The model learns the opponent's behavior over time.
- The cumulative score shows whether the learner gains advantage.
- The learned transition matrix approximates the true one.

## Usage
Run the simulation:

```bash
python main.py
