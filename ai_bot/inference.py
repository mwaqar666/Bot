import os
import pandas as pd
from typing import Dict
from stable_baselines3 import PPO
from .data_engine.feature_engineer import FeatureEngineer


class AI_Analyst:
    """
    The Inference Engine.
    Loads the trained model and predicts the next move.
    """

    def __init__(self, model_path="ai_bot/models/ppo_trading_bot"):
        self.model_path = model_path
        self.model = None
        self.engineer = FeatureEngineer()

    def load_model(self):
        try:
            # Check if model exists
            if not os.path.exists(self.model_path + ".zip"):
                print("No trained model found. AI will be disabled until trained.")
                return

            self.model = PPO.load(self.model_path)
            print("AI Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load AI model: {e}")

    def analyze(self, df: pd.DataFrame):
        """
        Analyzes the market data and returns a trading signal.
        Args:
            df: 15m Dataframe
        Returns:
            action (str): 'buy', 'sell', 'hold'
            confidence (float): Probability of the action
        """
        if self.model is None:
            return "hold", 0.0

        try:
            # Feature Engineering (Level 2 Max)
            # We must pass the context dataframes here!
            df_processed = self.engineer.process_data(df)

            # Get the latest state
            latest_state = df_processed.iloc[-1].values

            # Predict Action
            action, _states = self.model.predict(latest_state, deterministic=True)
            action_idx = int(action)

            # Convert to tensor observation
            obs_tensor, _ = self.model.policy.obs_to_tensor(latest_state.reshape(1, -1))

            # Get distribution
            distribution = self.model.policy.get_distribution(obs_tensor)

            # Extract probabilities
            probs = distribution.distribution.probs.detach().cpu().numpy()[0]
            confidence = float(probs[action_idx])

            algo_action = ["hold", "buy", "sell"][action_idx]
            return algo_action, confidence

        except Exception as e:
            print(f"AI Analysis Error: {e}")
            return "hold", 0.0
