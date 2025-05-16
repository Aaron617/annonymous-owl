import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Callable, Optional


class OWLNetwork(nn.Module):
    """
    Network architecture for OWL framework with both planner and actor policies
    """

    def __init__(
        self, state_dim, meta_action_dim, sub_state_dim, sub_action_dim, hidden_dim=128
    ):
        super(OWLNetwork, self).__init__()

        # Shared embedding network
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Planner policy network
        self.planner_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, meta_action_dim),
        )

        # Actor policy network for sub-agents
        self.actor_policy = nn.Sequential(
            nn.Linear(sub_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sub_action_dim),
        )

    def planner_forward(self, x):
        """Forward pass for planner"""
        features = self.shared_encoder(x)
        logits = self.planner_policy(features)
        return logits

    def actor_forward(self, sub_state):
        """Forward pass for actor/tool agent"""
        logits = self.actor_policy(sub_state)
        return logits

    def get_planner_log_prob(self, state, meta_action):
        """Compute log probability of meta-action given state"""
        logits = self.planner_forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, meta_action.unsqueeze(1)).squeeze(1)

    def get_actor_log_prob(self, sub_state, sub_action):
        """Compute log probability of sub-action given sub-state"""
        logits = self.actor_forward(sub_state)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, sub_action.unsqueeze(1)).squeeze(1)


class SubContext:
    """Container for a sub-agent's episode context and trajectory"""

    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.states = [initial_state]
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def add_transition(self, action, next_state, reward, log_prob):
        self.actions.append(action)
        self.states.append(next_state)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def get_return(self, gamma=0.99):
        """Calculate discounted return for this sub-context"""
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return sum(returns) if returns else 0


class OWLAgent:
    """
    Implementation of OWL hierarchical MDP agent with weighted SFT approach
    """

    def __init__(
        self,
        state_dim: int,
        meta_action_dim: int,
        sub_state_dim: int,
        sub_action_dim: int,
        num_sub_agents: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        context_mapping_fn: Callable = None,
        sft_weights: Optional[Dict] = None,
    ):
        self.network = OWLNetwork(
            state_dim, meta_action_dim, sub_state_dim, sub_action_dim
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.num_sub_agents = num_sub_agents

        # Function to map meta-actions to sub-contexts
        self.context_mapping_fn = context_mapping_fn

        # SFT weights for different components (weighted loss)
        self.sft_weights = sft_weights or {"planner": 1.0, "actors": 1.0}

    def select_meta_action(self, state):
        """Planner selects a meta-action"""
        with torch.no_grad():
            logits = self.network.planner_forward(state)
            probs = F.softmax(logits, dim=-1)
            meta_action = torch.multinomial(probs, 1).item()
        return meta_action

    def select_sub_action(self, sub_state):
        """Actor selects a sub-action"""
        with torch.no_grad():
            logits = self.network.actor_forward(sub_state)
            probs = F.softmax(logits, dim=-1)
            sub_action = torch.multinomial(probs, 1).item()
        return sub_action

    def generate_sub_contexts(self, meta_action):
        """Generate sub-contexts based on meta-action"""
        if self.context_mapping_fn is not None:
            return self.context_mapping_fn(meta_action, self.num_sub_agents)
        else:
            # Default: just replicate meta_action as the initial state for each sub-agent
            return [
                torch.tensor([meta_action], dtype=torch.float32)
                for _ in range(self.num_sub_agents)
            ]

    def train_step(self, state, meta_action, sub_contexts, planner_reward):
        """
        Perform a training step with weighted SFT approach

        Args:
            state: Global state
            meta_action: Selected meta-action
            sub_contexts: List of SubContext objects for each sub-agent
            planner_reward: Reward for the planner
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        meta_action_tensor = torch.LongTensor([meta_action])

        # Compute planner log probability
        planner_log_prob = self.network.get_planner_log_prob(
            state_tensor, meta_action_tensor
        )

        # Calculate global return
        global_return = planner_reward
        for context in sub_contexts:
            global_return += context.get_return(self.gamma)

        # Planner gradient (REINFORCE)
        planner_loss = -self.sft_weights["planner"] * planner_log_prob * global_return

        # Actor gradients (separated for each sub-agent)
        actor_losses = []

        for i, context in enumerate(sub_contexts):
            # For each action in the sub-agent's trajectory
            for t, log_prob in enumerate(context.log_probs):
                # We could implement baseline subtraction here as described in the paper
                # For now, using the simplest form without baseline
                actor_loss = -self.sft_weights["actors"] * log_prob * global_return
                actor_losses.append(actor_loss)

        # Combine losses
        total_loss = planner_loss
        if actor_losses:
            total_loss = total_loss + sum(actor_losses)

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "planner_loss": planner_loss.item(),
            "actor_losses": [loss.item() for loss in actor_losses],
            "global_return": global_return,
        }

    def train_with_weighted_sft(self, trajectories, weights=None):
        """
        Train with weighted SFT using collected trajectories

        Args:
            trajectories: List of dictionaries containing trajectories
            weights: Optional weighting for different trajectories (for SFT)
        """
        if weights is None:
            weights = [1.0] * len(trajectories)

        total_loss = 0

        for trajectory, weight in zip(trajectories, weights):
            state = trajectory["state"]
            meta_action = trajectory["meta_action"]
            sub_contexts = trajectory["sub_contexts"]
            planner_reward = trajectory["planner_reward"]

            # Apply the trajectory weight to the SFT weights
            original_weights = self.sft_weights.copy()
            for k in self.sft_weights:
                self.sft_weights[k] *= weight

            # Train on this trajectory
            loss_info = self.train_step(
                state, meta_action, sub_contexts, planner_reward
            )
            total_loss += loss_info["total_loss"]

            # Restore original weights
            self.sft_weights = original_weights

        return {"total_weighted_loss": total_loss}


# Example of how to use the implementation
def example_usage():
    # Define dimensions and parameters
    state_dim = 10
    meta_action_dim = 5
    sub_state_dim = 8
    sub_action_dim = 4
    num_sub_agents = 3

    # Create the agent
    owl_agent = OWLAgent(
        state_dim=state_dim,
        meta_action_dim=meta_action_dim,
        sub_state_dim=sub_state_dim,
        sub_action_dim=sub_action_dim,
        num_sub_agents=num_sub_agents,
        sft_weights={"planner": 1.5, "actors": 0.8},  # Weighted SFT approach
    )

    # Example of context mapping function
    def example_context_mapping(meta_action, num_agents):
        # This would typically create meaningful sub-contexts based on meta_action
        # For this example, we'll just create dummy contexts
        return [
            torch.ones(sub_state_dim) * (meta_action + i) for i in range(num_agents)
        ]

    owl_agent.context_mapping_fn = example_context_mapping

    # Example of collecting a trajectory
    state = np.random.rand(state_dim)
    meta_action = owl_agent.select_meta_action(torch.FloatTensor(state).unsqueeze(0))

    # Generate sub-contexts
    initial_sub_states = owl_agent.generate_sub_contexts(meta_action)
    sub_contexts = [SubContext(state) for state in initial_sub_states]

    # Simulate sub-agent episodes
    for i, context in enumerate(sub_contexts):
        for t in range(5):  # 5 timesteps per sub-episode
            current_state = context.states[-1]
            sub_action = owl_agent.select_sub_action(current_state.unsqueeze(0))

            # Simulate environment step (in a real environment, you'd get this from the env)
            next_state = current_state + 0.1 * torch.randn_like(current_state)
            reward = float(torch.sum(next_state) * 0.01)

            # Compute log probability
            sub_action_tensor = torch.LongTensor([sub_action])
            log_prob = owl_agent.network.get_actor_log_prob(
                current_state.unsqueeze(0), sub_action_tensor
            )

            # Add to trajectory
            context.add_transition(sub_action, next_state, reward, log_prob)

    # Simulate planner reward
    planner_reward = sum([c.get_return() for c in sub_contexts]) * 0.1

    # Create a trajectory
    trajectory = {
        "state": state,
        "meta_action": meta_action,
        "sub_contexts": sub_contexts,
        "planner_reward": planner_reward,
    }

    # Train with this trajectory
    owl_agent.train_step(state, meta_action, sub_contexts, planner_reward)

    # Alternatively, you can use the weighted SFT approach with multiple trajectories
    trajectories = [
        trajectory
    ] * 3  # Just using the same trajectory 3 times for the example
    weights = [1.0, 0.8, 1.2]  # Different weights for different trajectories

    loss_info = owl_agent.train_with_weighted_sft(trajectories, weights)
    print(f"Training loss: {loss_info['total_weighted_loss']}")


if __name__ == "__main__":
    example_usage()
