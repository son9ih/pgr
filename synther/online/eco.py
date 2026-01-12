"""
ECO (Episodic Curiosity Objective) implementation for proprioceptive states.
Based on "Episodic Curiosity through Reachability" (Savinov et al., 2018).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import random


class EmbeddingNetwork(nn.Module):
    """Embedding network E: maps observation to 512-dim embedding.
    For proprioceptive states, we use MLP instead of ResNet-18.
    """
    def __init__(self, input_dim: int, embedding_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # MLP for proprioceptive states (similar to ResNet-18 but simpler)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] observation tensor
        Returns:
            embedding: [batch, embedding_dim] embedding tensor
        """
        return self.network(x)


class ComparatorNetwork(nn.Module):
    """Comparator network C: computes reachability probability between two embeddings.
    4-layer MLP as specified in the paper.
    """
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 4-layer MLP comparator
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),  # Concatenate two embeddings
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Binary classification: reachable/not reachable
            nn.Softmax(dim=1)
        )
    
    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1: [batch, embedding_dim] first embedding
            emb2: [batch, embedding_dim] second embedding
        Returns:
            similarity: [batch, 2] probability distribution (dissimilar=0, similar=1)
        """
        x = torch.cat([emb1, emb2], dim=1)
        return self.network(x)


class RNetwork(nn.Module):
    """R-Network: Siamese architecture combining embedding and comparator networks.
    """
    def __init__(self, input_dim: int, embedding_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_net = EmbeddingNetwork(input_dim, embedding_dim, hidden_dim)
        self.comparator_net = ComparatorNetwork(embedding_dim, hidden_dim)
    
    def embed_observation(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a single observation.
        Args:
            x: [batch, input_dim] observation tensor
        Returns:
            embedding: [batch, embedding_dim]
        """
        return self.embedding_net(x)
    
    def embedding_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between two embeddings.
        Args:
            emb1: [batch, embedding_dim] first embedding
            emb2: [batch, embedding_dim] second embedding
        Returns:
            similarity: [batch] probability that states are reachable (0~1)
        """
        output = self.comparator_net(emb1, emb2)
        return output[:, 1]  # Return probability of "similar" class
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Full forward pass: two observations -> similarity probability.
        Args:
            x1: [batch, input_dim] first observation
            x2: [batch, input_dim] second observation
        Returns:
            similarity: [batch] probability that states are reachable
        """
        emb1 = self.embed_observation(x1)
        emb2 = self.embed_observation(x2)
        return self.embedding_similarity(emb1, emb2)


class EpisodicMemory:
    """Episodic memory M: stores embeddings of past observations.
    Paper: |M| = 200, replacement='random' (geometric distribution)
    """
    def __init__(self, capacity: int = 200, replacement: str = 'random'):
        self.capacity = capacity
        self.replacement = replacement  # 'random' or 'fifo'
        self.reset()
    
    def reset(self):
        """Reset episodic memory."""
        self._count = 0
        self._obs_memory = []  # List of embeddings
        self._memory_age = []  # Age of each memory entry
    
    def __len__(self):
        return min(self._count, self.capacity)
    
    def add(self, embedding: np.ndarray):
        """Add an embedding to memory.
        Args:
            embedding: [embedding_dim] numpy array
        """
        embedding = np.array(embedding).flatten()
        
        if self._count < self.capacity:
            # Memory not full yet
            index = self._count
            if len(self._obs_memory) <= index:
                self._obs_memory.append(embedding)
                self._memory_age.append(self._count)
            else:
                self._obs_memory[index] = embedding
                self._memory_age[index] = self._count
        else:
            # Memory full, need replacement
            if self.replacement == 'random':
                # Random replacement (geometric distribution)
                index = np.random.randint(0, self.capacity)
            elif self.replacement == 'fifo':
                # FIFO replacement (circular buffer)
                index = self._count % self.capacity
            else:
                raise ValueError(f'Invalid replacement scheme: {self.replacement}')
            
            self._obs_memory[index] = embedding
            self._memory_age[index] = self._count
        
        self._count += 1
    
    def similarity(self, embedding: np.ndarray, r_network: RNetwork, device: torch.device) -> np.ndarray:
        """Compute similarity between embedding and all memories.
        Args:
            embedding: [embedding_dim] numpy array
            r_network: RNetwork instance
            device: torch device
        Returns:
            similarities: [memory_size] numpy array of similarities
        """
        memory_length = len(self)
        if memory_length == 0:
            return np.array([])
        
        # Convert to tensors
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(device)  # [1, embedding_dim]
        memory_embeddings = torch.FloatTensor(np.array(self._obs_memory[:memory_length])).to(device)  # [memory_size, embedding_dim]
        
        # Replicate embedding for batch comparison
        embedding_batch = embedding_tensor.repeat(memory_length, 1)  # [memory_size, embedding_dim]
        
        # Compute similarities
        with torch.no_grad():
            similarities = r_network.embedding_similarity(embedding_batch, memory_embeddings)
        
        return similarities.cpu().numpy()


def similarity_to_memory(embedding: np.ndarray, episodic_memory: EpisodicMemory, 
                         r_network: RNetwork, device: torch.device,
                         similarity_aggregation: str = 'percentile') -> float:
    """Compute aggregated similarity to episodic memory.
    Paper: F = percentile-90
    
    Args:
        embedding: [embedding_dim] numpy array
        episodic_memory: EpisodicMemory instance
        r_network: RNetwork instance
        device: torch device
        similarity_aggregation: 'percentile', 'max', 'nth_largest', 'relative_count'
    
    Returns:
        aggregated_similarity: scalar (0~1)
    """
    memory_length = len(episodic_memory)
    if memory_length == 0:
        return 0.0
    
    similarities = episodic_memory.similarity(embedding, r_network, device)
    
    if similarity_aggregation == 'max':
        aggregated = np.max(similarities)
    elif similarity_aggregation == 'nth_largest':
        n = min(10, memory_length)
        aggregated = np.partition(similarities, -n)[-n]
    elif similarity_aggregation == 'percentile':
        percentile = 90
        aggregated = np.percentile(similarities, percentile)
    elif similarity_aggregation == 'relative_count':
        count = np.sum(similarities > 0.5)
        aggregated = float(count) / len(similarities)
    else:
        raise ValueError(f'Invalid similarity_aggregation: {similarity_aggregation}')
    
    return float(aggregated)


class ECO:
    """ECO (Episodic Curiosity Objective) for computing intrinsic rewards.
    
    Formula: F(s, a, s', r) = α (β - F(C(E(s), E(si)))) ∀si ∈ M
    where:
        - E: Embedding network
        - C: Comparator network
        - M: Episodic memory (|M| = 200)
        - F: Aggregation function (percentile-90)
        - α: Scale factor (0.03)
        - β: Bias term (0.5)
    """
    def __init__(self, obs_dim: int, embedding_dim: int = 512, hidden_dim: int = 512,
                 memory_capacity: int = 200, replacement: str = 'random',
                 alpha: float = 0.03, beta: float = 0.5,
                 similarity_threshold: float = 0.5,
                 similarity_aggregation: str = 'percentile',
                 device: Optional[torch.device] = None):
        """
        Args:
            obs_dim: Observation dimension
            embedding_dim: Embedding dimension (default: 512)
            hidden_dim: Hidden layer dimension (default: 512)
            memory_capacity: Episodic memory capacity (default: 200)
            replacement: Memory replacement strategy ('random' or 'fifo')
            alpha: Scale factor for intrinsic reward (default: 0.03)
            beta: Bias term (default: 0.5)
            similarity_threshold: Threshold for adding to memory (default: 0.5)
            similarity_aggregation: Aggregation method ('percentile', 'max', etc.)
            device: torch device (auto-detect if None)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.beta = beta
        self.similarity_threshold = similarity_threshold
        self.similarity_aggregation = similarity_aggregation
        
        # Initialize R-network
        self.r_network = RNetwork(obs_dim, embedding_dim, hidden_dim).to(device)
        self.r_network.eval()  # Evaluation mode by default
        
        # Initialize episodic memory
        self.episodic_memory = EpisodicMemory(capacity=memory_capacity, replacement=replacement)
        
        # Track episode state
        self.current_episode_done = False
    
    def reset_episode(self):
        """Reset episodic memory at the start of a new episode."""
        self.episodic_memory.reset()
        self.current_episode_done = False
    
    def compute_reward(self, current_obs: torch.Tensor, done: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute ECO intrinsic reward for current observation.
        
        According to the paper: "The episodic curiosity (EC) module takes the current observation o as input"
        
        Args:
            current_obs: [batch, obs_dim] or [obs_dim] tensor - current observation at time t
            done: [batch] or scalar tensor indicating episode end (optional)
        
        Returns:
            intrinsic_reward: [batch] or scalar tensor
        """
        # Handle single observation
        if len(current_obs.shape) == 1:
            current_obs = current_obs.unsqueeze(0)
            single_obs = True
        else:
            single_obs = False
        
        batch_size = current_obs.shape[0]
        rewards = []
        
        for i in range(batch_size):
            obs_i = current_obs[i]
            done_i = done[i].item() if done is not None and len(done.shape) > 0 else (done.item() if done is not None else False)
            
            # Embed observation
            with torch.no_grad():
                embedding = self.r_network.embed_observation(obs_i.unsqueeze(0))  # [1, embedding_dim]
                embedding_np = embedding.cpu().numpy().squeeze()  # [embedding_dim]
            
            # Compute similarity to memory
            similarity = similarity_to_memory(
                embedding_np, self.episodic_memory, self.r_network, self.device,
                self.similarity_aggregation
            )
            
            # Compute intrinsic reward: α (β - similarity)
            if done_i:
                intrinsic_reward = 0.0
            else:
                intrinsic_reward = self.alpha * (self.beta - similarity)
            
            # Update episodic memory
            if not done_i:
                # Only add if similarity is below threshold
                if similarity < self.similarity_threshold:
                    self.episodic_memory.add(embedding_np)
            else:
                # Episode ended: reset memory and add first state of new episode
                self.episodic_memory.reset()
                self.episodic_memory.add(embedding_np)
            
            rewards.append(intrinsic_reward)
        
        result = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        return result.squeeze() if single_obs else result
    
    def train_r_network(self, optimizer, positive_pairs, negative_pairs, epochs: int = 10):
        """Train R-network on positive and negative pairs.
        
        Args:
            optimizer: torch optimizer for R-network
            positive_pairs: List of (obs1, obs2) tuples (reachable pairs)
            negative_pairs: List of (obs1, obs2) tuples (not reachable pairs)
            epochs: Number of training epochs
        """
        self.r_network.train()
        
        # Prepare data
        all_x1 = []
        all_x2 = []
        all_labels = []
        
        for obs1, obs2 in positive_pairs:
            all_x1.append(obs1)
            all_x2.append(obs2)
            all_labels.append(1)  # Reachable
        
        for obs1, obs2 in negative_pairs:
            all_x1.append(obs1)
            all_x2.append(obs2)
            all_labels.append(0)  # Not reachable
        
        # Convert to tensors
        x1_tensor = torch.FloatTensor(np.array(all_x1)).to(self.device)
        x2_tensor = torch.FloatTensor(np.array(all_x2)).to(self.device)
        labels_tensor = torch.LongTensor(all_labels).to(self.device)
        labels_onehot = F.one_hot(labels_tensor, num_classes=2).float()
        
        # Training loop
        batch_size = 64
        num_batches = (len(all_x1) + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Shuffle data
            indices = torch.randperm(len(all_x1)).to(self.device)
            x1_shuffled = x1_tensor[indices]
            x2_shuffled = x2_tensor[indices]
            labels_shuffled = labels_onehot[indices]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_x1))
                
                x1_batch = x1_shuffled[start_idx:end_idx]
                x2_batch = x2_shuffled[start_idx:end_idx]
                labels_batch = labels_shuffled[start_idx:end_idx]
                
                # Forward pass
                emb1 = self.r_network.embed_observation(x1_batch)
                emb2 = self.r_network.embed_observation(x2_batch)
                output = self.r_network.comparator_net(emb1, emb2)
                
                # Loss: categorical cross-entropy
                loss = F.cross_entropy(output, labels_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.r_network.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 100 == 0:
                print(f'R-network training epoch {epoch}/{epochs}, loss: {total_loss/num_batches:.4f}')
        
        self.r_network.eval()

