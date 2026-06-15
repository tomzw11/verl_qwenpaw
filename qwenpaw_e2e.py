
"""
QwenPaw + verl Agent Framework Integration

This file demonstrates how QwenPaw integrates with verl's Agent Framework:

1. verl's OpenAICompatibleAgentFramework manages the rollout process
2. verl's GatewayServingRuntime provides OpenAI-compatible HTTP interface
3. QwenPaw acts as an external agent that uses the Gateway's API
4. The Gateway internally uses vLLM for inference and collects token-level trajectories

Architecture:
verl Framework -> Gateway (vLLM inside) <-> QwenPaw Agent
"""
from __future__ import annotations
import asyncio
import json
import os
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

import httpx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# verl imports
from verl.agent.framework.framework import OpenAICompatibleAgentFramework
from verl.agent.framework.types import SessionHandle, Trajectory, SessionRuntime
from verl.agent.gateway.runtime import GatewayServingRuntime
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput
from verl.utils.tensordict_utils import get_tensordict


# ==========================================
# Configuration
# ==========================================
@dataclass
class Config:
    model_path: str = "models/qwen2.5-0.5b-instruct"
    qwenpaw_url: str = "http://127.0.0.1:52143"
    use_dummy_inference: bool = True  # Set to False when vLLM is available
    max_steps: int = 3
    batch_size: int = 1
    learning_rate: float = 2e-5
    grpo_n: int = 4  # Number of rollouts per prompt for GRPO group sampling
    clip_eps: float = 0.2  # Clip ratio for GRPO


config = Config()


# ==========================================
# Mock Components (for demonstration without vLLM)
# ==========================================
class MinimalTokenizer:
    """Simple tokenizer for demonstration when full model is not available."""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        if tokenize:
            return [i % self.vocab_size for i in range(len(str(messages)))]
        return str(messages)
        
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join([f"tok_{i}" for i in token_ids])
        
    def encode(self, text, add_special_tokens=False):
        return [hash(word) % self.vocab_size for word in text.split()]


class DummyRolloutBackend:
    """
    Dummy inference backend that simulates vLLM's generate interface.
    
    In production, this is replaced by:
    - verl.workers.rollout.vllm_rollout.vllm_async_server.VLLMRolloutServer
    - Or sglang backend
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    async def generate(
        self,
        request_id: str,
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
        image_data: List[Any] | None = None,
        video_data: List[Any] | None = None,
    ) -&gt; TokenOutput:
        """
        Simulate generation. In production, this calls vLLM internally.
        
        Returns token_ids and log_probs like real vLLM would.
        """
        print(f"  [DUMMY] vLLM-style generate for request {request_id}")
        
        # Generate dummy but reasonable output
        max_tokens = sampling_params.get("max_tokens", 64)
        token_ids = [i % 1000 for i in range(len(prompt_ids), len(prompt_ids) + max_tokens)]
        
        # Dummy log_probs
        log_probs = [-0.1 for _ in range(len(token_ids))]
        
        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason="stop"
        )


class DummySessionRuntime:
    """
    Dummy SessionRuntime that simulates GatewayServingRuntime.
    
    In production, use verl.agent.gateway.runtime.GatewayServingRuntime
    which internally manages vLLM and GatewayActors.
    """
    def __init__(self, tokenizer, backend):
        self.tokenizer = tokenizer
        self.backend = backend
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._session_counter = 0
        
    async def create_session(self, session_id: str, **kwargs) -&gt; SessionHandle:
        """
        Create a new session. In real Gateway, this starts a GatewayActor
        and returns its base_url.
        """
        print(f"  [DUMMY] create_session(session_id={session_id})")
        
        # In real Gateway, base_url would point to a real HTTP server
        handle = SessionHandle(
            session_id=session_id,
            base_url=f"http://dummy-gateway-{self._session_counter}:8000/v1"
        )
        
        self._session_counter += 1
        self.sessions[session_id] = {
            "handle": handle,
            "trajectories": [],
            "completed": False
        }
        
        return handle
        
    async def complete_session(self, session_id: str) -&gt; None:
        """Mark session as completed."""
        if session_id in self.sessions:
            self.sessions[session_id]["completed"] = True
            
    async def finalize_session(self, session_id: str) -&gt; List[Trajectory]:
        """
        Finalize session and return collected trajectories.
        
        In real Gateway, this gets all the token-level trajectories
        collected by the GatewayActor.
        """
        print(f"  [DUMMY] finalize_session(session_id={session_id})")
        
        if session_id not in self.sessions:
            return []
            
        session = self.sessions[session_id]
        
        # If no trajectories yet, create a dummy one
        if not session["trajectories"]:
            dummy_trajectory = Trajectory(
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6, 7, 8],
                response_mask=[1, 1, 1, 1, 1],
                response_logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
                reward_info={},
                reward_score=0.0,
                num_turns=1,
                multi_modal_data=None,
                extra_fields={}
            )
            session["trajectories"] = [dummy_trajectory]
            
        return session["trajectories"]
        
    async def abort_session(self, session_id: str) -&gt; None:
        """Abort a session."""
        pass
        
    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -&gt; None:
        """Wait for session completion."""
        pass
        
    async def shutdown(self) -&gt; None:
        """Shutdown the runtime."""
        pass


# ==========================================
# QwenPaw Agent Runner
# ==========================================
class QwenPawAgent:
    """
    QwenPaw agent that uses verl's Gateway API.
    
    This is the agent_runner that verl's framework calls.
    """
    def __init__(self, qwenpaw_url: str = config.qwenpaw_url):
        self.qwenpaw_url = qwenpaw_url
        self._lock = asyncio.Lock()
        
    async def chat_with_qwenpaw(self, prompt: str) -&gt; str:
        """
        Call QwenPaw to get the user's input/response.
        
        QwenPaw acts as the UI, and the actual inference is done
        by verl's Gateway + vLLM.
        """
        async with self._lock:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.qwenpaw_url}/api/console/chat",
                        json={
                            "text": prompt,
                            "regenerate": False,
                            "sessionId": "test-session"
                        }
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return data.get("text", "")
            except Exception as e:
                print(f"  [QwenPaw] Error calling QwenPaw: {e}")
                
        # Fallback to dummy response
        return f"This is a dummy response to: {prompt[:50]}..."
        
    async def run(
        self,
        raw_prompt: List[Dict[str, Any]],
        session: SessionHandle,
        sample_index: int,
        **kwargs
    ) -&gt; None:
        """
        Agent runner function called by verl's OpenAICompatibleAgentFramework.
        
        This is the main entry point for QwenPaw integration:
        1. Get initial prompt from verl
        2. Use QwenPaw as UI to get user interaction
        3. Call Gateway's /v1/chat/completions for inference
        4. Call /complete when done
        
        Args:
            raw_prompt: OpenAI-style messages list
            session: SessionHandle with base_url pointing to Gateway
            sample_index: Index of the current sample
        """
        print(f"\n  [QwenPaw Agent] Starting session {session.session_id} for sample {sample_index}")
        print(f"  [QwenPaw Agent] Gateway URL: {session.base_url}")
        
        # Extract user question from raw_prompt
        user_question = "Hello"
        if raw_prompt and len(raw_prompt) &gt; 0:
            last_msg = raw_prompt[-1]
            if isinstance(last_msg, dict):
                user_question = last_msg.get("content", "Hello")
        
        print(f"  [QwenPaw Agent] User question: {user_question}")
        
        # Step 1: Get user interaction from QwenPaw
        qwenpaw_response = await self.chat_with_qwenpaw(user_question)
        print(f"  [QwenPaw Agent] QwenPaw says: {qwenpaw_response[:100]}...")
        
        # Step 2: Call Gateway's /v1/chat/completions for real inference
        # In this dummy implementation, we simulate the HTTP call
        # In production, this would be a real httpx call to session.base_url
        print(f"  [QwenPaw Agent] Calling Gateway /v1/chat/completions (simulated)")
        
        # Simulate chat completions
        assistant_response = {
            "role": "assistant",
            "content": qwenpaw_response
        }
        
        # Step 3: Call /complete to finalize the session
        print(f"  [QwenPaw Agent] Calling Gateway /complete (simulated)")
        
        # In production, this would be a real httpx call to:
        # session.base_url.removesuffix("/v1") + "/complete"
        
        print(f"  [QwenPaw Agent] Session {session.session_id} completed")


# ==========================================
# GSM8K Reward Function
# ==========================================
def compute_gsm8k_reward(
    trajectories: List[Trajectory],
    ground_truth: str,
    tokenizer
) -&gt; List[float]:
    """
    Compute GSM8K-style reward for trajectories.
    
    In real GRPO, this is done by verl's RewardLoopWorker.
    """
    rewards = []
    for traj in trajectories:
        # Decode response
        response_text = tokenizer.decode(traj.response_ids, skip_special_tokens=True)
        
        # Extract answer (simplified)
        is_correct = "correct" in response_text.lower() or "yes" in response_text.lower()
        
        reward = 1.0 if is_correct else 0.0
        rewards.append(reward)
        
    return rewards


# ==========================================
# Mock Core Algorithms (GRPO Advantage, GRPO Loss)
# ==========================================
def compute_mock_grpo_advantage(
    rewards: List[float],
    group_indices: List[int],
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True
) -&gt; List[float]:
    """
    Mock GRPO advantage computation.
    
    In production, use:
    verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage
    """
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 1.0
    
    if norm_adv_by_std_in_grpo:
        advantages = [(r - mean_reward) / (std_reward + epsilon) for r in rewards]
    else:
        advantages = [r - mean_reward for r in rewards]
    
    return advantages


def compute_mock_grpo_loss(
    log_probs: List[float],
    old_log_probs: List[float],
    advantages: List[float],
    clip_eps: float = 0.2
) -&gt; float:
    """
    Mock GRPO policy loss computation.
    
    GRPO loss is similar to PPO clip loss but without value loss.
    It also usually includes KL loss from reference policy.
    
    In production, use verl's policy loss function.
    """
    # GRPO policy loss (clip)
    policy_loss = 0.0
    for log_prob, old_log_prob, adv in zip(log_probs, old_log_probs, advantages):
        # Probability ratio
        ratio = torch.exp(torch.tensor(log_prob) - torch.tensor(old_log_prob))
        # Clipped ratio
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        # Surrogate loss
        surr1 = ratio * torch.tensor(adv)
        surr2 = clipped_ratio * torch.tensor(adv)
        # Policy loss (take min and negate for gradient ascent)
        policy_loss += -torch.min(surr1, surr2).item()
    
    # Average loss
    if log_probs:
        policy_loss /= len(log_probs)
    
    return policy_loss


# ==========================================
# Policy Model Training (GRPO)
# ==========================================
class PolicyTrainer:
    def __init__(self, model_path: str = config.model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [PolicyTrainer] Loading model from {model_path}")
        
        # Load model and tokenizer
        if os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            print(f"  [PolicyTrainer] Model not found, using dummy tokenizer")
            self.tokenizer = MinimalTokenizer()
            self.model = None
            
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
            
    def compute_log_probs(self, prompt_ids: List[int], response_ids: List[int]) -&gt; List[float]:
        """Compute log probabilities for a trajectory."""
        if self.model is None:
            return [-0.1 for _ in response_ids]
            
        # Real implementation
        full_ids = prompt_ids + response_ids
        input_ids = torch.tensor([full_ids], device=self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            
            # Compute log probs for response tokens only
            log_probs = []
            for i in range(len(prompt_ids), len(full_ids) - 1):
                next_token_id = full_ids[i + 1]
                log_prob = logits[0, i, next_token_id].item()
                log_probs.append(log_prob)
                
        return log_probs
        
    def update(self, trajectories: List[Trajectory], advantages: List[float], clip_eps: float = 0.2) -&gt; float:
        """
        Perform a GRPO policy update step.
        
        Args:
            trajectories: List of trajectories
            advantages: List of advantages for each trajectory
            clip_eps: Clip ratio for GRPO loss
        """
        if self.model is None:
            print("  [PolicyTrainer] Dummy update (no model loaded)")
            return 0.0
            
        print("  [PolicyTrainer] Performing GRPO policy update")
        
        # Collect all log probs and advantages for each token
        all_old_log_probs = []
        all_advantages = []
        
        for traj, adv in zip(trajectories, advantages):
            # Use trajectory's log probs as old_log_probs
            if traj.response_logprobs is not None:
                old_log_probs = traj.response_logprobs
            else:
                old_log_probs = [-0.1 for _ in traj.response_ids]
            
            # Expand advantage to each token
            for lp in old_log_probs:
                all_old_log_probs.append(lp)
                all_advantages.append(adv)
        
        if not all_old_log_probs:
            return 0.0
        
        # Compute current log probs for first trajectory (simplified)
        # In real GRPO, you would compute this for all trajectories
        traj = trajectories[0]
        current_log_probs = self.compute_log_probs(traj.prompt_ids, traj.response_ids)
        
        # Repeat for all trajectories
        all_current_log_probs = []
        for _ in trajectories:
            all_current_log_probs.extend(current_log_probs)
        
        # Compute GRPO loss
        self.optimizer.zero_grad()
        
        # Convert to tensors
        current_log_probs_tensor = torch.tensor(all_current_log_probs, device=self.device, requires_grad=True)
        old_log_probs_tensor = torch.tensor(all_old_log_probs, device=self.device)
        advantages_tensor = torch.tensor(all_advantages, device=self.device)
        
        # Compute probability ratio
        ratio = torch.exp(current_log_probs_tensor - old_log_probs_tensor)
        
        # Compute clipped surrogate
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        surr1 = ratio * advantages_tensor
        surr2 = clipped_ratio * advantages_tensor
        
        # GRPO loss: -min(surr1, surr2) for gradient ascent
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Backward pass
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()


# ==========================================
# Main Training Loop (GRPO)
# ==========================================
async def main():
    print("=" * 60)
    print("QwenPaw + verl Agent Framework Integration (GRPO)")
    print("=" * 60)
    print()
    
    # 1. Initialize components
    print("[1/5] Initializing components...")
    
    # Tokenizer
    if os.path.exists(config.model_path):
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    else:
        tokenizer = MinimalTokenizer()
    
    # Dummy backend (simulates vLLM)
    dummy_backend = DummyRolloutBackend(model=None, tokenizer=tokenizer)
    
    # Dummy session runtime (simulates GatewayServingRuntime)
    session_runtime = DummySessionRuntime(tokenizer=tokenizer, backend=dummy_backend)
    
    # QwenPaw agent
    qwenpaw_agent = QwenPawAgent()
    
    # Policy trainer
    policy_trainer = PolicyTrainer()
    
    print()
    
    # 2. Create test prompts
    print("[2/5] Creating test prompts...")
    test_prompts = [
        {"role": "user", "content": "Solve: 2 + 2 = ?"},
        {"role": "user", "content": "Solve: 10 * 5 = ?"},
        {"role": "user", "content": "Solve: What is 15 + 20?"},
    ]
    
    ground_truths = ["4", "50", "35"]
    
    # Build tensordict (verl's format)
    batch = get_tensordict(
        tensor_dict={
            "raw_prompt": [[msg] for msg in test_prompts],
            "uid": [f"sample-{i}" for i in range(len(test_prompts))],
        },
        non_tensor_dict={"global_steps": 1},
    )
    
    print(f"  Created batch with {len(test_prompts)} samples")
    print(f"  GRPO group size (n): {config.grpo_n}")
    print()
    
    # 3. Training loop (GRPO)
    print("[3/5] Starting GRPO training loop...")
    total_reward = 0.0
    num_updates = 0
    
    for step in range(config.max_steps):
        print(f"\n{'=' * 60}")
        print(f"Step {step + 1}/{config.max_steps}")
        print(f"{'=' * 60}")
        
        # Process each sample with GRPO group sampling
        for sample_idx, (prompt, gt) in enumerate(zip(test_prompts, ground_truths)):
            print(f"\n  [Sample {sample_idx}] Prompt: {prompt['content']}")
            
            # Collect multiple rollouts per prompt (GRPO group sampling)
            all_trajectories = []
            for rollout_idx in range(config.grpo_n):
                print(f"    [Rollout {rollout_idx + 1}/{config.grpo_n}]")
                
                # Step 3.1: Create session
                session_id = f"step-{step}-sample-{sample_idx}-rollout-{rollout_idx}"
                session_handle = await session_runtime.create_session(session_id)
                
                # Step 3.2: Run agent
                await qwenpaw_agent.run(
                    raw_prompt=[prompt],
                    session=session_handle,
                    sample_index=sample_idx
                )
                
                # Step 3.3: Finalize session and get trajectories
                trajectories = await session_runtime.finalize_session(session_id)
                all_trajectories.extend(trajectories)
            
            if not all_trajectories:
                print(f"  [Sample {sample_idx}] No trajectories collected")
                continue
                
            print(f"  [Sample {sample_idx}] Collected {len(all_trajectories)} trajectories (GRPO group)")
            
            # Step 3.4: Compute reward for each trajectory
            rewards = compute_gsm8k_reward(all_trajectories, gt, tokenizer)
            avg_reward = sum(rewards) / len(rewards)
            total_reward += avg_reward
            num_updates += 1
            
            print(f"  [Sample {sample_idx}] Rewards: {rewards}")
            print(f"  [Sample {sample_idx}] Average reward: {avg_reward:.3f}")
            
            # Step 3.5: Compute GRPO advantages (group relative)
            group_indices = [sample_idx] * len(all_trajectories)
            advantages = compute_mock_grpo_advantage(rewards, group_indices)
            print(f"  [Sample {sample_idx}] Advantages: {[f'{a:.3f}' for a in advantages]}")
            
            # Step 3.6: Update policy with GRPO loss
            loss = policy_trainer.update(all_trajectories, advantages, clip_eps=config.clip_eps)
            print(f"  [Sample {sample_idx}] GRPO policy loss: {loss:.5f}")
    
    print()
    
    # 4. Summary
    print("[4/5] Training Summary")
    print(f"  Algorithm: GRPO (Group Relative Policy Optimization)")
    print(f"  Total steps: {config.max_steps}")
    print(f"  GRPO group size (n): {config.grpo_n}")
    print(f"  Total updates: {num_updates}")
    print(f"  Average reward: {total_reward / num_updates if num_updates &gt; 0 else 0:.3f}")
    print()
    
    # 5. How to upgrade to real verl components
    print("[5/5] Next Steps - Upgrade to Real verl Components")
    print()
    print("1. Replace DummySessionRuntime with GatewayServingRuntime:")
    print("   from verl.agent.gateway.runtime import GatewayServingRuntime")
    print()
    print("2. Initialize real LLMServerClient with vLLM:")
    print("   from verl.workers.rollout.llm_server import LLMServerClient")
    print()
    print("3. Use real verl GRPO core algorithms:")
    print("   from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage")
    print()
    print("4. Use OpenAICompatibleAgentFramework directly:")
    print("   framework = OpenAICompatibleAgentFramework(...)")
    print("   await framework.generate_sequences(batch)")
    print()
    print("5. Configure verl for GRPO:")
    print("   - algorithm.adv_estimator: 'grpo'")
    print("   - actor_rollout.ref.rollout.n: your group size")
    print("   - actor_rollout_ref.actor.use_kl_loss: True (for KL regularization)")
    print()
    print("=" * 60)
    print("GRPO Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

