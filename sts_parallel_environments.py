#!/usr/bin/env python3
"""
Parallel Environment System for STS Two-Network Training

This module provides vectorized environments for efficient batch training
with multiple STS game instances running in parallel.
"""

import torch
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Process
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time
import traceback

import slaythespire
from sts_data_collection import STSEnvironmentWrapper
from sts_two_network_trainer import Experience, ScreenState


@dataclass
class ParallelExperience:
    """Experience from parallel environment with environment ID."""
    env_id: int
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    screen_state: ScreenState
    game_context: Dict
    network_specific_data: Dict


class EnvironmentWorker:
    """Worker process for running a single STS environment."""

    def __init__(self, worker_id: int, input_queue: Queue, output_queue: Queue,
                 character_class=None, ascension=0, reward_function='simple'):
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.character_class = character_class or slaythespire.CharacterClass.IRONCLAD
        self.ascension = ascension
        self.reward_function = reward_function
        self.env = None

    def run(self):
        """Main worker loop."""
        try:
            # Initialize environment in worker process
            self.env = STSEnvironmentWrapper(
                character_class=self.character_class,
                ascension=self.ascension,
                reward_function=self.reward_function
            )

            # Send ready signal
            self.output_queue.put(('ready', self.worker_id, None))

            while True:
                try:
                    # Wait for command
                    command = self.input_queue.get(timeout=30)

                    if command[0] == 'reset':
                        state = self.env.reset()
                        self.output_queue.put(('reset_done', self.worker_id, state))

                    elif command[0] == 'step':
                        action = command[1]
                        try:
                            next_state, reward, done, info = self.env.step(action)
                            self.output_queue.put(('step_done', self.worker_id,
                                                 (next_state, reward, done, info)))
                        except Exception as e:
                            # Handle step errors gracefully
                            self.output_queue.put(('step_error', self.worker_id, str(e)))

                    elif command[0] == 'get_context':
                        context_data = {
                            'hp': self.env.game_context.cur_hp,
                            'max_hp': self.env.game_context.max_hp,
                            'gold': self.env.game_context.gold,
                            'floor': self.env.game_context.floor_num,
                            'act': self.env.game_context.act
                        }
                        self.output_queue.put(('context_done', self.worker_id, context_data))

                    elif command[0] == 'close':
                        break

                except Exception as e:
                    self.output_queue.put(('error', self.worker_id, str(e)))

        except Exception as e:
            self.output_queue.put(('worker_error', self.worker_id, str(e)))


class VectorizedStsEnvironment:
    """
    Vectorized environment wrapper for parallel STS training.

    Manages multiple STS environment workers for efficient batch training.
    """

    def __init__(self, num_envs: int = 4, character_class=None, ascension=0,
                 reward_function='simple', timeout=10.0):
        self.num_envs = num_envs
        self.character_class = character_class or slaythespire.CharacterClass.IRONCLAD
        self.ascension = ascension
        self.reward_function = reward_function
        self.timeout = timeout

        # Communication queues
        self.input_queues = [Queue() for _ in range(num_envs)]
        self.output_queue = Queue()

        # Worker processes
        self.workers = []
        self.processes = []

        # Environment state tracking
        self.env_states = [None] * num_envs
        self.env_dones = [True] * num_envs  # Start with all environments needing reset

        self._start_workers()

    def _start_workers(self):
        """Start all worker processes."""
        for i in range(self.num_envs):
            worker = EnvironmentWorker(
                worker_id=i,
                input_queue=self.input_queues[i],
                output_queue=self.output_queue,
                character_class=self.character_class,
                ascension=self.ascension,
                reward_function=self.reward_function
            )

            process = Process(target=worker.run)
            process.start()

            self.workers.append(worker)
            self.processes.append(process)

        # Wait for all workers to be ready
        ready_count = 0
        while ready_count < self.num_envs:
            try:
                response = self.output_queue.get(timeout=self.timeout)
                if response[0] == 'ready':
                    ready_count += 1
                elif response[0] == 'worker_error':
                    raise RuntimeError(f"Worker {response[1]} failed to start: {response[2]}")
            except:
                raise RuntimeError("Failed to start environment workers")

    def reset(self, env_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Reset specified environments (or all if None).

        Args:
            env_ids: List of environment IDs to reset, or None for all

        Returns:
            Array of initial states with shape (num_envs, state_dim)
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        # Send reset commands
        for env_id in env_ids:
            self.input_queues[env_id].put(('reset',))

        # Collect responses
        states = [None] * len(env_ids)
        responses_received = 0

        while responses_received < len(env_ids):
            try:
                response = self.output_queue.get(timeout=self.timeout)
                if response[0] == 'reset_done':
                    env_id = response[1]
                    if env_id in env_ids:
                        idx = env_ids.index(env_id)
                        states[idx] = response[2]
                        self.env_states[env_id] = response[2]
                        self.env_dones[env_id] = False
                        responses_received += 1
            except:
                raise RuntimeError("Timeout waiting for environment resets")

        return np.array(states)

    def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Take steps in all environments.

        Args:
            actions: List of actions for each environment

        Returns:
            Tuple of (next_states, rewards, dones, infos)
        """
        assert len(actions) == self.num_envs, f"Expected {self.num_envs} actions, got {len(actions)}"

        # Send step commands to all active environments
        active_envs = []
        for env_id, action in enumerate(actions):
            if not self.env_dones[env_id]:
                self.input_queues[env_id].put(('step', action))
                active_envs.append(env_id)

        # Collect responses
        next_states = [None] * self.num_envs
        rewards = [0.0] * self.num_envs
        dones = [True] * self.num_envs
        infos = [{}] * self.num_envs

        responses_received = 0

        while responses_received < len(active_envs):
            try:
                response = self.output_queue.get(timeout=self.timeout)

                if response[0] == 'step_done':
                    env_id = response[1]
                    if env_id in active_envs:
                        next_state, reward, done, info = response[2]
                        next_states[env_id] = next_state
                        rewards[env_id] = reward
                        dones[env_id] = done
                        infos[env_id] = info

                        self.env_states[env_id] = next_state
                        self.env_dones[env_id] = done

                        responses_received += 1

                elif response[0] == 'step_error':
                    env_id = response[1]
                    if env_id in active_envs:
                        # Mark environment as done on error
                        next_states[env_id] = self.env_states[env_id]  # Use last known state
                        rewards[env_id] = -1.0  # Penalty for error
                        dones[env_id] = True
                        infos[env_id] = {'error': response[2]}

                        self.env_dones[env_id] = True
                        responses_received += 1

            except:
                # Timeout - mark remaining environments as done
                for env_id in active_envs:
                    if next_states[env_id] is None:
                        next_states[env_id] = self.env_states[env_id]
                        rewards[env_id] = -1.0
                        dones[env_id] = True
                        infos[env_id] = {'error': 'timeout'}
                        self.env_dones[env_id] = True
                break

        # Fill in states for environments that were already done
        for env_id in range(self.num_envs):
            if next_states[env_id] is None:
                next_states[env_id] = self.env_states[env_id]

        return (np.array(next_states),
                np.array(rewards),
                np.array(dones),
                infos)

    def get_game_contexts(self) -> List[Dict]:
        """Get game context data from all environments."""
        # Send context requests
        for env_id in range(self.num_envs):
            if not self.env_dones[env_id]:
                self.input_queues[env_id].put(('get_context',))

        # Collect responses
        contexts = [{}] * self.num_envs
        responses_received = 0
        active_envs = sum(1 for done in self.env_dones if not done)

        while responses_received < active_envs:
            try:
                response = self.output_queue.get(timeout=self.timeout)
                if response[0] == 'context_done':
                    env_id = response[1]
                    contexts[env_id] = response[2]
                    responses_received += 1
            except:
                break

        return contexts

    def close(self):
        """Close all environment workers."""
        # Send close commands
        for queue in self.input_queues:
            try:
                queue.put(('close',))
            except:
                pass

        # Wait for processes to finish
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
                if process.is_alive():
                    process.kill()

        # Clean up queues
        try:
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
        except:
            pass

        for queue in self.input_queues:
            try:
                while not queue.empty():
                    queue.get_nowait()
            except:
                pass


def test_vectorized_environment():
    """Test the vectorized environment system."""
    print("=== Testing Vectorized STS Environment ===\n")

    num_envs = 3
    print(f"Creating {num_envs} parallel environments...")

    try:
        vec_env = VectorizedStsEnvironment(
            num_envs=num_envs,
            reward_function='simple',
            timeout=5.0
        )

        print("✓ Environments created successfully")

        # Test reset
        print("\n--- Testing Reset ---")
        initial_states = vec_env.reset()
        print(f"Initial states shape: {initial_states.shape}")
        print(f"Sample state range: [{initial_states[0].min():.3f}, {initial_states[0].max():.3f}]")

        # Test steps
        print("\n--- Testing Steps ---")
        for step in range(3):
            # Random actions for testing
            actions = [np.random.randint(0, 4) for _ in range(num_envs)]

            next_states, rewards, dones, infos = vec_env.step(actions)

            print(f"Step {step + 1}:")
            print(f"  Actions: {actions}")
            print(f"  Rewards: {rewards}")
            print(f"  Dones: {dones}")
            print(f"  States shape: {next_states.shape}")

            # Reset done environments
            done_envs = [i for i, done in enumerate(dones) if done]
            if done_envs:
                print(f"  Resetting environments: {done_envs}")
                vec_env.reset(done_envs)

        # Test game contexts
        print("\n--- Testing Game Contexts ---")
        contexts = vec_env.get_game_contexts()
        for i, context in enumerate(contexts):
            if context:
                print(f"  Env {i}: HP={context.get('hp', 'N/A')}, Gold={context.get('gold', 'N/A')}, Floor={context.get('floor', 'N/A')}")

        print("\n--- Cleanup ---")
        vec_env.close()
        print("✓ Environments closed")

        print("\n✓ Vectorized environment test completed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_vectorized_environment()