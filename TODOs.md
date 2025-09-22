# Slay the Spire Neural Agent - Development Roadmap

## 🔄 **MAJOR ARCHITECTURE PIVOT** (Current Direction)

**NEW APPROACH**: Two separate specialized networks instead of multi-head architecture
- **Events/Planning Network**: Event choices, map actions, rewards, shops, rest sites (value-based selection)
- **Combat Network**: Combat actions with softmax over possibilities (action masking)

## Priority 1: Two-Network Architecture Implementation

### 1.1 Shared Infrastructure & Embeddings
- [ ] **Card Embedding System**
  - Design unified card embeddings for both networks
  - Handle card IDs, upgrades, costs, types, rarities
  - Support for deck pile embeddings (draw, discard, exhaust)
  - Maximum 75 cards per pile, order-independent

- [ ] **Relic Embedding System**
  - Design unified relic embeddings for both networks
  - Handle relic IDs, counters, stack counts
  - Support for relic interactions and synergies

- [ ] **Game State Vector Processing**
  - HP, max HP, gold, floor number, ascension level
  - Current boss type (categorical)
  - Potions (categorical)
  - Map state representation (with Wing Boots support)
  - Events seen (binary flags)
  - Previous elite (categorical)
  - Act information (may be redundant with floor number)

### 1.2 Events/Planning Network
- [ ] **Network Architecture**
  - Value-based network for discrete choice evaluation
  - Input: Game state + Full deck + Event vector
  - Output: Single value per choice option
  - Loop over all possible actions, select highest value

- [ ] **Event Vector Processing**
  - Current event as categorical encoding
  - Generic choice vectors (4 choices × 20 features each):
    - Choice type flags (8 features)
    - Binary flags (2 features)
    - Card ID, Relic ID, Gold amount, HP change, Max HP change
    - Gold cost, HP cost, Padding (3 features)
  - Handle variable choice counts (2-4 options typical)

- [ ] **Action Space Coverage**
  - Event choices and outcomes
  - Map navigation decisions
  - Card reward selections
  - Relic reward selections
  - Shop purchase decisions
  - Rest site choices (rest, upgrade, recall)

### 1.3 Combat Network
- [ ] **Network Architecture**
  - Policy network with softmax over action space
  - Input: Game state + Full deck + Combat vector
  - Output: Action probabilities with masking
  - Unified action space for all combat possibilities

- [ ] **Combat Vector Processing**
  - Enemy information (categorical)
  - Enemy intent (categorical + numerical values)
  - Enemy debuffs (categorical with counters)
  - Enemy health + max health
  - Current player buffs (categorical)
  - Energy (numerical)
  - Hand cards (IDs + costs)
  - Draw pile embedding
  - Discard pile embedding
  - Exhaust pile embedding
  - Block amount (numerical)
  - Powers in play (categorical, max ~15-20 powers)
  - Turn number (numerical)

- [ ] **Action Space & Masking**
  - Unified action space for all combat actions
  - Dynamic action masking based on:
    - Available energy for card costs
    - Valid targets for targeted cards
    - Potion availability
    - Card playability restrictions
  - Handle complex cards (Seek, Exhume, etc.)

### 1.4 Integration & Training
- [ ] **Network Coordination**
  - Determine when to use Events vs Combat network
  - Handle screen state transitions
  - Shared embedding consistency

- [ ] **Training Strategy**
  - Separate training loops for each network
  - Shared embedding updates
  - Experience replay tailored to each network type
  - Evaluation metrics for both networks

## 🗂️ Previous Multi-Head Architecture (DEPRECATED)

### 1.1 Multi-Head Value Network Implementation (COMPLETED BUT DEPRECATED)
- [x] **Replace Actor-Critic with Value-Based Architecture** ✅ COMPLETED
  - ✅ Implement `STSMultiHeadValueNetwork` with shared backbone
  - ✅ Create separate heads for combat vs meta-game actions
  - ✅ Design action encoding for both combat and meta actions
  - ✅ Add value prediction heads for different game contexts

- [x] **Action Encoding Design** ✅ COMPLETED
  - ✅ Define `combat_action_encoding_size` (64-dim: cards, targets, potions)
  - ✅ Define `meta_action_encoding_size` (32-dim: events, shops, map choices)
  - ✅ Create embedding layers for card IDs, relic IDs, etc.
  - ✅ Implement one-hot encoding for discrete choices

- [x] **Training Algorithm Updates** ✅ COMPLETED
  - ✅ Switch from PPO to value-based learning (Double DQN)
  - ✅ Implement prioritized experience replay buffer
  - ✅ Add target networks and double Q-learning
  - ✅ Design loss functions for multi-head architecture

### 1.2 Combat Action System Integration
- [x] **Battle Action Decoder** ✅ COMPLETED
  - ✅ Extend action decoder to handle `BattleAction` class
  - ✅ Map combat actions to generic action space (240-255 range)
  - ✅ Handle card play actions (source, target, upgrades)
  - ✅ Handle potion usage and end-turn actions

- [ ] **Combat State Representation** ⚠️ PARTIAL
  - ⚠️ Enhance observation space for battle context (using existing 550-dim space)
  - ❌ Include hand cards, energy, player/monster statuses (requires BattleContext)
  - ❌ Add monster intent and HP information (requires BattleContext)
  - ❌ Include combat-specific relics and powers (requires BattleContext)

### 1.3 Enhanced Reward Function
- [ ] **Granular Combat Rewards**
  - Damage dealt/received ratios
  - Block efficiency and energy usage
  - Status effect utilization (poison, vulnerable, etc.)
  - Turn efficiency metrics
  - Combat outcome bonuses/penalties

- [ ] **Progressive Reward Shaping**
  - Floor progression rewards
  - Deck composition improvements
  - Elite/boss defeat bonuses
  - Relic acquisition value
  - Route efficiency (pathing optimization)

## Priority 2: Game State Parsing & Action Handling

### 2.1 Event System Integration
- [ ] **Generic Choice Vector Implementation**
  - Parse all event choices into standardized format
  - Handle variable choice counts (2-4 options typical)
  - Map event outcomes to choice indices
  - Create event-specific reward calculations

- [ ] **Event State Recognition**
  - Identify current event type from game state
  - Extract choice descriptions and consequences
  - Handle special events (Neow, shops, rest sites)
  - Add event history tracking

### 2.2 Advanced Game State Features
- [ ] **Map Path Planning**
  - Encode available paths and room types
  - Add path risk/reward analysis
  - Implement lookahead for route planning
  - Consider floor objectives (keys, bosses)

- [ ] **Shop Economics**
  - Price evaluation and gold optimization
  - Card/relic value assessment
  - Removal priority calculations
  - Shop upgrade decisions

## Priority 3: Training Infrastructure

### 3.1 Self-Play and Evaluation
- [x] **Training Pipeline Optimization** ✅ COMPLETED
  - ✅ Implement parallel environment runners (STSEnvironmentWrapper)
  - ❌ Add curriculum learning (ascension levels)
  - ✅ Create evaluation benchmarks vs existing agents
  - ✅ Add model checkpointing and versioning

- [x] **Data Collection Improvements** ✅ COMPLETED
  - ✅ Balance combat vs non-combat experiences
  - ✅ Prioritized experience replay based on action type
  - ❌ Add demonstration data from expert play
  - ❌ Implement data augmentation for rare scenarios

### 3.2 Model Architecture Enhancements
- [ ] **Attention Mechanisms**
  - Card-card interactions in hand
  - Relic synergy modeling
  - Monster targeting decisions
  - Event choice dependencies

- [ ] **Hierarchical Decision Making**
  - High-level strategy planning (deck archetype)
  - Mid-level tactical decisions (floor goals)
  - Low-level execution (combat micro)

## Priority 4: Advanced Features (Post-Initial Training)

### 4.1 Tree Search Integration
- [ ] **Monte Carlo Tree Search (MCTS)**
  - Integrate value network with MCTS for combat
  - Design action expansion policies
  - Implement UCB selection with neural priors
  - Add simulation rollouts with learned policy

- [ ] **Search-Based Planning**
  - Route planning with search
  - Deck building strategy trees
  - Event choice optimization
  - Resource allocation planning

### 4.2 Multi-Character Support
- [ ] **Character-Specific Models**
  - Separate heads for Ironclad/Silent/Defect/Watcher
  - Character-specific action encodings
  - Unique card and strategy embeddings
  - Transfer learning between characters

### 4.3 Meta-Learning and Adaptation
- [ ] **Opponent Modeling** (for future multiplayer/mods)
  - Monster behavior prediction
  - Boss pattern recognition
  - Adaptive difficulty scaling

- [ ] **Few-Shot Learning**
  - Quick adaptation to new cards/relics
  - Mod support and custom content
  - Dynamic strategy adjustment

## Priority 5: Production & Deployment

### 5.1 Performance Optimization
- [ ] **Model Efficiency**
  - Model pruning and quantization
  - ONNX conversion for fast inference
  - Batch processing optimization
  - Memory usage profiling

- [ ] **Inference Speed**
  - Real-time decision making (<100ms)
  - Efficient action masking
  - Cached computations for repeated states
  - GPU utilization optimization

### 5.2 Integration & Testing
- [ ] **End-to-End Testing**
  - Full game completion tests
  - Ascension level progression
  - Win rate benchmarking
  - Regression testing for updates

- [ ] **User Interface**
  - Action explanation system
  - Confidence/reasoning display
  - Interactive training visualization
  - Model comparison tools

## Technical Debt & Infrastructure

### Code Quality
- [ ] **Refactoring**
  - Modularize neural network components
  - Standardize configuration management
  - Improve error handling and logging
  - Add comprehensive unit tests

- [ ] **Documentation**
  - Architecture decision records
  - Training procedure documentation
  - API documentation for all modules
  - Performance benchmarking results

### Development Tools
- [ ] **Debugging & Visualization**
  - Tensorboard integration
  - Action probability visualization
  - Value function heatmaps
  - Training curve analysis

- [ ] **Experiment Management**
  - Hyperparameter optimization
  - A/B testing framework
  - Model performance tracking
  - Automated experiment scheduling

## Long-Term Research Directions

### 5.3 Novel Approaches
- [ ] **Transformer Architectures**
  - Sequence modeling for game history
  - Attention over card sequences
  - Event chain reasoning

- [ ] **Multimodal Learning**
  - Visual card representation learning
  - UI element recognition
  - Natural language event processing

- [ ] **Continual Learning**
  - Learning new expansions without forgetting
  - Online adaptation during play
  - Meta-learning for quick adaptation

## 🎯 **NEW Implementation Priority Order**

1. **Phase 1 (Immediate)**: Two-Network Architecture Foundation ⏳ **CURRENT**
   - Shared embeddings (cards, relics)
   - Game state vector processing
   - Events/Planning network implementation
   - Combat network with action masking

2. **Phase 2 (Short-term)**: Integration & Training
   - Network coordination and screen transitions
   - Separate training pipelines
   - Experience replay for both networks
   - Evaluation and debugging tools

3. **Phase 3 (Medium-term)**: Advanced Features
   - Enhanced combat state representation (BattleContext integration)
   - Map processing with Wing Boots and complex pathing
   - Relic/debuff counter handling
   - Complex card interactions (Seek, Exhume, etc.)

4. **Phase 4 (Long-term)**: Optimization & Production
   - Performance optimization
   - Multi-character support
   - MCTS integration (especially for combat)
   - Production deployment

## 🚧 **Critical Implementation Notes**

### Technical Considerations
- **Pile Embeddings**: 75 card limit, order-independent (use set embeddings or averaging)
- **Powers Limit**: ~15-20 powers maximum in combat
- **Generic Choice Format**: 4 choices × 20 features (from observation space analysis)
- **Shared Embeddings**: Cards and relics must be consistent between networks
- **Action Masking**: Combat network needs dynamic masking for energy/targets
- **Map State**: Need to handle Wing Boots and complex pathfinding

### Questions to Resolve
- [ ] Map state representation - how to encode current paths and Wing Boots effect?
- [ ] Generic choice validation - ensure the 4×20 format covers all event types?
- [ ] Relic counter handling - how to represent stacks/counters in embeddings?
- [ ] Combat action space size - how many total combat actions exist?

## 📚 **Previous Multi-Head Architecture Reference**

Each phase should include comprehensive testing and evaluation before moving to the next phase.

## ✅ Recently Completed (Current Session)

### Core Architecture ✅ DONE
- [x] Multi-head value network with combat/meta heads
- [x] Action encoding system (64-dim combat, 32-dim meta)
- [x] Double DQN training with prioritized experience replay
- [x] Action context classification system
- [x] Combat action integration (actions 240-255)
- [x] End-to-end training pipeline
- [x] Model saving/loading and evaluation system

### Files Created/Updated ✅
- `sts_multihead_network.py` - Multi-head value network
- `sts_action_classifier.py` - Action context classification
- `sts_dqn_trainer.py` - Double DQN training system
- `train_multihead_agent.py` - Complete training pipeline
- `sts_action_decoder.py` - Enhanced with combat actions
- Updated existing files for value-based learning

### Testing Status ✅ VERIFIED
- ✅ Multi-head network forward/backward passes
- ✅ Action encoding for combat and meta actions
- ✅ Context classification (META→COMBAT transitions)
- ✅ DQN training loop with experience replay
- ✅ End-to-end system integration
- ✅ Screen progression: EVENT→REWARDS→MAP→BATTLE