from ..utils.config import EPISODES, BATCH_SIZE

def validate_agent(env, agent, max_steps=1000):
    """
    Validate the agent's performance on a validation environment
    
    Args:
        env: The validation environment
        agent: The trained agent
    
    Returns:
        float: The total reward achieved during validation
    """
    print("Starting validation...")
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        steps += 1
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        
        if steps % 100 == 0:  # Print progress every 100 steps
            print(f"  Validation step {steps}: Current reward = {total_reward:.2f}")
    
    print(f"Validation completed: {total_reward:.2f} reward in {steps} steps")
    return total_reward

def train_agent(train_env, val_env, agent, episodes=100, checkpoint_freq=10):
    """Train agent with checkpointing"""
    
    # Try to load latest checkpoint
    latest_episode = 0
    for ep in range(episodes, -1, -1):
        if agent.load_checkpoint(ep):
            latest_episode = ep
            break
    
    history = {
        'train_rewards': [],
        'val_metrics': [],
        'best_val_reward': float('-inf')
    }
    
    for episode in range(latest_episode, episodes):
        # Training phase
        train_reward = run_episode(train_env, agent, training=True)
        history['train_rewards'].append(train_reward)
        
        # Validation phase
        val_metrics = validate_agent(val_env, agent)
        history['val_metrics'].append(val_metrics)
        
        # Save checkpoint periodically
        if episode % checkpoint_freq == 0:
            agent.save_checkpoint(episode)
            
        # Save best model based on validation
        if val_metrics > history['best_val_reward']:
            history['best_val_reward'] = val_metrics
            agent.save_checkpoint('best')
            
        print(f"\nEpisode {episode} completed:")
        print(f"  Training reward: {train_reward:.2f}")
        print(f"  Validation reward: {val_metrics:.2f}")
        print(f"  Epsilon: {agent.epsilon:.4f}")
    
    return history 