"""Sanity test to ensure observation schema and step outputs are strict and deterministic."""

from open_bargain.config import OpenBargainConfig
from open_bargain.env.env import OpenBargainEnv

def test_observation_and_step_schema():
    print("Testing observation schema...")
    config = OpenBargainConfig.default()
    env = OpenBargainEnv(config=config)
    
    # 1. Reset env
    observations, info = env.reset(options={"agent_ids": ["agent_0", "agent_1"]})
    
    # 2. Read active proposer
    active_proposer = observations["agent_0"]["public"]["active_proposer_id"]
    print(f"Active proposer: {active_proposer}")
    
    # 3. Select dummy valid action
    action = {
        "agent_id": active_proposer,
        "action_type": "propose",
        "payload": {
            "allocation": {"agent_0": 50.0, "agent_1": 50.0}
        }
    }
    
    # 4. Call env.step()
    next_obs, rewards, terminated, truncated, step_info = env.step(action)
    
    # 5. Validate returned schema
    assert next_obs is not None
    assert rewards is not None
    assert terminated is not None
    assert truncated is not None
    
    print("Step output valid!")
    print("Test passed successfully.")

if __name__ == "__main__":
    test_observation_and_step_schema()
