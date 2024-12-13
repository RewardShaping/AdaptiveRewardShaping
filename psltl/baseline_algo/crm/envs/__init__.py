from gym.envs.registration import register
import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'Half-Cheetah-RM1-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
    if 'Half-Cheetah-RM2-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
    
    for i in range(11):
        w_id = 'Water-single-M%d-v0'%i
        if w_id in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
    for i in range(11):
        w_id = 'Water-M%d-v0'%i
        if w_id in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]

        


# ----------------------------------------- Half-Cheetah

register(
    id='Half-Cheetah-RM1-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM1',
    max_episode_steps=1000,
)
register(
    id='Half-Cheetah-RM2-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM2',
    max_episode_steps=1000,
)



# ----------------------------------------- WATER
for i in range(11):
    w_id = 'Water-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

for i in range(11):
    w_id = 'Water-single-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRM10EnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

# # ----------------------------------------- OFFICE
# register(
#     id='Office-v0',
#     entry_point='envs.grids.grid_environment:OfficeRMEnv',
#     max_episode_steps=1000
# )

register(
    id='Office-single-v0',
    entry_point='envs.grids.grid_environment:OfficeRM3Env',
    max_episode_steps=100
)

# # ----------------------------------------- CRAFT
# for i in range(11):
#     w_id = 'Craft-M%d-v0'%i
#     w_en = 'envs.grids.grid_environment:CraftRMEnvM%d'%i
#     register(
#         id=w_id,
#         entry_point=w_en,
#         max_episode_steps=1000
#     )

# for i in range(11):
#     w_id = 'Craft-single-M%d-v0'%i
#     w_en = 'envs.grids.grid_environment:CraftRM10EnvM%d'%i
#     register(
#         id=w_id,
#         entry_point=w_en,
#         max_episode_steps=1000
#     )

# ----------------------------------------- Taxi
register(
    id='Taxi-v0',
    entry_point='envs.grids.grid_environment:TaxiRMEnv',
    max_episode_steps=200
)
