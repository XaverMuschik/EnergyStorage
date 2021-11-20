from gym.envs.registration import register

register(
    id='energy_storage-v0',
    entry_point='gym_energy_storage.envs:EnergyStorageEnv',
)
register(
    id='energy-storage-extrahard-v0',
    entry_point='gym_energy_storage.envs:EnergyStorageExtraHardEnv',
)
