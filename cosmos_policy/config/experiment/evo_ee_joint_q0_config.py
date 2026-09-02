"""Stage-two Evo teleop training with shared EE + joint q0 supervision."""

from __future__ import annotations

import os

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.datasets.evo_ee_joint_q0_aloha_dataset import EvoEEJointQ0Dataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel


EVO_EE_JOINT_Q0_DATA_DIR = os.environ.get("EVO_EE_JOINT_Q0_DATA_DIR", "./evo_ee_joint_q0")
EVO_EE_JOINT_Q0_STATS_DIR = os.environ.get("EVO_EE_JOINT_Q0_STATS_DIR", EVO_EE_JOINT_Q0_DATA_DIR)
EVO_EE_JOINT_Q0_S3_CREDENTIALS = os.environ.get(
    "COSMOS_S3_CREDENTIALS", os.path.expanduser("~/.secrets/cosmos_s3.json")
)

evo_ee_joint_q0_dataset = L(EvoEEJointQ0Dataset)(
    data_dir=EVO_EE_JOINT_Q0_DATA_DIR,
    statistics_dir=EVO_EE_JOINT_Q0_STATS_DIR,
    t5_text_embeddings_path=os.path.join(EVO_EE_JOINT_Q0_DATA_DIR, "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    lazy_video_decompression=True,
    lazy_video_frame_access=True,
    prefer_indexed_frame_store=True,
    # Preserve the established Cosmos/YAM 3:1 interleaved schedule: 75% demo
    # samples supervise action + future state and 25% copied-success samples
    # supervise future state only.
    treat_demos_as_success_rollouts=True,
    demonstration_sampling_prob=0.75,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=True,
    p_world_model=1.0,
    gamma=0.998,
)
evo_ee_joint_q0_val_dataset = L(EvoEEJointQ0Dataset)(
    data_dir=EVO_EE_JOINT_Q0_DATA_DIR,
    statistics_dir=EVO_EE_JOINT_Q0_STATS_DIR,
    t5_text_embeddings_path=os.path.join(EVO_EE_JOINT_Q0_DATA_DIR, "t5_embeddings.pkl"),
    is_train=False,
    chunk_size=50,
    use_image_aug=False,
    use_stronger_image_aug=False,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    lazy_video_decompression=True,
    lazy_video_frame_access=True,
    prefer_indexed_frame_store=True,
    # Validation measures the demo objective directly. It does not duplicate
    # episodes into the auxiliary future-state-only population.
    treat_demos_as_success_rollouts=False,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=0.0,
    return_value_function_returns=True,
    p_world_model=1.0,
    gamma=0.998,
)


cosmos_predict2_2b_480p_evo_ee6d_joint35_teleop = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        trainer=dict(
            max_iter=12000,
            run_validation=True,
            run_validation_on_start=False,
            validation_iter=500,
            max_val_iter=None,
        ),
        scheduler=dict(
            cycle_lengths=[12000, 100000000000000],
            warm_up_steps=[400, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint=dict(
            save_iter=500,
            save_to_object_store=dict(
                enabled=True,
                credentials=EVO_EE_JOINT_Q0_S3_CREDENTIALS,
                bucket="policy-training",
            ),
            load_from_object_store=dict(
                enabled=True,
                credentials=EVO_EE_JOINT_Q0_S3_CREDENTIALS,
                bucket="policy-training",
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(mask_value_prediction_loss_for_policy_prediction=True),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=evo_ee_joint_q0_dataset,
            sampler=L(DistributedSampler)(
                dataset=evo_ee_joint_q0_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        dataloader_val=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=evo_ee_joint_q0_val_dataset,
            sampler=L(DistributedSampler)(
                dataset=evo_ee_joint_q0_val_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=False,
                seed=0,
            ),
            batch_size=25,
            drop_last=False,
        ),
        job=dict(
            project="cosmos-policy-towel-transfer",
            group="evo-ee6d-joint35-teleop",
            name="predict2-2b-evo-ee6d-joint35-teleop",
        ),
    )
)
