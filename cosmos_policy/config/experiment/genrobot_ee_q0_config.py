"""Training configuration for GenRobot q0-local EE-6D towel data."""

from __future__ import annotations

import os

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.datasets.genrobot_ee_q0_aloha_dataset import GenRobotEEQ0Dataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel


GENROBOT_EE_Q0_DATA_DIR = os.environ.get("GENROBOT_EE_Q0_DATA_DIR", "./genrobot_towel_ee6d")
GENROBOT_EE_Q0_STATS_DIR = os.environ.get("GENROBOT_EE_Q0_STATS_DIR", GENROBOT_EE_Q0_DATA_DIR)
GENROBOT_EE_Q0_S3_CREDENTIALS = os.environ.get(
    "COSMOS_S3_CREDENTIALS", os.path.expanduser("~/.secrets/cosmos_s3.json")
)

genrobot_towel_dataset = L(GenRobotEEQ0Dataset)(
    data_dir=GENROBOT_EE_Q0_DATA_DIR,
    statistics_dir=GENROBOT_EE_Q0_STATS_DIR,
    t5_text_embeddings_path=os.path.join(GENROBOT_EE_Q0_DATA_DIR, "t5_embeddings.pkl"),
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
    # Same deterministic dataset mixture as the prior YAM/Evo q0 recipe:
    # 75% demo indices predict action + future state and 25% copied-success
    # indices predict future state only. DistributedSampler shuffles this fixed
    # 3:1 index population each epoch; it is not a time-based curriculum.
    treat_demos_as_success_rollouts=True,
    demonstration_sampling_prob=0.75,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=True,
    p_world_model=1.0,
    gamma=0.998,
)

cosmos_predict2_2b_480p_genrobot_towel_ee6d35 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        trainer=dict(max_iter=12000),
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
                credentials=GENROBOT_EE_Q0_S3_CREDENTIALS,
                bucket="policy-training",
            ),
            load_from_object_store=dict(
                enabled=False,
                credentials=GENROBOT_EE_Q0_S3_CREDENTIALS,
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
            dataset=genrobot_towel_dataset,
            sampler=L(DistributedSampler)(
                dataset=genrobot_towel_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            project="cosmos-policy-towel-transfer",
            group="genrobot-ee6d35",
            name="predict2-2b-genrobot-towel-ee6d35",
        ),
    )
)
