"""Training configuration for EVO observed-state q0-anchored actions."""

from __future__ import annotations

import os

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.datasets.evo_q0_aloha_dataset import EVOQ0AnchoredDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel


EVO_Q0_DATA_DIR = os.environ.get("EVO_Q0_DATA_DIR", "./evo_q0_towel")
EVO_Q0_STATS_DIR = os.environ.get("EVO_Q0_STATS_DIR", EVO_Q0_DATA_DIR)
EVO_Q0_S3_CREDENTIALS = os.environ.get(
    "COSMOS_S3_CREDENTIALS",
    os.path.expanduser("~/.secrets/cosmos_s3.json"),
)

evo_q0_towel_dataset = L(EVOQ0AnchoredDataset)(
    data_dir=EVO_Q0_DATA_DIR,
    statistics_dir=EVO_Q0_STATS_DIR,
    t5_text_embeddings_path=os.path.join(EVO_Q0_DATA_DIR, "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    lazy_video_decompression=True,
    # The same episodes serve two objectives. Demo samples are action-only;
    # duplicated success-rollout samples are future-state-only, giving the
    # effective 75% / 25% split below. Predict2's tokenizer contract still
    # requires the final value segment to make a 41-frame sequence, but
    # p_world_model=1.0 prevents value samples and the model loss mask below
    # gives the structural value segment zero loss.
    treat_demos_as_success_rollouts=True,
    demonstration_sampling_prob=0.75,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=True,
    p_world_model=1.0,
)

cosmos_predict2_2b_480p_evo_q0_state17 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        trainer=dict(max_iter=8000),
        scheduler=dict(
            cycle_lengths=[8000, 100000000000000],
            warm_up_steps=[400, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint=dict(
            save_iter=500,
            save_to_object_store=dict(
                enabled=True,
                credentials=EVO_Q0_S3_CREDENTIALS,
                bucket="policy-training",
            ),
            load_from_object_store=dict(
                enabled=False,
                credentials=EVO_Q0_S3_CREDENTIALS,
                bucket="policy-training",
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                mask_loss_for_action_future_state_prediction=True,
                mask_value_prediction_loss_for_policy_prediction=False,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=evo_q0_towel_dataset,
            sampler=L(DistributedSampler)(
                dataset=evo_q0_towel_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            project="cosmos-policy-blue-towel",
            group="evo-q0",
            name="predict2-2b-evo-q0-state17",
        ),
    )
)
