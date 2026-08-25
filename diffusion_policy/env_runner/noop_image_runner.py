from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class NoopImageRunner(BaseImageRunner):
    """Placeholder until the LeHome simulator rollout adapter is added.

    Offline validation loss and checkpoints remain fully functional.  It is
    intentionally not a fake success metric.
    """

    def __init__(self, output_dir=None, **kwargs):
        self.output_dir = output_dir

    def run(self, policy):
        return {}
