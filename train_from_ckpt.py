from pytorch_lightning.cli import LightningCLI


if __name__ == "__main__":
    cli = LightningCLI()
    ckpt_path = "/cto_studio/lijingyi/vocos/logs96_decoder/lightning_logs/version_9/checkpoints/last.ckpt"
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
