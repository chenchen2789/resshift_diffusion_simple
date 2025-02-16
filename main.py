from omegaconf import OmegaConf
from model.ResShift_model import ResShiftTrainer


if __name__ == "__main__":
    path = r".\config\realsr_swinunet_realesrgan256_journal_simple.yaml"
    configs = OmegaConf.load(path)
    Trainer = ResShiftTrainer(configs=configs)
    # Trainer.evaluate() ##测试模型推理
    Trainer.train()  ##重新训练模型


