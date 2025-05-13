import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot.common import envs
from lerobot.common.optim import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.common.utils.hub import HubMixin
from lerobot.common.utils.utils import auto_select_torch_device, is_amp_available
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig

TRAIN_CONFIG_NAME = "train_config.json"

# wwq注释2025/2/17 训练配置类：训练参数及输出目录等配置
@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None      #  定义了与策略（预训练策略）相关的配置
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None          # 如果指定的目录已存在且 resume 设置为 False，则会抛出错误，以防止覆盖现有目录。如果未指定目录，则会自动生成一个基于当前时间和任务名称的目录。
    job_name: str | None = None             # 设置任务的名称
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False            
    device: str | None = None  # cuda | cpu | mp
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = True
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Number of workers for the dataloader.
    num_workers: int = 4            # 数据加载器使用的线程数。
    batch_size: int = 16
    steps: int = 100000            # 设置训练的总步数  
    eval_freq: int = 10000         # 设置训练过程中 评估的频率   20_000 只是为了让数字更易读，下划线并不会改变数字的实际值
    log_freq: int = 200             # 设置日志记录的频率
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20000         # 设置检查点保存的频率   
    use_policy_training_preset: bool = True     # 是否使用策略的训练预设。如果设置为 True，则使用策略提供的优化器和调度器预设
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # __post_init__ 是 dataclass 提供的一个特殊方法，它在数据TrainPipelineConfig 类实例化之后、返回实例之前被自动调用
    def __post_init__(self):
        self.checkpoint_path = None

    # 检查和设置配置属性的有效性和合理性，确保在训练过程开始前，所有必要的参数都已正确定义
    def validate(self):     # self 代表当前类的实例
        if not self.device:
            logging.warning("No device specified, trying to infer device automatically")
            device = auto_select_torch_device()
            self.device = device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logging.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        # 处理预训练路径
        policy_path = parser.get_path_arg("policy")         # 命令行参数解析获取 策略路径  --policy.path=lerobot/pi0  
        # 如果提供了路径，则加载相关的策略配置
        if policy_path:
            # Only load the policy config
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path           # 将策略路径存储在self.policy.pretrained_path
        
        # 处理恢复运行
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError("A config_path is expected when resuming a run.")
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )
            policy_path = Path(config_path).parent
            self.policy.pretrained_path = policy_path
            self.checkpoint_path = policy_path.parent

        # 如果没有指定作业名称，则根据策略类型或环境类型自动生成名称。
        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        # 检查输出目录是否存在，如果存在且 resume 为 False，则抛出错误。
        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} alreay exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        # 如果输出目录未设置，则根据当前时间和任务名称生成一个新的输出目录。
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        # 数据集检查： 如果数据集的 repo_id 是一个列表，抛出未实现的错误
        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        # 检查是否使用规则策略训练，如果不使用，则验证优化器和调度器是否已设置
        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict:
        return draccus.encode(self)

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    # 用于从预训练的配置文件中加载训练管道配置
    # 帮助用户从多个来源（本地路径、文件、远程模型）加载训练时的配置文件，便于继续之前的训练或者进行新的训练
    @classmethod
    def from_pretrained(
        cls: Type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,        # 指定预训练模型的路径或名称，可以是本地路径或远程名称
        *,
        force_download: bool = False,               # 如果为 True，则强制重新下载模型，即使本地已经存在
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,        # 预训练模型缓存目录。
        local_files_only: bool = False,             # 如果为 True，则只下载本地文件，不从网上下载。
        revision: str | None = None,
        **kwargs,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)     # 将 pretrained_name_or_path 转换为字符串格式
        config_file: str | None = None
        # 如果 model_id 是一个目录，检查该目录下是否存在 TRAIN_CONFIG_NAME（即 train_config.json）
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        # 如果 model_id 是一个文件，直接将 config_file 设置为该文件。
        elif Path(model_id).is_file():
            config_file = model_id
        # 如果 model_id 是一个远程名称(既不是目录也不是文件)，则从 HuggingFace Hub 下载配置文件。
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        cfg = draccus.parse(cls, config_file, args=cli_args)

        return cfg
