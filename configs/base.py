from easydict import EasyDict as edict

config = edict()

config.seed = 42
config.output = "./output"
config.augmentation = None

config.batch_size = 4
config.num_workers = 0

config.network = "vit_s_qs"
config.dropout = 0.0
config.fp16 = False
config.embedding_size = 512

config.margin_list = [1.0, 0.5, 0.0]
config.interclass_filtering_threshold = 0
config.num_classes = 28

config.optimizer = "adamw"
config.lr = 1e-4
config.weight_decay = 5e-4

config.num_image = 1000
config.warmup_epoch = 1
config.num_epoch = 2

config.sample_rate = 1.0
config.alpha = 0.5

config.gradient_acc = 1
config.verbose = 10

config.save_all_states = False
config.dali = False

config.val_targets = []
config.using_wandb = False