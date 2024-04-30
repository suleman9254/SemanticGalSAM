from torch.utils.data import DataLoader
from modules.dataset import SAMDataset
from modules.model import SAM
from modules.lora import fetch_lora_regex
from modules.utils import reset_seed, seed_worker, generator

from accelerate import Accelerator
from torchvision.transforms import v2
from torchmetrics import Accuracy, F1Score, JaccardIndex, MetricCollection

import optuna

seed = 42

num_classes = 5
image_size = 512
vit_patch_size = 8
pretrained_path = 'facebook/sam-vit-base'
lora_regex, normal_layers = fetch_lora_regex('vision_encoder')

scheduler_patience = 5
scheduler_threshold = 0.01
early_stopping_patience = 10
early_stopping_threshold = 0.01

metric = MetricCollection([Accuracy(task='multiclass', num_classes=num_classes, average='macro'), 
                           F1Score(task="multiclass", num_classes=num_classes, average='macro'),
                           JaccardIndex(task='multiclass', num_classes=num_classes, average='macro')])

transforms = v2.Compose([v2.RandomRotation(degrees=180)])
root, annFile = '../data/train', '../data/annotations/train.json'
trainset = SAMDataset(root, annFile, image_size, transforms=transforms, means=[-1.8163, -1.9570, -1.7297], stds=[0.8139, 0.4834, 0.4621])

root, annFile = '../data/val', '../data/annotations/val.json'
valset = SAMDataset(root, annFile, image_size, means=[-1.8168, -1.9573, -1.7303], stds=[0.8127, 0.4824, 0.4602])

def objective(trial):
    global model
    reset_seed(n=seed)
    generator.manual_seed(seed)

    lr = trial.suggest_float("lr", low=5e-5, high=5e-3, log=True)
    batch_size = trial.suggest_int("batch_size", low=2, high=22, step=5)
    epochs = trial.suggest_int("epochs", low=10, high=200, step=25)
    lora_rank = trial.suggest_int("lora_rank", low=4, high=32, step=4)
    weight_decay = trial.suggest_float("weight_decay", low=0.01, high=0.03, log=True)
    lambda_dice = trial.suggest_float("lambda_dice", low=0.05, high=1, log=True)
    lora_alpha = lora_rank * 2

    save_path = f'saves/lr_{lr}_batch_size_{batch_size}_epochs_{epochs}_rank_{lora_rank}_weight_decay_{weight_decay}_lambda_dice_{lambda_dice}.ckpt'

    accelerator = Accelerator(gradient_accumulation_steps=batch_size)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, worker_init_fn=seed_worker, generator=generator)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=generator)

    model = SAM(pretrained_path, 
            num_classes=num_classes, 
            image_size=image_size, 
            vit_patch_size=vit_patch_size, 
            lora_regex=lora_regex, 
            normal_regex=normal_layers, 
            lora_rank=lora_rank, 
            lora_alpha=lora_alpha)

    cfg = {'trainloader': trainloader, 
           'valloader': valloader, 
           'epochs': epochs, 
           'lr': lr,
           'save_path': save_path, 
           'accelerator': accelerator, 
           'metric': metric, 
           'weight_decay': weight_decay, 
           'scheduler_patience': scheduler_patience, 
           'scheduler_threshold': scheduler_threshold, 
           'lambda_dice': lambda_dice, 
           'early_stopping_threshold': early_stopping_threshold, 
           'early_stopping_patience': early_stopping_patience,
           'monitored_metric': 'MulticlassJaccardIndex',
           'wandb': False}

    vScores = model.fit(cfg)
    return vScores['MulticlassJaccardIndex']

save_path = 'saves/best_optimizer'
def save_model(study, trial):
    if study.best_trial == trial:
        model.save(save_path)

study = optuna.create_study(study_name="optimizing_parameters", 
                            direction='maximize', 
                            load_if_exists=True, 
                            storage="sqlite:///optimizer_sql.db")
study.optimize(objective, n_trials=50, callbacks=[save_model])
