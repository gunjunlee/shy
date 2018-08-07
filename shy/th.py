import time
import copy
import os
import torch
from tqdm import tqdm
from torchvision import datasets

class Trainer():
    def __init__(self):
        pass

    def make_dataloaders(self, data_dir, data_transforms, phases=['train', 'val', 'test'], batch_size=1, num_workers=4, shuffle=True):
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in datasets}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) for x in datasets}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in datasets}
        self.class_names = self.image_datasets['train'].classes
        self.phases = phases

    def train_model(self, model, criterion, optimizer, scheduler, num_epochs, model_save_path='./ckpt/model.pth'):

        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in self.phases:
                if phase == 'train':
                    scheduler.step()
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in tqdm(self.dataloaders[phase]):

                    inputs = inputs.float()
                    inputs, labels = inputs.cuda(), labels.cuda()

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs).view(-1, 2)
                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs.squeeze(-1), labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)

                    running_corrects += torch.sum(preds == labels.data)
                    
                epoch_loss = running_loss/self.dataset_sizes[phase]
                epoch_acc =  running_corrects.item()/self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), model_save_path)
            print()
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)

        return model