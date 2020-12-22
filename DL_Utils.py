
import torch
import torchvision
from timeit import default_timer as timer
import sys

class Validation: 
    def __init__(self, data_loader, device, criterion):
        self.data_loader = data_loader
        self.device = device
        self.criterion = criterion
    
    # Validation Method for the model
    def validate(self, model):
        val_loss = 0
        correct_pixel = 0
        total_pixel = 0

        for images, masks, _ in self.data_loader:
            images = images.to(self.device)
            masks = masks.type(torch.LongTensor)
            masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
            masks = masks.to(self.device)

            outputs = model(images)
            val_loss += self.criterion(outputs, masks).item()

            _, predicted = torch.max(outputs.data, 1)
            correct_pixel += (predicted == masks).sum().item()

            b, h, w = masks.shape
            batch_total_pixel = b*h*w

            total_pixel += batch_total_pixel

        acc = correct_pixel/total_pixel
        return val_loss, acc, len(self.data_loader)


# TODO
class Train:
    def __init__(self, data_loader, device, criterion, optimizer, validation=None, scheduler=None):
        self.data_loader = data_loader 
        self.device = device 
        self.criterion = criterion 
        self.optimizer = optimizer 
        self.validation = validation 
        self.scheduler = scheduler
    
    # Training of the Model
    def start(self, model, epochs, batch_extender, validation_on, scheduler_on, loss_print_per_epoch):
        if validation_on and self.validation is None:
            sys.exit('Validation function is not provided. It cannot validate!')
        
        if scheduler_on and self.scheduler is None:
            sys.exit('No scheduler provided. Create instance with a scheduler!')

        total_steps = len(self.data_loader)
        print(f"{epochs} Epochs & {total_steps} Total Steps per Epoch")

        start_time = timer()
        print(f'Training Started in {start_time} sec.')
        model.train()
        try:
            batch_step = 0
            for epoch in range(epochs):
                for idx, (images, masks, _) in enumerate(self.data_loader, 1):
                    images = images.to(self.device)  # Sends to GPU
                    masks = masks.type(torch.LongTensor)
                    masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3])
                    masks = masks.to(self.device)    # Sends to GPU

                    # Forward pass
                    predicts = model(images)
                    loss = self.criterion(predicts, masks)

                    # This doubles our batch size
                    if batch_extender:
                        if batch_step == 0:
                            self.optimizer.zero_grad()
                            loss.backward()
                            batch_step = 1
                        elif batch_step == 1:
                            loss.backward()
                            self.optimizer.step()
                            batch_step = 0
                    else:
                        # Backward and optimize
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    if idx % int(total_steps/loss_print_per_epoch) == 0:
                        if validation_on:
                            acc = 0
                            # Validation Part
                            model.eval()
                            with torch.no_grad():
                                validation_loss, validation_accuracy, length_validation = self.validation.validate(model)
                            model.train()
                        valstr = f'\tValid. Loss: {(validation_loss/length_validation):.4f}\tValid. Acc.: {validation_accuracy * 100:.3f}%' if validation_on else ''
                        print(f'Epoch: {epoch + 1}/{epochs}\tSt: {idx}/{total_steps}\tLast.Loss: {loss.item():4f}{valstr}')
                if scheduler_on:
                    self.scheduler.step(acc) # -> ReduceLROnPlateau

                    # scheduler.step() # -> StepLR
                    # print(f'LR: {scheduler.get_last_lr()}')
        
        except Exception as e:
            print(e)

        print('Execution time:', '{:5.2f}'.format(timer() - start_time), 'seconds')
        return model


# TODO
def build_model():
    return ''