import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torch
import plateModule
from tqdm import tqdm


class PlateDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(os.path.join(path, 'labels\\data.csv'))
        self.img_path = os.path.join(path, 'images')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.img_path, self.data.iloc[item, 1]))
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1).view(3, 48, 128)
        label = self.data.iloc[item, 2]
        label = label2tensor(label)
        return img, label


def label2tensor(label: str,):
    label_index = [plateModule.char_dict[c] for c in label]
    label_tensor = torch.tensor(label_index)
    label_len = torch.tensor(label_tensor.shape[0], )
    label_tensor = torch.nn.functional.pad(label_tensor, (0, 9 - label_len), value=plateModule.char_len - 1)
    return torch.cat([label_tensor, label_len.unsqueeze(0)], dim=-1)


'''ctc_loss = torch.nn.CTCLoss()
def loss_f(output, label, label_len):
    output = torch.log_softmax(output, dim=1)

    loss = ctc_loss(output, label, torch.tensor(14), torch.tensor(label_len))
    return loss'''


def train(model: plateModule.PlateOcr, dataset: Dataset, epochs=10, lr=0.001, device='cpu', eval_epoch=3,
          model_save_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CTCLoss(blank=plateModule.char_len-1)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size, num_workers=8, persistent_workers=True)

    train_loss, test_loss = [], []
    model.to(device)

    for epoch in range(epochs):
        train_loss.append(0.0)
        test_loss.append(0.0)

        model.train()
        pbar = tqdm(train_dataloader, unit="batch", desc=f'epoch [{epoch+1}/{epochs}] train', colour='#66ccff')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _batch_size = inputs.size(0)
            input_lengths = torch.full((_batch_size,), 14, dtype=torch.long)
            outputs = outputs.view(_batch_size, -1, plateModule.char_len)
            outputs = outputs.permute(1, 0, 2).contiguous()
            outputs = torch.log_softmax(outputs, dim=2)

            loss = criterion(outputs, labels[:, :-2], input_lengths, labels[:, -1])

            loss.backward()
            optimizer.step()

            train_loss[-1] += loss.item() * _batch_size
            pbar.set_postfix({'loss': f'{train_loss[-1]:.4f}'}, refresh=False)
        train_loss[-1] /= len(train_dataset)

        if epoch % eval_epoch != 0 and epoch != epochs-1:
            test_loss[-1] += test_loss[-2]
            continue

        model.eval()
        pbar = tqdm(test_dataloader, unit="batch", desc=f'epoch [{epoch + 1}/{epochs}] test ', colour='#66ff66')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _batch_size = inputs.size(0)
            input_lengths = torch.full((_batch_size,), 14, dtype=torch.long)
            outputs = outputs.view(_batch_size, -1, plateModule.char_len)
            outputs = outputs.permute(1, 0, 2).contiguous()
            outputs = torch.log_softmax(outputs, dim=2)

            loss = criterion(outputs, labels[:, :-2], input_lengths, labels[:, -1])

            test_loss[-1] += loss.item() * _batch_size
            pbar.set_postfix({'loss': f'{test_loss[-1]:.4f}'}, refresh=False)
        test_loss[-1] /= len(test_dataset)

    save = input(' save model?[n/model name]')
    if save == 'n':
        return
    torch.save(model.state_dict(), os.path.join(model_save_path, save+'.pt'))
    df = pd.DataFrame({"train_loss": train_loss, "test_loss": test_loss, })
    df.to_csv(os.path.join(model_save_path, save+'.csv'), index=False)


if __name__ == '__main__':

    dataset_path = 'dataset\\CBLPRD-330k'
    model_save_path = 'ocrmodel'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    eval_epoch = 5

    dataset = PlateDataset(dataset_path)
    model = plateModule.PlateOcr()
    model.load_state_dict(torch.load('ocrmodel\\ocr-4-1.pt'))

    train(model, dataset, epochs=25, device=device, model_save_path=model_save_path,
          eval_epoch=eval_epoch,
          lr=0.0001
          )



