import argparse
from data.dataset import DatasetMetr
from metrics.mae import masked_mae
from metrics.mape import masked_mape
from metrics.rmse import masked_rmse
from model import SUSTeR
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import uuid



DATA_PATH = Path('data/METR-LA/')
BATCH_SIZE = 64
SHUFFLE = True
LR = 1e-3
L2_WEIGHT = 1e-5
GRAD_MAX_CLIP = 5




def main():
   
    ap = argparse.ArgumentParser()
    ap.add_argument('--dropout', help= 'The dropout from the created dataset.', default='0.99', type=str)
    ap.add_argument('--slurmid', help= 'Passing a slurm id which will help to identify the learned model.', default=None)
    ap.add_argument('--graphnodes', help= 'The number of graph nodes.', default= 10, type = int)
    ap.add_argument('--context', help='If Xbar is dependent on the context.', default=True, type= bool)
    ap.add_argument('--factor', help= 'The fraction of neurons that the wrapped STGCN can work with. None is equal is an avergae of the graphs as described in the paper.', default = 1, type= int)
    ap.add_argument('--train_percentage', help= 'Define with how much of the training data SUSTeR will be trained.', default=1., type=float)
    ap.add_argument('--reps', help= 'Number of repetitions', default= 1, type= int)
    ap.add_argument('--embed_dim', help='The latent embedding dimension', default=32, type= int)
    ap.add_argument('--epochs', default=50, type = int)


    args = ap.parse_args()

    if args.slurmid is not None:
        model_id = f'slurm{args.slurmid}'
    else :
        model_id = f'model{str(uuid.uuid1())}'

    check_point_dir = Path(f'models/{model_id}')
    check_point_dir.parent.mkdir(exist_ok= True)
    check_point_dir.mkdir()


    for model_index in range(args.reps):
        execute_experiment(args.dropout, args.graphnodes, args.embed_dim, args.epochs, check_point_dir / f'm{model_index}', context= args.context, factor= args.factor, t_percent= args.train_percentage)


def execute_experiment(dropout_rate, n_proxy, embed_dim, epochs, save_path: Path, mean_beta=.3, context = True, factor = 1, t_percent = 1.):

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'

    data_file = DATA_PATH / f'DROP{dropout_rate}' / 'data_in12_out1.pkl'
    index_file = DATA_PATH / f'DROP{dropout_rate}' / 'index_in12_out1.pkl'

    training_dataset = DatasetMetr(data_file, index_file, 'train', t_percent)
    validation_dataset = DatasetMetr(data_file, index_file, 'valid')
    test_dataset = DatasetMetr(data_file, index_file, 'test')

    training_loader = DataLoader(training_dataset, BATCH_SIZE, SHUFFLE, num_workers= 2, pin_memory= False)
    validation_loader = DataLoader(validation_dataset, BATCH_SIZE, False, num_workers= 2, pin_memory= False)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, False, num_workers= 2, pin_memory= False)

    ## Constant by construction of the data set 
    n_channels = 5 
    output_target = 207


    suster = SUSTeR(n_proxy, embed_dim, n_channels, output_target, context, norm_adj= False, factor = factor)

    optimizer = torch.optim.Adam(suster.parameters(),lr = LR, weight_decay=  L2_WEIGHT)
    suster.to(device)

    loss_per_epoch = []
    val_loss_per_epoch = []
    
    for _ in (epochbar := trange(0,epochs,1, position=0)):
        losses = []
        suster.train()


        for train_input, train_output in (pbar := tqdm(training_loader, position= 1, leave= False)):
            optimizer.zero_grad()

            rand_indices = torch.randperm(207)

            train_prediction, mean_prediction = suster(train_input[:,:, rand_indices].to(device), train_output[:,0, :, -4:].to(device))
           
            loss = masked_mae(train_prediction[:, [0]], train_output[..., [0]].to(device)) 
            mean_loss = masked_mae(mean_prediction[:, [0]], train_output[..., [0]].to(device)) 
            
            loss = loss + mean_beta * mean_loss
            loss.backward()
            
            torch.nn.utils.clip_grad.clip_grad_norm_(suster.parameters(), max_norm=GRAD_MAX_CLIP)
            optimizer.step()

            losses.append(loss.item())
            pbar.set_description(f'Train Loss: {losses[-1]:.3f}')

        loss_per_epoch.append(np.mean(losses))

        metrics = []
        suster.eval()
        for val_input, val_output in validation_loader:
            val_prediction, _ = suster(val_input.to(device), val_output[:,0, :, -4:].to(device))
            metric = masked_mae(val_prediction[:, [0]], val_output[..., [0]].to(device))
            metrics.append(metric.item())
        
        val_loss_per_epoch.append(np.mean(metrics))
        epochbar.set_description(f'V: {val_loss_per_epoch[-1]:.2f}')


        if len(val_loss_per_epoch) < 2 or val_loss_per_epoch[-1] < np.min(val_loss_per_epoch[:-1]):
            if not save_path.is_dir():
                save_path.mkdir()

            if (save_path / 'best_model.pth').is_file():
                os.remove(save_path / 'best_model.pth')
            torch.save(suster.state_dict(), save_path / 'best_model.pth')
    
    del suster

    best_model = SUSTeR(n_proxy, embed_dim, n_channels, output_target, context, factor= factor)
    best_model.load_state_dict(torch.load(save_path / 'best_model.pth'))

    test_model(best_model, test_loader, test_dataset.scaler_mean, test_dataset.scaler_scale , device)


def test_model(model, dataloader, scaler_mean, scaler_scale, device):


    mae_metrics = []
    rmse_metrics = []
    mape_metrics = []

    model.eval()
    model.to(device)
    for input, output in dataloader:
        prediction, _ = model(input.to(device), output[:,0, :, -4:].to(device))
        
        truth = output[..., [0]].to(device) * scaler_scale.to(device) + scaler_mean.to(device)
        prediction = prediction[:, [0]] * scaler_scale.to(device) + scaler_mean.to(device)

        mae = masked_mae(prediction, truth)
        rmse = masked_rmse(prediction, truth)
        mape = masked_mape(prediction, truth)
               
        mae_metrics.append(mae.item())
        rmse_metrics.append(rmse.item())
        mape_metrics.append(mape.item())
    
        
    print(f'Performance: Test MAE {np.mean(mae_metrics):.3f} RMSE {np.mean(rmse_metrics):.3f}  MAPE {np.mean(mape_metrics):.3f}')



        

if __name__ == '__main__':
    main()
    

