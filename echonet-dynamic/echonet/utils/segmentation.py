"""Functions for training and running segmentation."""

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm

import echonet


@click.command("segmentation")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.segmentation.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.segmentation.__dict__[name]))),
    default="deeplabv3_resnet50")
@click.option("--pretrained/--random", default=False)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--save_video/--skip_video", default=False)
@click.option("--num_epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-5)
@click.option("--weight_decay", type=float, default=0)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,

    model_name="deeplabv3_resnet50",#https://debuggercafe.com/semantic-segmentation-using-pytorch-deeplabv3-resnet50/
    pretrained=False,
    weights=None,

    run_test=False,
    save_video=False,
    num_epochs=50,
    lr=1e-5,
    weight_decay=1e-5,
    lr_step_period=None,
    num_train_patients=None,
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
):
    """Trains/tests segmentation model.

    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/segmentation/<model_name>_<pretrained/random>/.
        model_name (str, optional): Name of segmentation model. One of ``deeplabv3_resnet50'',
            ``deeplabv3_resnet101'', ``fcn_resnet50'', or ``fcn_resnet101''
            (options are torchvision.models.segmentation.<model_name>)
            Defaults to ``deeplabv3_resnet50''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to False.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        save_video (bool, optional): Whether to save videos with segmentations.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training  The function trains for the specified number of epochs and after each epoch runs a full validation step. It also keeps track of the best performing model (in terms of validation accuracy), and at the end of training returns the best performing model. After each epoch, the training and validation accuracies are printed.
            Defaults to 50.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-5.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 0.
        lr_step_period (int or None, optional): Period of learning rate decay ???SMISAO DECAYANJA LEARNING RATEA???
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to math.inf (never decay learning rate).
        num_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation", "{}_{}".format(model_name, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations -> SEMINAR
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.segmentation.__dict__[model_name](pretrained=pretrained, aux_loss=False)

    #vRAĆA ZADNJI CLASSIFIER                //in_channels-> prvi argument-> broj kanala ulazne slike, VELICINA KERNELA 2D KONVOLUCIJE, VEĆINA FILTERA KONVOLUCIJE
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    #uCITAJ PRETRENIRANE TEZINE AKO JE TAKO SPECIFICRANO
    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer -> STOHASTICKI ALGORITAM GRADIJEJTNOG SPUSTA, OPTIMIZER DEFINIRA METODLOGIJU DOLASKA DO OPTIMALNOG RJESENJA/MINIMAZIJA FUNKCIJE KOSTANJA
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)#MIJENJA LEARNING RATE POSTEPENO S ZADANIM PERIODOM-> STA IDU DALJE EPOHE CILJ JE SMANJIVAT LR ILI KADA ZAPENEMO TOKOM TRENIRANJA -> NPR NE SMANJUJE NAM SE CV GRESKA ILI VIDIMO DA JE LR PREVELIK itd-> TREBALI BI SE PRIBLIZAVAT OPTIMUMU

    # Compute mean and std ZA TRAINING SET
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std
              }

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint= u njoj zapisano SVO STANJE MODELA U TOM TRENUTKU TRENIRANJA
            #Vidi jel vec postoji zapisan checkpoint na kojem smo stali ili treba ic ispocetka
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        #Treniramo u for petlji ili preostale epohe ta su ostale ili od 0
        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            #u svako j EPOHI RADIMO PRVO TRENIRANJE OPET KROZ CIJELI DATASET I ONDA VALIDACIJU
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):#SEMINAR
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                #The most important argument of DataLoader constructor is dataset, which indicates a dataset object to load data from. PyTorch supports two different types of datasets:
                #map-style datasets
                #iterable-style datasets.
                #A map-style dataset is one that implements the __getitem__() and __len__() protocols, and represents a map from (possibly non-integral) indices/keys to data samples.
                #ovo je naš slučaj, mi imamo ovakvi tip dataseta -> DEFINIRANO U /datasets/echo.py

                dataloader = torch.utils.data.DataLoader(                           #SEMINAR
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                #POKRENI EPOHU
                loss, large_inter, large_union, small_inter, small_union = echonet.utils.segmentation.run_epoch(model, dataloader, phase == "train", optim, device)
                overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())#DICE SIMILIRATY COEFFICENT-> DAJE STATISTIČKU MJERU SLIČNOSTI PODATAKA OD 2 OBJEKTA/TESNROA
                large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                    phase,
                                                                    loss,
                                                                    overall_dice,
                                                                    large_dice,
                                                                    small_dice,
                                                                    time.time() - start_time,
                                                                    large_inter.size,
                                                                    sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                    sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                    batch_size))
                #SCHDULER.STEP()->updates the parameters
                f.flush()
            scheduler.step()

            #IZASLI IZ for petlje -> gotovi i treniranje i validacija ZA TRENUTNU EPOHU
            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            #cilj-> minimizacija funkcije koštanja
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        #IZASLI IZ FOR PETLJE KOJA PROLAZI EPOHE -> PROSLI SVE EPOHE -> ZAVRŠENO TRENIRANJE(ne validacija i testiranje)
        # Load best weights-> uzmi optimalni trenutak ODNOSNO NAJBOLJU EPOHU-> s najmanjim greškama epoha
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            #ukoliko želimo odradit VALIDACIJU I TESTIRANJE
            # split u for petlji definirana da ćemo uzet dedicated dataset za validaciju i za testiranje
            # nećemo radit opet sve epohe na validaciji pa se njoj fittat nego ćemo KOD VALIDATION I TEST DATASET proć SAMO JEDNOM(1 epoha) i vidit PERFORMANSE NAŠEG PRETHODNO ISTRENIRANOG MODELA NA novim podacima koji nisu bili ukljuceni u treniranje odnosno koliko dobro GENEIRALIZIRA(signalizacija overfitta)
            # Run on validation and test
            for split in ["val", "test"]:
                dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                #run_epoch TRENIRA + DAJE PARAMETRE ZA EVALUACIJU
                loss, large_inter, large_union, small_inter, small_union = echonet.utils.segmentation.run_epoch(model, dataloader, False, None, device)

                overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)
                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))

                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)), echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(large_inter, large_union, echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(small_inter, small_union, echonet.utils.dice_similarity_coefficient)))
                f.flush()

    # Saving videos with segmentations -> buduci da je testiranje BILO ZADNJE -> NA NJEMU JE ODRADENA EPOHA I NJEGOVI PODACI PREDSTAVLJAJU SEGMENTIRANE VIDEE -> na njima je odrađena segmentacija u videu
    dataset = echonet.datasets.Echo(root=data_dir, split="test",
                                    target_type=["Filename", "LargeIndex", "SmallIndex"],  # Need filename for saving, and human-selected frames to annotate
                                    mean=mean, std=std,  # Normalization -> ZA NORMALIZACIJU VRIJEDNOSTI MEAN I STD SU IZRACUNATE NA TRAINING SETU -> get_mean_and_std funkcija
                                    length=None, max_length=None, period=1  # Take all frames
                                    )
    #u ovom dataloader se koristi video_collate_fn koji će ako vidimo video_collate_fn mergat videa po dimneziji frameova
    #Naime -> dataset __getitem() metdoa vraća format u obliku [c,f,h,w] ali će video collate njih mergat po dimneziji frameova tako da dobije kontinuiran video koji će se spremit
    #Zato je dole prva dimenzija frame a ne chanel odnosno video_coolate_fn radi transofmraciju iz [c,f,h,w] -> [f,c,h,win]
    #data loader će za svaki item iz batcha vratit ono sta smo specificirali u datasets.Echo konsturktoru
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=num_workers, shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)
    #batch = koliko videa ćemo uzet kod treniranja i obrade u modelu
    # video_collate_fn ce spojit sve videe u jednom batch_sizeu u jedan video radi bržeg učitavanja
    #x će zapravo sadržažavat 10 spojenih videa po dimenziji frameova odnosno sve spojene fameove od 10 videa
    # Save videos with segmentation
    if save_video and not all(os.path.isfile(os.path.join(output, "videos", f)) for f in dataloader.dataset.fnames):
        # Only run if missing videos

        model.eval()#GOTOVI S TRENIRANJEM-> EVALUACIJA MOELA ZAPOČINJE

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                #iteriramo po svim batchevima u datasetu koji je ucitan
                #ova 3 parametra koja se vracaju su psecificirani u video_collate_fn funkciji 
                #prvi parmaetar su spojeni videi u jedan, drugi parametar su (filename,large_index,small_index) za svaki video koji je bio spojen i length su duljine svakog videa(kasnije ce bit pretvorene u offset parametar)
                for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
                    #X JE VIDEO OBJEKT-> VIDEO JE 4D TENSOR-> [f,c,h,w] nakon video_coolate_fn
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together
                    #All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
                    #Vidimio da model ocekuje format [f,3,h,w]
                    #modelu dajemo 10 po 10 frameova
                    # #batch_size nam definira BROJ VIDEA KOJE ĆEMO UZET KOD UČITAVANJA PODATAKA + ROJ FRAMEOVA KOJE ĆEMO DAT MODELU                                                                                  #x,shape[0] je broj frameova ukupni
                    y = np.concatenate([model(x[i:(i + batch_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], batch_size)])

                    start = 0
                    x = x.numpy()
                    #budući da x i y unutar sebe imaju spojenih 10 videa onda je potrebno dvojit taj spojeni niz u zasebne videe akon šta je model vratija rezultat
                    #to radimo tako šta svaki file za sebe ima offset odnosno svoju velicinu
                    #za svaki file/video od njih 10 u batchu će vratit filename i offset njegov
                    for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                        # Extract one video and segmentation predictions(logit-> 0 ili 1)
                        #to extractamo jer za svaki file/video u batchu imamo njegove offsete
                        #na taj nacin obraujemo svaki bideo posebno
                        video = x[start:(start + offset), ...]
                        logit = y[start:(start + offset), 0, :, :]#samo 1 kanal ima -> 0 ili 1

                        # Un-normalize video-> vrati nazad u prave vrijendosti po formuli za jedinicnu normalnu varijablu
                        video *= std.reshape(1, 3, 1, 1)#pretvori u tensro ovih dimenzija
                        video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  # pylint: disable=W0612
                        assert c == 3#r,g,b

                        # Put two copies of the video side by side
                        video = np.concatenate((video, video), 3)#concatenatmo po zadnjoj osi-> zadnja os je width-> po stupcima concatenate

                        # If a pixel is in the segmentation, saturate blue channel -> PLACA JE NA PRVOM INDEKSU= 255 VRIJEDNOST -> INVERTIRAN REDSOLIJED KOD UCITAVANJA
                        # Leave alone otherwise
                        #drugi video cemo segmentirat -> uzimamo raspon w: pa nadalje -> to će bit naš video koji cemo bojat/segmentirat po onom sta nam je model da, ovo prije je stvarni video
                        video[:, 0, :, w:] = np.maximum(255. * (logit > 0), video[:, 0, :, w:])  # pylint: disable=E1111

                        # Add blank canvas under pair of videos-> za graf

                        video = np.concatenate((video, np.zeros_like(video)), 2)#spoji po osi 2 -> visini -> po retcima-> ovi blnk canvas ide ispod videa

                        # Compute size of segmentation per frame
                        size = (logit > 0).sum((1, 2))#ZA SVAKI FRAME CE SUMA JEDINICA ODGOVOARAT VELICINI VENTRICULA U PIXELIMA-> IMAMO POVRSINU

                        # Identify systole frames with peak detectiON
                        #SORITRANO UZLAZNO-> PO OVOJ FORMULI SA EXPR PROCIJENIMO DI CE BIT SYSTOLIC A DI DIJASTOLIC FRAME
                        trim_min = sorted(size)[round(len(size) ** 0.05)]#exponenntiramo
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])

                        # Write sizes and frames to file
                        for (frame, s) in enumerate(size):
                            #upis u size.csv
                            g.write("{},{},{},{},{},{}\n".format(filename, frame, s, 1 if frame == large_index[i] else 0, 1 if frame == small_index[i] else 0, 1 if frame in systole else 0))

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf"))
                        plt.close(fig)

                        # Normalize size to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate the frames in this video
                        for (f, s) in enumerate(size):

                            # On all frames, mark a pixel for the size of the frame
                            video[:, :, int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))] = 255.

                            if f in systole:
                                # If frame is computer-selected systole, mark with a line
                                video[:, :, 115:224, int(round(f / len(size) * 200 + 10))] = 255.

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on
                                    x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf
                            d = dash(115, 224)

                            if f == large_index[i]:
                                # If frame is human-selected diastole, mark with green dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 225, 0]).reshape((1, 3, 1))
                            if f == small_index[i]:
                                # If frame is human-selected systole, mark with red dashed line on all frames
                                video[:, :, d, int(round(f / len(size) * 200 + 10))] = np.array([0, 0, 225]).reshape((1, 3, 1))

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.disk((int(round(115 + 100 * s)), int(round(f / len(size) * 200 + 10))), 4.1)

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.

                        # Rearrange dimensions and save
                        # vrati u c,f,w,h kao sta je u __get_item__ -> POGRANA U TOM FORMATU
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        echonet.utils.savevideo(os.path.join(output, "videos", filename), video, 50)

                        # Move to next video u batchu
                        start += offset


def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.
    n = 0

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0

    model.train(train)

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    with torch.set_grad_enabled(train):
        #TQDM JE ZA PROGRESS BAR
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            #ovo odgovara formatu koji vraća __get_item__ funkcija-> prvi clan je video, drugi clan je tuple s odredenim target parametrima koji su specificirani
            #defualtno ce ako ne specificiramo svoju collate funkciju vratit tensore koji ce unutar sebe imat podatke za svaku instancu
            #(large_frame, small_frame, large_trace, small_trace) podaci su isto u oblika niza odnosno sadrze sve ove parametre za svaki video iz batcha
            for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader:
                #EPOHA-> IDEMO KROZ SVE PODATKE/VIDEE
                #u jednoj iteraciji dataloadera dobivamo batch_size videa i njihovih parametara
                # Count number of pixels in/out of human segmentation
                #pos broji pozitivne pixele-> UKUPAN BROJ I OD LARGE I OD SMALL
                #large_trace(DIJASTOLA) predstavlja oznacenu masku videa s pixelima u 1 tamo di se nalazi i 0 di nije, ista stvar i za small_trace(SISTOLA)
                pos += (large_trace == 1).sum().item()#item() kopira u skalar, zbrajamo samo one di je 1
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()

                #to mice podatke u memoriju zadanog device argumenta-> ili u memeoriju kojoj moze pristupit CPU ili GPU
                # Count number of pixels in/out of computer segmentation
                #As mentioned before, np.ndarray object does not have this extra "computational graph" layer and therefore, when converting a torch.tensor to np.ndarray you must explicitly remove the computational graph of the tensor using the detach() command.
                #0 specificira os po kojoj zbrajamo-> 0 -> zbrajamo po STUPCIMA
                #AL OVO NE BI TRIBALO IC NAKON STA MODEL VRATI REZULTATE PREDIKCIJE
                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                # Run prediction for diastolic frames-> large frame ide u model and compute loss
                #ucitaj na GPU podatke/tensor
                large_frame = large_frame.to(device)
                large_trace = large_trace.to(device)
                y_large = model(large_frame)["out"]#to je pytorch tensoro objekt pa ga trib detahc da bi se moga s numpy koristit
                #Function that measures Binary Cross Entropy between target and input logits
                #TARGET JE ONO STA SMO ZELJELI DOBIT A INPUT ONO STA JEDAO MODEL
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                #The reason for requiring explicit .cpu() is that CPU tensors and the converted numpy arrays share memory-> prebicvianje u host memeory
                large_inter += np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                                                                                                                                                #SUM()=> If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple -> zbraja sve osi -> ovo odgovaora zapravo ciloj matrici
                large_inter_list.extend(np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                large_union_list.extend(np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device)
                small_trace = small_trace.to(device)
                y_small = model(small_frame)["out"]
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                small_inter += np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Take gradient step if training
                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    #BACKWARD PROPAGACIJA I RACUNANJE LOSSA NA OSTALIN NEURONIMA UNAZAD
                    # In backprop, the NN adjusts its parameters proportionate to the error in its guess. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (gradients), and optimizing the parameters using gradient descent
                    #https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step
                    loss.backward()
                    optim.step()

                # Accumulate losses-> DODAJ NA PRETHODNE and compute baselines
                total += loss.item()
                n += large_trace.size(0)#GLEDAJ X AXIS-> BROJ REDAKA
                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                # Show info on process bar
                pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(total / n / 112 / 112, loss.item() / large_trace.size(0) / 112 / 112, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    return (total / n / 112 / 112,#OVO JE FORMULA ZA LOSS
            large_inter_list,
            large_union_list,
            small_inter_list,
            small_union_list,
            )


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))#zamijeni prvu(channels) i drugu(frames) os tako da rezultat bude [frame, channel, h, w]

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i
